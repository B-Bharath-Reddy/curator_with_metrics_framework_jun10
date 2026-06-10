from __future__ import annotations

import os
from contextlib import asynccontextmanager

# ADD BackgroundTasks for post-response BigQuery insert

from fastapi import FastAPI, HTTPException, BackgroundTasks
from opentelemetry.trace import Status, StatusCode, SpanKind
from opentelemetry import trace as otel_trace

import logging

from app.config import settings
from app.curator_models import (
    CuratorReviewRequest,
    CuratorFeedbackResponse,
    ReflectorRow,
)
from app.curator_service import curate_feedback
from app.curator_input_store import bigquery_input_store
from app.curator_output_store import curator_output_store
from app.observability import setup_phoenix, get_tracer, shutdown_phoenix


logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    phoenix_enabled = os.getenv("PHOENIX_ENABLED", "false").lower() == "true"

    if phoenix_enabled:
        setup_phoenix(app)

    try:
        yield
    finally:
        if phoenix_enabled:
            shutdown_phoenix()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Curator Agent — normalizes Reflector events and builds DPO training examples",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {
        "status":                   "ok",
        "app":                      settings.app_name,
        "version":                  settings.app_version,
        "bigquery_input_available": bigquery_input_store.is_available(),
        "bigquery_output_available": curator_output_store.is_available(),
    }



def store_curator_output_background(
        response: CuratorFeedbackResponse,
        row: ReflectorRow,
        parent_context,) -> None:
    """
    Runs after the HTTP response is sent.
    BigQuery insert failures are logged but do not fail the API request.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span("curator.store_response",context=parent_context,) as span:
        span.set_attribute("curator.request_id", response.request_id)
        stored = curator_output_store.store(response, row)
        span.set_attribute("curator.output_stored", stored)
        if not stored:
            span.set_status(Status(StatusCode.ERROR, "Curator output insert failed"))
            logger.error("Failed to store curator output in BigQuery. request_id=%s", response.request_id)
        else:
            span.set_status(Status(StatusCode.OK))



@app.post(
    "/v1/ace/curator:prepare-dpo-candidate",
    response_model=CuratorFeedbackResponse,
    summary="Prepare DPO candidate",
    description=(
        "Production endpoint that accepts a request_id, fetches the full Reflector "
        "row from BigQuery, prepares the Curator response, and stores output asynchronously."
    ),
)
async def prepare_dpo_candidate(req: CuratorReviewRequest,background_tasks: BackgroundTasks):
    tracer = get_tracer()
    with tracer.start_as_current_span("curator.workflow", kind=SpanKind.SERVER) as root_span:
        # Set input.value attribute with stringified JSON payload
        root_span.set_attribute("input.value", req.model_dump_json())
        root_span.set_attribute("curator.request_id", req.request_id)
        root_span.set_attribute("span.kind", "server")
        root_span.set_attribute("service.name", "ace-curator-agent")

        try:
            row = await bigquery_input_store.fetch_by_request_id(req.request_id)

            if not row:
                # SET STATUS BEFORE RAISING
                root_span.set_status(Status(StatusCode.ERROR, "No data found"))
                raise HTTPException(
                    status_code=404,
                    detail=f"No data found for request_id={req.request_id}",
                )

            reflector_row = ReflectorRow(**row)


            # Build response without waiting for BigQuery output insert.
            response = await curate_feedback(reflector_row, store_output=False)


            parent_context = otel_trace.set_span_in_context(root_span)
            # Schedule BigQuery output insert after the response is returned.
            background_tasks.add_task(
                store_curator_output_background,
                response,
                reflector_row,
                parent_context,
            )


            # Output BEFORE Status
            root_span.set_attribute("output.value", response.model_dump_json())

            # SET STATUS LAST in success path
            root_span.set_status(Status(StatusCode.OK))

            return response

        except HTTPException:
            # EMPTY HANDLER - status already set before raise
            raise
        except (KeyError, ValueError) as e:
            # SET STATUS BEFORE RAISING
            root_span.record_exception(e)
            root_span.set_status(Status(StatusCode.ERROR, "Validation Error"))
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            # SET STATUS BEFORE RAISING
            root_span.record_exception(e)
            root_span.set_status(Status(StatusCode.ERROR, str(e)))
            raise HTTPException(status_code=500, detail=str(e))







#  MANUAL TESTING endpoint
# POST the full ReflectorRow directly — no BigQuery fetch needed
# Use the real BigQuery row JSON you already have for testing

@app.post(
    "/v1/ace/curator:prepare-dpo-candidate-direct",
    response_model=CuratorFeedbackResponse,
    summary="Prepare DPO candidate directly",
    description=(
        "Developer/testing endpoint that accepts the full Reflector row payload "
        "directly, bypasses BigQuery input fetch, returns the Curator response, "
        "and does not store output in BigQuery."
    ),
)
async def prepare_dpo_candidate_direct(req: ReflectorRow):
    try:
        # Direct endpoint is for local validation with full payload.
        # It bypasses BigQuery input fetch and still exercises the same curator logic.
        return await curate_feedback(req)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
