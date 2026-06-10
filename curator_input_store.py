from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional
import asyncio

from google.cloud import bigquery
from opentelemetry.trace import Status, StatusCode

from app.config import settings
from app.observability import get_tracer

logger = logging.getLogger(__name__)


class BigQueryInputStore:
    """
    Fetches Reflector payload from BigQuery reflection_events table by request_id.

    """

    def __init__(self):
        self.client = None
        self._initialize()

    def _initialize(self) -> None:
        # Fail fast if table ID not configured
        if not settings.bigquery_input_table_id:
            logger.warning(
                "BigQueryInputStore: BIGQUERY_INPUT_TABLE_ID not set. Store unavailable."
            )
            return
        try:
            self.client = bigquery.Client(project=settings.google_cloud_project)
            logger.info(
                "BigQueryInputStore initialized. table=%s",
                settings.bigquery_input_table_id,
            )
        except Exception as e:
            logger.error("BigQueryInputStore init failed: %s", e)

    def is_available(self) -> bool:
        # Both client AND table id must be present
        return self.client is not None and bool(settings.bigquery_input_table_id)

    async def fetch_by_request_id(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch full row from reflection_events by request_id.
        Returns the raw row as a dict if found, None otherwise.
        JSON columns are parsed into Python objects automatically.
        """
        tracer = get_tracer()
        with tracer.start_as_current_span("curator.input_store.fetch") as span:
            span.set_attribute("curator.request_id", request_id)

            if not self.is_available():
                logger.warning("BigQueryInputStore not available. Skipping fetch.")
                span.set_status(Status(StatusCode.ERROR, "BigQuery input store unavailable"))
                return None

            try:
                query = """
                    SELECT
                        request_id,
                        event_id,
                        tenant_id,
                        site_id,
                        timestamp,
                        decision,
                        disposition,
                        expected_route,
                        actual_route,
                        expected_scope_key,
                        actual_scope_key,
                        impression_text,
                        generated_impression,
                        final_impression,
                        human_impression,
                        TO_JSON_STRING(input_payload_json) AS input_payload_json,
                        TO_JSON_STRING(issues_json)        AS issues_json,
                        has_embedding
                    FROM `{table}`
                    WHERE request_id = @request_id
                    LIMIT 1
                """.format(table=settings.bigquery_input_table_id)

                job_config = bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("request_id", "STRING", request_id)
                    ]
                )

                started = time.perf_counter()
                rows = await asyncio.to_thread(
                    lambda: list(self.client.query(query, job_config=job_config).result())
                )
                elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
                span.set_attribute("curator.fetch.row_count", len(rows))
                span.set_attribute("curator.fetch_latency_ms", elapsed_ms)

                if not rows:
                    logger.warning("No row found for request_id=%s", request_id)
                    span.set_status(Status(StatusCode.OK))
                    return None

                row = dict(rows[0])

                for col in ("input_payload_json", "issues_json"):
                    raw = row.get(col)
                    if raw and isinstance(raw, str):
                        try:
                            row[col] = json.loads(raw)
                        except Exception as e:
                            logger.warning(
                                "Failed to parse %s for request_id=%s — keeping as None",
                                col, request_id,
                            )
                            row[col] = None
                            span.set_attribute(f"curator.fetch.parse_failed.{col}", True)
                            span.record_exception(e)

                span.set_attribute("curator.tenant_id", row.get("tenant_id") or "")
                span.set_attribute("curator.site_id", row.get("site_id") or "")
                span.set_attribute("curator.decision", row.get("decision") or "")
                span.set_status(Status(StatusCode.OK))

                logger.info(
                    "Fetched row. request_id=%s decision=%s",
                    request_id, row.get("decision"),
                )
                return row

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"BigQuery fetch failed: {str(e)}"))
                logger.error("BigQuery fetch failed. request_id=%s error=%s", request_id, e)
                return None




bigquery_input_store = BigQueryInputStore()
