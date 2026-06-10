import os
from typing import Optional

from phoenix.otel import register
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry import trace
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.sdk.resources import Resource




_TRACING_INITIALIZED = False
_TRACER = None
_TRACER_PROVIDER = None
_DEFAULT_PROJECT_NAME = "ace-curator-agent"


def setup_phoenix(app) -> Optional[object]:
    global _TRACING_INITIALIZED, _TRACER, _TRACER_PROVIDER

    if _TRACING_INITIALIZED:
        return _TRACER

    project_name = os.getenv("PHOENIX_PROJECT_NAME", _DEFAULT_PROJECT_NAME)

    # CRITICAL: Add service resource for proper Kind display
    resource = Resource.create({
        "service.name": "ace-curator-agent",
        "service.version": "1.0.0",
        "telemetry.sdk.language": "python",
        "environment": os.getenv("ENVIRONMENT", "production"),
    })

    tracer_provider = register(
        project_name=project_name,
        endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT"),
        api_key=os.getenv("PHOENIX_API_KEY"),
        protocol=os.getenv("PHOENIX_PROTOCOL", "http/protobuf"),
        auto_instrument=False,
        batch=True,
        resource=resource,
    )

    LangChainInstrumentor().instrument()

    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=tracer_provider,
        excluded_urls="health",
    )

    _TRACER_PROVIDER = tracer_provider
    _TRACER = trace.get_tracer(project_name)
    _TRACING_INITIALIZED = True
    return _TRACER


def get_tracer():
    global _TRACER
    if _TRACER is None:
        project_name = os.getenv("PHOENIX_PROJECT_NAME", _DEFAULT_PROJECT_NAME)
        _TRACER = trace.get_tracer(project_name)
    return _TRACER


def shutdown_phoenix() -> None:
    global _TRACING_INITIALIZED, _TRACER, _TRACER_PROVIDER

    if _TRACER_PROVIDER is None:
        return

    _TRACER_PROVIDER.force_flush()
    _TRACER_PROVIDER.shutdown()
    _TRACING_INITIALIZED = False
    _TRACER = None
    _TRACER_PROVIDER = None
