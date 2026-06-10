from __future__ import annotations

import os
from pydantic import BaseModel
from dotenv import load_dotenv


load_dotenv()


class Settings(BaseModel):
    app_name:    str = os.getenv("APP_NAME", "ace-curator-agent")
    app_version: str = os.getenv("APP_VERSION", "1.0.0")
    env:         str = os.getenv("ENV", "development")

    # Google Cloud
    google_cloud_project:  str | None = os.getenv("GOOGLE_CLOUD_PROJECT")
    google_cloud_location: str        = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    # Phoenix Observability
    phoenix_enabled:      bool       = os.getenv("PHOENIX_ENABLED", "false").lower() == "true"
    phoenix_collector_endpoint: str | None = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    phoenix_project_name: str        = os.getenv("PHOENIX_PROJECT_NAME", "ace-curator-agent")
    phoenix_experiment_name: str | None = os.getenv("PHOENIX_EXPERIMENT_NAME")
    phoenix_protocol: str = os.getenv("PHOENIX_PROTOCOL", "http/protobuf")
    phoenix_api_key:      str | None = os.getenv("PHOENIX_API_KEY")
    phoenix_client_headers: str | None = os.getenv("PHOENIX_CLIENT_HEADERS")
    phoenix_capture_payloads: bool = os.getenv("PHOENIX_CAPTURE_PAYLOADS", "true").lower() == "true"
    phoenix_capture_embeddings: bool = os.getenv("PHOENIX_CAPTURE_EMBEDDINGS", "false").lower() == "true"
    phoenix_trace_sample_rate: float = float(os.getenv("PHOENIX_TRACE_SAMPLE_RATE", "1.0"))

    # BigQuery — Reflector input table (READ)
    # Set to: project-cadb0b11-dcf4-4c53-aa3.ace_reflector.reflection_events
    bigquery_input_table_id: str | None = os.getenv("BIGQUERY_INPUT_TABLE_ID")

    # BigQuery — Curator output table (WRITE)
    bigquery_table_id: str | None = os.getenv("BIGQUERY_TABLE_ID")

    # LLM Clinical Accuracy Evaluator (Layer 1)
    # CLINICAL_ACCURACY_MODE env var selects python vs llm at runtime
    clinical_accuracy_mode: str = os.getenv("CLINICAL_ACCURACY_MODE", "python")
    llm_provider:   str = os.getenv("LLM_PROVIDER", "vertexai")
    llm_model_name: str = os.getenv("LLM_MODEL_NAME", "")


settings = Settings()

