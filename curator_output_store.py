from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

from app.observability import get_tracer
from opentelemetry.trace import Status, StatusCode

from google.cloud import bigquery

from app.config import settings
from app.curator_models import CuratorFeedbackResponse, ReflectorRow

logger = logging.getLogger(__name__)


def _extract_exam_dimensions(source_row: ReflectorRow) -> tuple[str | None, str | None]:
    payload = source_row.input_payload_json if isinstance(source_row.input_payload_json, dict) else {}
    case = payload.get("case") if isinstance(payload.get("case"), dict) else {}
    exam = case.get("exam") if isinstance(case.get("exam"), dict) else {}
    return exam.get("modality"), exam.get("body_part")


class CuratorOutputStore:
    """
    Writes Curator output rows to BigQuery.

    The table stores a few filterable columns, the clean dpo_record when ready,
    and the full curator_response JSON for audit/debug/reporting.
    """

    def __init__(self):
        self.client = None
        self.table = None
        self._initialize()

    def _initialize(self) -> None:
        if not settings.bigquery_table_id:
            logger.warning(
                "CuratorOutputStore: BIGQUERY_TABLE_ID not set. Output writes are disabled."
            )
            return

        try:
            self.client = bigquery.Client(project=settings.google_cloud_project)
            self.table = self.client.get_table(settings.bigquery_table_id)
            logger.info(
                "CuratorOutputStore initialized. table=%s",
                settings.bigquery_table_id,
            )
        except Exception:
            logger.exception("CuratorOutputStore initialization failed")
            self.client = None
            self.table = None

    def is_available(self) -> bool:
        return self.client is not None and self.table is not None

    def store(self, response: CuratorFeedbackResponse, source_row: ReflectorRow) -> bool:
        if not self.is_available():
            logger.error(
                "CuratorOutputStore is not available. Check BIGQUERY_TABLE_ID and credentials."
            )
            return False
        tracer = get_tracer()
        with tracer.start_as_current_span("curator.output_store.store") as span:
            span.set_attribute("curator.request_id", response.request_id)
            span.set_attribute("curator.is_dpo_ready", response.readiness.is_dpo_ready)
            span.set_attribute("curator.has_dpo_pair", response.readiness.has_dpo_pair)
            span.set_attribute("curator.store_table", settings.bigquery_table_id)

            try:
                row = self._build_row(response, source_row)
                started = time.perf_counter()
                errors = self.client.insert_rows_json(settings.bigquery_table_id, [row])
                elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
                span.set_attribute("curator.store_latency_ms", elapsed_ms)
                if errors:
                    logger.error("BQ insert errors [%s]: %s", response.request_id, errors)
                    span.set_status(Status(StatusCode.ERROR, str(errors)))
                    return False
                span.set_attribute("curator.stored", True)
                span.set_status(Status(StatusCode.OK))
                logger.info("Stored curator response [%s]", response.request_id)
                return True

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"Failed to store: {str(e)}"))
                logger.exception("Failed to store [%s]", response.request_id)
                return False



    def _build_row(
            self,
            response: CuratorFeedbackResponse,
            source_row: ReflectorRow,
    ) -> dict:
        training = response.training_example
        readiness = response.readiness
        trace = response.trace or {}
        modality, body_part = _extract_exam_dimensions(source_row)

        dpo_record = None
        if readiness.is_dpo_ready and training:
            dpo_record = {
                "prompt": training.prompt_input.get("prompt_text"),
                "chosen": training.preferred_output,
                "rejected": training.dispreferred_output
            }

        return {
            "request_id": response.request_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "event_id": source_row.event_id,
            "tenant_id": response.tenant_id,
            "site_id": source_row.site_id,
            "modality": modality,
            "body_part": body_part,
            "decision": response.decision,
            "disposition": source_row.disposition,
            "actual_route": source_row.actual_route,
            "expected_route": source_row.expected_route,
            "actual_scope_key": source_row.actual_scope_key,
            "expected_scope_key": source_row.expected_scope_key,

            "prompt_text": response.prompt_text,
            "preferred_output": response.impressions.human_impression,
            "dispreferred_output": response.impressions.generated_impression,

            "has_dpo_pair": readiness.has_dpo_pair,
            "is_dpo_ready": readiness.is_dpo_ready,
            "missing_fields": json.dumps(readiness.missing_fields),

            "reward_score": response.reward_score,
            "reward_label": response.reward_label,

            "latency_ms": trace.get("latency_ms"),

            "has_generated_output": response.analytics.has_generated_output,
            "has_human_reference": response.analytics.has_human_reference,
            "is_dpo_training_ready": response.analytics.is_dpo_training_ready,
            "analytics_json": json.dumps(response.analytics.model_dump(mode="json")),

            # Human feedback metrics (flattened for BQ)
            "is_human_overridden": response.analytics.human_feedback.is_human_overridden
            if response.analytics.human_feedback else None,
            "is_human_accepted": response.analytics.human_feedback.is_human_accepted
            if response.analytics.human_feedback else None,
            "human_feedback_edit_distance": response.analytics.human_feedback.human_feedback_edit_distance
            if response.analytics.human_feedback else None,
            "human_feedback_normalized_edit_distance": response.analytics.human_feedback.normalized_edit_distance
            if response.analytics.human_feedback else None,
            "human_reward_score": response.analytics.human_feedback.human_reward_score
            if response.analytics.human_feedback else None,
            "curator_reward_score": response.analytics.human_feedback.curator_reward_score
            if response.analytics.human_feedback else None,
            "human_feedback_json": json.dumps(
                response.analytics.human_feedback.model_dump(mode="json")
            ) if response.analytics.human_feedback else None,

            # Clinical accuracy metrics (flattened for BQ)
            "clinical_accuracy_json": json.dumps(
                response.analytics.clinical_accuracy.model_dump(mode="json")
            ) if response.analytics.clinical_accuracy else None,
            "clinical_accuracy_score": response.analytics.clinical_accuracy.clinical_accuracy_score
            if response.analytics.clinical_accuracy else None,
            "finding_recall_generated": response.analytics.clinical_accuracy.finding_recall_generated
            if response.analytics.clinical_accuracy else None,
            "finding_precision_generated": response.analytics.clinical_accuracy.finding_precision_generated
            if response.analytics.clinical_accuracy else None,
            "false_negative_rate_generated": response.analytics.clinical_accuracy.false_negative_rate_generated
            if response.analytics.clinical_accuracy else None,
            "critical_finding_capture_rate_generated": response.analytics.clinical_accuracy.critical_finding_capture_rate_generated
            if response.analytics.clinical_accuracy else None,
            "laterality_accuracy_generated": response.analytics.clinical_accuracy.laterality_accuracy_generated
            if response.analytics.clinical_accuracy else None,
            "severity_accuracy_generated": response.analytics.clinical_accuracy.severity_accuracy_generated
            if response.analytics.clinical_accuracy else None,
            "impression_agreement_score": response.analytics.clinical_accuracy.impression_agreement_score
            if response.analytics.clinical_accuracy else None,
            "missed_findings_generated": json.dumps(
                response.analytics.clinical_accuracy.missed_findings_generated
            ) if response.analytics.clinical_accuracy else None,
            "missed_critical_generated": json.dumps(
                response.analytics.clinical_accuracy.missed_critical_generated
            ) if response.analytics.clinical_accuracy else None,
            "finding_checks_json": json.dumps(
                [check.model_dump(mode="json") for check in response.analytics.clinical_accuracy.finding_checks]
            ) if response.analytics.clinical_accuracy else None,
            "hallucinated_findings_generated": json.dumps(
                response.analytics.clinical_accuracy.hallucinated_findings_generated
            ) if response.analytics.clinical_accuracy else None,

            # Field performance analytics (Layer 3 — per-field error rates)
            "field_performance_json": json.dumps(
                response.analytics.field_performance.model_dump(mode="json")
            ) if response.analytics.field_performance else None,
            "laterality_error_rate": response.analytics.field_performance.laterality_error_rate
            if response.analytics.field_performance else None,
            "severity_error_rate": response.analytics.field_performance.severity_error_rate
            if response.analytics.field_performance else None,
            "projection_error_rate": response.analytics.field_performance.projection_error_rate
            if response.analytics.field_performance else None,
            "measurement_error_rate": response.analytics.field_performance.measurement_error_rate
            if response.analytics.field_performance else None,
            "overall_field_error_rate": response.analytics.field_performance.overall_field_error_rate
            if response.analytics.field_performance else None,

            # Operational metrics (Layer 4 — per-request passthrough)
            "total_tokens": response.analytics.total_tokens,
            "prompt_tokens": response.analytics.prompt_tokens,
            "completion_tokens": response.analytics.completion_tokens,
            "estimated_cost_usd": response.analytics.estimated_cost_usd,
            "generator_model_version": response.analytics.generator_model_version,
            "reflector_model_version": response.analytics.reflector_model_version,
            "clinical_accuracy_mode": response.analytics.clinical_accuracy_mode,
            "reflector_triggered": response.analytics.reflector_triggered,
            "agent_latency_ms": response.analytics.agent_latency_ms,
            "eval_latency_ms": response.analytics.eval_latency_ms,
            "submitted_to_queue_at": response.analytics.submitted_to_queue_at,
            "human_approved_at": response.analytics.human_approved_at,

            "dpo_record": json.dumps(dpo_record) if dpo_record else None,
            "curator_response": json.dumps(response.model_dump(mode="json")),
        }


curator_output_store = CuratorOutputStore()
