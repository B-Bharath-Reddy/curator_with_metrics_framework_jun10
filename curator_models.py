from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class ReflectorRow(BaseModel):
    """
    Maps directly to a row from reflection_events BigQuery table.
    This is the Curator input — fetched by request_id.
    """
    request_id:           str
    event_id:             Optional[str]  = None
    tenant_id:            Optional[str]  = None
    site_id:              Optional[str]  = None
    decision:             Optional[str]  = None
    disposition: Optional[str] = None
    expected_route:       Optional[str]  = None
    actual_route:         Optional[str]  = None
    expected_scope_key:   Optional[str]  = None
    actual_scope_key:     Optional[str]  = None
    impression_text:      Optional[str]  = None
    generated_impression: Optional[str]  = None
    final_impression:     Optional[str]  = None
    human_impression:     Optional[str]  = None   # radiologist finalized — confirmed preferred_output
    input_payload_json:   Optional[Any]  = None   # parsed dict after fetch
    issues_json:          Optional[Any]  = None   # parsed list after fetch
    has_embedding:        Optional[bool] = None

    @field_validator("input_payload_json", "issues_json", mode="before")
    @classmethod
    def parse_json_columns(cls, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return json.loads(stripped)
            except Exception:
                return None
        return value


class TrainingExample(BaseModel):
    """
    DPO training example.

    prompt_input     -> clinical context plus prompt_text when supplied by upstream.

    preferred_output → human_impression (radiologist finalized — confirmed by manager)
    dispreferred     → generated_impression (DPO rejected side)
    """
    prompt_input:        Dict[str, Any]
    preferred_output:    str
    dispreferred_output: Optional[str] = None
    metadata:            Dict[str, Any] = Field(default_factory=dict)


class ImpressionSet(BaseModel):
    """All 3 impression sources preserved as-is for traceability."""
    generated_impression: Optional[str] = None
    final_impression:     Optional[str] = None
    human_impression:     Optional[str] = None


class ReadinessFlags(BaseModel):
    """Reports what data is available and what is still missing."""
    has_prompt:           bool      = False
    has_generated_output: bool      = False
    has_human_output:     bool      = False
    has_dpo_pair:         bool      = False
    is_dpo_ready:         bool      = False
    missing_fields:       List[str] = Field(default_factory=list)


class CuratorScores(BaseModel):
    """Simple quality/readiness scores for reporting."""
    data_completeness_score:   float = 0.0
    prompt_completeness_score: float = 0.0
    dpo_readiness_score:       float = 0.0




class FindingAccuracyCheck(BaseModel):
    """
    Per-finding accuracy assessment for one structured finding.
    Stored for audit traceability — dashboards aggregate these, but
    individual checks let you drill down into which specific finding
    was missed and in which impression.
    """
    finding_id: Optional[str] = None
    label: str
    presence: str
    severity: Optional[str] = None
    location: Optional[str] = None
    normalized_finding_type: Optional[str] = None
    mentioned_in_generated: bool
    mentioned_in_human: bool
    laterality_match_generated: Optional[bool] = None
    laterality_match_human: Optional[bool] = None
    severity_match_generated: Optional[bool] = None
    severity_match_human: Optional[bool] = None
    is_critical: bool
    critical_captured_generated: bool
    critical_captured_human: bool
    generated_evidence: Optional[str] = None
    human_evidence: Optional[str] = None
    reasoning: Optional[str] = None


class ClinicalAccuracyMetrics(BaseModel):
    """
    Per-request clinical accuracy metrics comparing impression text
    against structured findings (the authoritative clinical ground truth).
    All rates are None when denominator is zero (data unavailable).
    """
    # Layer 1 metrics (generated impression vs structured findings)
    finding_recall_generated: Optional[float] = None
    finding_precision_generated: Optional[float] = None
    false_negative_rate_generated: Optional[float] = None
    critical_finding_capture_rate_generated: Optional[float] = None
    laterality_accuracy_generated: Optional[float] = None
    severity_accuracy_generated: Optional[float] = None
    impression_agreement_score: Optional[float] = None

    # Same metrics for human impression (comparison baseline)
    finding_recall_human: Optional[float] = None
    finding_precision_human: Optional[float] = None
    false_negative_rate_human: Optional[float] = None
    critical_finding_capture_rate_human: Optional[float] = None
    laterality_accuracy_human: Optional[float] = None
    severity_accuracy_human: Optional[float] = None

    # Absent finding accuracy
    absent_finding_accuracy_generated: Optional[float] = None
    absent_finding_accuracy_human: Optional[float] = None

    # Counts for dashboard denominators
    total_findings_checked: int = 0
    present_findings_count: int = 0
    critical_findings_count: int = 0
    findings_with_laterality: int = 0
    findings_with_severity: int = 0
    absent_findings_count: int = 0


    # LLM-enriched counts (decoupled gen/human denominators)
    findings_with_laterality_generated: int = 0
    findings_with_laterality_human: int = 0
    findings_with_severity_generated: int = 0
    findings_with_severity_human: int = 0
    generated_supported_mentions_count: int = 0
    generated_unsupported_mentions_count: int = 0
    human_supported_mentions_count: int = 0
    human_unsupported_mentions_count: int = 0

    # Miss/hallucination lists for drill-down
    missed_findings_generated: List[str] = Field(default_factory=list)
    missed_findings_human: List[str] = Field(default_factory=list)
    missed_critical_generated: List[str] = Field(default_factory=list)
    missed_critical_human: List[str] = Field(default_factory=list)
    hallucinated_findings_generated: List[Dict[str, str]] = Field(default_factory=list)
    hallucinated_findings_human: List[Dict[str, str]] = Field(default_factory=list)

    # Per-finding audit trail
    finding_checks: List[FindingAccuracyCheck] = Field(default_factory=list)

    # Composite score
    clinical_accuracy_score: Optional[float] = None


class HumanFeedbackMetrics(BaseModel):
    """
    Per-request human feedback / curator learning metrics.
    Derived from the upstream disposition field and clinical accuracy outputs.
    All fields are None when disposition is not yet available.
    """
    disposition: Optional[str] = None
    is_human_overridden: Optional[bool] = None
    is_human_accepted: Optional[bool] = None
    human_feedback_edit_distance: Optional[int] = None
    normalized_edit_distance: Optional[float] = None
    human_reward_score: Optional[float] = None
    curator_reward_score: Optional[float] = None


class FieldPerformanceCheck(BaseModel):
    """
    Per-finding field-level error check for Layer 3 analytics.
    """
    label: str
    presence: str
    location: Optional[str] = None
    severity: Optional[str] = None
    laterality_applicable: bool = False
    laterality_error: bool = False
    severity_applicable: bool = False
    severity_error: bool = False
    projection_applicable: bool = False
    projection_error: bool = False
    measurement_applicable: bool = False
    measurement_error: bool = False
    measurement_expected_mm: Optional[float] = None
    measurement_mentioned_mm: Optional[float] = None



class FieldPerformanceMetrics(BaseModel):
    """
    Layer 3 — Field Performance Analytics for a single request.
    Per-field error rates for structured data elements.
    """
    field_checks: List[FieldPerformanceCheck] = Field(default_factory=list)
    laterality_applicable: int = 0
    laterality_errors: int = 0
    laterality_error_rate: Optional[float] = None
    severity_applicable: int = 0
    severity_errors: int = 0
    severity_error_rate: Optional[float] = None
    projection_applicable: int = 0
    projection_errors: int = 0
    projection_error_rate: Optional[float] = None
    measurement_applicable: int = 0
    measurement_errors: int = 0
    measurement_error_rate: Optional[float] = None
    total_field_checks: int = 0
    total_field_errors: int = 0
    overall_field_error_rate: Optional[float] = None

class CuratorAnalytics(BaseModel):
    """Request-level analytics stored for dashboards and monitoring."""
    has_generated_output:          bool = False
    has_human_reference:           bool = False
    is_dpo_training_ready:         bool = False
    clinical_accuracy: Optional[ClinicalAccuracyMetrics] = None
    human_feedback: Optional[HumanFeedbackMetrics] = None
    field_performance: Optional[FieldPerformanceMetrics] = None

    # Layer 4 — Operational passthrough (per-request).
    # Token usage from LLM evaluation; all zero when mode=python or no LLM call.
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: Optional[float] = None

    # Generator model version from upstream payload (NULL when upstream does not provide it).
    generator_model_version: Optional[str] = None
    # Reflector model version from upstream (NULL when not provided).
    reflector_model_version: Optional[str] = None

    # Clinical accuracy mode used for this request ("python" or "llm").
    clinical_accuracy_mode: Optional[str] = None

    # Reflector flagged issues (True when issues_json is non-empty).
    reflector_triggered: Optional[bool] = None

    # Upstream timing passthrough (NULL when upstream does not provide).
    agent_latency_ms: Optional[float] = None
    eval_latency_ms: Optional[float] = None

    # Upstream queue timestamps (NULL when upstream does not provide).
    submitted_to_queue_at: Optional[str] = None
    human_approved_at: Optional[str] = None


class CuratorFeedbackResponse(BaseModel):
    request_id:       str
    tenant_id:        Optional[str]   = None
    decision:         Optional[str]   = None

    # All 3 impressions preserved
    impressions:      ImpressionSet

    # Upstream reward from Reflector — not recomputed
    reward_score:     Optional[float] = None
    reward_label:     Optional[str]   = None
    reward_reason:    Optional[str]   = None
    issue_codes:      List[str]       = Field(default_factory=list)

    prompt_text:      Optional[str]   = None
    scores:           CuratorScores   = Field(default_factory=CuratorScores)
    analytics:        CuratorAnalytics = Field(default_factory=CuratorAnalytics)

    # DPO training example
    # Built only when prompt_text + human_impression + generated_impression exist.
    training_example: Optional[TrainingExample] = None

    # Readiness — tells downstream what is still missing
    readiness:        ReadinessFlags

    provenance:       Dict[str, Any]  = Field(default_factory=dict)
    trace:            Dict[str, Any]  = Field(default_factory=dict)


class CuratorReviewRequest(BaseModel):
    """
    What Reflector POSTs to curator_callback_url.
    Curator fetches full payload from BigQuery using this request_id.
    """
    request_id: str
