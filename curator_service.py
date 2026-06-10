from __future__ import annotations


import os
import time
import logging
from dataclasses import asdict as _dc_asdict
from typing import Any, Dict, List, Optional, Tuple

from opentelemetry.trace import Status, StatusCode

from app.curator_models import (
    ClinicalAccuracyMetrics,
    CuratorAnalytics,
    CuratorFeedbackResponse,
    CuratorScores,
    HumanFeedbackMetrics,
    ImpressionSet,
    ReadinessFlags,
    ReflectorRow,
    TrainingExample,
    FieldPerformanceMetrics,
)
from app.curator_metrics import calculate_curator_scores
from app.clinical_accuracy import calculate_clinical_accuracy_metrics
from app.llm_clinical_accuracy import (
    calculate_clinical_accuracy_metrics_llm,
    LLMEvaluationResult,
)
from app.human_feedback_metrics import calculate_human_feedback_metrics
from app.field_performance import calculate_field_performance_metrics
from app.observability import get_tracer

logger = logging.getLogger(__name__)


# Helpers

def _derive_reward_label(score: Optional[float]) -> str:
    if score is None:
        return "unknown"
    if score >= 0.6:
        return "good"
    if score >= 0.0:
        return "acceptable"
    return "bad"


def _extract_issue_codes(issues: Any) -> List[str]:
    if not issues or not isinstance(issues, list):
        return []
    return [i.get("code", "") for i in issues if isinstance(i, dict) and i.get("code")]


def _as_dict(value: Any) -> Dict:
    return value if isinstance(value, dict) else {}


def _extract_generator_text_with_source(row: ReflectorRow) -> tuple[Optional[str], Optional[str], bool]:
    """
    Generator text with fallback chain:
    1. generated_impression   — BQ flat column
    2. input_payload_json.generator_response.output.impression_text
    3. input_payload_json.impression_text
    4. impression_text        — BQ flat column (fallback)
    """
    # 1. BQ flat column
    if row.generated_impression:
        return row.generated_impression, "generated_impression", False

    payload = _as_dict(row.input_payload_json)
    if isinstance(payload, dict):
        # 2. generator_response.output.impression_text
        gen_resp = _as_dict(payload.get("generator_response"))
        output   = _as_dict(gen_resp.get("output"))
        text     = output.get("impression_text")
        if text:
            return text, "input_payload_json.generator_response.output.impression_text", True

        # 3. top-level impression_text in payload
        text = payload.get("impression_text")
        if text:
            return text, "input_payload_json.impression_text", True

    # 4. BQ flat impression_text fallback
    if row.impression_text:
        return row.impression_text, "impression_text", True

    return None, None, False


def _extract_generator_text(row: ReflectorRow) -> Optional[str]:
    text, _, _ = _extract_generator_text_with_source(row)
    return text


def _extract_prompt_text(payload: Any) -> Optional[str]:
    """
    Prompt text fallback chain for the full-payload phase:
    1. input_payload_json.prompt_text
    2. input_payload_json.generator_request.prompt_text
    3. input_payload_json.prompt
    4. input_payload_json.generator_request.prompt
    """
    payload = _as_dict(payload)
    for key in ("prompt_text", "prompt"):
        text = payload.get(key)
        if isinstance(text, str) and text.strip():
            return text

    gen_req = _as_dict(payload.get("generator_request"))
    for key in ("prompt_text", "prompt"):
        text = gen_req.get(key)
        if isinstance(text, str) and text.strip():
            return text

    return None


def _extract_case(payload: Any) -> Dict:
    """
    Extract case/exam with fallback:
    1. input_payload_json.case
    2. input_payload_json.generator_request.case
    """
    if not isinstance(payload, dict):
        return {}
    case = payload.get("case")
    if isinstance(case, dict) and case:
        return case
    gen_req = _as_dict(payload.get("generator_request"))
    return gen_req.get("case") or {}


def _extract_inputs(payload: Any) -> Dict:
    """
    Extract inputs object with fallback:
    1. input_payload_json.inputs
    2. input_payload_json.generator_request.inputs

    Contains structured_findings and findings_text when stored by Reflector.
    Not always present — only available in newer Reflector rows.
    """
    if not isinstance(payload, dict):
        return {}
    inputs = payload.get("inputs")
    if isinstance(inputs, dict) and inputs:
        return inputs
    gen_req = _as_dict(payload.get("generator_request"))
    return gen_req.get("inputs") or {}


def _extract_generator_model_version(payload: Any) -> Optional[str]:
    """
    Extract the generator model version from the upstream payload.

    ASSUMPTION: Upstream Reflector payload contains the model version
    that generated the impression. If upstream does not populate any
    of these fields, returns None and the BQ column will be NULL.

    Fallback chain:
    1. input_payload_json.generator_response.model_version
    2. input_payload_json.generator_response.model
    3. input_payload_json.generator_request.model
    4. input_payload_json.model
    """
    p = _as_dict(payload)
    gen_resp = _as_dict(p.get("generator_response"))
    for key in ("model_version", "model"):
        version = gen_resp.get(key)
        if isinstance(version, str) and version.strip():
            return version.strip()
    gen_req = _as_dict(p.get("generator_request"))
    for key in ("model", "model_version"):
        version = gen_req.get(key)
        if isinstance(version, str) and version.strip():
            return version.strip()
    for key in ("model", "model_version"):
        version = p.get(key)
        if isinstance(version, str) and version.strip():
            return version.strip()
    return None


def _extract_reflector_model_version(payload: Any) -> Optional[str]:
    """
    Extract the reflector model version from the upstream payload.

    ASSUMPTION: Upstream Reflector payload may contain the reflector's
    own model version. If not provided, returns None (BQ column = NULL).

    Fallback chain:
    1. input_payload_json.reflector_response.model_version
    2. input_payload_json.reflector_response.model
    3. input_payload_json.reflector_model
    """
    p = _as_dict(payload)
    refl_resp = _as_dict(p.get("reflector_response"))
    for key in ("model_version", "model"):
        version = refl_resp.get(key)
        if isinstance(version, str) and version.strip():
            return version.strip()
    for key in ("reflector_model", "reflector_model_version"):
        version = p.get(key)
        if isinstance(version, str) and version.strip():
            return version.strip()
    return None


def _extract_reflector_triggered(issues_json: Any) -> Optional[bool]:
    """
    Derive whether the Reflector flagged issues (triggering review).
    True when issues_json is a non-empty list.
    None when issues_json is not available.
    """
    if issues_json is None:
        return None
    if isinstance(issues_json, list) and len(issues_json) > 0:
        return True
    return False


def _extract_latency_breakdown(payload: Any) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract agent and eval latency from upstream payload.
    Returns (agent_latency_ms, eval_latency_ms).
    Both are NULL when upstream does not provide timing breakdown.
    """
    p = _as_dict(payload)
    timing = _as_dict(p.get("timing") or p.get("latency_breakdown"))
    agent_ms = timing.get("agent_latency_ms") or timing.get("generation_ms")
    eval_ms = timing.get("eval_latency_ms") or timing.get("evaluation_ms")
    return (
        float(agent_ms) if isinstance(agent_ms, (int, float)) else None,
        float(eval_ms) if isinstance(eval_ms, (int, float)) else None,
    )


def _extract_queue_timestamps(payload: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract queue timestamps from upstream payload.
    Returns (submitted_to_queue_at, human_approved_at).
    Both are NULL when upstream does not provide them.
    """
    p = _as_dict(payload)
    submitted = p.get("submitted_to_queue_at") or p.get("queued_at")
    approved = p.get("human_approved_at") or p.get("approved_at")
    return (
        str(submitted) if submitted else None,
        str(approved) if approved else None,
    )


# Token cost rates per 1K tokens (USD)
_TOKEN_COST_PER_1K = {
    "vertexai": {"prompt": 0.000125, "completion": 0.000375},
    "genai":    {"prompt": 0.000125, "completion": 0.000375},
    "default":  {"prompt": 0.0001,   "completion": 0.0002},
}


def _compute_estimated_cost(
    prompt_tokens: int,
    completion_tokens: int,
    provider: Optional[str] = None,
) -> Optional[float]:
    """
    Estimate USD cost from LLM token counts.
    Returns None when no tokens were consumed (mode=python or no LLM call).
    """
    if prompt_tokens == 0 and completion_tokens == 0:
        return None
    rates = _TOKEN_COST_PER_1K.get(
        (provider or "default").strip().lower(),
        _TOKEN_COST_PER_1K["default"],
    )
    cost = (prompt_tokens / 1000 * rates["prompt"]) + (completion_tokens / 1000 * rates["completion"])
    return round(cost, 6)


# Impression normalization

def normalize_impressions(row: ReflectorRow) -> ImpressionSet:
    """Collect all 3 impression sources as-is for traceability."""
    return ImpressionSet(
        generated_impression=_extract_generator_text(row),
        final_impression=row.final_impression,
        human_impression=row.human_impression,
    )


#  Readiness assessment

def assess_readiness(
    impressions: ImpressionSet,
    prompt_text: Optional[str],
) -> ReadinessFlags:
    """
    Report what is available and what is still missing.
    For the full-payload phase, DPO-ready means:
    prompt_text + human_impression + generated_impression.
    """
    missing = []

    has_generated_output = bool(impressions.generated_impression)
    if not has_generated_output:
        missing.append("generated_impression")

    has_human_output = bool(impressions.human_impression)
    if not has_human_output:
        missing.append("human_impression")

    has_prompt = bool(prompt_text)
    if not has_prompt:
        missing.append("prompt_text")

    has_dpo_pair = has_generated_output and has_human_output
    is_dpo_ready = has_dpo_pair and has_prompt

    return ReadinessFlags(
        has_prompt=has_prompt,
        has_generated_output=has_generated_output,
        has_human_output=has_human_output,
        has_dpo_pair=has_dpo_pair,
        is_dpo_ready=is_dpo_ready,
        missing_fields=missing,
    )


# Training example

def build_training_example(
    row: ReflectorRow,
    impressions: ImpressionSet,
    reward_score: Optional[float],
    reward_label: str,
    issue_codes: List[str],
    payload: Dict,
    prompt_text: Optional[str],
    findings_text: Optional[str],
    structured_findings: Optional[List],
) -> TrainingExample:
    """
    Builds DPO training example.

    Confirmed by manager:
      preferred_output  = human_impression  (radiologist finalized)
      dispreferred      = generated_impression (model output)

    prompt_input captures the exact prompt_text when supplied by upstream.
    findings_text and structured_findings remain useful audit context, but DPO
    readiness is gated by prompt_text + chosen/rejected outputs.
    """
    case = _extract_case(payload)
    exam = case.get("exam") or {} if isinstance(case, dict) else {}

    preferred_output    = impressions.human_impression or ""
    dispreferred_output = impressions.generated_impression or ""

    return TrainingExample(
        prompt_input={
            # Available now
            "request_id":         row.request_id,
            "modality":           exam.get("modality"),
            "body_part":          exam.get("body_part"),
            "procedure_code":     exam.get("procedure_code"),
            "scope_key":          row.actual_scope_key,
            "clinical_indication": case.get("clinical_indication"),
            "history":            case.get("history"),

            # Populated when input_payload_json.inputs exists (newer Reflector rows)
            "findings_text":      findings_text,
            "structured_findings": structured_findings,

            "prompt_text":        prompt_text,
        },
        preferred_output=preferred_output,
        dispreferred_output=dispreferred_output or None,
        metadata={
            "tenant_id":    row.tenant_id,
            "site_id":      row.site_id,
            "decision":     row.decision,
            "reward_score": reward_score,
            "reward_label": reward_label,
            "issue_codes":  issue_codes,
            "actual_route": row.actual_route,
            "scope_key":    row.actual_scope_key,
        },
    )


# Main service

async def curate_feedback(
    row: ReflectorRow,
    store_output: bool = True,
) -> CuratorFeedbackResponse:
    """
    Curator pipeline:
    1. Check send_to_curator flag — skip if False
    2. Extract prompt_text and inputs if available
    3. Normalize all 3 impressions and generator fallback source
    4. Preserve upstream reward + issues from Reflector
    5. Assess readiness
    6. Build TrainingExample only when is_dpo_ready is True
       - preferred_output  = human_impression  (confirmed by manager)
       - dispreferred      = generated_impression
       - DPO-ready requires prompt_text + preferred + dispreferred
    7. Optionally store response in BigQuery for request-id production flow
    8. Return response
    """
    tracer = get_tracer()
    with tracer.start_as_current_span("curator.prepare_dpo_dataset") as span:
        started = time.perf_counter()
        span.set_attribute("curator.request_id",       row.request_id)
        span.set_attribute("curator.tenant_id",        row.tenant_id or "")
        span.set_attribute("curator.site_id",          row.site_id or "")
        span.set_attribute("curator.decision",         row.decision or "")
        span.set_attribute("curator.actual_route",     row.actual_route or "")
        span.set_attribute("curator.expected_route",   row.expected_route or "")
        span.set_attribute("curator.actual_scope_key", row.actual_scope_key or "")
        span.set_attribute("curator.expected_scope_key", row.expected_scope_key or "")

        try:
            payload    = row.input_payload_json or {}
            issues_raw = row.issues_json or []
            reward     = payload.get("reward_signal") or {} if isinstance(payload, dict) else {}

            # Step 1: Check send_to_curator flag
            # Reflector sets send_to_curator=True only for rows worth training on.
            # Skip rows that Reflector did not flag.
            if not reward.get("send_to_curator"):
                logger.info(
                    "Skipping — send_to_curator=False. request_id=%s",
                    row.request_id,
                )
                span.set_attribute("curator.skipped", True)
                span.set_attribute("curator.skip_reason", "send_to_curator=False")
                span.set_status(Status(StatusCode.OK))
                response = CuratorFeedbackResponse(
                    request_id=row.request_id,
                    tenant_id=row.tenant_id,
                    decision=row.decision,
                    impressions=ImpressionSet(),
                    readiness=ReadinessFlags(
                        missing_fields=["send_to_curator=False — row skipped"]
                    ),
                    training_example=None,
                    provenance={},
                    trace={
                        "skipped":     True,
                        "skip_reason": "send_to_curator=False",
                    },
                )
                return response

            reward_score  = reward.get("score")
            reward_label  = _derive_reward_label(reward_score)
            reward_reason = reward.get("reason")
            issue_codes   = _extract_issue_codes(issues_raw)

            span.set_attribute("curator.send_to_curator", True)


            # Step 2: Extract prompt and inputs if available
            inputs       = _extract_inputs(payload)
            source_text  = inputs.get("source_text") or {} if isinstance(inputs, dict) else {}
            obs_payload  = inputs.get("observation_payload") or {} if isinstance(inputs, dict) else {}
            findings_text        = source_text.get("findings_text")
            structured_findings  = obs_payload.get("structured_findings")
            prompt_text          = _extract_prompt_text(payload)

            span.set_attribute("curator.has_inputs",              bool(inputs))
            span.set_attribute("curator.has_findings_text",       bool(findings_text))
            span.set_attribute("curator.has_structured_findings", bool(structured_findings))
            span.set_attribute("curator.has_prompt",              bool(prompt_text))

            with tracer.start_as_current_span("curator.extract_prompt") as prompt_span:
                prompt_span.set_attribute("curator.prompt_present", bool(prompt_text))
                prompt_span.set_attribute(
                    "curator.prompt_length",
                    len(prompt_text) if prompt_text else 0,
                )
                prompt_span.set_status(Status(StatusCode.OK))

            #   Step 3: Normalize impressions
            with tracer.start_as_current_span("curator.normalize_feedback") as normalize_span:
                generated_text, generator_output_source, fallback_used = (
                    _extract_generator_text_with_source(row)
                )
                impressions = ImpressionSet(
                    generated_impression=generated_text,
                    final_impression=row.final_impression,
                    human_impression=row.human_impression,
                )
                normalize_span.set_attribute(
                    "curator.generated_impression_present",
                    bool(impressions.generated_impression),
                )
                normalize_span.set_attribute(
                    "curator.generator_output_source",
                    generator_output_source or "",
                )
                normalize_span.set_attribute("curator.fallback_used", fallback_used)
                normalize_span.set_attribute(
                    "curator.final_impression_present",
                    bool(impressions.final_impression),
                )
                normalize_span.set_attribute(
                    "curator.human_impression_present",
                    bool(impressions.human_impression),
                )
                normalize_span.set_status(Status(StatusCode.OK))

            # Step 4: Assess readiness
            with tracer.start_as_current_span("curator.assess_readiness") as readiness_span:
                readiness = assess_readiness(impressions, prompt_text)
                readiness_span.set_attribute(
                    "curator.has_generated_output", readiness.has_generated_output
                )
                readiness_span.set_attribute(
                    "curator.has_human_output",
                    readiness.has_human_output,
                )
                readiness_span.set_attribute("curator.has_prompt", readiness.has_prompt)
                readiness_span.set_attribute("curator.has_dpo_pair",       readiness.has_dpo_pair)
                readiness_span.set_attribute("curator.is_dpo_ready", readiness.is_dpo_ready)
                readiness_span.set_attribute(
                    "curator.missing_fields_count", len(readiness.missing_fields)
                )
                readiness_span.set_attribute(
                    "curator.missing_fields", readiness.missing_fields
                )
                readiness_span.set_status(Status(StatusCode.OK))

            # Step 5: Build training example
            training_example = None
            with tracer.start_as_current_span("curator.build_training_example") as training_span:
                if readiness.is_dpo_ready:
                    training_example = build_training_example(
                        row=row,
                        impressions=impressions,
                        reward_score=reward_score,
                        reward_label=reward_label,
                        issue_codes=issue_codes,
                        payload=payload,
                        prompt_text=prompt_text,
                        findings_text=findings_text,
                        structured_findings=structured_findings,
                    )
                training_span.set_attribute(
                    "curator.has_training_example", training_example is not None
                )
                training_span.set_status(Status(StatusCode.OK))

            with tracer.start_as_current_span("curator.score_record") as score_span:
                scores = calculate_curator_scores(
                    request_id=row.request_id,
                    send_to_curator=reward.get("send_to_curator"),
                    prompt_text=prompt_text,
                    preferred_output=impressions.human_impression,
                    dispreferred_output=impressions.generated_impression,
                )
                score_span.set_attribute(
                    "curator.data_completeness_score",
                    scores.data_completeness_score,
                )
                score_span.set_attribute(
                    "curator.prompt_completeness_score",
                    scores.prompt_completeness_score,
                )
                score_span.set_attribute(
                    "curator.dpo_readiness_score",
                    scores.dpo_readiness_score,
                )
                score_span.set_attribute("curator.score.has_prompt", readiness.has_prompt)
                score_span.set_attribute("curator.score.has_generated_output", readiness.has_generated_output)
                score_span.set_attribute("curator.score.has_human_output", readiness.has_human_output)
                score_span.set_attribute("curator.score.has_dpo_pair", readiness.has_dpo_pair)
                score_span.set_attribute("curator.score.issue_count", len(issue_codes))
                score_span.set_attribute("curator.score.missing_fields_count", len(readiness.missing_fields))
                score_span.set_status(Status(StatusCode.OK))

            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

            # Extract generator model version (Layer 4/5 operational passthrough)
            generator_model_version = _extract_generator_model_version(payload)
            reflector_model_version = _extract_reflector_model_version(payload)
            reflector_triggered = _extract_reflector_triggered(row.issues_json)
            agent_latency_ms, eval_latency_ms = _extract_latency_breakdown(payload)
            submitted_to_queue_at, human_approved_at = _extract_queue_timestamps(payload)

            # Calculate clinical accuracy metrics if structured findings available
            # CLINICAL_ACCURACY_MODE env var: "python" (default) | "llm"
            clinical_accuracy_result = None
            clinical_accuracy_mode = os.getenv("CLINICAL_ACCURACY_MODE", "python").strip().lower()
            llm_token_usage: Dict[str, int] = {}
            if structured_findings:
                if clinical_accuracy_mode == "llm":
                    case = _extract_case(payload)
                    exam = case.get("exam") if isinstance(case, dict) else None
                    llm_result = calculate_clinical_accuracy_metrics_llm(
                        structured_findings=structured_findings,
                        generated_impression=impressions.generated_impression,
                        human_impression=impressions.human_impression,
                        exam=exam,
                    )
                    if llm_result is not None:
                        clinical_accuracy_result = llm_result.metrics
                        llm_token_usage = {
                            "prompt_tokens": llm_result.timing.prompt_tokens,
                            "completion_tokens": llm_result.timing.completion_tokens,
                            "total_tokens": llm_result.timing.total_tokens,
                        }
                    logger.info(
                        "Clinical accuracy: LLM mode. request_id=%s timing_ms=%.1f tokens=%d",
                        row.request_id,
                        llm_result.timing.total_ms if llm_result else 0.0,
                        llm_result.timing.total_tokens if llm_result else 0,
                    )
                else:
                    clinical_accuracy_result = calculate_clinical_accuracy_metrics(
                        structured_findings=structured_findings,
                        generated_impression=impressions.generated_impression,
                        human_impression=impressions.human_impression,
                    )
                    logger.info(
                        "Clinical accuracy: Python mode. request_id=%s",
                        row.request_id,
                    )

            # Calculate field performance metrics (Layer 3) if structured findings available
            field_performance_result = None
            if structured_findings:
                case = _extract_case(payload)
                exam = case.get("exam") or {} if isinstance(case, dict) else {}
                field_performance_result = calculate_field_performance_metrics(
                    structured_findings=structured_findings,
                    generated_impression=impressions.generated_impression,
                    human_impression=impressions.human_impression,
                    exam=exam,
                )


            # Calculate human feedback metrics from upstream disposition
            human_feedback_result = calculate_human_feedback_metrics(
                disposition=row.disposition,
                generated_impression=impressions.generated_impression,
                human_impression=impressions.human_impression,
                clinical_accuracy=clinical_accuracy_result,
            )

            # DPO training ready: data is ready AND human did not reject
            is_dpo_training_ready = bool(
                readiness.is_dpo_ready
                and human_feedback_result.disposition is not None
                and human_feedback_result.disposition != "rejected"
            )

            with tracer.start_as_current_span("curator.calculate_analytics") as analytics_span:
                # Build CuratorAnalytics directly (Pydantic model).
                # Dataclass → dict → Pydantic conversion.
                ca_pydantic = None
                if clinical_accuracy_result is not None:
                    ca_pydantic = ClinicalAccuracyMetrics.model_validate(
                        _dc_asdict(clinical_accuracy_result)
                    )
                hf_pydantic = None
                if human_feedback_result is not None:
                    hf_pydantic = HumanFeedbackMetrics.model_validate(
                        _dc_asdict(human_feedback_result)
                    )
                fp_pydantic = None
                if field_performance_result is not None:
                    fp_pydantic = FieldPerformanceMetrics.model_validate(
                        _dc_asdict(field_performance_result)
                    )
                analytics_result = CuratorAnalytics(
                    has_generated_output=bool(impressions.generated_impression),
                    has_human_reference=bool(impressions.human_impression),
                    is_dpo_training_ready=is_dpo_training_ready,
                    clinical_accuracy=ca_pydantic,
                    human_feedback=hf_pydantic,
                    field_performance=fp_pydantic,
                    total_tokens=llm_token_usage.get("total_tokens", 0),
                    prompt_tokens=llm_token_usage.get("prompt_tokens", 0),
                    completion_tokens=llm_token_usage.get("completion_tokens", 0),
                    estimated_cost_usd=_compute_estimated_cost(
                        llm_token_usage.get("prompt_tokens", 0),
                        llm_token_usage.get("completion_tokens", 0),
                        os.getenv("LLM_PROVIDER", "vertexai"),
                    ),
                    generator_model_version=generator_model_version,
                    reflector_model_version=reflector_model_version,
                    clinical_accuracy_mode=clinical_accuracy_mode,
                    reflector_triggered=reflector_triggered,
                    agent_latency_ms=agent_latency_ms,
                    eval_latency_ms=eval_latency_ms,
                    submitted_to_queue_at=submitted_to_queue_at,
                    human_approved_at=human_approved_at,
                )

                # Set span attributes for observability
                analytics_span.set_attribute(
                    "curator.analytics.disposition",
                    human_feedback_result.disposition or "unknown",
                )
                analytics_span.set_attribute(
                    "curator.analytics.clinical_accuracy.mode",
                    clinical_accuracy_mode,
                )
                analytics_span.set_attribute(
                    "curator.analytics.is_dpo_training_ready", is_dpo_training_ready,
                )
                if human_feedback_result.curator_reward_score is not None:
                    analytics_span.set_attribute(
                        "curator.analytics.curator_reward_score",
                        human_feedback_result.curator_reward_score,
                    )
                if human_feedback_result.human_reward_score is not None:
                    analytics_span.set_attribute(
                        "curator.analytics.human_reward_score",
                        human_feedback_result.human_reward_score,
                    )
                if human_feedback_result.is_human_accepted is not None:
                    analytics_span.set_attribute(
                        "curator.analytics.is_human_accepted",
                        human_feedback_result.is_human_accepted,
                    )
                if human_feedback_result.is_human_overridden is not None:
                    analytics_span.set_attribute(
                        "curator.analytics.is_human_overridden",
                        human_feedback_result.is_human_overridden,
                    )
                if clinical_accuracy_result is not None:
                    ca = clinical_accuracy_result
                    if ca.clinical_accuracy_score is not None:
                        analytics_span.set_attribute(
                            "curator.analytics.clinical_accuracy.score",
                            ca.clinical_accuracy_score,
                        )
                    if ca.finding_recall_generated is not None:
                        analytics_span.set_attribute(
                            "curator.analytics.clinical_accuracy.finding_recall_generated",
                            ca.finding_recall_generated,
                        )
                    if ca.critical_finding_capture_rate_generated is not None:
                        analytics_span.set_attribute(
                            "curator.analytics.clinical_accuracy.critical_capture_rate_generated",
                            ca.critical_finding_capture_rate_generated,
                        )
                    analytics_span.set_attribute(
                        "curator.analytics.clinical_accuracy.total_findings_checked",
                        ca.total_findings_checked,
                    )
                if field_performance_result is not None:
                    fp = field_performance_result
                    if fp.overall_field_error_rate is not None:
                        analytics_span.set_attribute(
                            "curator.analytics.field_performance.overall_error_rate",
                            fp.overall_field_error_rate,
                        )
                    if fp.laterality_error_rate is not None:
                        analytics_span.set_attribute(
                            "curator.analytics.field_performance.laterality_error_rate",
                            fp.laterality_error_rate,
                        )
                    if fp.severity_error_rate is not None:
                        analytics_span.set_attribute(
                            "curator.analytics.field_performance.severity_error_rate",
                            fp.severity_error_rate,
                        )
                    if fp.projection_error_rate is not None:
                        analytics_span.set_attribute(
                            "curator.analytics.field_performance.projection_error_rate",
                            fp.projection_error_rate,
                        )
                    if fp.measurement_error_rate is not None:
                        analytics_span.set_attribute(
                            "curator.analytics.field_performance.measurement_error_rate",
                            fp.measurement_error_rate,
                        )
                    analytics_span.set_attribute(
                        "curator.analytics.field_performance.total_checks",
                        fp.total_field_checks,
                    )



                analytics_span.set_status(Status(StatusCode.OK))

            if readiness.missing_fields:
                logger.info(
                    "Curator readiness incomplete. request_id=%s missing=%s",
                    row.request_id, readiness.missing_fields,
                )

            span.set_attribute("curator.reward_label",    reward_label)
            span.set_attribute("curator.has_input_payload", bool(payload))
            span.set_attribute("curator.has_dpo_pair",        readiness.has_dpo_pair)
            span.set_attribute("curator.is_dpo_ready",        readiness.is_dpo_ready)
            span.set_attribute("curator.is_dpo_training_ready", is_dpo_training_ready)
            span.set_attribute("curator.has_training_example", training_example is not None)
            span.set_attribute("curator.data_completeness_score", scores.data_completeness_score)
            span.set_attribute("curator.prompt_completeness_score", scores.prompt_completeness_score)
            span.set_attribute("curator.dpo_readiness_score", scores.dpo_readiness_score)
            span.set_attribute("curator.latency_ms",          elapsed_ms)
            if human_feedback_result.disposition:
                span.set_attribute("curator.disposition", human_feedback_result.disposition)
            if human_feedback_result.curator_reward_score is not None:
                span.set_attribute("curator.curator_reward_score", human_feedback_result.curator_reward_score)
            if human_feedback_result.human_reward_score is not None:
                span.set_attribute("curator.human_reward_score", human_feedback_result.human_reward_score)
            if reward_score is not None:
                span.set_attribute("curator.reward_score", reward_score)
            span.set_status(Status(StatusCode.OK))

            response = CuratorFeedbackResponse(
                request_id=row.request_id,
                tenant_id=row.tenant_id,
                decision=row.decision,
                impressions=impressions,
                reward_score=reward_score,
                reward_label=reward_label,
                reward_reason=reward_reason,
                issue_codes=issue_codes,
                prompt_text=prompt_text,
                scores=CuratorScores(
                    data_completeness_score=scores.data_completeness_score,
                    prompt_completeness_score=scores.prompt_completeness_score,
                    dpo_readiness_score=scores.dpo_readiness_score,
                ),
                analytics=analytics_result,
                training_example=training_example,
                readiness=readiness,
                provenance={
                    "actual_route":       row.actual_route,
                    "expected_route":     row.expected_route,
                    "actual_scope_key":   row.actual_scope_key,
                    "expected_scope_key": row.expected_scope_key,
                    "reward_reason":      reward_reason,
                    "send_to_curator":    reward.get("send_to_curator"),
                },
                trace={
                    "latency_ms":               elapsed_ms,
                    "status":                   "ready" if readiness.is_dpo_ready else "incomplete",
                    "has_training_example":     bool(training_example),
                    "generator_output_source":  generator_output_source,
                    "fallback_used":            fallback_used,
                },
            )
            # Output storage is intentionally no longer performed here.
            # The API endpoint schedules BigQuery output persistence as a FastAPI background task
            # so the response can return without waiting for the insert.
            if store_output:
                logger.warning(
                    "store_output=True is ignored in curate_feedback; output storage should be scheduled by the API layer. request_id=%s",
                    row.request_id,
                )


            return response

        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
