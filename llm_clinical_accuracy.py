"""
LLM-Based Clinical Accuracy Metrics

Computes the 7 clinical accuracy metrics using Gemini / Vertex AI as the
clinical semantic judge and deterministic Python arithmetic for final rates.


Provider config:
  LLM_PROVIDER=vertexai | genai       default: vertexai
  LLM_MODEL_NAME=<model>              optional
  GOOGLE_CLOUD_PROJECT=<project>      required for Vertex AI
  GOOGLE_CLOUD_LOCATION=us-central1   optional for Vertex AI
  GOOGLE_API_KEY=<key>                required for Gemini API fallback

The LLM decides finding-level clinical judgments:
- mention/capture
- hallucinations
- laterality correctness
- severity correctness
- criticality
- impression agreement inputs

Python calculates:
- Critical Finding Capture Rate
- False Negative Rate
- Finding-level Precision
- Finding-level Recall
- Impression Agreement Score
- Laterality Accuracy
- Severity Accuracy
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


_SUPPORTED_PROVIDERS = ("vertexai", "genai")
_DEFAULT_MODELS = {
    "vertexai": "gemini-2.5-flash",
    "genai": "gemini-2.5-flash",
}


@dataclass(frozen=True)
class FindingAccuracyCheck:
    """Per-finding audit trail for one structured finding."""

    finding_id: str
    label: str
    presence: str
    severity: Optional[str]
    location: Optional[str]
    normalized_finding_type: Optional[str]
    mentioned_in_generated: bool
    mentioned_in_human: bool
    laterality_match_generated: Optional[bool]
    laterality_match_human: Optional[bool]
    severity_match_generated: Optional[bool]
    severity_match_human: Optional[bool]
    is_critical: bool
    critical_captured_generated: bool
    critical_captured_human: bool
    generated_evidence: Optional[str] = None
    human_evidence: Optional[str] = None
    reasoning: Optional[str] = None


@dataclass(frozen=True)
class ClinicalAccuracyMetrics:
    """Section 1 clinical accuracy metrics from LLM judgments."""

    finding_recall_generated: Optional[float]
    finding_precision_generated: Optional[float]
    false_negative_rate_generated: Optional[float]
    critical_finding_capture_rate_generated: Optional[float]
    laterality_accuracy_generated: Optional[float]
    severity_accuracy_generated: Optional[float]
    impression_agreement_score: Optional[float]

    finding_recall_human: Optional[float]
    finding_precision_human: Optional[float]
    false_negative_rate_human: Optional[float]
    critical_finding_capture_rate_human: Optional[float]
    laterality_accuracy_human: Optional[float]
    severity_accuracy_human: Optional[float]

    absent_finding_accuracy_generated: Optional[float]
    absent_finding_accuracy_human: Optional[float]

    total_findings_checked: int
    present_findings_count: int
    critical_findings_count: int
    findings_with_laterality_generated: int
    findings_with_laterality_human: int
    findings_with_severity_generated: int
    findings_with_severity_human: int
    absent_findings_count: int

    generated_supported_mentions_count: int
    generated_unsupported_mentions_count: int
    human_supported_mentions_count: int
    human_unsupported_mentions_count: int

    missed_findings_generated: List[str]
    missed_findings_human: List[str]
    missed_critical_generated: List[str]
    missed_critical_human: List[str]
    hallucinated_findings_generated: List[Dict[str, str]]
    hallucinated_findings_human: List[Dict[str, str]]
    finding_checks: List[FindingAccuracyCheck]
    clinical_accuracy_score: Optional[float]


@dataclass(frozen=True)
class LLMTimingInfo:
    """Latency and token breakdown for one clinical metrics evaluation."""

    clinical_evaluation_ms: float = 0.0
    total_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class LLMEvaluationResult:
    """LLM metric output plus timing metadata."""

    metrics: ClinicalAccuracyMetrics
    timing: LLMTimingInfo


class ImpressionFindingJudgment(BaseModel):
    """LLM judgment for one structured finding against one impression."""

    mention_status: str = Field(
        description="mentioned, negated, not_mentioned, or uncertain"
    )
    assertion_sentence: Optional[str] = None
    negation_sentence: Optional[str] = None
    laterality_applicable: bool = False
    expected_laterality: Optional[str] = None
    stated_laterality: Optional[str] = None
    laterality_correct: Optional[bool] = None
    severity_applicable: bool = False
    expected_severity: Optional[str] = None
    stated_severity: Optional[str] = None
    severity_correct: Optional[bool] = None
    critical_captured: bool = False
    evidence: Optional[str] = None


class FindingJudgment(BaseModel):
    """LLM judgment for one structured finding across both impressions."""

    finding_id: str
    label: str
    presence: str
    normalized_finding_type: Optional[str] = None
    is_critical: bool = False
    generated: ImpressionFindingJudgment = Field(default_factory=ImpressionFindingJudgment)
    human: ImpressionFindingJudgment = Field(default_factory=ImpressionFindingJudgment)
    reasoning: Optional[str] = None


class UnsupportedFinding(BaseModel):
    """Finding mentioned in an impression but unsupported by ground truth."""

    label: str
    location: Optional[str] = None
    severity: Optional[str] = None
    evidence: Optional[str] = None
    confidence: Optional[str] = None


class ClinicalLLMEvaluation(BaseModel):
    """Structured output returned by the LLM judge."""

    finding_judgments: List[FindingJudgment] = Field(default_factory=list)
    hallucinated_findings_generated: List[UnsupportedFinding] = Field(default_factory=list)
    hallucinated_findings_human: List[UnsupportedFinding] = Field(default_factory=list)
    agreement_common_finding_ids: List[str] = Field(default_factory=list)
    agreement_generated_only_finding_ids: List[str] = Field(default_factory=list)
    agreement_human_only_finding_ids: List[str] = Field(default_factory=list)
    reasoning: Optional[str] = None


_LLM_PROMPT = """
You are a board-certified radiologist evaluating clinical accuracy metrics.

Compare structured findings, an AI-generated impression, and a human/radiologist
impression. Use clinical meaning, not exact text overlap. Handle synonyms,
abbreviations, negation, paraphrase, and implied findings.

Return one judgment per structured finding using the exact finding_id values.
Also list hallucinated findings in each impression: clinical findings stated in
an impression but not supported by any structured finding.


INPUT FORMAT:
- structured_findings are provided as JSON lines, one per finding.
  Each has: finding_id, label, presence (present/absent), location, severity,
  measurements_size_mm, critical_flags, and evidence.
- Only populated fields are included — if a field is absent from a finding's
  JSON, it means that data is not available for that finding.
- presence="present" means the finding exists and should appear in the impression.
- presence="absent" means the finding does NOT exist. A well-formed impression
  may explicitly rule it out (e.g., "No pneumothorax") or simply omit it.
  Both are acceptable — neither is a false negative.


Rules:
- Ground truth is structured_findings.
- For presence="present", mention_status="mentioned" means the impression
  clinically captures that finding.
- For presence="absent", a negated mention or omission is not a false negative.
  Only mark mention_status="mentioned" if the impression asserts the finding
  as actually present (not just ruling it out).
- "cannot exclude X" is uncertain and should not be treated as negation.
- "no evidence of X", "without X", or "X is absent" is negation.
- Evaluate laterality and severity only within the assertion sentence for that
  finding, not across unrelated sentences.
- Laterality applies when location/laterality is available or clinically required.
- Severity applies when severity/grade is available or clinically meaningful.
- Mark is_critical=true for urgent/safety-significant findings, including but not
  limited to pneumothorax, tension physiology, clinically important fracture,
  pulmonary embolism, hemorrhage, severe obstruction, TB concern, malignancy
  concern, or any severe/large/displaced finding requiring urgent attention.
- critical_captured=true only if is_critical=true and the impression clinically
  captures the critical finding.
- Agreement lists should use finding_id values for present structured findings
  mentioned by generated and/or human impressions.
- Do not calculate percentages or rates. Python will compute all rates.
""".strip()


_llm_client = None
_llm_provider: Optional[str] = None


def _get_llm():
    """Initialize and cache the Gemini / Vertex AI chat client."""
    global _llm_client, _llm_provider

    if _llm_client is not None:
        return _llm_client

    provider = os.getenv("LLM_PROVIDER", "vertexai").strip().lower()
    if provider not in _SUPPORTED_PROVIDERS:
        provider = "vertexai"

    model_name = os.getenv("LLM_MODEL_NAME", "").strip()
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    if provider == "vertexai" and os.getenv("GOOGLE_CLOUD_PROJECT"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            model = model_name or _DEFAULT_MODELS["vertexai"]
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
            _llm_client = ChatGoogleGenerativeAI(
                model=model,
                temperature=0.0,
                max_output_tokens=8192,
            )
            _llm_provider = "vertexai"
            logger.info("LLM clinical evaluator using Vertex AI (genai) model=%s", model)
            return _llm_client
        except Exception as exc:
            logger.warning("Vertex AI (genai) init failed: %s. Trying Gemini API.", exc)

    if os.getenv("GOOGLE_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            os.environ.pop("GOOGLE_GENAI_USE_VERTEXAI", None)
            model = model_name or _DEFAULT_MODELS["genai"]
            _llm_client = ChatGoogleGenerativeAI(
                model=model,
                temperature=0.0,
                max_output_tokens=8192,
            )
            _llm_provider = "genai"
            logger.info("LLM clinical evaluator using Gemini API model=%s", model)
            return _llm_client
        except Exception as exc:
            logger.error("Gemini API init failed: %s", exc)

    raise RuntimeError(
        "No LLM provider available. Set GOOGLE_CLOUD_PROJECT for Vertex AI "
        "or GOOGLE_API_KEY for Gemini API."
    )



def _strip_json_fence(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = _strip_json_fence(text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON root must be an object")
    return parsed


def _invoke_structured(
        prompt: str, schema: type[BaseModel],
) -> Tuple[BaseModel, Dict[str, int]]:
    """
    Call LLM with structured output.
    Retries once on transient failure. Raises RuntimeError on persistent failure.
    """
    llm = _get_llm()

    for attempt in range(2):
        try:
            token_usage: Dict[str, int] = {}
            structured_llm = llm.with_structured_output(schema, include_raw=True)
            result = structured_llm.invoke(prompt)

            parsed = result.get("parsed")
            raw_message = result.get("raw")

            if raw_message and hasattr(raw_message, "usage_metadata") and raw_message.usage_metadata:
                meta = raw_message.usage_metadata
                token_usage = {
                    "prompt_tokens": meta.get("input_tokens", 0),
                    "completion_tokens": meta.get("output_tokens", 0),
                    "total_tokens": meta.get("total_tokens", 0),
                }

            if isinstance(parsed, schema):
                return parsed, token_usage
            elif isinstance(parsed, dict):
                return schema.model_validate(parsed), token_usage
            else:
                raise RuntimeError(f"Unexpected structured output type: {type(parsed)}")

        except Exception as exc:
            if attempt == 0:
                logger.warning("LLM call attempt 1 failed: %s. Retrying.", exc)
                continue
            raise RuntimeError(f"LLM structured evaluation failed after 2 attempts: {exc}") from exc




def _safe_rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def _as_bool(value: Any) -> bool:
    return bool(value)


def _is_mentioned(judgment: ImpressionFindingJudgment) -> bool:
    return judgment.mention_status.strip().lower() == "mentioned"


def _normalize_key(value: Any) -> str:
    """Normalize LLM-returned identifiers for tolerant lookup."""
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _label_signature(value: Any) -> str:
    """Label fallback key; used only when the label is unique in the payload."""
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _format_findings(findings: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for idx, finding in enumerate(findings, 1):
        finding_id = str(finding.get("finding_id") or f"finding_{idx}")
        measurements = finding.get("measurements")
        size = None
        if isinstance(measurements, dict):
            size = measurements.get("size_mm")

        record: Dict[str, Any] = {
            "finding_id": finding_id,
            "label": finding.get("label"),
            "presence": finding.get("presence"),
        }
        if finding.get("location"):
            record["location"] = finding["location"]
        if finding.get("severity"):
            record["severity"] = finding["severity"]
        if size is not None:
            record["measurements_size_mm"] = size
        if finding.get("critical_flags"):
            record["critical_flags"] = finding["critical_flags"]
        if finding.get("evidence"):
            record["evidence"] = finding["evidence"]

        lines.append(json.dumps(record, ensure_ascii=False))
    return "\n".join(lines)


def _prepare_findings(structured_findings: Any) -> List[Dict[str, Any]]:
    valid: List[Dict[str, Any]] = []
    if not isinstance(structured_findings, list):
        return valid
    for idx, finding in enumerate(structured_findings, 1):
        if not isinstance(finding, dict):
            continue
        label = finding.get("label")
        if not isinstance(label, str) or not label.strip():
            continue
        copied = dict(finding)
        copied["finding_id"] = str(copied.get("finding_id") or copied.get("id") or f"finding_{idx}")
        copied["label"] = label.strip()
        copied["presence"] = str(copied.get("presence") or "unknown").strip().lower()
        valid.append(copied)
    return valid


def _unsupported_to_dict(items: List[UnsupportedFinding]) -> List[Dict[str, str]]:
    output: List[Dict[str, str]] = []
    for item in items:
        row: Dict[str, str] = {"label": item.label}
        if item.location:
            row["location"] = item.location
        if item.severity:
            row["severity"] = item.severity
        if item.evidence:
            row["evidence"] = item.evidence
        if item.confidence:
            row["confidence"] = item.confidence
        output.append(row)
    return output


def _evaluate_clinically(
    *,
    structured_findings: List[Dict[str, Any]],
    generated_impression: Optional[str],
    human_impression: Optional[str],
    exam: Optional[Dict[str, Any]],
) -> Tuple[ClinicalLLMEvaluation, float, Dict[str, int]]:
    payload = {
        "structured_findings": _format_findings(structured_findings),
        "exam_context": exam or {},
        "generated_impression": generated_impression or "",
        "human_impression": human_impression or "",
    }
    prompt = _LLM_PROMPT + "\n\nINPUT:\n" + json.dumps(payload, ensure_ascii=False, indent=2)

    start = time.perf_counter()
    result, token_usage = _invoke_structured(prompt, ClinicalLLMEvaluation)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    return result, elapsed_ms, token_usage


def _compute_metrics(*,valid_findings: List[Dict[str, Any]],evaluation: ClinicalLLMEvaluation,) -> ClinicalAccuracyMetrics:
    judgments_by_id = {j.finding_id: j for j in evaluation.finding_judgments}
    judgments_by_normalized_id = {
        _normalize_key(j.finding_id): j for j in evaluation.finding_judgments
    }

    label_counts: Dict[str, int] = {}
    for finding in valid_findings:
        label_key = _label_signature(finding.get("label"))
        label_counts[label_key] = label_counts.get(label_key, 0) + 1

    judgments_by_unique_label: Dict[str, FindingJudgment] = {}
    for judgment in evaluation.finding_judgments:
        label_key = _label_signature(judgment.label)
        if label_counts.get(label_key) == 1:
            judgments_by_unique_label[label_key] = judgment

    finding_checks: List[FindingAccuracyCheck] = []
    present_count = 0
    absent_count = 0
    critical_count = 0

    gen_recall_hits = 0
    human_recall_hits = 0
    absent_correct_gen = 0
    absent_correct_human = 0
    critical_captured_gen = 0
    critical_captured_human = 0

    lat_app_gen = 0
    lat_app_human = 0
    lat_correct_gen = 0
    lat_correct_human = 0
    sev_app_gen = 0
    sev_app_human = 0
    sev_correct_gen = 0
    sev_correct_human = 0

    gen_supported_mentions = 0
    human_supported_mentions = 0
    gen_mentioned_present_ids: set[str] = set()
    human_mentioned_present_ids: set[str] = set()

    missed_gen: List[str] = []
    missed_human: List[str] = []
    missed_critical_gen: List[str] = []
    missed_critical_human: List[str] = []

    for finding in valid_findings:
        finding_id = str(finding["finding_id"])
        label = str(finding["label"])
        presence = str(finding.get("presence") or "unknown").strip().lower()
        severity = finding.get("severity") if isinstance(finding.get("severity"), str) else None
        location = finding.get("location") if isinstance(finding.get("location"), str) else None

        judgment = judgments_by_id.get(finding_id)
        if judgment is None:
            judgment = judgments_by_normalized_id.get(_normalize_key(finding_id))
        if judgment is None:
            judgment = judgments_by_unique_label.get(_label_signature(label))
        if judgment is None:
            raise RuntimeError(
                f"LLM response missing judgment for finding_id={finding_id} label={label}"
            )

        gen_mentioned = _is_mentioned(judgment.generated)
        human_mentioned = _is_mentioned(judgment.human)
        is_critical = _as_bool(judgment.is_critical)

        if presence == "present": # Only check findings that actually exist
            present_count += 1    # Total present findings (denominator for Recall, FNR)
            if gen_mentioned:      # LLM says generated impression captured this finding
                gen_recall_hits += 1   # Recall numerator: how many present findings did gen capture
                gen_supported_mentions += 1  # Precision numerator: gen mentioned it AND it's real (supported)
                gen_mentioned_present_ids.add(finding_id)  # Track for Agreement score
            else:
                missed_gen.append(label)

            if human_mentioned:
                human_recall_hits += 1
                human_supported_mentions += 1
                human_mentioned_present_ids.add(finding_id)
            else:
                missed_human.append(label)

            if is_critical:   # Is this finding urgent/life-threatening?
                critical_count += 1  # Total critical findings (denominator for CFCR)
                if judgment.generated.critical_captured:
                    critical_captured_gen += 1
                else:
                    missed_critical_gen.append(label)
                if judgment.human.critical_captured:
                    critical_captured_human += 1
                else:
                    missed_critical_human.append(label)

        elif presence == "absent":
            absent_count += 1
            if not gen_mentioned:
                absent_correct_gen += 1
            if not human_mentioned:
                absent_correct_human += 1

        if judgment.generated.laterality_applicable and judgment.generated.laterality_correct is not None:
            lat_app_gen += 1
            if judgment.generated.laterality_correct:
                lat_correct_gen += 1
        if judgment.human.laterality_applicable and judgment.human.laterality_correct is not None:
            lat_app_human += 1
            if judgment.human.laterality_correct:
                lat_correct_human += 1

        if judgment.generated.severity_applicable and judgment.generated.severity_correct is not None:
            sev_app_gen += 1
            if judgment.generated.severity_correct:
                sev_correct_gen += 1
        if judgment.human.severity_applicable and judgment.human.severity_correct is not None:
            sev_app_human += 1
            if judgment.human.severity_correct:
                sev_correct_human += 1

        finding_checks.append(FindingAccuracyCheck(
            finding_id=finding_id,
            label=label,
            presence=presence,
            severity=severity,
            location=location,
            normalized_finding_type=judgment.normalized_finding_type,
            mentioned_in_generated=gen_mentioned,
            mentioned_in_human=human_mentioned,
            laterality_match_generated=judgment.generated.laterality_correct,
            laterality_match_human=judgment.human.laterality_correct,
            severity_match_generated=judgment.generated.severity_correct,
            severity_match_human=judgment.human.severity_correct,
            is_critical=is_critical,
            critical_captured_generated=judgment.generated.critical_captured,
            critical_captured_human=judgment.human.critical_captured,
            generated_evidence=judgment.generated.evidence,
            human_evidence=judgment.human.evidence,
            reasoning=judgment.reasoning,
        ))

    gen_hallucinations = _unsupported_to_dict(evaluation.hallucinated_findings_generated)
    human_hallucinations = _unsupported_to_dict(evaluation.hallucinated_findings_human)

    gen_precision_denom = gen_supported_mentions + len(gen_hallucinations)
    human_precision_denom = human_supported_mentions + len(human_hallucinations)

    common_ids = set(evaluation.agreement_common_finding_ids)
    gen_only_ids = set(evaluation.agreement_generated_only_finding_ids)
    human_only_ids = set(evaluation.agreement_human_only_finding_ids)
    agreement_denom = len(common_ids | gen_only_ids | human_only_ids)
    if agreement_denom > 0:
        agreement = _safe_rate(len(common_ids), agreement_denom)
    else:
        union = gen_mentioned_present_ids | human_mentioned_present_ids
        agreement = _safe_rate(len(gen_mentioned_present_ids & human_mentioned_present_ids), len(union))

    recall_gen = _safe_rate(gen_recall_hits, present_count)
    recall_human = _safe_rate(human_recall_hits, present_count)
    fnr_gen = _safe_rate(present_count - gen_recall_hits, present_count)
    fnr_human = _safe_rate(present_count - human_recall_hits, present_count)
    precision_gen = _safe_rate(gen_supported_mentions, gen_precision_denom)
    precision_human = _safe_rate(human_supported_mentions, human_precision_denom)
    crit_cap_gen = _safe_rate(critical_captured_gen, critical_count)
    crit_cap_human = _safe_rate(critical_captured_human, critical_count)
    lat_acc_gen = _safe_rate(lat_correct_gen, lat_app_gen)
    lat_acc_human = _safe_rate(lat_correct_human, lat_app_human)
    sev_acc_gen = _safe_rate(sev_correct_gen, sev_app_gen)
    sev_acc_human = _safe_rate(sev_correct_human, sev_app_human)
    absent_acc_gen = _safe_rate(absent_correct_gen, absent_count)
    absent_acc_human = _safe_rate(absent_correct_human, absent_count)

    components: List[Tuple[float, float]] = []
    if recall_gen is not None:
        components.append((recall_gen, 0.30))
    if crit_cap_gen is not None:
        components.append((crit_cap_gen, 0.25))
    if fnr_gen is not None:
        components.append((1.0 - fnr_gen, 0.20))
    if lat_acc_gen is not None:
        components.append((lat_acc_gen, 0.15))
    if sev_acc_gen is not None:
        components.append((sev_acc_gen, 0.10))

    clinical_accuracy_score: Optional[float] = None
    if components:
        total_weight = sum(weight for _, weight in components)
        clinical_accuracy_score = round(
            sum(score * weight for score, weight in components) / total_weight,
            4,
        )

    return ClinicalAccuracyMetrics(
        finding_recall_generated=recall_gen,
        finding_precision_generated=precision_gen,
        false_negative_rate_generated=fnr_gen,
        critical_finding_capture_rate_generated=crit_cap_gen,
        laterality_accuracy_generated=lat_acc_gen,
        severity_accuracy_generated=sev_acc_gen,
        impression_agreement_score=agreement,
        finding_recall_human=recall_human,
        finding_precision_human=precision_human,
        false_negative_rate_human=fnr_human,
        critical_finding_capture_rate_human=crit_cap_human,
        laterality_accuracy_human=lat_acc_human,
        severity_accuracy_human=sev_acc_human,
        absent_finding_accuracy_generated=absent_acc_gen,
        absent_finding_accuracy_human=absent_acc_human,
        total_findings_checked=len(finding_checks),
        present_findings_count=present_count,
        critical_findings_count=critical_count,
        findings_with_laterality_generated=lat_app_gen,
        findings_with_laterality_human=lat_app_human,
        findings_with_severity_generated=sev_app_gen,
        findings_with_severity_human=sev_app_human,
        absent_findings_count=absent_count,
        generated_supported_mentions_count=gen_supported_mentions,
        generated_unsupported_mentions_count=len(gen_hallucinations),
        human_supported_mentions_count=human_supported_mentions,
        human_unsupported_mentions_count=len(human_hallucinations),
        missed_findings_generated=missed_gen,
        missed_findings_human=missed_human,
        missed_critical_generated=missed_critical_gen,
        missed_critical_human=missed_critical_human,
        hallucinated_findings_generated=gen_hallucinations,
        hallucinated_findings_human=human_hallucinations,
        finding_checks=finding_checks,
        clinical_accuracy_score=clinical_accuracy_score,
    )


def calculate_clinical_accuracy_metrics_llm(
    structured_findings: Any,
    generated_impression: Optional[str],
    human_impression: Optional[str],
    exam: Optional[Dict[str, Any]] = None,
) -> Optional[LLMEvaluationResult]:
    """
    Calculate all 7 clinical accuracy metrics using Gemini / Vertex AI.

    Returns None when data is insufficient. Raises RuntimeError if the LLM call
    fails or returns incomplete structured judgments, because silently treating
    failed evaluation as clinical misses would corrupt metrics.
    """
    valid_findings = _prepare_findings(structured_findings)
    if not valid_findings:
        return None

    has_generated = bool(generated_impression and generated_impression.strip())
    has_human = bool(human_impression and human_impression.strip())
    if not has_generated and not has_human:
        return None

    evaluation, elapsed_ms, token_usage = _evaluate_clinically(
        structured_findings=valid_findings,
        generated_impression=generated_impression,
        human_impression=human_impression,
        exam=exam,
    )
    metrics = _compute_metrics(valid_findings=valid_findings, evaluation=evaluation)
    timing = LLMTimingInfo(
        clinical_evaluation_ms=elapsed_ms,
        total_ms=elapsed_ms,
        prompt_tokens=token_usage.get("prompt_tokens", 0),
        completion_tokens=token_usage.get("completion_tokens", 0),
        total_tokens=token_usage.get("total_tokens", 0),
    )
    return LLMEvaluationResult(metrics=metrics, timing=timing)





if __name__ == "__main__":
    import argparse
    from dataclasses import asdict

    def _as_dict(value: Any) -> Dict[str, Any]:
        return value if isinstance(value, dict) else {}

    parser = argparse.ArgumentParser(
        description="Calculate Gemini/Vertex LLM clinical accuracy metrics from a JSON payload."
    )
    parser.add_argument("payload", help="Path to Reflector/Curator JSON payload")
    args = parser.parse_args()

    with open(args.payload, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    row = payload[0] if isinstance(payload, list) and payload else payload
    input_payload = _as_dict(row.get("input_payload_json")) or _as_dict(row)
    case = _as_dict(input_payload.get("case"))
    exam = _as_dict(case.get("exam"))
    inputs = _as_dict(input_payload.get("inputs"))
    obs_payload = _as_dict(inputs.get("observation_payload"))
    structured = obs_payload.get("structured_findings")

    generator_response = _as_dict(input_payload.get("generator_response"))
    generator_output = _as_dict(generator_response.get("output"))
    generated = (
        row.get("generated_impression")
        or generator_output.get("impression_text")
        or input_payload.get("impression_text")
        or row.get("impression_text")
    )
    human = row.get("human_impression") or row.get("final_impression")

    result = calculate_clinical_accuracy_metrics_llm(
        structured_findings=structured,
        generated_impression=generated,
        human_impression=human,
        exam=exam,
    )
    print(json.dumps(asdict(result) if result else None, indent=2))
