"""
Field Performance Analytics Module — Layer 3

Computes per-field error rates for structured data elements in radiology
impressions.  These metrics identify WHERE the model performs poorly to
drive retraining decisions.

FIELD CHECKS
---------------------
1. Laterality Error Rate:  Wrong left/right/bilateral attribution?
2. Severity Error Rate:    Wrong mild/moderate/severe grading?
3. Projection Error Rate:  Wrong PA/AP/lateral view reference?
4. Measurement Error Rate: Wrong or missing lesion size?

RELATIONSHIP TO LAYER 1 (llm_clinical_accuracy.py)
-----------------------------------------------
Layer 1 computes composite scores (Recall, Precision, CFCR) and per-finding
accuracy checks for clinical accuracy evaluation.

Layer 3 reuses the same per-finding mention-detection infrastructure but
focuses on per-FIELD error rates for structured data quality — a distinct
concern that drives retraining decisions rather than clinical safety.

UPSTREAM REQUIREMENTS
----------------------
- Laterality:   finding.location (already populated)
- Severity:     finding.severity (already populated)
- Projection:   case.exam.projection (NEW — added to ExamContext)
- Measurement:  finding.measurements.size_mm (existing field — populate value)

ASSUMPTIONS
-----------
A1: Projection is study-level (from case.exam.projection), not finding-level.
    All findings in the same study share the same expected projection.
A2: Measurement concordance uses ±10% tolerance to account for rounding
    differences (e.g., 15 mm reported as "1.5 cm").
A3: Projection checks are only meaningful for X-ray modalities (DX, CR, MG).
    For CT/MR/US, exam.projection will be null → check skipped.
A4: Multi-view projections (e.g., "PA and lateral") are decomposed into
    discrete views; EACH required view must be found in the impression.
A5: Attribute accuracy is conditional on the finding being mentioned.
    If the AI omitted the finding entirely, Layer 3 field checks are
    skipped for that finding.  The omission is a Layer 1 Recall issue,
    not a Layer 3 attribute error.  This prevents corrupted denominators
    where laterality/severity get free passes and measurement gets
    double-penalized for the same omission.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.text_utils import (
    LATERALITY_KEYWORDS,
    LOCATION_LATERALITY,
    SEVERITY_KEYWORDS,
    _check_laterality,
    _check_severity,
    _find_assertion_sentence,
    _is_label_mentioned,
    _normalize,
    _split_into_sentences,
    _word_in_text,
)


# SECTION 1: PROJECTION KEYWORDS

PROJECTION_KEYWORDS: Dict[str, List[str]] = {
    "pa":       ["pa", "posteroanterior", "posterior-anterior"],
    "ap":       ["ap", "anteroposterior", "anterior-posterior"],
    "lateral":  ["lateral"],
    "oblique":  ["oblique"],
    "frontal":  ["frontal", "pa", "ap", "posteroanterior", "anteroposterior"],
    "portable": ["portable", "upright"],
    "upright":  ["upright", "erect", "standing"],
    "supine":   ["supine", "recumbent"],
}


# SECTION 2: MEASUREMENT EXTRACTION

_MEASUREMENT_PATTERN: re.Pattern = re.compile(
    r'(\d+\.?\d*)\s*(?:x|by)?\s*(\d+\.?\d*)?\s*(cm|mm|centimeters?|millimeters?)',
    re.IGNORECASE,
)

_CM_TO_MM: float = 10.0


# SECTION 3: DATA MODELS


@dataclass(frozen=True)
class FieldPerformanceCheck:
    """
    Per-finding field-level error assessment for one structured finding.

    Captures laterality, severity, projection, and measurement checks
    in a single flat record per finding for BQ ingestion.
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


@dataclass(frozen=True)
class FieldPerformanceMetrics:
    """
    Layer 3 — Field Performance Analytics for a single request.

    Per-finding checks feed BQ UNNEST for dashboard aggregation.
    Summary error rates provide request-level quick stats.
    """
    field_checks: List[FieldPerformanceCheck]

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


# SECTION 4: PROJECTION CHECK (Multi-View Decomposition)


def _extract_required_views(projection_text: str) -> List[str]:
    """
    Decompose a multi-view projection string into discrete view keys.

    Uses word-boundary matching to avoid substring false positives
    (e.g., "ap" must not match inside "lateral").

    Examples:
        "PA and lateral"  → ["pa", "lateral"]
        "AP portable"     → ["ap", "portable"]
        "frontal"         → ["frontal"]
    """
    proj_lower = projection_text.strip().lower()
    required: List[str] = []
    for view_key in PROJECTION_KEYWORDS:
        if _word_in_text(proj_lower, view_key):
            required.append(view_key)
    return required


def _check_projection(
    expected_projection: str,
    assertion_sentence: str,
) -> Tuple[bool, bool]:
    """
    Check if the assertion sentence conveys ALL required projection views.

    Multi-view decomposition: "PA and lateral" requires both "pa" AND
    "lateral" to appear in the assertion sentence.  If any required
    view is missing → error.

    Args:
        expected_projection: From case.exam.projection.
        assertion_sentence: The specific sentence where the finding is
            positively asserted (from _find_assertion_sentence).

    Returns:
        (applicable, error):
        - applicable: True if projection check was meaningful.
        - error: True if any required view was missing.
    """
    if not expected_projection or not isinstance(expected_projection, str):
        return False, False
    if not expected_projection.strip():
        return False, False

    required_views = _extract_required_views(expected_projection)
    if not required_views:
        return False, False

    for view in required_views:
        view_matched = False
        for keyword in PROJECTION_KEYWORDS[view]:
            if _word_in_text(assertion_sentence, keyword):
                view_matched = True
                break
        if not view_matched:
            return True, True

    return True, False


# SECTION 5: MEASUREMENT CHECK (Multi-Dimensional Extraction)


def _extract_measurement_from_sentence(
    sentence: str,
) -> List[Tuple[float, str]]:
    """
    Extract measurement values with units from a sentence.

    Handles single values ("15 mm", "1.5 cm") and multi-dimensional
    measurements ("12 x 15 mm", "3.0 by 2.5 cm").

    Returns:
        List of (value_in_mm, original_unit) tuples.
    """
    matches = _MEASUREMENT_PATTERN.findall(sentence)
    results: List[Tuple[float, str]] = []
    for first_val, second_val, unit in matches:
        try:
            value = float(first_val)
        except ValueError:
            continue
        unit_lower = unit.lower().rstrip("s")
        if unit_lower in ("cm", "centimeter"):
            value *= _CM_TO_MM
            unit = "mm"
        results.append((value, unit))
        if second_val:
            try:
                value2 = float(second_val)
            except ValueError:
                continue
            unit2 = "mm"
            if unit_lower in ("cm", "centimeter"):
                value2 *= _CM_TO_MM
            results.append((value2, unit2))
    return results


def _check_measurement(
    finding: Dict[str, Any],
    normalized_text: str,
    label: str,
    label_mentioned: bool,
) -> Tuple[bool, bool, Optional[float], Optional[float]]:
    """
    Check if the impression conveys the correct measurement for a finding.

    Uses sentence-level isolation: measurement keywords are only searched
    within the assertion sentence where the finding label is positively
    asserted.

    Args:
        finding: Structured finding dict with 'measurements' field.
        normalized_text: Lowercase impression text.
        label: Finding label for locating assertion sentence.
        label_mentioned: Whether the label is positively asserted.

    Returns:
        (applicable, error, expected_mm, mentioned_mm)
    """
    measurements = finding.get("measurements")
    if not isinstance(measurements, dict) or not measurements:
        return False, False, None, None

    expected_mm = measurements.get("size_mm")
    if not isinstance(expected_mm, (int, float)):
        return False, False, None, None

    tolerance = measurements.get("tolerance_pct", 10.0)

    if not label_mentioned:
        return True, True, float(expected_mm), None

    sentences = _split_into_sentences(normalized_text)
    label_lower = label.strip().lower()
    assertion_sentence = _find_assertion_sentence(label_lower, sentences)

    if assertion_sentence is None:
        return True, True, float(expected_mm), None

    extracted = _extract_measurement_from_sentence(assertion_sentence)
    if not extracted:
        return True, True, float(expected_mm), None

    for value_mm, _ in extracted:
        diff_pct = (abs(value_mm - expected_mm) / expected_mm) * 100
        if diff_pct <= tolerance:
            return True, False, float(expected_mm), value_mm

    return True, True, float(expected_mm), extracted[0][0] if extracted else None


# SECTION 6: APPLICABILITY HELPERS


def _is_laterality_applicable(finding: Dict[str, Any]) -> bool:
    """Check if finding has location data with recognizable laterality."""
    location = finding.get("location")
    if not isinstance(location, str) or not location.strip():
        return False
    location_lower = location.strip().lower()
    for loc_term, _ in LOCATION_LATERALITY:
        if loc_term in location_lower:
            return True
    return False


def _is_severity_applicable(finding: Dict[str, Any]) -> bool:
    """Check if finding has a recognizable severity value."""
    severity = finding.get("severity")
    if not isinstance(severity, str) or not severity.strip():
        return False
    severity_lower = severity.strip().lower()
    for sev_key in SEVERITY_KEYWORDS:
        if sev_key in severity_lower or severity_lower in sev_key:
            return True
    return False


def _has_measurements(finding: Dict[str, Any]) -> bool:
    """Check if finding has measurement data with a numeric size."""
    measurements = finding.get("measurements")
    if not isinstance(measurements, dict) or not measurements:
        return False
    return isinstance(measurements.get("size_mm"), (int, float))


# SECTION 7: MAIN ENTRY POINT


def _safe_rate(numerator: int, denominator: int) -> Optional[float]:
    """Compute a rate, returning None when denominator is zero."""
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def calculate_field_performance_metrics(
    *,
    structured_findings: Any,
    generated_impression: Optional[str],
    human_impression: Optional[str],
    exam: Optional[Dict[str, Any]] = None,
) -> Optional[FieldPerformanceMetrics]:
    """
    Compute field-level performance analytics for one request.

    Checks 4 structured field dimensions against impression text:
    1. Laterality — from finding.location
    2. Severity   — from finding.severity
    3. Projection — from case.exam.projection (study-level)
    4. Measurement — from finding.measurements.size_mm

    All checks use sentence-level isolation via _find_assertion_sentence()
    to prevent cross-contamination between findings.

    CONDITIONAL PROBABILITY INVARIANT:
    Attribute accuracy is conditional on the finding being mentioned by
    the AI.  If the AI omitted the finding entirely (False Negative),
    all field checks for that finding are skipped.  The omission is
    tracked by clinical metrics (Recall / FNR), not here.  This prevents
    corrupted denominators where laterality/severity get free passes
    and measurement gets double-penalized for the same omission.

    Args:
        structured_findings: List from observation_payload.structured_findings.
        generated_impression: AI-generated impression text.
        human_impression: Radiologist-finalized impression text.
        exam: case.exam dict with optional 'projection' field.

    Returns:
        FieldPerformanceMetrics with per-finding checks and summary rates.
        None when data is insufficient.
    """
    if not isinstance(structured_findings, list) or len(structured_findings) == 0:
        return None

    has_generated = bool(generated_impression and generated_impression.strip())
    has_human = bool(human_impression and human_impression.strip())
    if not has_generated and not has_human:
        return None

    gen_text = _normalize(generated_impression)
    human_text = _normalize(human_impression)

    exam = exam or {}
    expected_projection = exam.get("projection")
    if not isinstance(expected_projection, str):
        expected_projection = None

    checks: List[FieldPerformanceCheck] = []

    lat_applicable = 0
    lat_errors = 0
    sev_applicable = 0
    sev_errors = 0
    proj_applicable = 0
    proj_errors = 0
    meas_applicable = 0
    meas_errors = 0
    total_applicable = 0
    total_errors = 0

    for finding in structured_findings:
        if not isinstance(finding, dict):
            continue

        label = finding.get("label", "")
        presence = str(finding.get("presence", "")).strip().lower()

        if not isinstance(label, str) or not label.strip():
            continue
        if presence not in ("present", "absent"):
            continue

        label = label.strip()

        gen_mentioned, _ = _is_label_mentioned(label, gen_text) \
            if has_generated else (False, False)

        # Layer 3 invariant: attribute accuracy is conditional on mention.
        # If the AI omitted the finding, skip all field checks for it.
        # The omission is a Layer 1 Recall issue, not a Layer 3 error.
        if not gen_mentioned:
            continue

        #  Laterality (sentence-isolated)
        lat_app, lat_err = False, False
        if _is_laterality_applicable(finding):
            lat_app = True
            if has_generated:
                lat_match = _check_laterality(finding, gen_text, True)
                if lat_match is False:
                    lat_err = True

        # Severity (sentence-isolated)
        sev_app, sev_err = False, False
        if _is_severity_applicable(finding):
            sev_app = True
            if has_generated:
                sev_match = _check_severity(finding, gen_text, True)
                if sev_match is False:
                    sev_err = True

        #  Projection (sentence-isolated, multi-view)
        proj_app, proj_err = False, False
        if expected_projection and has_generated:
            label_lower = label.lower()
            sentences = _split_into_sentences(gen_text)
            assertion_sentence = _find_assertion_sentence(label_lower, sentences)
            if assertion_sentence is not None:
                proj_app, proj_err = _check_projection(
                    expected_projection, assertion_sentence,
                )

        # Measurement (sentence-isolated, multi-dimensional)
        m_app, m_err, m_expected, m_mentioned = False, False, None, None
        if _has_measurements(finding):
            m_app, m_err, m_expected, m_mentioned = _check_measurement(
                finding, gen_text, label, True,
            )

        # Accumulate counts
        for app_flag, err_flag in [
            (lat_app, lat_err), (sev_app, sev_err),
            (proj_app, proj_err), (m_app, m_err),
        ]:
            if app_flag:
                total_applicable += 1
                if err_flag:
                    total_errors += 1

        if lat_app:
            lat_applicable += 1
            if lat_err:
                lat_errors += 1
        if sev_app:
            sev_applicable += 1
            if sev_err:
                sev_errors += 1
        if proj_app:
            proj_applicable += 1
            if proj_err:
                proj_errors += 1
        if m_app:
            meas_applicable += 1
            if m_err:
                meas_errors += 1

        checks.append(FieldPerformanceCheck(
            label=label,
            presence=presence,
            location=finding.get("location") if isinstance(finding.get("location"), str) else None,
            severity=finding.get("severity") if isinstance(finding.get("severity"), str) else None,
            laterality_applicable=lat_app,
            laterality_error=lat_err,
            severity_applicable=sev_app,
            severity_error=sev_err,
            projection_applicable=proj_app,
            projection_error=proj_err,
            measurement_applicable=m_app,
            measurement_error=m_err,
            measurement_expected_mm=m_expected,
            measurement_mentioned_mm=m_mentioned,
        ))

    return FieldPerformanceMetrics(
        field_checks=checks,
        laterality_applicable=lat_applicable,
        laterality_errors=lat_errors,
        laterality_error_rate=_safe_rate(lat_errors, lat_applicable),
        severity_applicable=sev_applicable,
        severity_errors=sev_errors,
        severity_error_rate=_safe_rate(sev_errors, sev_applicable),
        projection_applicable=proj_applicable,
        projection_errors=proj_errors,
        projection_error_rate=_safe_rate(proj_errors, proj_applicable),
        measurement_applicable=meas_applicable,
        measurement_errors=meas_errors,
        measurement_error_rate=_safe_rate(meas_errors, meas_applicable),
        total_field_checks=total_applicable,
        total_field_errors=total_errors,
        overall_field_error_rate=_safe_rate(total_errors, total_applicable),
    )