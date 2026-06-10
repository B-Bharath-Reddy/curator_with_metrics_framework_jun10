"""
Clinical Accuracy Metrics Module

Computes 7 clinically grounded accuracy metrics by comparing impression
text against structured findings (the authoritative clinical ground truth).

GROUND TRUTH
-------------
Structured findings from the upstream observation pipeline are the
authoritative clinical truth.  Both the Generator and the radiologist
write narrative impressions based on these SAME findings.  The
structured findings do not change — only the impression text differs.

MATCHING STRATEGY
-----------------
Primary matching is EXACT case-insensitive label check within
sentence-level boundaries.  Since the Generator receives structured
findings as input, it uses the same label vocabulary in the impression
text.  A small abbreviation map covers common radiology shorthand
(PTX, fx, PE, etc.) for edge cases.

SENTENCE-LEVEL ISOLATION
------------------------
All mention detection and modifier (laterality, severity) matching
operates on individual sentences, not the entire impression block.
This prevents cross-contamination in multi-finding reports where, for
example, "left" in sentence 1 should not be attributed to a finding
in sentence 3.  The impression is split into sentences using a
regex-aware splitter that preserves decimal values (e.g., "1.5 cm").

NEGATION DETECTION
------------------
Each sentence is checked for clinical negation cues ("no", "not",
"without", "absent", "free of", etc.) that precede or follow the
finding label.  A negated mention ("no pneumothorax") is NOT counted
as a successful finding capture — it is treated as an intentional
omission or ruling-out of the finding.  This prevents false-positive
inflation of Recall and Critical Capture rates.

This approach is SELF-ADAPTING: it works for ANY payload regardless
of body part, finding types, or taxonomy changes.  No pre-built
body-part-specific label sets are needed — the module extracts its
vocabulary directly from each payload's structured findings.

THE 7 METRICS (Layer 1 — Clinical Accuracy)
--------------------------------------------
1. Finding-level Recall: What fraction of present findings are mentioned?
2. Finding-level Precision: Are mentioned findings actually present?
3. False Negative Rate: What fraction of present findings are omitted?
4. Critical Finding Capture Rate: Life-threatening findings captured?
5. Laterality Accuracy: Correct left/right/bilateral attribution?
6. Severity Accuracy: Correct mild/moderate/severe attribution?
7. Impression Agreement Score: Do gen and human mention same findings?

ASSUMPTIONS (explicitly documented per project convention)
-----------------------------------------------------------
A1: The Generator uses the same finding labels in impression text
    as in the structured findings.  Verified against test payloads.

A2: Negation tracking is implemented.  "No pneumothorax" is recognized
    via sentence-level negation boundary detection and does NOT count
    as a successful finding capture.  This prevents false-positive
    inflation of Recall and Critical Finding Capture Rate when the
    impression explicitly rules out a condition that is present in the
    structured findings ground truth.  Negation keywords include
    standard clinical denial phrases: "no", "not", "without", "absent",
    "free of", "clear of", "no evidence of", "negative for", etc.

A3: Modifier attribution (laterality and severity) is isolated to the
    sentence-level block where the finding label was asserted.  In a
    multi-sentence report such as "1. Severe right pneumothorax.
    2. Left pleural effusion.", the laterality "right" is attributed
    only to pneumothorax and "left" only to pleural effusion.  This
    eliminates cross-contamination that would occur with paragraph-wide
    keyword scanning.  If a finding is negated or absent from the
    impression, modifier checks return None immediately.

CLARIFICATION NEEDED
--------------------
C1: The critical finding taxonomy (CRITICAL_FINDINGS) should be validated
    by a clinical advisor.  It is derived from the classifier's "always
    complex" routing rules, which capture clinical significance but may
    not exactly match patient-safety urgency levels.

C2: For Finding-level Precision, hallucination detection currently only
    checks labels from the structured findings list.  If the Generator
    mentions a finding NOT in the structured list using a term we don't
    know about, we won't catch it.  Full precision would require the
    Generator to output structured findings, or a clinical NER module.
"""

from __future__ import annotations


from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
from app.text_utils import (
    ABBREVIATIONS,
    LATERALITY_KEYWORDS,
    LOCATION_LATERALITY,
    NEGATION_KEYWORDS,
    SEVERITY_KEYWORDS,
    _check_laterality,
    _check_severity,
    _find_assertion_sentence,
    _is_label_mentioned,
    _is_sentence_negated,
    _normalize,
    _split_into_sentences,
    _word_in_text,
)




# SECTION 1: CRITICAL FINDING TAXONOMY



"""
Simplified critical finding detection using only upstream structured
finding data.  No pre-built taxonomy required.

A finding is considered critical when:
  1. It has a non-empty label, AND
  2. Its severity field contains a clinically urgent term.

Urgent severity terms: "severe", "moderate", "large", "displaced",
"displacement", "angulated", "angulation", "tension".

This is SELF-ADAPTING — works for any body part or finding type
because it reads directly from each payload's structured findings.
"""

CRITICAL_SEVERITY_TERMS: List[str] = [
    "severe",
    "moderate",
    "large",
    "displaced",
    "displacement",
    "angulated",
    "angulation",
    "tension",
]





# SECTION 1: DATA MODELS



@dataclass(frozen=True)
class FindingAccuracyCheck:
    """
    Per-finding accuracy assessment for one structured finding.

    Stored for audit traceability — dashboards aggregate these, but
    individual checks let you drill down into which specific finding
    was missed and in which impression.

    Example:
        FindingAccuracyCheck(
            label="Pneumothorax",
            presence="present",
            severity="severe",
            location="bilateral",
            mentioned_in_generated=True,
            mentioned_in_human=True,
            laterality_match_generated=False,  # gen said "right" not "bilateral"
            laterality_match_human=True,       # human said "bilateral"
            severity_match_generated=True,
            severity_match_human=True,
            is_critical=True,
            critical_captured_generated=True,
            critical_captured_human=True,
        )
    """
    label: str
    presence: str
    severity: Optional[str]
    location: Optional[str]
    mentioned_in_generated: bool
    mentioned_in_human: bool
    laterality_match_generated: Optional[bool]
    laterality_match_human: Optional[bool]
    severity_match_generated: Optional[bool]
    severity_match_human: Optional[bool]
    is_critical: bool
    critical_captured_generated: bool
    critical_captured_human: bool


@dataclass(frozen=True)
class ClinicalAccuracyMetrics:
    """
    Per-request clinical accuracy metrics comparing impression text
    against structured findings (the authoritative clinical ground truth).

    All rates are None when the denominator is zero (data unavailable).
    Store None as NULL in BigQuery — never as 0.0, to distinguish
    "not computed" from "computed as 0.0".

    METRICS DEFINITIONS:

    finding_recall_generated / finding_recall_human:
        What fraction of present findings are mentioned in the impression?
        Formula: #mentioned / #present_in_structured_findings
        Target: > 95%.  Low recall = missed findings.

    finding_precision_generated / finding_precision_human:
        Of findings mentioned in the impression, how many are actually
        present in structured findings?  Detects hallucinations.
        Formula: #mentioned_and_present / #mentioned
        Target: > 95%.  Low precision = hallucinated findings.

    false_negative_rate_generated / false_negative_rate_human:
        What fraction of present findings are completely omitted?
        Formula: #not_mentioned / #present
        Target: < 2%.  FNR = 1 - Recall.

    critical_finding_capture_rate_generated / _human:
        Among findings classified as critical (life-threatening),
        what fraction are mentioned in the impression?
        Formula: #critical_mentioned / #critical_total
        Target: > 99%.  A missed critical finding is a safety event.

    laterality_accuracy_generated / _human:
        Among findings with laterality (left/right/bilateral),
        does the impression convey the correct side?
        Formula: #laterality_correct / #with_laterality
        Target: > 98%.  Wrong-side errors are dangerous.

    severity_accuracy_generated / _human:
        Among findings with severity (mild/moderate/severe),
        does the impression convey the correct severity level?
        Formula: #severity_correct / #with_severity
        Target: > 90%.

    impression_agreement_score:
        Finding-level Jaccard similarity between generated and human
        impressions.  Do they mention the same findings?
        Formula: |gen_mentions ∩ human_mentions| / |gen_mentions ∪ human_mentions|
        This is finding-level overlap, NOT text similarity.
        Two impressions worded differently but same findings → 1.0.

    clinical_accuracy_score:
        Weighted composite score for quick comparison.
        Formula: 0.30×recall + 0.25×critical_capture + 0.20×(1-FNR)
                 + 0.15×laterality + 0.10×severity
        Unavailable components excluded; remaining weights scale up.
    """
    # --- Layer 1 metrics (generated impression vs structured findings) ---
    finding_recall_generated: Optional[float]
    finding_precision_generated: Optional[float]
    false_negative_rate_generated: Optional[float]
    critical_finding_capture_rate_generated: Optional[float]
    laterality_accuracy_generated: Optional[float]
    severity_accuracy_generated: Optional[float]
    impression_agreement_score: Optional[float]

    # --- Same metrics for human impression (comparison baseline) ---
    finding_recall_human: Optional[float]
    finding_precision_human: Optional[float]
    false_negative_rate_human: Optional[float]
    critical_finding_capture_rate_human: Optional[float]
    laterality_accuracy_human: Optional[float]
    severity_accuracy_human: Optional[float]

    # --- Absent finding accuracy ---
    absent_finding_accuracy_generated: Optional[float]
    absent_finding_accuracy_human: Optional[float]

    # --- Counts for dashboard denominators ---
    total_findings_checked: int
    present_findings_count: int
    critical_findings_count: int
    findings_with_laterality: int
    findings_with_severity: int
    absent_findings_count: int

    # --- Miss/hallucination lists for drill-down ---
    missed_findings_generated: List[str]
    missed_findings_human: List[str]
    missed_critical_generated: List[str]
    missed_critical_human: List[str]

    # --- Per-finding audit trail ---
    finding_checks: List[FindingAccuracyCheck]

    # --- Composite score ---
    clinical_accuracy_score: Optional[float]


# SECTION 3: LAYER 1 HELPERS


def _is_critical_finding(finding: Dict[str, Any]) -> bool:
    """
    Check if a structured finding is critical using only upstream data.

    A finding is critical when it has a non-empty label AND its
    severity field contains a clinically urgent term.

    This is a simplified, self-adapting approach — no pre-built
    taxonomy or qualifier mapping needed.

    Args:
        finding: Structured finding dict with 'label' and 'severity'.

    Returns:
        True if the finding is clinically urgent, False otherwise.
    """
    label = finding.get("label", "")
    if not isinstance(label, str) or not label.strip():
        return False

    severity = str(finding.get("severity") or "").lower().strip()
    if not severity:
        return False

    for term in CRITICAL_SEVERITY_TERMS:
        if term in severity:
            return True

    return False


def _extract_mentioned_labels(
    reference_labels: List[str],
    normalized_text: str,
) -> Set[str]:
    """
    Check which reference labels are positively asserted in impression
    text (negation-aware).

    Self-adapting: uses the labels from the current payload's structured
    findings, not a pre-built list.  This works for any body part or
    finding combination without configuration.

    A label is included in the returned set only if it is found in a
    non-negated sentence.  Labels that appear only in negated contexts
    (e.g., "no pleural effusion") are excluded.

    Returns a set of labels (original casing) that are positively
    asserted.
    """
    mentioned: Set[str] = set()
    for label in reference_labels:
        is_mentioned, _ = _is_label_mentioned(label, normalized_text)
        if is_mentioned:
            mentioned.add(label)
    return mentioned


def _safe_rate(numerator: int, denominator: int) -> Optional[float]:
    """Compute a rate, returning None when denominator is zero."""
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)



# MAIN ENTRY POINT



def calculate_clinical_accuracy_metrics(
    structured_findings: Any,
    generated_impression: Optional[str],
    human_impression: Optional[str],
) -> Optional[ClinicalAccuracyMetrics]:
    """
    Compute per-request clinical accuracy metrics.

    Self-adapting: extracts all vocabulary from the payload's structured
    findings.  Works for any body part, any finding types, any
    combination — no body-part-specific configuration needed.

    Args:
        structured_findings: List of finding dicts from
            inputs.observation_payload.structured_findings.
            This is the authoritative clinical ground truth.
        generated_impression: AI-generated impression text.
        human_impression: Radiologist-finalized impression text.

    Returns:
        ClinicalAccuracyMetrics with all 7 metrics + per-finding checks.
        Returns None when data is insufficient (no structured findings
        or no impression text available).
    """
    # Validate input: must have structured findings
    if not isinstance(structured_findings, list) or len(structured_findings) == 0:
        return None

    # Validate input: must have at least one impression
    has_generated = bool(generated_impression and generated_impression.strip())
    has_human = bool(human_impression and human_impression.strip())
    if not has_generated and not has_human:
        return None

    # Normalize impression text for matching
    gen_text = _normalize(generated_impression)
    human_text = _normalize(human_impression)

    # Extract all finding labels from this payload
    # These are the vocabulary for this specific request.
    # No pre-built label sets needed — adapts to any payload.
    all_labels: List[str] = []
    for finding in structured_findings:
        if not isinstance(finding, dict):
            continue
        label = finding.get("label")
        if isinstance(label, str) and label.strip():
            all_labels.append(label.strip())

    # Extract labels mentioned in each impression
    # Used for Precision and Agreement Score.
    gen_mentioned_all = _extract_mentioned_labels(all_labels, gen_text) \
        if has_generated else set()
    human_mentioned_all = _extract_mentioned_labels(all_labels, human_text) \
        if has_human else set()

    # --- Build reference sets from structured findings ---
    # Only findings with presence="present" count as ground truth.
    reference_present: Set[str] = set()
    for finding in structured_findings:
        if not isinstance(finding, dict):
            continue
        label = finding.get("label", "")
        presence = str(finding.get("presence", "")).strip().lower()
        if isinstance(label, str) and label.strip() and presence == "present":
            reference_present.add(label.strip())


    # Per-finding evaluation

    finding_checks: List[FindingAccuracyCheck] = []

    present_count = 0
    gen_recall_hits = 0
    human_recall_hits = 0
    absent_count = 0
    absent_correct_gen = 0
    absent_correct_human = 0
    critical_count = 0
    critical_captured_gen = 0
    critical_captured_human = 0
    lat_applicable = 0
    lat_correct_gen = 0
    lat_correct_human = 0
    sev_applicable = 0
    sev_correct_gen = 0
    sev_correct_human = 0

    missed_gen: List[str] = []
    missed_human: List[str] = []
    missed_critical_gen: List[str] = []
    missed_critical_human: List[str] = []

    for finding in structured_findings:
        if not isinstance(finding, dict):
            continue

        label = finding.get("label", "")
        presence = str(finding.get("presence", "")).strip().lower()
        severity = finding.get("severity")
        location = finding.get("location")

        if not isinstance(label, str) or not label.strip():
            continue

        label = label.strip()

        # Is this finding positively asserted in each impression?
        # _is_label_mentioned returns (is_mentioned, is_negated).
        # is_mentioned=True only when the label appears in a non-negated
        # sentence. is_negated=True when the label appears only in
        # negated sentences (useful for absent-finding accuracy).
        gen_mentioned, gen_negated = _is_label_mentioned(label, gen_text) \
            if has_generated else (False, False)
        human_mentioned, human_negated = _is_label_mentioned(label, human_text) \
            if has_human else (False, False)

        # Laterality check
        lat_gen = _check_laterality(finding, gen_text, gen_mentioned) \
            if has_generated else None
        lat_human = _check_laterality(finding, human_text, human_mentioned) \
            if has_human else None

        # Severity check
        sev_gen = _check_severity(finding, gen_text, gen_mentioned) \
            if has_generated else None
        sev_human = _check_severity(finding, human_text, human_mentioned) \
            if has_human else None

        # Critical finding classification
        is_critical = _is_critical_finding(finding)

        # Accumulate counts by presence type
        crit_cap_gen = False
        crit_cap_human = False

        if presence == "present":
            present_count += 1

            if gen_mentioned:
                gen_recall_hits += 1
            else:
                missed_gen.append(label)

            if human_mentioned:
                human_recall_hits += 1
            else:
                missed_human.append(label)

            if is_critical:
                critical_count += 1
                if gen_mentioned:
                    critical_captured_gen += 1
                    crit_cap_gen = True
                else:
                    missed_critical_gen.append(label)
                if human_mentioned:
                    critical_captured_human += 1
                    crit_cap_human = True
                else:
                    missed_critical_human.append(label)

        elif presence == "absent":
            # Negation-aware absent finding check.
            # A finding with presence="absent" is correctly handled when:
            #   - The impression does NOT mention the label at all, OR
            #   - The impression mentions it in a negated context
            #     (e.g., "no pleural effusion"), which correctly reflects
            #     the absent status.

            absent_count += 1
            if has_generated and (not gen_mentioned):
                absent_correct_gen += 1
            if has_human and (not human_mentioned):
                absent_correct_human += 1

        # Laterality/severity accumulation (only if applicable)
        if lat_gen is not None:
            lat_applicable += 1
            if lat_gen:
                lat_correct_gen += 1
            if lat_human:
                lat_correct_human += 1

        if sev_gen is not None:
            sev_applicable += 1
            if sev_gen:
                sev_correct_gen += 1
            if sev_human:
                sev_correct_human += 1

        finding_checks.append(FindingAccuracyCheck(
            label=label,
            presence=presence,
            severity=severity if isinstance(severity, str) else None,
            location=location if isinstance(location, str) else None,
            mentioned_in_generated=gen_mentioned,
            mentioned_in_human=human_mentioned,
            laterality_match_generated=lat_gen,
            laterality_match_human=lat_human,
            severity_match_generated=sev_gen,
            severity_match_human=sev_human,
            is_critical=is_critical,
            critical_captured_generated=crit_cap_gen,
            critical_captured_human=crit_cap_human,
        ))


    # Aggregate metrics


    # 1. Recall & False Negative Rate
    # Recall = #mentioned / #present.  FNR = 1 - Recall.
    recall_gen = _safe_rate(gen_recall_hits, present_count)
    recall_human = _safe_rate(human_recall_hits, present_count)
    fnr_gen = _safe_rate(present_count - gen_recall_hits, present_count)
    fnr_human = _safe_rate(present_count - human_recall_hits, present_count)

    # 2. Precision
    # Precision = #mentioned_and_actually_present / #mentioned.
    # gen_mentioned_all = all labels from this payload found in gen text.
    # reference_present = labels with presence="present" in structured findings.
    # precision = |mentioned ∩ reference_present| / |mentioned|
    # Detects hallucinations: findings mentioned but not in ground truth.
    gen_true_pos = gen_mentioned_all & reference_present
    human_true_pos = human_mentioned_all & reference_present
    precision_gen = _safe_rate(len(gen_true_pos), len(gen_mentioned_all))
    precision_human = _safe_rate(len(human_true_pos), len(human_mentioned_all))

    # 3. Critical Finding Capture Rate
    # Among critical findings, what fraction are mentioned?
    crit_cap_gen = _safe_rate(critical_captured_gen, critical_count)
    crit_cap_human = _safe_rate(critical_captured_human, critical_count)

    #  4. Laterality Accuracy
    # Among findings with laterality, is the correct side mentioned?
    lat_acc_gen = _safe_rate(lat_correct_gen, lat_applicable)
    lat_acc_human = _safe_rate(lat_correct_human, lat_applicable)

    # 5. Severity Accuracy
    # Among findings with severity, is the correct level mentioned?
    sev_acc_gen = _safe_rate(sev_correct_gen, sev_applicable)
    sev_acc_human = _safe_rate(sev_correct_human, sev_applicable)

    #  6. Absent Finding Accuracy
    # Among absent findings, are they correctly omitted from impressions?
    absent_acc_gen = _safe_rate(absent_correct_gen, absent_count)
    absent_acc_human = _safe_rate(absent_correct_human, absent_count)

    # 7. Impression Agreement Score (finding-level Jaccard)
    # Do generated and human impressions mention the same findings?
    # This is finding-level overlap, NOT text similarity.
    # Jaccard = |gen ∩ human| / |gen ∪ human|
    # Clinically more meaningful than edit distance:
    # - Two impressions worded differently but same findings → 1.0
    # - Two impressions differ by "right" vs "left" but same labels → 1.0
    #   (laterality error is caught by laterality_accuracy, not here)
    if gen_mentioned_all or human_mentioned_all:
        intersection = gen_mentioned_all & human_mentioned_all
        union = gen_mentioned_all | human_mentioned_all
        agreement = round(len(intersection) / len(union), 4) \
            if len(union) > 0 else None
    else:
        agreement = None


    # Composite Clinical Accuracy Score

    # Weighted composite for the GENERATED impression.
    # Unavailable components are excluded; remaining weights scale up.

    # Weights:
    #   0.30 — Recall: primary clinical utility.
    #   0.25 — Critical capture: patient safety.
    #   0.20 — (1 - FNR): reinforces the safety signal.
    #   0.15 — Laterality: wrong-side errors are dangerous.
    #   0.10 — Severity: important but less critical than misses.
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
        total_weight = sum(w for _, w in components)
        if total_weight > 0:
            clinical_accuracy_score = round(
                sum(s * w for s, w in components) / total_weight, 4
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
        findings_with_laterality=lat_applicable,
        findings_with_severity=sev_applicable,
        absent_findings_count=absent_count,

        missed_findings_generated=missed_gen,
        missed_findings_human=missed_human,
        missed_critical_generated=missed_critical_gen,
        missed_critical_human=missed_critical_human,

        finding_checks=finding_checks,
        clinical_accuracy_score=clinical_accuracy_score,
    )
