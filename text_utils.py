"""
Shared Radiology Text Utilities

Pure NLP utilities for radiology impression text analysis.
No metric computation — these are building blocks used by both
Layer 1 (clinical accuracy) and Layer 3 (field performance).

SENTENCE-LEVEL ISOLATION
------------------------
All mention detection and modifier matching operates on individual
sentences, not the entire impression block. This prevents cross-
contamination in multi-finding reports.

NEGATION DETECTION
------------------
Each sentence is checked for clinical negation cues. A negated
mention ("no pneumothorax") is NOT counted as a successful finding
capture.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


ABBREVIATIONS: Dict[str, str] = {
    "ptx":  "pneumothorax",
    "fx":   "fracture",
    "pe":   "pulmonary embolism",
    "ett":  "endotracheal tube",
    "sbo":  "small bowel obstruction",
    "ich":  "intracranial hemorrhage",
    "sdh":  "subdural hematoma",
    "edh":  "epidural hematoma",
    "sah":  "subarachnoid hemorrhage",
    "tb":   "tuberculosis",
    "oa":   "osteoarthritis",
    "druj": "distal radioulnar joint",
}

_ABBREV_REVERSE: Dict[str, str] = {v: k for k, v in ABBREVIATIONS.items()}


SEVERITY_KEYWORDS: Dict[str, List[str]] = {
    "mild":     ["mild", "small", "minimal", "trace"],
    "moderate": ["moderate"],
    "severe":   ["severe", "large", "extensive", "massive"],
    "displaced": ["displaced", "displacement", "angulation", "angulated"],
}

LATERALITY_KEYWORDS: Dict[str, List[str]] = {
    "left":      ["left"],
    "right":     ["right"],
    "bilateral": ["bilateral", "bilaterally", "both"],
    "midline":   ["midline", "central"],
}

LOCATION_LATERALITY: List[Tuple[str, str]] = [
    ("left upper lobe",  "left"),
    ("left lower lobe",  "left"),
    ("right upper lobe", "right"),
    ("right lower lobe", "right"),
    ("left",             "left"),
    ("right",            "right"),
    ("bilateral",        "bilateral"),
]


NEGATION_KEYWORDS: List[str] = [
    "no",
    "not",
    "none",
    "negative",
    "clear of",
    "without",
    "free of",
    "absent",
    "denies",
    "no evidence of",
    "no sign of",
    "no signs of",
    "ruled out",
    "excluded",
]

_SENTENCE_SPLIT_PATTERN: re.Pattern = re.compile(
    r'(?<=[.!?])(?!\d)\s+'
)

_LIST_MARKER_PATTERN: re.Pattern = re.compile(r'^\d+\.\s*')


def _normalize(text: Optional[str]) -> str:
    """Lowercase and collapse whitespace. Preserves punctuation."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def _word_in_text(text: str, term: str) -> bool:
    """Check if a term appears in text using word-boundary regex."""
    return bool(re.search(r"\b" + re.escape(term) + r"\b", text))


def _split_into_sentences(text: str) -> List[str]:
    """
    Split impression text into individual sentences.

    Handles radiology-specific text patterns:
    - Preserves decimal values: "1.5 cm" is NOT split.
    - Handles numbered lists: "1. Severe pneumothorax. 2. Cardiomegaly."
    - Splits on '.', '!', '?' followed by whitespace.
    """
    if not text or not text.strip():
        return []

    raw_sentences = _SENTENCE_SPLIT_PATTERN.split(text.strip())

    sentences: List[str] = []
    for sentence in raw_sentences:
        cleaned = sentence.strip()
        if not cleaned:
            continue
        cleaned = _LIST_MARKER_PATTERN.sub("", cleaned).strip()
        if cleaned:
            sentences.append(cleaned)

    return sentences


def _is_sentence_negated(sentence: str) -> bool:
    """Check whether a sentence contains clinical negation cues."""
    for keyword in NEGATION_KEYWORDS:
        if _word_in_text(sentence, keyword):
            return True
    return False


def _find_assertion_sentence(
    label_lower: str,
    sentences: List[str],
) -> Optional[str]:
    """
    Locate the sentence where a finding label is positively asserted.

    Returns the FIRST sentence that contains the label (word-boundary
    match) and is NOT negated. Checks abbreviation forms as a second
    pass. Returns None if the label is only found in negated contexts.
    """
    for sentence in sentences:
        if not _word_in_text(sentence, label_lower):
            continue
        if _is_sentence_negated(sentence):
            continue
        return sentence

    abbrev = _ABBREV_REVERSE.get(label_lower)
    full_term = ABBREVIATIONS.get(label_lower)

    for sentence in sentences:
        has_abbrev = abbrev and _word_in_text(sentence, abbrev)
        has_full = full_term and _word_in_text(sentence, full_term)
        if not (has_abbrev or has_full):
            continue
        if _is_sentence_negated(sentence):
            continue
        return sentence

    return None


def _is_label_mentioned(
    label: str,
    normalized_text: str,
) -> Tuple[bool, bool]:
    """
    Check whether a finding label is positively asserted in impression
    text using sentence-level negation-aware detection.

    Returns:
        (is_mentioned, is_negated):
        - is_mentioned: True if positively asserted in at least one sentence.
        - is_negated: True if the label appears only in negated sentence(s).
    """
    label_lower = label.strip().lower()
    sentences = _split_into_sentences(normalized_text)

    if not sentences:
        return False, False

    found_positively = False
    found_negated = False

    candidate_terms: List[str] = [label_lower]
    abbrev = _ABBREV_REVERSE.get(label_lower)
    if abbrev:
        candidate_terms.append(abbrev)
    full_term = ABBREVIATIONS.get(label_lower)
    if full_term:
        candidate_terms.append(full_term)

    for sentence in sentences:
        term_found = any(
            _word_in_text(sentence, term) for term in candidate_terms
        )
        if not term_found:
            continue

        if _is_sentence_negated(sentence):
            found_negated = True
        else:
            found_positively = True

    return found_positively, found_negated


def _check_laterality(
    finding: Dict[str, Any],
    normalized_text: str,
    label_mentioned: bool,
) -> Optional[bool]:
    """
    Check if the impression conveys the correct laterality for a finding,
    using sentence-level isolation.

    Returns:
        True  — correct laterality in assertion sentence.
        False — laterality mismatch.
        None  — no location data, finding not mentioned, or negated.
    """
    location = finding.get("location")
    if not isinstance(location, str) or not location.strip():
        return None

    if not label_mentioned:
        return None

    label_lower = str(finding.get("label", "")).strip().lower()
    sentences = _split_into_sentences(normalized_text)
    assertion_sentence = _find_assertion_sentence(label_lower, sentences)

    if assertion_sentence is None:
        return None

    location_lower = location.strip().lower()

    expected_laterality: Optional[str] = None
    for loc_term, lat_value in LOCATION_LATERALITY:
        if loc_term in location_lower:
            expected_laterality = lat_value
            break

    if expected_laterality is None:
        return None

    expected_terms = LATERALITY_KEYWORDS.get(expected_laterality, [])
    for term in expected_terms:
        if _word_in_text(assertion_sentence, term):
            return True

    return False


def _check_severity(
    finding: Dict[str, Any],
    normalized_text: str,
    label_mentioned: bool,
) -> Optional[bool]:
    """
    Check if the impression conveys the correct severity for a finding,
    using sentence-level isolation.

    Returns:
        True  — correct severity in assertion sentence.
        False — severity mismatch.
        None  — no severity data, finding not mentioned, or negated.
    """
    severity = finding.get("severity")
    if not isinstance(severity, str) or not severity.strip():
        return None

    if not label_mentioned:
        return None

    label_lower = str(finding.get("label", "")).strip().lower()
    sentences = _split_into_sentences(normalized_text)
    assertion_sentence = _find_assertion_sentence(label_lower, sentences)

    if assertion_sentence is None:
        return None

    severity_lower = severity.strip().lower()

    for sev_key, keywords in SEVERITY_KEYWORDS.items():
        if sev_key in severity_lower or severity_lower in sev_key:
            for term in keywords:
                if _word_in_text(assertion_sentence, term):
                    return True
            return False

    return None