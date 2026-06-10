from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

from Levenshtein import distance as _levenshtein_distance


@dataclass(frozen=True)
class CuratorScores:
    data_completeness_score: float
    prompt_completeness_score: float
    dpo_readiness_score: float


def normalize_impression_text(value: Optional[str]) -> str:
    """
    Normalize impression text for comparison.
    Lowercases, collapses whitespace, normalizes punctuation spacing.
    Used by human_feedback_metrics for edit distance computation.
    """
    if not value:
        return ""
    normalized = value.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s*([,.;:])\s*", r"\1 ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def calculate_edit_distance(left: str, right: str) -> int:
    """
    Levenshtein edit distance between two strings.
    Delegates to the Levenshtein library (C-optimized via rapidfuzz).
    Used by human_feedback_metrics for edit distance metric.
    """
    return _levenshtein_distance(left, right)


def _present(value: object) -> bool:
    """Check whether a value is present (non-None, non-empty string)."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def calculate_prompt_completeness_score(prompt_text: Optional[str]) -> float:
    """Binary score: 1.0 if prompt_text is present, 0.0 otherwise."""
    return 1.0 if _present(prompt_text) else 0.0


def calculate_dpo_readiness_score(
    prompt_text: Optional[str],
    preferred_output: Optional[str],
    dispreferred_output: Optional[str],
) -> float:
    """Binary score: 1.0 if all 3 DPO components are present, 0.0 otherwise."""
    if _present(prompt_text) and _present(preferred_output) and _present(dispreferred_output):
        return 1.0
    return 0.0


def calculate_data_completeness_score(
    request_id: Optional[str],
    send_to_curator: Optional[bool],
    prompt_text: Optional[str],
    preferred_output: Optional[str],
    dispreferred_output: Optional[str],
) -> float:
    """Ratio score: what fraction of the 5 required fields are present?"""
    required = [
        _present(request_id),
        send_to_curator is True,
        _present(prompt_text),
        _present(preferred_output),
        _present(dispreferred_output),
    ]
    return round(sum(required) / len(required), 4)


def calculate_curator_scores(
    request_id: Optional[str],
    send_to_curator: Optional[bool],
    prompt_text: Optional[str],
    preferred_output: Optional[str],
    dispreferred_output: Optional[str],
) -> CuratorScores:
    """Compute all 3 curator quality scores for a single request."""
    return CuratorScores(
        data_completeness_score=calculate_data_completeness_score(
            request_id=request_id,
            send_to_curator=send_to_curator,
            prompt_text=prompt_text,
            preferred_output=preferred_output,
            dispreferred_output=dispreferred_output,
        ),
        prompt_completeness_score=calculate_prompt_completeness_score(prompt_text),
        dpo_readiness_score=calculate_dpo_readiness_score(
            prompt_text=prompt_text,
            preferred_output=preferred_output,
            dispreferred_output=dispreferred_output,
        ),
    )
