from __future__ import annotations
import re
from Levenshtein import distance as _levenshtein_distance
from dataclasses import dataclass
from typing import Any, Optional


_DISPOSITION_ACCEPTED = "accepted"
_DISPOSITION_EDITED = "edited"
_DISPOSITION_MODIFIED = "modified"
_DISPOSITION_REJECTED = "rejected"

_EDIT_DISTANCE_MINOR_THRESHOLD = 0.10

_REWARD_ACCEPTED = 5.0
_REWARD_MINOR_EDIT = 3.0
_PENALTY_CRITICAL_MISS = -10.0
_PENALTY_WRONG_LATERALITY = -4.0
_PENALTY_REJECTION = -8.0


@dataclass(frozen=True)
class HumanFeedbackMetrics:
    """
    Per-request human feedback / curator learning metrics derived from the
    upstream Reflector disposition field and clinical accuracy outputs.

    Metrics:
      1. is_human_overridden      — radiologist edited the AI impression
      2. is_human_accepted        — radiologist did NOT reject (accepted or edited)
      3. human_feedback_edit_distance — raw Levenshtein between generated and human
      4. normalized_edit_distance — edit distance normalized to [0, 1]
      5. human_reward_score       — signal-level reward (0.0 / +5.0 / -8.0)
      6. curator_reward_score     — composite reward including clinical penalties
    """
    disposition: Optional[str]
    is_human_overridden: Optional[bool]
    is_human_accepted: Optional[bool]
    human_feedback_edit_distance: Optional[int]
    normalized_edit_distance: Optional[float]
    human_reward_score: Optional[float]
    curator_reward_score: Optional[float]




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





def _has_critical_miss(clinical_accuracy: Any) -> bool:
    if clinical_accuracy is None:
        return False
    missed = getattr(clinical_accuracy, "missed_critical_generated", None)
    return bool(missed and len(missed) > 0)


def _has_laterality_error(clinical_accuracy: Any) -> bool:
    if clinical_accuracy is None:
        return False
    lat_acc = getattr(clinical_accuracy, "laterality_accuracy_generated", None)
    if lat_acc is None:
        return False
    lat_count = getattr(clinical_accuracy, "findings_with_laterality_generated", None)
    if lat_count is None:
        # Backward compatibility for the deterministic Layer 1 implementation.
        lat_count = getattr(clinical_accuracy, "findings_with_laterality", 0)
    return lat_count > 0 and lat_acc < 1.0


def calculate_human_feedback_metrics(
    *,
    disposition: Optional[str],
    generated_impression: Optional[str],
    human_impression: Optional[str],
    clinical_accuracy: Any = None,
) -> HumanFeedbackMetrics:
    """
    Compute all 6 human feedback / curator learning metrics.

    Disposition values: 'accepted' | 'edited' | 'modified' | 'rejected' | None

    When disposition is None (upstream has not yet provided human review),
    ALL metrics return None — never 0 or False.

    Curator Reward Score formula (manager spec):
      +5.0  accepted (unchanged)
      +3.0  minor edit  (normalized_edit_distance <= 0.10)
       0.0  major edit  (normalized_edit_distance > 0.10)
     -10.0  critical miss   (added to base when disposition != rejected)
      -4.0  wrong laterality (added to base when disposition != rejected)
      -8.0  rejection        (subsumes clinical penalties — no stacking)
    """
    if disposition is None or not str(disposition).strip():
        return HumanFeedbackMetrics(
            disposition=None,
            is_human_overridden=None,
            is_human_accepted=None,
            human_feedback_edit_distance=None,
            normalized_edit_distance=None,
            human_reward_score=None,
            curator_reward_score=None,
        )

    disp = str(disposition).strip().lower()
    # Boolean flags: is_human_overridden / is_human_accepted
    is_overridden = disp in (_DISPOSITION_EDITED, _DISPOSITION_MODIFIED)
    is_accepted = disp in (_DISPOSITION_ACCEPTED, _DISPOSITION_EDITED, _DISPOSITION_MODIFIED)

    # Edit distance metrics (shared by both reward scores)
    gen_norm = normalize_impression_text(generated_impression)
    human_norm = normalize_impression_text(human_impression)

    edit_dist: Optional[int] = None
    norm_edit: Optional[float] = None
    has_pair = bool(gen_norm and human_norm)

    if has_pair:
        edit_dist = calculate_edit_distance(gen_norm, human_norm)
        max_len = max(len(gen_norm), len(human_norm), 1)
        norm_edit = round(edit_dist / max_len, 4)
    elif disp == _DISPOSITION_REJECTED:
        norm_edit = 1.0


    # HUMAN REWARD SCORE  (pure radiologist signal, no clinical penalties)
    #   +5.0  accepted
    #    0.0  edited / modified
    #   -8.0  rejected

    if disp == _DISPOSITION_REJECTED:
        human_reward = _PENALTY_REJECTION
    elif disp == _DISPOSITION_ACCEPTED:
        human_reward = _REWARD_ACCEPTED
    else:
        human_reward = 0.0

    # CURATOR REWARD SCORE
    #   Composite score = human base reward + clinical penalties.
    #   Starts from a disposition-based reward, then STACKS:
    #     -10.0  critical finding missed
    #      -4.0  laterality was wrong
    #   Exception: rejection (-8.0) SUBSUMES all penalties (no stacking).

    if disp == _DISPOSITION_REJECTED:
        curator_reward = _PENALTY_REJECTION
    elif disp == _DISPOSITION_ACCEPTED:
        curator_reward = _REWARD_ACCEPTED
        if _has_critical_miss(clinical_accuracy):
            curator_reward += _PENALTY_CRITICAL_MISS
        if _has_laterality_error(clinical_accuracy):
            curator_reward += _PENALTY_WRONG_LATERALITY
    elif disp in (_DISPOSITION_EDITED, _DISPOSITION_MODIFIED):
        if norm_edit is not None and norm_edit <= _EDIT_DISTANCE_MINOR_THRESHOLD:
            curator_reward = _REWARD_MINOR_EDIT
        else:
            curator_reward = 0.0
        if _has_critical_miss(clinical_accuracy):
            curator_reward += _PENALTY_CRITICAL_MISS
        if _has_laterality_error(clinical_accuracy):
            curator_reward += _PENALTY_WRONG_LATERALITY
    else:
        curator_reward = 0.0

    return HumanFeedbackMetrics(
        disposition=disp,
        is_human_overridden=is_overridden,
        is_human_accepted=is_accepted,
        human_feedback_edit_distance=edit_dist,
        normalized_edit_distance=norm_edit,
        human_reward_score=human_reward,
        curator_reward_score=curator_reward,
    )
