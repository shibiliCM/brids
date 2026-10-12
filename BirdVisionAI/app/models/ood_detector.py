"""models/ood_detector.py
Out-of-Distribution (OOD) detection for BirdVision AI.

Algorithms Supported:
---------------------
1. Maximum Softmax Probability (MSP - Hendrycks & Gimpel, 2017):
   Flags images whose top-1 probability is below the configured threshold.
2. Foundation Model Cross-Verification (CLIP Consensus):
   Detects out-of-distribution species (e.g., Peacock, Pigeon, Crow) where
   the CUB-200 specialist model falsely produces high softmax confidence
   for an unrelated CUB class due to color or plumage similarities.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from models.clip_fallback import CLIPVerification


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OODResult:
    """Immutable result returned by :func:`detect_ood`.

    Attributes:
        is_ood:             True when the image is classified as out-of-distribution.
        top_confidence:     The highest softmax probability (0-100 %, float).
        threshold_pct:      The OOD threshold that was applied (0-100 %, float).
        top5:               Top-5 ``(display_name, confidence_pct)`` tuples.
        ood_reason:         Explanation of why OOD was flagged (if applicable).
        clip_override_name: Species name identified by foundation model consensus (e.g. 'Peacock').
        clip_confidence:    Confidence percentage of the foundation model identification.
        clip_suggestions:   List of top suggestions from the foundation model.
    """
    is_ood:             bool
    top_confidence:     float                              # percentage  0-100
    threshold_pct:      float                              # percentage  0-100
    top5:               list[tuple[str, float]] = field(default_factory=list)
    ood_reason:         str | None = None
    clip_override_name: str | None = None
    clip_confidence:    float | None = None
    clip_suggestions:   list[tuple[str, float]] = field(default_factory=list)


# ── Core detection function ───────────────────────────────────────────────────

def detect_ood(
    raw_predictions:   list[tuple[int, float]],
    class_names:       list[str],
    ood_threshold:     float,
    name_formatter:    Callable[[str], str] | None = None,
    clip_verification: "CLIPVerification | None" = None,
) -> OODResult:
    """Decide whether *raw_predictions* represent a known or unknown species.

    Args:
        raw_predictions:   List of ``(class_index, probability)`` tuples as
                           returned by ``models.predictor.predict()``.
        class_names:       Ordered list of CUB-200 class name strings.
        ood_threshold:     Decision boundary in the range 0–1.
        name_formatter:    Optional callable that formats a raw class name.
        clip_verification: Optional :class:`CLIPVerification` from foundation model.

    Returns:
        :class:`OODResult` with all fields populated.
    """
    if not raw_predictions:
        return OODResult(
            is_ood=True,
            top_confidence=0.0,
            threshold_pct=ood_threshold * 100,
            top5=[],
            ood_reason="No predictions exceeded the minimum confidence filter.",
        )

    # ── Format display names ──────────────────────────────────────────────────
    fmt = name_formatter if name_formatter is not None else lambda n: n

    top5: list[tuple[str, float]] = [
        (fmt(class_names[idx]), prob * 100)        # prob → percentage
        for idx, prob in raw_predictions
    ]

    top_confidence_pct = top5[0][1]

    # ── 1. Check Foundation Model Cross-Verification ──────────────────────────
    # If the foundation model strongly identifies an out-of-distribution species
    # that contradicts the CUB-200 specialist prediction (e.g. Peacock vs Green Violetear),
    # prioritize the foundation model's identification.
    if clip_verification is not None and clip_verification.is_discrepancy:
        return OODResult(
            is_ood=True,
            top_confidence=top_confidence_pct,
            threshold_pct=ood_threshold * 100,
            top5=top5,
            ood_reason=clip_verification.reason,
            clip_override_name=clip_verification.clip_top_name,
            clip_confidence=clip_verification.clip_top_pct,
            clip_suggestions=clip_verification.suggestions,
        )

    # ── 2. Standard Maximum Softmax Probability (MSP) ─────────────────────────
    is_ood = top_confidence_pct < (ood_threshold * 100)
    reason = None
    if is_ood:
        reason = (
            f"Confidence ({top_confidence_pct:.1f}%) is below the configured "
            f"OOD threshold ({ood_threshold * 100:.1f}%)."
        )

    return OODResult(
        is_ood=is_ood,
        top_confidence=top_confidence_pct,
        threshold_pct=ood_threshold * 100,
        top5=top5,
        ood_reason=reason,
        clip_suggestions=clip_verification.suggestions if clip_verification else [],
    )
