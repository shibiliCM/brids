"""models/ood_detector.py
Out-of-Distribution (OOD) detection for BirdVision AI.

Algorithm
---------
We use **Maximum Softmax Probability (MSP)** — the simplest and most
interpretable OOD heuristic (Hendrycks & Gimpel, 2017).

After a standard Softmax pass:
  * If  max(P) >= threshold  → the image is IN-distribution → known species.
  * If  max(P) <  threshold  → the image is OUT-of-distribution → unknown bird.

This works because a model trained only on CUB-200 species concentrates its
probability mass when it recognises a known bird, but spreads it thinly across
all 200 classes when the input is unfamiliar.

Limitations
-----------
MSP is a lightweight heuristic.  For production systems consider:
  - Temperature scaling (calibrated probabilities)
  - Energy-based OOD scoring (Liu et al., 2020)
  - Mahalanobis distance in feature space (Lee et al., 2018)
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OODResult:
    """Immutable result returned by :func:`detect_ood`.

    Attributes:
        is_ood:         True when the image is classified as out-of-distribution.
        top_confidence: The highest softmax probability (0-100 %, float).
        threshold_pct:  The OOD threshold that was applied (0-100 %, float).
        top5:           Top-5 ``(display_name, confidence_pct)`` tuples,
                        always populated regardless of OOD status (for debug).
    """
    is_ood:         bool
    top_confidence: float                              # percentage  0-100
    threshold_pct:  float                              # percentage  0-100
    top5:           list[tuple[str, float]] = field(default_factory=list)


# ── Core detection function ───────────────────────────────────────────────────

def detect_ood(
    raw_predictions: list[tuple[int, float]],
    class_names:     list[str],
    ood_threshold:   float,
    name_formatter:  "Callable[[str], str] | None" = None,
) -> OODResult:
    """Decide whether *raw_predictions* represent a known or unknown species.

    Args:
        raw_predictions: List of ``(class_index, probability)`` tuples as
                         returned by ``models.predictor.predict()``.
                         Probabilities are in the range 0–1.
        class_names:     Ordered list of CUB-200 class name strings.
        ood_threshold:   Decision boundary in the range 0–1.
                         Images whose top-1 probability is below this value
                         are flagged as OOD.
        name_formatter:  Optional callable that formats a raw class name into
                         a human-readable display string.  If ``None``, raw
                         names are used as-is.

    Returns:
        :class:`OODResult` with all fields populated.

    Example::

        result = detect_ood(predictions, class_names, ood_threshold=0.70)
        if result.is_ood:
            print(f"Unknown bird — confidence only {result.top_confidence:.1f}%")
        else:
            print(f"Known species: {result.top5[0][0]}")
    """
    if not raw_predictions:
        # Edge-case: predictor returned nothing (e.g. all probs below the
        # display threshold).  Treat as OOD with 0 % confidence.
        return OODResult(is_ood=True, top_confidence=0.0,
                         threshold_pct=ood_threshold * 100, top5=[])

    # ── Format display names ──────────────────────────────────────────────────
    fmt = name_formatter if name_formatter is not None else lambda n: n

    top5: list[tuple[str, float]] = [
        (fmt(class_names[idx]), prob * 100)        # prob → percentage
        for idx, prob in raw_predictions
    ]

    # ── OOD decision ──────────────────────────────────────────────────────────
    # top_confidence is already in percent (multiplied above).
    top_confidence_pct = top5[0][1]

    # Compare against the threshold, which is stored as 0-1 in settings.
    is_ood = top_confidence_pct < (ood_threshold * 100)

    return OODResult(
        is_ood=is_ood,
        top_confidence=top_confidence_pct,
        threshold_pct=ood_threshold * 100,
        top5=top5,
    )
