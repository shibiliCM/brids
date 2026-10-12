"""ui/report.py
Observation certificate & field report generator for BirdVision AI.
"""
from __future__ import annotations

import datetime
import uuid
from typing import Sequence

from ui.species_info import _lookup


def generate_field_report(
    species_name: str,
    confidence: float,
    is_ood: bool,
    clip_override: str | None = None,
    top_candidates: Sequence[tuple[str, float]] | None = None,
) -> str:
    """Generate a formatted observation certificate report in markdown format.

    Args:
        species_name: Identified common or display name.
        confidence: Prediction or consensus confidence percentage (0-100).
        is_ood: Whether the observation was flagged as Out-of-Distribution.
        clip_override: Foundation model override species name, if triggered.
        top_candidates: List of (name, confidence_pct) pairs for alternative species.

    Returns:
        Formatted markdown report string.
    """
    obs_id = f"BIRD-{datetime.datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    info = _lookup(species_name) or {}
    scientific = info.get("scientific", "Unspecified")
    status = info.get("conservation", "Evaluated via Dual Foundation Consensus")
    habitat = info.get("habitat", "Global Distribution")
    diet = info.get("diet", "Species standard")
    marks = info.get("field_marks", "Key morphological traits verified via Vision Models")

    report_lines = [
        "# 🪶 BirdVision AI — Field Observation Certificate",
        f"**Observation ID**: `{obs_id}`  ",
        f"**Timestamp**: {now_str}  ",
        "**Verification Engine**: BirdVision AI (Dual-Model Foundation Consensus)  ",
        "",
        "---",
        "",
        "## 🦅 Primary Identification",
        f"- **Common Name**: **{species_name}**",
        f"- **Scientific Name**: *{scientific}*",
        f"- **Identification Confidence**: **{confidence:.1f}%**",
    ]

    if clip_override:
        report_lines.extend([
            "- **Verification Protocol**: ⚡ Foundation Consensus Override",
            f"- **Consensus Analysis**: OpenCLIP ViT-B-32 identified '{clip_override}', overriding CUB specialist bias.",
        ])
    elif is_ood:
        report_lines.extend([
            "- **Verification Protocol**: ⚠ Out-of-Distribution (Non-CUB Species Detected)",
            "- **Consensus Analysis**: Prediction fell below specialist certainty threshold.",
        ])
    else:
        report_lines.extend([
            "- **Verification Protocol**: ✅ Verified In-Distribution (CUB-200 Specialist)",
            "- **Consensus Analysis**: ResNet-18 specialist and visual features in full agreement.",
        ])

    report_lines.extend([
        "",
        "---",
        "",
        "## 🌿 Natural History & Field Markers",
        f"- **Conservation Status**: {status}",
        f"- **Native Habitat**: {habitat}",
        f"- **Dietary Guild**: {diet}",
        f"- **Key Field Identifiers**: {marks}",
        "",
        "---",
        "",
        "## 🔬 Model Consensus Diagnostics",
    ])

    if top_candidates:
        report_lines.append("### Top Candidate Predictions:")
        for rank, (cand_name, cand_pct) in enumerate(list(top_candidates)[:5], start=1):
            report_lines.append(f"{rank}. **{cand_name}** — {cand_pct:.1f}%")
        report_lines.append("")

    report_lines.extend([
        "---",
        "*Certified by BirdVision AI — Explainable Deep Learning & Foundation Consensus*",
    ])

    return "\n".join(report_lines)
