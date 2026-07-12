"""ui/templates.py
Pure HTML template functions for BirdVision AI.

Every function returns a raw HTML string.
No Streamlit calls are made here — callers are responsible for
passing the string to ``st.markdown(..., unsafe_allow_html=True)``.
"""
from __future__ import annotations


# ── Page header ───────────────────────────────────────────────────────────────

def page_header(icon_svg: str) -> str:
    """Top-of-page logo + title block."""
    return f"""
<div style='display:flex;align-items:center;gap:16px;margin-bottom:10px'>
  <div style='width:56px;height:56px;flex-shrink:0;border-radius:14px;overflow:hidden'>
    {icon_svg}
  </div>
  <div>
    <div style='font-family:Syne,sans-serif;font-size:28px;font-weight:800;
    color:#c8d8a0;line-height:1;letter-spacing:-0.02em'>Bird Vision AI</div>
    <div style='font-size:12px;color:#4a5238;margin-top:5px'>
    Fine-grained species classification &middot; 200 classes &middot; CUB-200-2011</div>
  </div>
</div>
<hr style='margin:14px 0 20px'/>
"""


# ── Sidebar components ────────────────────────────────────────────────────────

def sidebar_logo(icon_svg: str) -> str:
    """Sidebar logo banner with brand name and dataset label."""
    return f"""
<div style='display:flex;align-items:center;gap:11px;margin-bottom:22px;
padding-bottom:18px;border-bottom:1px solid rgba(255,255,255,0.05)'>
  <div style='width:38px;height:38px;flex-shrink:0;border-radius:10px;overflow:hidden'>
    {icon_svg}
  </div>
  <div>
    <div style='font-family:Syne,sans-serif;font-size:14px;font-weight:700;
    color:#c8d8a0;line-height:1.15'>BirdVision AI</div>
    <div style='font-size:10px;color:#4a5238;letter-spacing:.06em;
    text-transform:uppercase;margin-top:2px'>CUB-200-2011</div>
  </div>
</div>
"""


def sidebar_section_label(text: str) -> str:
    """Small uppercase section divider label used in the sidebar."""
    return (
        f"<div style='font-size:10px;color:#4a5238;letter-spacing:.08em;"
        f"text-transform:uppercase;margin-bottom:6px'>{text}</div>"
    )


def sidebar_divider() -> str:
    """Thin horizontal rule."""
    return "<div style='height:1px;background:rgba(255,255,255,0.05);margin:14px 0 10px'></div>"


def sidebar_status_dot() -> str:
    """Glowing green 'Model loaded · Ready' indicator."""
    return """
<div class='status-dot' style='margin-top:2rem'>
  <span></span> Model loaded &middot; Ready
</div>
"""


# ── Section labels (main content) ─────────────────────────────────────────────

def section_label(text: str) -> str:
    """Small uppercase label above a content section."""
    return f"<div class='section-label'>{text}</div>"


# ── Classify page — known species ─────────────────────────────────────────────

def pred_top_card(display_name: str, pct: float) -> str:
    """Best-match prediction card — rank 1 (known species)."""
    bar_width = min(pct, 100)
    return f"""
<div class='pred-top'>
  <div class='pred-label'>Best match</div>
  <div class='pred-name'>{display_name}</div>
  <div class='pred-conf'>
    {pct:.2f}%
    <span class='pred-conf-unit'> confidence</span>
  </div>
  <div class='pred-bar-track'>
    <div style='height:100%;width:{bar_width:.1f}%;background:
    linear-gradient(90deg,#4a7a1c,#a8e060);border-radius:99px'></div>
  </div>
</div>
"""


def pred_row_card(rank: int, display_name: str, pct: float, bar_w: float) -> str:
    """Single prediction row for ranks 2–5 (known species)."""
    return f"""
<div class='pred-row'>
  <div style='display:flex;justify-content:space-between;align-items:center'>
    <span style='font-size:13px;font-weight:500;color:#a8b880'>#{rank} {display_name}</span>
    <span style='font-size:12px;color:#7aad38;font-weight:600'>{pct:.2f}%</span>
  </div>
  <div style='height:3px;background:rgba(255,255,255,0.05);
  border-radius:99px;margin-top:7px'>
    <div style='height:100%;width:{bar_w:.1f}%;background:
    linear-gradient(90deg,#3d6e18,#5a9028);border-radius:99px'></div>
  </div>
</div>
"""


# ── Classify page — OOD (unknown species) ────────────────────────────────────

def ood_card(top_confidence: float, threshold_pct: float) -> str:
    """Full 'Unknown Bird Species' alert card shown when OOD is detected.

    Uses amber/orange tones so users immediately see it differs from a
    normal (green) classification result.

    Args:
        top_confidence: Highest softmax probability in percent (0–100).
        threshold_pct:  The OOD threshold applied, in percent (0–100).
    """
    return f"""
<div class='ood-card'>
  <div class='ood-badge'>⚠ Out-of-Distribution Detected</div>
  <div class='ood-title'>Unknown Bird Species</div>
  <div class='ood-conf'>
    {top_confidence:.2f}%
    <span class='ood-conf-unit'> confidence</span>
  </div>
  <div class='ood-explanation'>
    This image does not closely match any of the <strong>200 bird species</strong>
    used during model training (CUB-200-2011).<br/><br/>
    The model's highest confidence was <strong>{top_confidence:.2f}%</strong>, which is
    below the OOD threshold of <strong>{threshold_pct:.0f}%</strong>.
    This typically means the bird is not represented in the training dataset
    (e.g. Crow, Pigeon, Parrot, Peacock).<br/><br/>
    <em>Adjust the OOD threshold in the sidebar to tune sensitivity.</em>
  </div>
</div>
"""


def ood_debug_header() -> str:
    """Small section label above the Top-5 debug rows."""
    return """
<div style='font-size:10px;color:#6a4a28;letter-spacing:.08em;
text-transform:uppercase;margin-bottom:6px'>
  Top-5 candidates (for debugging)
</div>
"""


def ood_debug_row(rank: int, display_name: str, pct: float, bar_w: float) -> str:
    """One candidate row inside the OOD debug section."""
    return f"""
<div class='ood-debug-row'>
  <div style='display:flex;justify-content:space-between;align-items:center'>
    <span style='font-size:13px;font-weight:500;color:#7a6040'>#{rank} {display_name}</span>
    <span style='font-size:12px;color:#c87040;font-weight:600'>{pct:.2f}%</span>
  </div>
  <div style='height:3px;background:rgba(255,255,255,0.04);
  border-radius:99px;margin-top:7px'>
    <div style='height:100%;width:{bar_w:.1f}%;background:
    linear-gradient(90deg,#7a3a10,#c87040);border-radius:99px'></div>
  </div>
</div>
"""


# ── Shared ────────────────────────────────────────────────────────────────────

def spinner_label() -> str:
    """Custom 'Analysing image…' text shown during inference."""
    return "<div style='color:#7aad38;font-size:13px;letter-spacing:.04em'>Analysing image…</div>"


def empty_state(emoji: str, message: str, sub: str = "") -> str:
    """Centred empty-state placeholder."""
    sub_html = f"<div style='font-size:12px;color:#3a4a28'>{sub}</div>" if sub else ""
    return f"""
<div style='display:flex;flex-direction:column;align-items:center;justify-content:center;
padding:5rem 0;gap:14px;opacity:0.5'>
  <div style='font-size:48px'>{emoji}</div>
  <div style='font-size:15px;color:#4a5238;letter-spacing:.04em'>{message}</div>
  {sub_html}
</div>
"""


def no_predictions_notice() -> str:
    """Shown when no predictions pass the confidence threshold."""
    return (
        "<div style='color:#4a5a28;font-size:13px;margin-top:1rem'>"
        "No predictions meet the confidence threshold. "
        "Try lowering the slider in the sidebar."
        "</div>"
    )


# ── Analytics page ────────────────────────────────────────────────────────────

def metric_card(value: str, label: str) -> str:
    """Single analytics metric card."""
    return f"""
<div class='metric-card'>
  <div class='metric-val'>{value}</div>
  <div class='metric-lbl'>{label}</div>
</div>
"""


# ── History page ──────────────────────────────────────────────────────────────

def history_row(name: str, conf: float, timestamp: str, is_ood: bool = False) -> str:
    """Single row in the classification history list.

    Args:
        name:      Display species name (or 'Unknown Bird Species').
        conf:      Confidence percentage (0–100).
        timestamp: HH:MM:SS string.
        is_ood:    When True renders confidence in amber (OOD colour).
    """
    conf_class = "hist-conf hist-ood" if is_ood else "hist-conf"
    icon = "❓" if is_ood else "🐦"
    return f"""
<div class='hist-row'>
  <div>
    <div class='hist-name'>{icon} {name}</div>
    <div class='hist-time'>{timestamp}</div>
  </div>
  <div class='{conf_class}'>{conf:.2f}%</div>
</div>
"""


# ── CLIP Fallback Card ────────────────────────────────────────────────────────

def clip_card_header() -> str:
    """Header for the CLIP fallback results section."""
    return """
<div class='clip-card'>
  <div class='clip-header'>
    <span class='clip-badge'>🔍 CLIP Fallback</span>
    <span class='clip-title'>Possible Species Matches</span>
  </div>
"""


def clip_card_footer() -> str:
    """Closing div for the CLIP card."""
    return "</div>"


def clip_suggestion_row(rank: int, name: str, pct: float, bar_w: float) -> str:
    """One CLIP suggestion row.

    Args:
        rank:  Position (1-based).
        name:  Species name (title case).
        pct:   Confidence in percent (0–100).
        bar_w: Bar fill width in percent relative to rank-1.
    """
    return f"""
<div class='clip-row'>
  <div style='display:flex;justify-content:space-between;align-items:center'>
    <span style='font-size:13px;font-weight:500;color:#a0c8e0'>#{rank} {name}</span>
    <span style='font-size:12px;color:#60b0e0;font-weight:600'>{pct:.2f}%</span>
  </div>
  <div style='height:3px;background:rgba(255,255,255,0.04);border-radius:99px;margin-top:7px'>
    <div style='height:100%;width:{bar_w:.1f}%;background:
    linear-gradient(90deg,#2060a0,#60b0e0);border-radius:99px'></div>
  </div>
</div>
"""


def clip_unavailable_notice(error: str = "") -> str:
    """Shown when CLIP model is not installed."""
    tip = f"<br><code style='font-size:10px;color:#3a4a58'>{error}</code>" if error else ""
    return f"""
<div class='clip-card'>
  <div class='clip-badge'>🔍 CLIP Fallback</div>
  <div class='clip-unavailable' style='margin-top:10px'>
    OpenCLIP model not available. Install with:<br>
    <code style='color:#60b0e0'>pip install open_clip_torch</code>{tip}
  </div>
</div>
"""


# ── Grad-CAM section labels ───────────────────────────────────────────────────

def gradcam_section_header() -> str:
    """Header div for the Grad-CAM explainability section."""
    return """
<div class='gradcam-section'>
  <div class='gradcam-label'>🔥 Grad-CAM — Model Attention Heatmap</div>
"""


def gradcam_section_footer() -> str:
    return "</div>"


def gradcam_caption(col: str) -> str:
    """Caption below each Grad-CAM image column."""
    return f"<div class='gradcam-caption'>{col}</div>"

