"""app/main.py
BirdVision AI v2 — application entry point.

Responsibilities:
  - Page configuration & CSS injection
  - Session-state bootstrap
  - Sidebar rendering & page routing
  - Inference pipeline: ResNet-18 → OOD → CLIP fallback
  - Explainability: Grad-CAM heatmap
  - Visualisation: Confidence chart, Species info panel
"""
import datetime
import sys
from pathlib import Path

import streamlit as st

# ── Path resolution ───────────────────────────────────────────────────────────
app_dir      = Path(__file__).resolve().parent
project_root = app_dir.parent
for p in (app_dir, project_root):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BirdVision AI",
    page_icon=str(app_dir / "assets" / "bird_icon.svg"),
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS injection ─────────────────────────────────────────────────────────────
from ui.css_loader import inject_css
inject_css(app_dir / "assets" / "styles.css")

# ── SVG icon ──────────────────────────────────────────────────────────────────
_icon_raw = (app_dir / "assets" / "bird_icon.svg").read_text(encoding="utf-8")
ICON = _icon_raw.replace(
    'width="120" height="120"',
    'width="100%" height="100%" style="display:block"',
)

# ── Project imports ───────────────────────────────────────────────────────────
from config.settings import (
    MODEL_PATH        as DEFAULT_MODEL_PATH,
    CLASS_NAMES_PATH,
    OOD_THRESHOLD     as DEFAULT_OOD_THRESHOLD,
    GRADCAM_ENABLED   as DEFAULT_GRADCAM,
    CLIP_ENABLED      as DEFAULT_CLIP,
    CLIP_TOP_K,
    NUM_CLASSES,
)
from models.model_loader      import load_model
from models.predictor         import predict
from models.ood_detector      import detect_ood
from models.gradcam           import get_gradcam, GradCAM
from models.clip_fallback     import run_clip_fallback
from utils.class_loader       import load_class_names
from preprocessing.transforms import get_transform
from utils.logger             import get_logger

from ui.sidebar           import render_sidebar
from ui.upload            import upload_image
from ui.results           import display_results
from ui.confidence_chart  import render_confidence_chart
from ui.species_info      import render_species_info
from ui.templates import (
    page_header,
    section_label,
    empty_state,
    spinner_label,
    metric_card,
    history_row,
    ood_card,
    ood_debug_header,
    ood_debug_row,
    clip_card_header,
    clip_card_footer,
    clip_suggestion_row,
    clip_unavailable_notice,
    gradcam_section_header,
    gradcam_section_footer,
    gradcam_caption,
    no_predictions_notice,
)

logger = get_logger(__name__)

# ── Session-state bootstrap ───────────────────────────────────────────────────
_DEFAULTS: dict = {
    "history":            [],
    "current_page":       "classify",
    "OOD_THRESHOLD":      DEFAULT_OOD_THRESHOLD,
    "CONFIDENCE_THRESHOLD": 0.01,
    "GRADCAM_ENABLED":    DEFAULT_GRADCAM,
    "CLIP_ENABLED":       DEFAULT_CLIP,
}
for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar ───────────────────────────────────────────────────────────────────
render_sidebar()

# ── Read session state ────────────────────────────────────────────────────────
model_path    = Path(st.session_state.get("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
ood_threshold = st.session_state.get("OOD_THRESHOLD",        DEFAULT_OOD_THRESHOLD)
conf_filter   = st.session_state.get("CONFIDENCE_THRESHOLD", 0.01)
gradcam_on    = st.session_state.get("GRADCAM_ENABLED",      DEFAULT_GRADCAM)
clip_on       = st.session_state.get("CLIP_ENABLED",         DEFAULT_CLIP)


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _get_model(path: Path):
    logger.info(f"Loading model from {path}")
    return load_model(path)


model       = _get_model(model_path)
class_names = load_class_names(CLASS_NAMES_PATH)
transform   = get_transform()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _fmt(raw: str) -> str:
    """Format CUB class name: '006.Least_Auklet' → '006. Least Auklet'."""
    name = raw.strip().replace("_", " ")
    if "." in name:
        num, rest = name.split(".", 1)
        return f"{num.strip()}. {rest.strip()}"
    return name


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(page_header(ICON), unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE ROUTER
# ═════════════════════════════════════════════════════════════════════════════
page = st.session_state.get("current_page", "classify")

# ─────────────────────────────────────────────────────────────────────────────
# 1. CLASSIFY PAGE
# ─────────────────────────────────────────────────────────────────────────────
if page == "classify":

    uploaded = upload_image()   # PIL Image or None

    if not uploaded:
        st.markdown(
            empty_state(
                "🪶",
                "Upload a bird photo to begin classification",
                "Supports JPG, JPEG, PNG",
            ),
            unsafe_allow_html=True,
        )
    else:
        col_img, col_preds = st.columns([1, 1.2], gap="large")

        with col_img:
            st.markdown(section_label("Uploaded Image"), unsafe_allow_html=True)
            st.image(uploaded, use_container_width=True)

        with col_preds:
            # ── Inference ─────────────────────────────────────────────────
            with st.spinner(""):
                st.markdown(spinner_label(), unsafe_allow_html=True)
                img_tensor = transform(uploaded).unsqueeze(0)
                raw_preds  = predict(model, img_tensor, conf_filter)

            # ── OOD detection ─────────────────────────────────────────────
            ood_result = detect_ood(
                raw_predictions=raw_preds,
                class_names=class_names,
                ood_threshold=ood_threshold,
                name_formatter=_fmt,
            )

            st.markdown(section_label("Prediction Result"), unsafe_allow_html=True)

            # ── KNOWN species branch ──────────────────────────────────────
            if not ood_result.is_ood:
                if ood_result.top5:
                    display_results(ood_result.top5)

                    # Confidence chart
                    st.markdown("<div class='chart-section'>", unsafe_allow_html=True)
                    st.markdown(section_label("Confidence Chart"), unsafe_allow_html=True)
                    render_confidence_chart(ood_result.top5)
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Record in history
                    top_name, top_pct = ood_result.top5[0]
                    st.session_state["history"].append({
                        "name":   top_name,
                        "conf":   top_pct,
                        "time":   datetime.datetime.now().strftime("%H:%M:%S"),
                        "is_ood": False,
                    })
                else:
                    st.markdown(no_predictions_notice(), unsafe_allow_html=True)

            # ── OOD (Unknown) branch ──────────────────────────────────────
            else:
                st.markdown(
                    ood_card(ood_result.top_confidence, ood_result.threshold_pct),
                    unsafe_allow_html=True,
                )

                # Top-5 debug rows
                if ood_result.top5:
                    st.markdown(ood_debug_header(), unsafe_allow_html=True)
                    top_pct = ood_result.top5[0][1]
                    for rank, (name, pct) in enumerate(ood_result.top5, start=1):
                        bar_w = min(pct / top_pct * 100, 100) if top_pct > 0 else 0
                        st.markdown(
                            ood_debug_row(rank, name, pct, bar_w),
                            unsafe_allow_html=True,
                        )

                # CLIP fallback
                clip_result = None
                if clip_on:
                    with st.spinner("Running CLIP fallback…"):
                        clip_result = run_clip_fallback(uploaded, top_k=CLIP_TOP_K)

                    if clip_result.available and clip_result.suggestions:
                        st.markdown(clip_card_header(), unsafe_allow_html=True)
                        top_clip_pct = clip_result.suggestions[0][1]
                        for rank, (name, pct) in enumerate(clip_result.suggestions, start=1):
                            bar_w = min(pct / top_clip_pct * 100, 100) if top_clip_pct > 0 else 0
                            st.markdown(
                                clip_suggestion_row(rank, name, pct, bar_w),
                                unsafe_allow_html=True,
                            )
                        st.markdown(clip_card_footer(), unsafe_allow_html=True)
                    else:
                        st.markdown(
                            clip_unavailable_notice(clip_result.error or ""),
                            unsafe_allow_html=True,
                        )

                # Record as OOD in history
                st.session_state["history"].append({
                    "name":   "Unknown Bird Species",
                    "conf":   ood_result.top_confidence,
                    "time":   datetime.datetime.now().strftime("%H:%M:%S"),
                    "is_ood": True,
                })

        # ── Grad-CAM (full-width below columns) ───────────────────────────
        if gradcam_on and not ood_result.is_ood and raw_preds:
            try:
                st.markdown(gradcam_section_header(), unsafe_allow_html=True)
                top_class_idx = raw_preds[0][0]  # highest-confidence class index

                with st.spinner("Generating Grad-CAM…"):
                    gcam    = get_gradcam(model)
                    heatmap = gcam.generate_heatmap(img_tensor, top_class_idx, NUM_CLASSES)
                    overlay = GradCAM.overlay_on_image(uploaded, heatmap)
                    gcam.remove_hooks()

                cam_col1, cam_col2 = st.columns(2, gap="medium")
                with cam_col1:
                    st.image(uploaded, use_container_width=True)
                    st.markdown(
                        gradcam_caption("Original Image"),
                        unsafe_allow_html=True,
                    )
                with cam_col2:
                    st.image(overlay, use_container_width=True)
                    st.markdown(
                        gradcam_caption("Grad-CAM Attention Heatmap"),
                        unsafe_allow_html=True,
                    )
                st.markdown(gradcam_section_footer(), unsafe_allow_html=True)

            except Exception as e:
                logger.warning(f"Grad-CAM failed: {e}")
                st.markdown(
                    "<div style='color:#4a5238;font-size:12px;margin-top:8px'>"
                    f"⚠ Grad-CAM unavailable: {e}</div>",
                    unsafe_allow_html=True,
                )

        # ── Species Info (always shown — uses best available name) ────────
        # For known species: use ResNet top-1 prediction.
        # For OOD birds: use CLIP top suggestion if available, else ResNet top candidate.
        _species_name_for_info: str | None = None
        if not ood_result.is_ood and ood_result.top5:
            _species_name_for_info = ood_result.top5[0][0]
        elif ood_result.is_ood:
            # Try CLIP first
            _clip_res = locals().get("clip_result")
            if (
                _clip_res is not None
                and getattr(_clip_res, "available", False)
                and getattr(_clip_res, "suggestions", [])
            ):
                _species_name_for_info = _clip_res.suggestions[0][0]
            elif ood_result.top5:
                # Fall back to closest ResNet candidate
                _species_name_for_info = ood_result.top5[0][0]

        if _species_name_for_info:
            render_species_info(_species_name_for_info)

# ─────────────────────────────────────────────────────────────────────────────
# 2. ANALYTICS PAGE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "analytics":
    import pandas as pd

    history  = st.session_state.get("history", [])
    st.markdown(section_label("Session Analytics"), unsafe_allow_html=True)

    total    = len(history)
    known    = [h for h in history if not h.get("is_ood", False)]
    unknown  = [h for h in history if h.get("is_ood",  False)]
    avg_conf = sum(h["conf"] for h in known) / len(known) if known else 0.0
    top_cls  = (
        max(set(h["name"] for h in known),
            key=lambda n: sum(1 for h in known if h["name"] == n))
        if known else "—"
    )

    c1, c2, c3, c4 = st.columns(4, gap="large")
    for col, val, lbl in [
        (c1, str(total),        "Total Scans"),
        (c2, str(len(known)),   "Known Species"),
        (c3, str(len(unknown)), "Unknown / OOD"),
        (c4, f"{avg_conf:.1f}%", "Avg Confidence"),
    ]:
        with col:
            st.markdown(metric_card(val, lbl), unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    if known:
        st.markdown(section_label("Confidence by Classification"), unsafe_allow_html=True)
        df = pd.DataFrame(known)
        st.bar_chart(df["conf"], use_container_width=True, height=220, color="#7aad38")
    elif not history:
        st.markdown(
            empty_state("📊", "No data yet — classify some birds first"),
            unsafe_allow_html=True,
        )

    if top_cls != "—":
        st.markdown(section_label("Top Species"), unsafe_allow_html=True)
        st.markdown(
            metric_card(top_cls[:28] + ("…" if len(top_cls) > 28 else ""), "Most Identified"),
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# 3. HISTORY PAGE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "history":
    history = st.session_state.get("history", [])

    hdr_col, btn_col = st.columns([4, 1])
    with hdr_col:
        st.markdown(section_label("Classification History"), unsafe_allow_html=True)
    with btn_col:
        if history and st.button("🗑 Clear"):
            st.session_state["history"] = []
            st.rerun()

    if history:
        for item in reversed(history):
            st.markdown(
                history_row(
                    name=item["name"],
                    conf=item["conf"],
                    timestamp=item["time"],
                    is_ood=item.get("is_ood", False),
                ),
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            empty_state("🗂", "No classifications yet"),
            unsafe_allow_html=True,
        )
