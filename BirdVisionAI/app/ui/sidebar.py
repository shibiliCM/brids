"""ui/sidebar.py
Sidebar renderer — navigation, model settings, OOD threshold, and status dot.
All HTML is sourced from ui/templates.py; no inline markup here.
"""
import streamlit as st
from pathlib import Path

from ui.templates import (
    sidebar_logo,
    sidebar_section_label,
    sidebar_divider,
    sidebar_status_dot,
)

# ── Navigation item definitions ───────────────────────────────────────────────
_NAV_ITEMS: list[tuple[str, str, str]] = [
    ("classify",  "🖼  Classify",  "classify"),
    ("analytics", "📊  Analytics", "analytics"),
    ("history",   "🗂  History",   "history"),
]


def render_sidebar() -> None:
    """Render the full sidebar.

    Side effects — writes to ``st.session_state``:
        - ``current_page``          (str)   selected nav page key
        - ``MODEL_PATH``            (str)   absolute path to chosen .pth
        - ``CONFIDENCE_THRESHOLD``  (float) display confidence filter
        - ``OOD_THRESHOLD``         (float) OOD decision boundary (0–1)
        - ``SELECTED_MODEL``        (str | None)
    """

    # ── Icon ──────────────────────────────────────────────────────────────────
    _icon_path = Path(__file__).resolve().parents[1] / "assets" / "bird_icon.svg"
    icon_svg = ""
    if _icon_path.exists():
        raw = _icon_path.read_text(encoding="utf-8")
        icon_svg = raw.replace(
            'width="120" height="120"',
            'width="100%" height="100%" style="display:block"',
        )

    # ── Logo banner ───────────────────────────────────────────────────────────
    st.sidebar.markdown(sidebar_logo(icon_svg), unsafe_allow_html=True)

    # ── Navigation ────────────────────────────────────────────────────────────
    st.sidebar.markdown(sidebar_section_label("Navigation"), unsafe_allow_html=True)

    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "classify"

    nav_labels = [label for _, label, _ in _NAV_ITEMS]
    nav_keys   = [key   for _, _,     key in _NAV_ITEMS]

    current_index = nav_keys.index(st.session_state["current_page"])

    selected_label = st.sidebar.radio(
        "",
        options=nav_labels,
        index=current_index,
        key="_nav_radio",
        label_visibility="collapsed",
    )
    st.session_state["current_page"] = nav_keys[nav_labels.index(selected_label)]

    # ── Model Settings ────────────────────────────────────────────────────────
    st.sidebar.markdown(sidebar_divider(), unsafe_allow_html=True)
    st.sidebar.markdown(sidebar_section_label("Model Settings"), unsafe_allow_html=True)

    checkpoint_dir = Path(__file__).resolve().parents[2] / "checkpoints"
    model_files = [p.name for p in checkpoint_dir.glob("*.pth")]

    if not model_files:
        st.sidebar.warning("No model checkpoints found in 'checkpoints' folder.")
        selected_model = None
    else:
        selected_model = st.sidebar.selectbox(
            "Model checkpoint",
            options=model_files,
            index=0,
            help="Choose which model checkpoint to load for inference.",
        )
        st.session_state["MODEL_PATH"] = str(checkpoint_dir / selected_model)

    st.session_state["SELECTED_MODEL"] = selected_model

    # ── Detection Settings ────────────────────────────────────────────────────
    st.sidebar.markdown(sidebar_divider(), unsafe_allow_html=True)
    st.sidebar.markdown(sidebar_section_label("Detection Settings"), unsafe_allow_html=True)

    # OOD threshold — primary decision boundary for unknown bird detection
    from config.settings import OOD_THRESHOLD as _DEFAULT_OOD
    ood_threshold = st.sidebar.slider(
        "OOD threshold",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.get("OOD_THRESHOLD", _DEFAULT_OOD),
        step=0.01,
        help=(
            "Images whose top-1 confidence is **below** this value are shown "
            "as 'Unknown Bird Species'. "
            "Increase to flag more images as unknown; decrease to be more permissive."
        ),
    )
    st.session_state["OOD_THRESHOLD"] = ood_threshold

    # Display confidence filter — hides low-ranked predictions from the UI
    confidence = st.sidebar.slider(
        "Display confidence filter",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.get("CONFIDENCE_THRESHOLD", 0.01),
        step=0.01,
        help="Hide predictions whose probability falls below this value in the Top-5 list.",
    )
    st.session_state["CONFIDENCE_THRESHOLD"] = confidence

    # ── Explainability Settings ────────────────────────────────────────────────
    st.sidebar.markdown(sidebar_divider(), unsafe_allow_html=True)
    st.sidebar.markdown(sidebar_section_label("Explainability"), unsafe_allow_html=True)

    from config.settings import GRADCAM_ENABLED as _DEFAULT_GRADCAM, CLIP_ENABLED as _DEFAULT_CLIP

    gradcam_on = st.sidebar.checkbox(
        "🔥 Show Grad-CAM heatmap",
        value=st.session_state.get("GRADCAM_ENABLED", _DEFAULT_GRADCAM),
        help="Visualise which image regions most influenced the model's prediction.",
    )
    st.session_state["GRADCAM_ENABLED"] = gradcam_on

    clip_on = st.sidebar.checkbox(
        "🔍 Enable CLIP fallback",
        value=st.session_state.get("CLIP_ENABLED", _DEFAULT_CLIP),
        help=(
            "When OOD is detected, run OpenCLIP zero-shot classification "
            "to suggest possible species. Requires `open_clip_torch`."
        ),
    )
    st.session_state["CLIP_ENABLED"] = clip_on

    # ── Status dot ────────────────────────────────────────────────────────────
    st.sidebar.markdown(sidebar_status_dot(), unsafe_allow_html=True)

