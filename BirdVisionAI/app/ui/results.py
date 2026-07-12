"""ui/results.py
Prediction results renderer.
All HTML sourced from ui/templates.py — no inline markup here.
"""
import streamlit as st

from ui.templates import (
    pred_top_card,
    pred_row_card,
    section_label,
    no_predictions_notice,
)


def display_results(top5: list[tuple[str, float]]) -> None:
    """Render top-5 predictions using premium card components.

    Args:
        top5: List of ``(display_name, confidence_pct)`` tuples sorted
              highest-first.  Confidence values are in **percent** (0–100).
    """
    if not top5:
        st.markdown(no_predictions_notice(), unsafe_allow_html=True)
        return

    top_name, top_pct = top5[0]

    # Rank 1 — best-match card
    st.markdown(pred_top_card(top_name, top_pct), unsafe_allow_html=True)

    # Ranks 2-5 — row cards
    if len(top5) > 1:
        st.markdown(section_label("Other candidates"), unsafe_allow_html=True)
        for rank, (name, pct) in enumerate(top5[1:5], start=2):
            bar_w = min(pct / top_pct * 100, 100) if top_pct > 0 else 0
            st.markdown(pred_row_card(rank, name, pct, bar_w), unsafe_allow_html=True)
