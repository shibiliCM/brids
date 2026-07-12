"""ui/confidence_chart.py
Confidence bar chart for Top-5 predictions.

Renders an interactive Plotly horizontal bar chart styled to match the
BirdVision AI dark theme.  Falls back to Streamlit's native bar_chart
if Plotly is unavailable.
"""
from __future__ import annotations

import streamlit as st


def render_confidence_chart(top5: list[tuple[str, float]]) -> None:
    """Render a horizontal bar chart of Top-5 confidence scores.

    Args:
        top5: List of ``(display_name, confidence_pct)`` tuples, sorted
              highest-first.  Confidence values are in percent (0–100).
    """
    if not top5:
        return

    labels = [f"#{i+1} {n[:30]}{'…' if len(n) > 30 else ''}"
              for i, (n, _) in enumerate(top5)]
    values = [round(c, 2) for _, c in top5]

    try:
        import plotly.graph_objects as go  # noqa: WPS433

        bar_colors = [
            "#a8e060", "#7aad38", "#5a9028", "#3d6e18", "#2a4e10"
        ][: len(top5)]

        fig = go.Figure(
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker_color=bar_colors,
                text=[f"{v:.1f}%" for v in values],
                textposition="outside",
                textfont=dict(color="#c8d8a0", size=11, family="DM Sans"),
                hovertemplate="%{y}<br>Confidence: %{x:.2f}%<extra></extra>",
            )
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=50, t=10, b=10),
            height=220,
            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.05)",
                tickfont=dict(color="#4a5238", size=10),
                range=[0, max(values) * 1.25],
                zeroline=False,
            ),
            yaxis=dict(
                tickfont=dict(color="#a8b880", size=11, family="DM Sans"),
                autorange="reversed",
            ),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    except ImportError:
        # Fallback: Streamlit native chart
        import pandas as pd

        df = pd.DataFrame({"Confidence (%)": values}, index=labels)
        st.bar_chart(df, color="#7aad38", height=220)
