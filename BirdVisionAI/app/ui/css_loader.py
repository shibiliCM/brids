"""ui/css_loader.py
Utility for injecting external CSS files into Streamlit pages.
"""
import streamlit as st
from pathlib import Path


def inject_css(css_path: Path | str) -> None:
    """Read *css_path* and inject it into the current Streamlit page.

    Args:
        css_path: Absolute or relative path to a ``.css`` file.

    Raises:
        FileNotFoundError: If the CSS file does not exist.
    """
    path = Path(css_path)
    if not path.exists():
        raise FileNotFoundError(f"CSS file not found: {path}")
    css = path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
