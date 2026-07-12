"""ui/species_info.py
Species information panel for BirdVision AI.

Loads data from data/species_db.json and renders a styled info card.
When a species is not in the database the card degrades gracefully,
showing only the species name with a 'Data not available' notice.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import streamlit as st


# ── DB path ───────────────────────────────────────────────────────────────────
_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "species_db.json"


@lru_cache(maxsize=1)
def _load_db() -> dict:
    """Load species DB once and cache it in memory."""
    if _DB_PATH.exists():
        with open(_DB_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _lookup(display_name: str) -> dict | None:
    """Fuzzy-match *display_name* against the species DB.

    Strips the numeric prefix (e.g. '006. Least Auklet' → 'Least Auklet'),
    then tries exact and partial key matches.
    """
    db = _load_db()
    if not db:
        return None

    # Normalise: strip leading 'NNN. '
    name = display_name.strip()
    if ". " in name:
        name = name.split(". ", 1)[1].strip()

    # Exact match
    if name in db:
        return db[name]

    # Case-insensitive exact match
    name_lower = name.lower()
    for key, val in db.items():
        if key.lower() == name_lower:
            return val

    # Partial: key is contained in name or vice-versa
    for key, val in db.items():
        if key.lower() in name_lower or name_lower in key.lower():
            return val

    return None


# ── Conservation status badge colour ─────────────────────────────────────────
_STATUS_COLOURS: dict[str, str] = {
    "Least Concern":     "#7aad38",
    "Near Threatened":   "#d4a017",
    "Vulnerable":        "#e87a30",
    "Endangered":        "#e84040",
    "Critically Endangered": "#c01010",
    "Extinct in the Wild":   "#800000",
    "Extinct":               "#400000",
}


def _status_color(status: str) -> str:
    for key, col in _STATUS_COLOURS.items():
        if key.lower() in status.lower():
            return col
    return "#4a5238"


# ── Public renderer ───────────────────────────────────────────────────────────

def render_species_info(display_name: str) -> None:
    """Render the species information panel for *display_name*.

    Args:
        display_name: The formatted species name from the classifier
                      (e.g. '047. American Crow').
    """
    info = _lookup(display_name)

    st.markdown(
        "<div class='section-label'>Species Information</div>",
        unsafe_allow_html=True,
    )

    if info is None:
        st.markdown(
            f"""
<div class='species-card'>
  <div class='species-name'>{display_name}</div>
  <div class='species-no-data'>
    ℹ️ Detailed species data not yet in the database.
  </div>
</div>""",
            unsafe_allow_html=True,
        )
        return

    conservation = info.get("conservation", "Unknown")
    status_col   = _status_color(conservation)

    st.markdown(
        f"""
<div class='species-card'>
  <div class='species-name'>{display_name}</div>
  <div class='species-sci'>{info.get('scientific_name', '')}</div>

  <div class='species-badge-row'>
    <span class='species-badge' style='background:rgba(122,173,56,0.12);
          border-color:rgba(122,173,56,0.3);color:#7aad38'>
      🏛 {info.get('family', '—')}
    </span>
    <span class='species-badge' style='background:rgba(80,160,220,0.10);
          border-color:rgba(80,160,220,0.25);color:#60b0e0'>
      🌍 {info.get('region', '—')}
    </span>
    <span class='species-badge'
          style='background:rgba(0,0,0,0.15);
                 border-color:{status_col}55;color:{status_col}'>
      ♻ {conservation}
    </span>
  </div>

  <div class='species-grid'>
    <div class='species-field'>
      <div class='species-field-lbl'>🌿 Habitat</div>
      <div class='species-field-val'>{info.get('habitat', '—')}</div>
    </div>
    <div class='species-field'>
      <div class='species-field-lbl'>🍽 Diet</div>
      <div class='species-field-val'>{info.get('diet', '—')}</div>
    </div>
  </div>

  <div class='species-desc'>{info.get('description', '')}</div>
</div>""",
        unsafe_allow_html=True,
    )
