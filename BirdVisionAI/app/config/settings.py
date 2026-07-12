import os
from pathlib import Path

# Base directory of the project (three levels up from this file)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH       = ROOT_DIR / "checkpoints" / "cub_resnet18.pth"
CLASS_NAMES_PATH = ROOT_DIR / "data" / "classes.txt"
SPECIES_DB_PATH  = ROOT_DIR / "data" / "species_db.json"

# ── Inference ─────────────────────────────────────────────────────────────────
IMAGE_SIZE: int   = 224        # Resize input images to this square size
TOP_K: int        = 5          # Number of top predictions to display
NUM_CLASSES: int  = 200        # CUB-200-2011 classes

# ── Out-of-Distribution (OOD) Detection ───────────────────────────────────────
# If the model's top-1 softmax probability is below this value the image is
# reported as "Unknown Bird Species" instead of a named CUB-200 class.
#
# Range  : 0.0 – 1.0   (float)
# Default: 0.70  (70 %)
OOD_THRESHOLD: float = 0.70

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
GRADCAM_ENABLED: bool   = True     # Show Grad-CAM heatmap by default
GRADCAM_ALPHA:   float  = 0.45     # Heatmap overlay opacity (0–1)

# ── CLIP Fallback ─────────────────────────────────────────────────────────────
# Activated when OOD is detected.  Requires: pip install open_clip_torch
CLIP_ENABLED:    bool  = True
CLIP_MODEL:      str   = "ViT-B-32"
CLIP_PRETRAINED: str   = "openai"
CLIP_TOP_K:      int   = 5
