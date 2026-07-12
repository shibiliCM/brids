"""models/clip_fallback.py
OpenCLIP zero-shot bird species fallback.

Activated when the ResNet-18 classifier flags an image as Out-of-Distribution.
Uses ViT-B-32 (OpenAI weights) for zero-shot matching against a curated list
of common bird species NOT in CUB-200-2011.

Install: pip install open_clip_torch
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from PIL import Image

try:
    from utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)


# ── Common birds NOT in CUB-200 (for zero-shot matching) ─────────────────────
BIRD_CANDIDATES: list[str] = [
    "crow", "house crow", "jungle crow", "carrion crow",
    "pigeon", "rock pigeon", "wood pigeon", "turtle dove",
    "sparrow", "house sparrow", "eurasian tree sparrow",
    "bald eagle", "golden eagle", "fish eagle", "sea eagle",
    "red-tailed hawk", "cooper's hawk", "eurasian sparrowhawk",
    "barn owl", "great horned owl", "snowy owl", "tawny owl",
    "african grey parrot", "rose-ringed parakeet", "green parrot",
    "indian peafowl", "peacock",
    "greater flamingo", "american flamingo",
    "emperor penguin", "african penguin", "little blue penguin",
    "mallard duck", "wood duck", "mandarin duck",
    "mute swan", "whooper swan", "trumpeter swan",
    "grey heron", "great blue heron", "purple heron",
    "common kingfisher", "belted kingfisher",
    "pileated woodpecker", "great spotted woodpecker",
    "american robin", "european robin",
    "goldfinch", "house finch", "purple finch",
    "european starling",
    "eurasian magpie", "black-billed magpie",
    "blue jay", "eurasian jay", "steller's jay",
    "common raven", "northern raven",
    "turkey vulture", "griffon vulture", "egyptian vulture",
    "peregrine falcon", "american kestrel",
    "brown pelican", "american white pelican",
    "white stork", "black stork",
    "sandhill crane", "common crane",
    "toco toucan", "keel-billed toucan",
    "scarlet macaw", "blue-and-yellow macaw", "green-winged macaw",
    "ruby-throated hummingbird", "anna's hummingbird",
    "barn swallow", "tree swallow",
    "yellow warbler", "black-and-white warbler",
    "song thrush", "american robin thrush",
    "common myna", "indian myna",
    "purple sunbird", "olive-backed sunbird",
    "domestic canary",
    "budgerigar", "monk parakeet",
    "cockatiel", "sulphur-crested cockatoo",
    "white ibis", "glossy ibis", "scarlet ibis",
    "great egret", "snowy egret", "cattle egret",
    "great cormorant", "double-crested cormorant",
    "atlantic puffin", "horned puffin",
    "herring gull", "laughing gull",
    "common tern", "arctic tern",
    "osprey",
    "roadrunner",
    "secretary bird",
    "hoopoe",
    "bee-eater", "european bee-eater",
]


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CLIPResult:
    """Result returned by :func:`run_clip_fallback`.

    Attributes:
        available:   True when OpenCLIP is installed and inference succeeded.
        suggestions: List of ``(species_name, confidence_pct)`` tuples.
        error:       Error message when ``available`` is False.
    """
    available: bool
    suggestions: list[tuple[str, float]] = field(default_factory=list)
    error: str | None = None


# ── Lazy singleton ────────────────────────────────────────────────────────────

_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_clip_loaded: bool = False


def _load_clip() -> bool:
    """Lazy-load the OpenCLIP ViT-B-32 model.  Returns True on success."""
    global _clip_model, _clip_preprocess, _clip_tokenizer, _clip_loaded

    if _clip_loaded:
        return _clip_model is not None

    _clip_loaded = True
    try:
        import open_clip  # noqa: WPS433
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        _clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        _clip_model.eval()
        logger.info("OpenCLIP ViT-B-32 (openai) loaded successfully.")
        return True
    except Exception as exc:
        logger.warning(f"OpenCLIP unavailable: {exc}")
        return False


# ── Public API ────────────────────────────────────────────────────────────────

def is_clip_available() -> bool:
    """Return True if the OpenCLIP model can be loaded."""
    return _load_clip()


def run_clip_fallback(pil_image: Image.Image, top_k: int = 5) -> CLIPResult:
    """Run zero-shot CLIP bird classification on *pil_image*.

    Args:
        pil_image: Original (un-preprocessed) PIL image.
        top_k:     Number of species suggestions to return.

    Returns:
        :class:`CLIPResult` containing availability flag and suggestions.
    """
    if not _load_clip():
        return CLIPResult(
            available=False,
            suggestions=[],
            error="open_clip_torch is not installed.  Run: pip install open_clip_torch",
        )

    try:
        texts   = [f"a photo of a {b}" for b in BIRD_CANDIDATES]
        tokens  = _clip_tokenizer(texts)
        img_t   = _clip_preprocess(pil_image).unsqueeze(0)

        with torch.no_grad():
            img_f  = _clip_model.encode_image(img_t)
            txt_f  = _clip_model.encode_text(tokens)
            img_f /= img_f.norm(dim=-1, keepdim=True)
            txt_f /= txt_f.norm(dim=-1, keepdim=True)
            sims   = (100.0 * img_f @ txt_f.T).softmax(dim=-1)

        scores  = sims[0].tolist()
        ranked  = sorted(zip(BIRD_CANDIDATES, scores),
                         key=lambda x: x[1], reverse=True)[:top_k]
        suggestions = [(name.title(), conf) for name, conf in ranked]

        logger.info(f"CLIP top-1 suggestion: {suggestions[0]}")
        return CLIPResult(available=True, suggestions=suggestions)

    except Exception as exc:
        logger.error(f"CLIP inference error: {exc}")
        return CLIPResult(available=False, suggestions=[], error=str(exc))
