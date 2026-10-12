"""models/clip_fallback.py
OpenCLIP zero-shot bird species fallback and cross-model verification.

Activated for Out-of-Distribution (OOD) detection and foundation-model cross-verification.
Uses ViT-B-32 (OpenAI weights) for zero-shot matching against a curated list
of common global bird species not covered in CUB-200-2011.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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
    "peacock", "indian peafowl", "green peafowl", "peahen",
    "crow", "house crow", "jungle crow", "carrion crow",
    "pigeon", "rock pigeon", "wood pigeon", "turtle dove", "mourning dove",
    "sparrow", "house sparrow", "eurasian tree sparrow",
    "bald eagle", "golden eagle", "fish eagle", "sea eagle",
    "red-tailed hawk", "cooper's hawk", "eurasian sparrowhawk",
    "barn owl", "great horned owl", "snowy owl", "tawny owl", "screech owl",
    "african grey parrot", "rose-ringed parakeet", "green parrot", "lovebird",
    "greater flamingo", "american flamingo", "lesser flamingo",
    "emperor penguin", "african penguin", "little blue penguin", "king penguin",
    "mallard duck", "wood duck", "mandarin duck", "domestic duck",
    "mute swan", "whooper swan", "trumpeter swan", "black swan",
    "grey heron", "great blue heron", "purple heron",
    "common kingfisher", "belted kingfisher", "pied kingfisher",
    "pileated woodpecker", "great spotted woodpecker",
    "american robin", "european robin",
    "goldfinch", "house finch", "purple finch",
    "european starling", "common starling",
    "eurasian magpie", "black-billed magpie",
    "blue jay", "eurasian jay", "steller's jay",
    "common raven", "northern raven",
    "turkey vulture", "griffon vulture", "egyptian vulture",
    "peregrine falcon", "american kestrel",
    "brown pelican", "american white pelican",
    "white stork", "black stork", "marabou stork",
    "sandhill crane", "common crane", "sarus crane",
    "toco toucan", "keel-billed toucan",
    "scarlet macaw", "blue-and-yellow macaw", "green-winged macaw",
    "ruby-throated hummingbird", "anna's hummingbird",
    "barn swallow", "tree swallow",
    "yellow warbler", "black-and-white warbler",
    "song thrush", "american robin thrush",
    "common myna", "indian myna", "crested myna",
    "purple sunbird", "olive-backed sunbird",
    "domestic canary", "atlantic canary",
    "budgerigar", "monk parakeet",
    "cockatiel", "sulphur-crested cockatoo", "palm cockatoo",
    "white ibis", "glossy ibis", "scarlet ibis",
    "great egret", "snowy egret", "cattle egret",
    "great cormorant", "double-crested cormorant",
    "atlantic puffin", "horned puffin",
    "herring gull", "laughing gull",
    "common tern", "arctic tern",
    "osprey",
    "roadrunner", "greater roadrunner",
    "secretary bird",
    "hoopoe", "eurasian hoopoe",
    "bee-eater", "european bee-eater",
    "ostrich", "common ostrich",
    "emu", "cassowary",
    "wild turkey", "domestic turkey",
    "rooster", "chicken",
    "common pheasant", "golden pheasant",
    "quail", "california quail",
]


# ── Result dataclasses ───────────────────────────────────────────────────────

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


@dataclass(frozen=True)
class CLIPVerification:
    """Consensus verification between ResNet-18 top-1 prediction and OpenCLIP."""
    is_discrepancy: bool
    clip_top_name: str
    clip_top_pct: float
    resnet_class_clip_pct: float
    suggestions: list[tuple[str, float]] = field(default_factory=list)
    reason: str | None = None


# ── Lazy singleton ────────────────────────────────────────────────────────────

_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_clip_loaded: bool = False
_candidate_text_features = None


def _load_clip() -> bool:
    """Lazy-load the OpenCLIP model and pre-compute candidate text features."""
    global _clip_model, _clip_preprocess, _clip_tokenizer, _clip_loaded, _candidate_text_features

    if _clip_loaded:
        return _clip_model is not None

    _clip_loaded = True
    try:
        import open_clip  # noqa: WPS433

        # Try fast local cached safetensors model first
        try:
            _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
                "hf-hub:timm/vit_base_patch32_clip_224.openai"
            )
            _clip_tokenizer = open_clip.get_tokenizer("hf-hub:timm/vit_base_patch32_clip_224.openai")
            logger.info("OpenCLIP loaded from local cached timm/vit_base_patch32_clip_224.openai.")
        except Exception:
            _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            _clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
            logger.info("OpenCLIP ViT-B-32 (openai) loaded successfully.")

        _clip_model.eval()

        # Pre-compute text features for all fixed candidates
        texts = [f"a photo of a {b}" for b in BIRD_CANDIDATES]
        tokens = _clip_tokenizer(texts)
        with torch.no_grad():
            txt_f = _clip_model.encode_text(tokens)
            _candidate_text_features = txt_f / txt_f.norm(dim=-1, keepdim=True)

        return True
    except Exception as exc:
        logger.warning(f"OpenCLIP unavailable: {exc}")
        return False


# ── Public API ────────────────────────────────────────────────────────────────

def is_clip_available() -> bool:
    """Return True if the OpenCLIP model can be loaded."""
    return _load_clip()


def run_clip_fallback(pil_image: Image.Image, top_k: int = 5) -> CLIPResult:
    """Run zero-shot CLIP bird classification on *pil_image*."""
    if not _load_clip():
        return CLIPResult(
            available=False,
            suggestions=[],
            error="open_clip_torch is not installed or failed to initialize.",
        )

    try:
        img_t = _clip_preprocess(pil_image).unsqueeze(0)

        with torch.no_grad():
            img_f = _clip_model.encode_image(img_t)
            img_f /= img_f.norm(dim=-1, keepdim=True)
            sims = (100.0 * img_f @ _candidate_text_features.T).softmax(dim=-1)

        scores = sims[0].tolist()
        ranked = sorted(
            zip(BIRD_CANDIDATES, scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]
        suggestions = [(name.title(), conf * 100.0) for name, conf in ranked]

        logger.info(f"CLIP top-1 suggestion: {suggestions[0]}")
        return CLIPResult(available=True, suggestions=suggestions)

    except Exception as exc:
        logger.error(f"CLIP inference error: {exc}")
        return CLIPResult(available=False, suggestions=[], error=str(exc))


def _clean_species_name(display_name: str) -> str:
    """Strip number prefix and format species name for prompting."""
    clean = re.sub(r"^\d+\.\s*", "", display_name).strip()
    return clean.replace("_", " ")


def verify_with_clip(
    pil_image: Image.Image,
    resnet_top_display_name: str,
    top_k: int = 5,
) -> CLIPVerification | None:
    """Cross-verify ResNet-18 top prediction against OpenCLIP foundation model.

    Detects if the image is an out-of-distribution species (like a Peacock)
    where ResNet-18 assigned high softmax probability to an unrelated CUB species
    (like Green Violetear) due to color/texture similarities.

    Returns:
        :class:`CLIPVerification` or None if CLIP is unavailable.
    """
    if not _load_clip():
        return None

    try:
        clean_resnet = _clean_species_name(resnet_top_display_name)
        resnet_prompt = f"a photo of a {clean_resnet.lower()}"

        # Combine fixed candidates and the specific ResNet predicted class
        all_candidates = list(BIRD_CANDIDATES)
        if clean_resnet.lower() not in [c.lower() for c in all_candidates]:
            all_candidates.append(clean_resnet.lower())

        prompts = [f"a photo of a {b}" for b in all_candidates]
        tokens = _clip_tokenizer(prompts)
        img_t = _clip_preprocess(pil_image).unsqueeze(0)

        with torch.no_grad():
            img_f = _clip_model.encode_image(img_t)
            txt_f = _clip_model.encode_text(tokens)
            img_f /= img_f.norm(dim=-1, keepdim=True)
            txt_f /= txt_f.norm(dim=-1, keepdim=True)
            sims = (100.0 * img_f @ txt_f.T).softmax(dim=-1)[0]

        scores = sims.tolist()
        ranked = sorted(
            zip(all_candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        top_cand, top_score = ranked[0]
        top_score_pct = top_score * 100.0

        # Find the score for the ResNet predicted class
        resnet_score_pct = 0.0
        for cand, sc in ranked:
            if cand.lower() == clean_resnet.lower():
                resnet_score_pct = sc * 100.0
                break

        top_suggestions = [
            (cand.title(), sc * 100.0) for cand, sc in ranked[:top_k]
        ]

        # Check for discrepancy:
        # If top CLIP match is an out-of-distribution species that strongly
        # dominates over the ResNet CUB class:
        is_same_bird = (
            top_cand.lower() == clean_resnet.lower()
            or clean_resnet.lower() in top_cand.lower()
            or top_cand.lower() in clean_resnet.lower()
        )

        # Discrepancy triggers when CLIP has strong confidence on an OOD species
        # and distinctly rejects the ResNet candidate
        is_discrepancy = (
            not is_same_bird
            and top_score_pct >= 20.0
            and (top_score_pct >= 2.0 * resnet_score_pct or resnet_score_pct < 5.0)
        )

        reason = None
        if is_discrepancy:
            reason = (
                f"Foundation Model Consensus: Identified as {top_cand.title()} "
                f"({top_score_pct:.1f}% confidence). The CUB-200 specialist model "
                f"predicted '{resnet_top_display_name}' ({resnet_score_pct:.1f}% CLIP score), "
                f"which lacks foundation model visual agreement."
            )

        return CLIPVerification(
            is_discrepancy=is_discrepancy,
            clip_top_name=top_cand.title(),
            clip_top_pct=top_score_pct,
            resnet_class_clip_pct=resnet_score_pct,
            suggestions=top_suggestions,
            reason=reason,
        )

    except Exception as exc:
        logger.error(f"Error during CLIP cross-verification: {exc}")
        return None
