import sys
from pathlib import Path
from PIL import Image

# Add project root and app directory to sys.path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "app"))
sys.path.insert(1, str(root))

from models.ood_detector import detect_ood
from models.clip_fallback import CLIPVerification, verify_with_clip, is_clip_available


def test_standard_msp_in_distribution():
    class_names = ["001.Black_footed_Albatross", "002.Laysan_Albatross", "070.Green_Violetear"]
    raw_preds = [(0, 0.85), (1, 0.10), (2, 0.05)]
    res = detect_ood(raw_preds, class_names, ood_threshold=0.70)
    assert not res.is_ood
    assert res.top_confidence == 85.0
    assert res.clip_override_name is None


def test_standard_msp_out_of_distribution():
    class_names = ["001.Black_footed_Albatross", "002.Laysan_Albatross", "070.Green_Violetear"]
    raw_preds = [(0, 0.45), (1, 0.35), (2, 0.20)]
    res = detect_ood(raw_preds, class_names, ood_threshold=0.70)
    assert res.is_ood
    assert res.top_confidence == 45.0


def test_foundation_model_override():
    class_names = ["070.Green_Violetear", "001.Black_footed_Albatross"]
    # Simulating ResNet overconfidence on an OOD bird (e.g. Peacock -> Green Violetear with 94%)
    raw_preds = [(0, 0.94), (1, 0.06)]
    
    # Simulating CLIP verification flagging discrepancy:
    clip_ver = CLIPVerification(
        is_discrepancy=True,
        clip_top_name="Peacock",
        clip_top_pct=65.5,
        resnet_class_clip_pct=1.2,
        suggestions=[("Peacock", 65.5), ("Indian Peafowl", 20.2)],
        reason="Foundation model consensus identified Peacock",
    )
    
    res = detect_ood(raw_preds, class_names, ood_threshold=0.70, clip_verification=clip_ver)
    assert res.is_ood
    assert res.clip_override_name == "Peacock"
    assert res.clip_confidence == 65.5
    assert "Peacock" in res.ood_reason


def test_field_report_generation():
    from ui.report import generate_field_report

    report = generate_field_report(
        species_name="Indian Peafowl",
        confidence=96.5,
        is_ood=True,
        clip_override="Indian Peafowl",
        top_candidates=[("Indian Peafowl", 96.5), ("Peacock", 2.1)],
    )
    assert "# 🪶 BirdVision AI — Field Observation Certificate" in report
    assert "Indian Peafowl" in report
    assert "96.5%" in report
    assert "Foundation Consensus Override" in report
    assert "BIRD-" in report

