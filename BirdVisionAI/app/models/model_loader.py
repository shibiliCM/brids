# models/model_loader.py
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

def load_model(model_path: Path):
    """Load the ResNet‑18 checkpoint for CUB‑200‑2011.
    Args:
        model_path (Path): Path to the .pth checkpoint.
    Returns:
        torch.nn.Module: Ready‑to‑use model.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = models.resnet18(pretrained=False)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Dynamically adjust model.fc to match checkpoint shapes
    if "fc.weight" in state_dict:
        num_classes = state_dict["fc.weight"].shape[0]
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model
