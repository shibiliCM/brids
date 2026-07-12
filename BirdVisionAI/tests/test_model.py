import sys
from pathlib import Path

# Add project root and app directory to sys.path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "app"))
sys.path.insert(1, str(root))

from config.settings import MODEL_PATH
from models.model_loader import load_model

from models.predictor import predict
import torch

def test_model_loading():
    if MODEL_PATH.exists():
        model = load_model(MODEL_PATH)
        assert model is not None
    else:
        # If the file isn't present in CI/CD environment, skip or pass
        pass

def test_predict():
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            # Return dummy logits where index 10 has highest logit
            logits = torch.zeros((1, 200))
            logits[0, 10] = 10.0
            logits[0, 5] = 5.0
            return logits

    model = DummyModel()
    dummy_input = torch.randn((1, 3, 224, 224))
    
    # Predict with threshold 0.5
    preds = predict(model, dummy_input, confidence_threshold=0.5)
    assert len(preds) > 0
    # The highest should be index 10
    top_idx, top_prob = preds[0]
    assert top_idx == 10
    assert top_prob > 0.9
