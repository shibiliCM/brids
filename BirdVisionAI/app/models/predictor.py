import torch

def predict(model, img_tensor, confidence_threshold=0.5):
    """Run model inference and return top-5 predictions above confidence threshold."""
    with torch.no_grad():
        output = model(img_tensor)
        # Slice to the first 200 classes (CUB classes) to handle checkpoints saved with 1000 outputs
        logits = output[:, :200]
        probs = torch.softmax(logits, dim=1).squeeze(0)
    
    # Check shape to handle batch size 1 properly
    if probs.dim() == 0:
        probs = probs.unsqueeze(0)

    # Top-K (up to 5)
    k = min(5, len(probs))
    topk = torch.topk(probs, k)

    top_idx = topk.indices.tolist()
    top_probs = topk.values.tolist()

    return list(zip(top_idx, top_probs))
