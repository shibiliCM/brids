"""models/gradcam.py
Grad-CAM (Gradient-weighted Class Activation Mapping) for ResNet-18.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization" (ICCV 2017).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class GradCAM:
    """Hook-based Grad-CAM for any CNN model.

    Args:
        model:        PyTorch model (eval mode recommended).
        target_layer: The conv layer to attach forward/backward hooks to.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._hooks: list = []
        self._register_hooks()

    # ── Hooks ──────────────────────────────────────────────────────────

    def _register_hooks(self) -> None:
        def _fwd(_module, _inp, out):
            self._activations = out.detach().clone()

        def _bwd(_module, _grad_in, grad_out):
            self._gradients = grad_out[0].detach().clone()

        self._hooks.append(self.target_layer.register_forward_hook(_fwd))
        self._hooks.append(
            self.target_layer.register_full_backward_hook(_bwd)
        )

    def remove_hooks(self) -> None:
        """Remove all registered hooks (call when done to avoid leaks)."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ── Heatmap ────────────────────────────────────────────────────────

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        class_idx: int,
        num_classes: int = 200,
    ) -> np.ndarray:
        """Compute a Grad-CAM saliency map for *class_idx*.

        Args:
            input_tensor: Preprocessed image tensor (1, C, H, W).
            class_idx:    Target class to explain.
            num_classes:  Logit slice size (200 for CUB).

        Returns:
            Float32 numpy array (h', w') normalised to [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward with gradients enabled
        output = self.model(input_tensor)
        logits = output[:, :num_classes]

        # Scalar score for target class → triggers backward
        score = logits[0, class_idx]
        score.backward()

        # Global-average-pool the gradients → channel weights
        weights = self._gradients.mean(dim=[0, 2, 3])      # (C,)
        acts    = self._activations[0]                      # (C, h, w)

        # Weighted sum + ReLU
        cam = (weights[:, None, None] * acts).sum(dim=0)   # (h, w)
        cam = torch.relu(cam).numpy()

        # Normalise to [0, 1]
        cam -= cam.min()
        if cam.max() > 1e-8:
            cam /= cam.max()

        return cam.astype(np.float32)

    # ── Overlay ────────────────────────────────────────────────────────

    @staticmethod
    def overlay_on_image(
        pil_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.45,
    ) -> Image.Image:
        """Blend a Grad-CAM heatmap with the original PIL image.

        Uses OpenCV's JET colormap for the standard red-hot rendering.

        Args:
            pil_image: Original RGB PIL image.
            heatmap:   Float32 array in [0, 1], any spatial size.
            alpha:     Heatmap opacity (0 = image only, 1 = heatmap only).

        Returns:
            Blended RGB PIL Image.
        """
        try:
            import cv2
            _has_cv2 = True
        except ImportError:
            _has_cv2 = False

        img_rgb = np.array(pil_image.convert("RGB"))
        h, w    = img_rgb.shape[:2]

        # --- Resize heatmap to image size ---
        if _has_cv2:
            hm_u8      = np.uint8(255 * heatmap)
            hm_resized = cv2.resize(hm_u8, (w, h), interpolation=cv2.INTER_LINEAR)
            hm_colored = cv2.applyColorMap(hm_resized, cv2.COLORMAP_JET)
            hm_rgb     = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)
            blended    = cv2.addWeighted(img_rgb, 1 - alpha, hm_rgb, alpha, 0)
        else:
            # Fallback: pure PIL/numpy (no OpenCV) — uses a simple red overlay
            from PIL import Image as PILImage
            hm_resized = np.array(
                PILImage.fromarray(np.uint8(255 * heatmap)).resize(
                    (w, h), PILImage.BILINEAR
                )
            ) / 255.0
            red = np.zeros_like(img_rgb, dtype=np.float32)
            red[..., 0] = 255
            img_f   = img_rgb.astype(np.float32)
            mask    = hm_resized[..., None]
            blended = (img_f * (1 - alpha * mask) + red * alpha * mask).clip(0, 255)
            blended = blended.astype(np.uint8)

        return Image.fromarray(blended)


# ── Convenience factory ───────────────────────────────────────────────────────

def get_gradcam(model: nn.Module) -> GradCAM:
    """Return a GradCAM hooked into ResNet-18's last conv layer (layer4[-1].conv2).

    This is the standard Grad-CAM target for ResNet-18: the last spatial
    feature map before global average pooling, producing the richest
    spatial attribution.
    """
    return GradCAM(model, target_layer=model.layer4[-1].conv2)
