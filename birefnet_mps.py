"""PyTorch MPS BiRefNet session with a rembg-compatible predict() API.

Uses the same ZhengPeng7/BiRefNet weights as rembg's birefnet-general, but runs
on Apple Silicon GPU via Metal. Deformable convolution may fall back to CPU
when mps-deform-conv is unavailable; inference is still much faster than ONNX.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / "work" / "hf-cache"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class BiRefNetMpsSession:
    """Drop-in for rembg sessions: predict(pil) -> [mask_pil].

    Preprocessing must match rembg: LANCZOS to 1024x1024 then ImageNet
    normalisation. Bilinear resizing thins narrow white parts such as the
    shuttle tail fin until the model stops predicting them.
    Set BRICK_MPS_PRECISION=fp32 to disable half precision.
    """

    def __init__(self, half=None):
        if half is None:
            half = os.environ.get("BRICK_MPS_PRECISION", "fp16").lower() != "fp32"
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        from transformers import AutoModelForImageSegmentation

        model = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True
        ).eval()
        model.to(self.device)
        # The published checkpoint is fp16; cast explicitly either way.
        self.half = bool(half) and self.device.type == "mps"
        model = model.half() if self.half else model.float()
        self.model = model

    def predict(self, img: Image.Image):
        rgb = img.convert("RGB")
        # LANCZOS matches rembg's preprocessing; bilinear thins narrow parts.
        resized = rgb.resize((1024, 1024), Image.Resampling.LANCZOS)
        arr = np.asarray(resized).astype(np.float32) / 255
        arr = (arr - MEAN) / STD
        x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
        if self.half:
            x = x.half()
        x = x.to(self.device)
        with torch.no_grad():
            preds = self.model(x)[-1].sigmoid()
        if self.device.type == "mps":
            torch.mps.synchronize()
        mask = preds[0, 0].float().cpu().numpy()
        mask_im = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8))
        mask_im = mask_im.resize(rgb.size, Image.Resampling.LANCZOS)
        return [mask_im]


def new_session(half=None):
    return BiRefNetMpsSession(half=half)
