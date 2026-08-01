"""Segmentation engine abstraction with lazy model loading.

Backends supported (controlled by env `SEGMENTATION_BACKEND`):
- legacy  : uses `water_detection.WaterDetectionAnalyzer` to produce masks (CPU, lightweight)
- deeplab : uses a DeepLabV3 model (torchvision) for semantic segmentation

Public API (module-level convenience wrappers use a singleton):
- load_model()
- get_mask(image) -> np.uint8 mask (0/255)
- get_confidence(image) -> float in [0,1]

This file intentionally avoids heavy imports at module import-time.
"""

import os
import threading
import logging
from typing import Optional

import numpy as np
import cv2

LOG = logging.getLogger(__name__)


class SegmentationEngine:
    def __init__(self, backend: Optional[str] = None):
        self.backend = backend or os.environ.get("SEGMENTATION_BACKEND", "legacy")
        self._model = None
        self._lock = threading.Lock()
        self._loaded_backend = None
        # deeplab-specific objects
        self._transform = None
        self._device = None
        self._deeplab_class_idx = int(os.environ.get("SEGMENTATION_DEEPLAB_CLASS_IDX", "1"))

    def load_model(self):
        """Lazily load the model for the configured backend.

        Safe to call multiple times; the model is loaded once.
        """
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            backend = self.backend.lower()
            LOG.info("SegmentationEngine: loading backend=%s", backend)

            if backend == "legacy":
                # Use the existing python heuristics as a legacy backend.
                try:
                    from water_detection import WaterDetectionAnalyzer

                    self._model = WaterDetectionAnalyzer()
                    self._loaded_backend = "legacy"
                    LOG.info("SegmentationEngine: legacy backend loaded")
                except Exception as e:
                    LOG.exception("Failed to load legacy segmentation backend: %s", e)
                    raise

            elif backend == "deeplab":
                try:
                    import torch
                    from torchvision import models, transforms
                    from PIL import Image

                    self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                    ckpt = os.environ.get("SEGMENTATION_DEEPLAB_CHECKPOINT")
                    if ckpt:
                        # Expecting a state_dict compatible with torchvision.deeplabv3_resnet50
                        model = models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True)
                        state = torch.load(ckpt, map_location="cpu")
                        model.load_state_dict(state)
                    else:
                        model = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)

                    model.to(self._device).eval()
                    self._model = model
                    # standard ImageNet normalization
                    self._transform = transforms.Compose([
                        transforms.Resize((512, 512)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                    self._loaded_backend = "deeplab"
                    LOG.info("SegmentationEngine: deeplab backend loaded on %s", self._device)
                except Exception as e:
                    LOG.exception("Failed to load deeplab backend: %s", e)
                    raise

            else:
                raise ValueError(f"Unknown SEGMENTATION_BACKEND: {backend}")

    def _ensure_loaded(self):
        if self._model is None:
            self.load_model()

    def get_mask(self, image: np.ndarray) -> np.ndarray:
        """Return a binary mask (H x W) uint8 values {0,255} for water.

        Parameters
        - image: BGR numpy array (as produced by OpenCV)
        """
        self._ensure_loaded()

        if self._loaded_backend == "legacy":
            # water_detection provides a combined mask under key 'water_mask'
            try:
                res = self._model.detect_water_surface(image, depth_map=None)
                mask = res.get("water_mask")
                if mask is None:
                    # fallback: create mask from water_percentage threshold (no per-pixel mask available)
                    h, w = image.shape[:2]
                    mask = np.zeros((h, w), dtype=np.uint8)
                else:
                    mask = mask.astype(np.uint8)

                # Normalize to 0/255
                mask = (mask > 0).astype(np.uint8) * 255
                return mask
            except Exception as e:
                LOG.exception("Legacy backend failed to produce mask: %s", e)
                raise

        elif self._loaded_backend == "deeplab":
            try:
                import torch
                from PIL import Image

                # Preprocess
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                inp = self._transform(pil).unsqueeze(0).to(self._device)

                with torch.no_grad():
                    out = self._model(inp)['out']  # [1, C, H, W]
                    probs = torch.softmax(out, dim=1)
                    # class index to treat as 'water' is configurable
                    class_idx = int(self._deeplab_class_idx)
                    if class_idx >= probs.shape[1]:
                        # fallback: use argmax as a single class
                        mask_pred = torch.argmax(probs, dim=1)[0]
                        mask = (mask_pred.cpu().numpy() > 0).astype(np.uint8) * 255
                    else:
                        prob_map = probs[0, class_idx]
                        # Resize prob_map back to original image size
                        prob_map = torch.nn.functional.interpolate(
                            prob_map.unsqueeze(0).unsqueeze(0),
                            size=(image.shape[0], image.shape[1]),
                            mode='bilinear',
                            align_corners=False,
                        ).squeeze()
                        mask = (prob_map.cpu().numpy() > 0.5).astype(np.uint8) * 255

                    return mask
            except Exception as e:
                LOG.exception("Deeplab inference failed: %s", e)
                raise

        else:
            raise RuntimeError("Segmentation backend not loaded")

    def get_confidence(self, image: np.ndarray) -> float:
        """Return a scalar confidence (0.0 - 1.0) representing segmentation certainty.

        For `legacy` backend this maps to the consensus confidence. For `deeplab` it
        is the mean per-pixel probability for the configured class index.
        """
        self._ensure_loaded()

        if self._loaded_backend == "legacy":
            try:
                res = self._model.detect_water_surface(image, depth_map=None)
                return float(res.get("confidence", 0.0))
            except Exception:
                return 0.0

        elif self._loaded_backend == "deeplab":
            try:
                import torch
                from PIL import Image

                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                inp = self._transform(pil).unsqueeze(0).to(self._device)

                with torch.no_grad():
                    out = self._model(inp)['out']
                    probs = torch.softmax(out, dim=1)
                    class_idx = int(self._deeplab_class_idx)
                    if class_idx >= probs.shape[1]:
                        # fallback: use max class probability averaged
                        max_prob = torch.max(probs, dim=1)[0]
                        mean_conf = float(torch.mean(max_prob).cpu().numpy())
                    else:
                        prob_map = probs[0, class_idx]
                        mean_conf = float(torch.mean(prob_map).cpu().numpy())

                    return float(np.clip(mean_conf, 0.0, 1.0))
            except Exception as e:
                LOG.exception("Deeplab confidence computation failed: %s", e)
                return 0.0

        else:
            return 0.0


# Module-level singleton and convenience wrappers
_ENGINE = SegmentationEngine()


def load_model():
    return _ENGINE.load_model()


def get_mask(image: np.ndarray) -> np.ndarray:
    return _ENGINE.get_mask(image)


def get_confidence(image: np.ndarray) -> float:
    return _ENGINE.get_confidence(image)


__all__ = ["SegmentationEngine", "load_model", "get_mask", "get_confidence"]
