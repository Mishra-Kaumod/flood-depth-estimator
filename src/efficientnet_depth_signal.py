"""Shared, unchanged EfficientNet checkpoint loading and preprocessing.

Extracted mechanically from the retained legacy pipeline. V6 loads only this
primary model; no legacy fusion, guards, classifier or fallback is executed.
"""
from pathlib import Path
from typing import Optional
import logging

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from src.settings import load_settings_dict

logger = logging.getLogger(__name__)


class EfficientNetDepthSignal:
    def __init__(self) -> None:
        self._efficientnet_model = None
        self._efficientnet_transform = None
        self._efficientnet_backend = "disabled"
        self._efficientnet_max_depth_cm = 100.0
        self._load_efficientnet_signal_if_available()
        if self._efficientnet_model is None or self._efficientnet_transform is None:
            raise RuntimeError("V6 EfficientNet checkpoint could not be loaded; check inference.efficientnet_signal configuration and checkpoint")

    def _build_efficientnet_depth_model(self) -> nn.Module:
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        return model

    def _load_efficientnet_signal_if_available(self) -> None:
        try:
            cfg = load_settings_dict().get("inference", {}).get("efficientnet_signal", {})
        except Exception as exc:
            logger.info("EfficientNet signal config unavailable: %s", exc)
            return

        if not bool(cfg.get("enabled", False)):
            return

        model_path = Path(str(cfg.get("model_path", "models/candidate/best_flood_model_water_aware.pth")))
        configured_max_depth_cm = float(cfg.get("max_depth_cm", 100.0))
        if not model_path.exists():
            logger.warning("EfficientNet signal checkpoint missing at %s", model_path)
            return

        try:
            device = torch.device("cuda" if torch.cuda.is_available() and str(cfg.get("device", "cpu")) == "cuda" else "cpu")
            model = self._build_efficientnet_depth_model().to(device)
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self._efficientnet_max_depth_cm = float(checkpoint.get("max_depth_cm", configured_max_depth_cm)) if isinstance(checkpoint, dict) else configured_max_depth_cm
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            self._efficientnet_model = model
            self._efficientnet_device = device
            self._efficientnet_backend = str(model_path)
            self._efficientnet_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            logger.info("Loaded EfficientNet depth signal from %s", model_path)
        except Exception as exc:
            logger.warning("EfficientNet depth signal unavailable: %s", exc)
            self._efficientnet_model = None
            self._efficientnet_transform = None
            self._efficientnet_backend = "unavailable"


    def _efficientnet_depth_signal(self, image_rgb: np.ndarray) -> Optional[float]:
        if self._efficientnet_model is None or self._efficientnet_transform is None:
            return None
        image = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
        tensor = self._efficientnet_transform(image).unsqueeze(0).to(self._efficientnet_device)
        with torch.no_grad():
            return round(float(self._efficientnet_model(tensor).squeeze().item()) * self._efficientnet_max_depth_cm, 2)

    def predict(self, image_rgb: np.ndarray) -> dict:
        """Supply the existing V6 signal contract, without executing V5."""
        return {"structured_features": {
            "efficientnet_candidate_depth_cm": self._efficientnet_depth_signal(image_rgb),
            "efficientnet_backend": self._efficientnet_backend,
            "input_image_height_px": image_rgb.shape[0],
        }}
