from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


def _extract_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        if isinstance(state_dict, Mapping):
            return state_dict
    if isinstance(checkpoint, Mapping):
        return checkpoint
    raise TypeError("Unsupported checkpoint format for severity model.")


def _build_severity_model(state_dict: Mapping[str, torch.Tensor]) -> torch.nn.Module:
    keys = set(state_dict.keys())

    if "classifier.1.weight" in keys:
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
        return model

    if "fc.weight" in keys:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 5)
        return model

    raise RuntimeError(
        "Unsupported severity checkpoint architecture. "
        "Expected EfficientNet-B0 or ResNet18 state dict."
    )


def load_severity_model(weights_path: Path, device: str) -> torch.nn.Module:
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)
    model = _build_severity_model(state_dict)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
