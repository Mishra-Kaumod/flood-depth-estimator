"""
Phase 3: MC-Dropout confidence estimation.

Runs N stochastic forward passes with dropout active (model.train() mode)
to produce a principled uncertainty estimate alongside the depth prediction.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

MC_PASSES = 12  # number of stochastic forward passes


def mc_dropout_confidence(
    model,
    tensor,
    n_passes: int = MC_PASSES,
) -> Tuple[float, float]:
    """
    Return (mean_output, confidence_score) using MC-Dropout.

    confidence_score = 1 - clipped_coefficient_of_variation
    so high agreement across passes -> high confidence.
    """
    import torch

    # Enable dropout layers (train mode) while keeping BN frozen
    _set_dropout_train(model)

    samples = []
    with torch.no_grad():
        for _ in range(n_passes):
            out = model(tensor)
            if hasattr(out, "logits"):
                out = out.logits
            val = float(out.squeeze())
            samples.append(val)

    model.eval()

    arr = np.array(samples)
    mean_val = float(arr.mean())
    std_val = float(arr.std())

    # Coefficient of variation (relative uncertainty); cap at 1.0
    if abs(mean_val) > 1e-6:
        cv = min(std_val / abs(mean_val), 1.0)
    else:
        cv = 1.0

    confidence = round(1.0 - cv, 4)
    confidence = max(0.0, min(1.0, confidence))
    return mean_val, confidence


def _set_dropout_train(model):
    """Set only Dropout layers to train mode; leave everything else in eval."""
    import torch.nn as nn
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
