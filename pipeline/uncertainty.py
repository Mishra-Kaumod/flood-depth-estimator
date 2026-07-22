# pipeline/uncertainty.py
"""
Stage 3b — MC-Dropout Uncertainty Quantification
==================================================
Wraps the depth model and runs N forward passes with dropout active to
estimate pixel-wise depth uncertainty via Monte Carlo dropout.

Scalar output: depth_uncertainty_score (0–1)
  = mean variance across water-mask pixels, normalised by depth range.
  0 = very confident (all passes agree), 1 = highly uncertain.

Usage:
  unc_stage = UncertaintyStage(depth_stage, n_passes=8)
  score = unc_stage.score(image_bgr, water_mask)

GPU cost: N full forward passes on the depth model.
  On CPU (stub mode) cost is ~0 since the stub is deterministic + cheap.
  On GPU with ViT-L: ~8 × 100ms = ~800ms — acceptable for 15-min batch cadence.
  If latency is tight, reduce n_passes=3 or enable mini_ensemble mode below.

Mini-ensemble fallback (config mini_ensemble=True):
  Instead of N dropout passes on one model, load 2-3 pre-trained depth
  checkpoints and run one pass each.  Ensemble variance across checkpoints
  is used instead.  More robust than dropout if checkpoints are available.
"""

import logging
import numpy as np
from typing import List

log = logging.getLogger("pipeline.uncertainty")


class UncertaintyStage:
    """
    Computes a scalar depth_uncertainty_score for each image.

    Args:
        depth_stage:  the DepthStage instance (already loaded).
        n_passes:     number of MC-dropout forward passes (default 8).
        mini_ensemble_paths: optional list of extra checkpoint paths.
                             When provided, runs one pass per checkpoint
                             instead of N dropout passes on the primary model.
    """

    def __init__(self, depth_stage, n_passes: int = 8,
                 mini_ensemble_paths: List[str] | None = None):
        self.depth_stage          = depth_stage
        self.n_passes             = max(n_passes, 2)
        self.mini_ensemble_paths  = mini_ensemble_paths or []
        self._ensemble_models: list = []

        if self.mini_ensemble_paths:
            self._load_ensemble()
            log.info("Uncertainty: mini-ensemble mode (%d checkpoints)", len(self._ensemble_models))
        else:
            log.info("Uncertainty: MC-dropout mode (%d passes)", self.n_passes)

    # ── Public API ─────────────────────────────────────────────────────────
    def score(
        self,
        image_bgr:  np.ndarray,
        water_mask: np.ndarray,
    ) -> float:
        """
        Return depth_uncertainty_score ∈ [0, 1].

        Lower = model is confident about the flood depth estimate.
        Higher = passes disagree widely — treat depth estimate with caution.
        """
        if self.mini_ensemble_paths and self._ensemble_models:
            return self._ensemble_uncertainty(image_bgr, water_mask)
        return self._mcdropout_uncertainty(image_bgr, water_mask)

    # ── MC-Dropout ────────────────────────────────────────────────────────
    def _mcdropout_uncertainty(
        self, image_bgr: np.ndarray, water_mask: np.ndarray
    ) -> float:
        """
        Run N forward passes with dropout active. Compute pixel-wise variance
        over passes, then average across the water-masked region.

        If the depth model is a stub (no real weights), variance is near-zero
        because the stub is deterministic — this correctly signals "no useful
        uncertainty estimate available" and the score returns 0.5 (unknown).
        """
        model = self.depth_stage._model
        if model is None:
            # Stub: no weights → can't measure real uncertainty
            return 0.5

        try:
            import torch
            import cv2

            device = self.depth_stage.device

            # Enable dropout (train mode), freeze BN layers
            model.train()
            self._freeze_batchnorm(model)

            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            depth_maps: list[np.ndarray] = []

            with torch.no_grad():
                for _ in range(self.n_passes):
                    depth = model.infer_image(rgb)           # H×W float32
                    d_min, d_max = depth.min(), depth.max()
                    norm = (depth - d_min) / (d_max - d_min + 1e-6)
                    depth_maps.append(norm)

            model.eval()  # restore eval mode

            stack    = np.stack(depth_maps, axis=0)          # N×H×W
            variance = np.var(stack, axis=0)                 # H×W

            return self._normalised_score(variance, water_mask, stack)

        except Exception:
            log.warning("MC-dropout uncertainty failed — returning 0.5", exc_info=True)
            model.eval()
            return 0.5

    # ── Mini-ensemble ─────────────────────────────────────────────────────
    def _load_ensemble(self) -> None:
        """Load each extra checkpoint as an independent depth model."""
        from .depth import DepthStage
        for path in self.mini_ensemble_paths:
            try:
                stage = DepthStage(model_path=path, device=self.depth_stage.device)
                if stage._model is not None:
                    self._ensemble_models.append(stage)
            except Exception:
                log.warning("Could not load ensemble checkpoint %s", path, exc_info=True)

    def _ensemble_uncertainty(
        self, image_bgr: np.ndarray, water_mask: np.ndarray
    ) -> float:
        """Run one pass per ensemble checkpoint; compute variance across predictions."""
        maps: list[np.ndarray] = []
        # Include the primary model
        for stage in [self.depth_stage] + self._ensemble_models:
            try:
                result = stage.predict(image_bgr)
                maps.append(result.depth_map)
            except Exception:
                log.warning("Ensemble member prediction failed", exc_info=True)
        if len(maps) < 2:
            return 0.5
        stack    = np.stack(maps, axis=0)
        variance = np.var(stack, axis=0)
        return self._normalised_score(variance, water_mask, stack)

    # ── Shared scoring ────────────────────────────────────────────────────
    @staticmethod
    def _normalised_score(
        variance:   np.ndarray,  # H×W
        water_mask: np.ndarray,  # H×W bool
        stack:      np.ndarray,  # N×H×W
    ) -> float:
        """
        Aggregate pixel-wise variance within the water mask into a scalar.

        Normalise by the max per-pixel variance (= 0.25, achievable when
        half passes say 0 and half say 1), so the score is always in [0,1].
        """
        masked_var = variance[water_mask] if water_mask.any() else variance.ravel()
        if masked_var.size == 0:
            return 0.0
        mean_var = float(np.mean(masked_var))
        # Max theoretical variance for values in [0,1] = 0.25
        normalised = min(mean_var / 0.25, 1.0)
        return round(normalised, 4)

    @staticmethod
    def _freeze_batchnorm(model) -> None:
        """Keep BN layers in eval mode so only dropout is stochastic."""
        try:
            import torch.nn as nn
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.eval()
        except Exception:
            pass
