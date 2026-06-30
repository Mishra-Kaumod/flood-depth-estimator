"""
WATER-AWARE TRAINING ENGINE
Trains model with water region detection for partial flooding scenarios.
Essential for handling images where one side is flooded, other is dry.
"""
import os
import yaml
import logging
import argparse
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from torchvision import models
from tqdm import tqdm
import numpy as np

# Ensure the repo root is importable when running this file directly.
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Local imports
from src.dataset import load_config, create_dataloaders
from src.train import build_model, EarlyStopping
from src.water_region_detector import WaterRegionDetector, RegionBasedDataLoader, RegionAwareTrainer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WATER-AWARE TRAINER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class WaterAwareTrainer:
    """
    Training engine that:
    1. Detects water regions in each image
    2. Only calculates loss on water pixels
    3. Focuses learning on flooded areas only
    4. Handles partially flooded images correctly
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: torch.device,
        detector: WaterRegionDetector = None,
        output_dir: str = "models"
    ):
        """
        Initialize water-aware trainer.
        
        Args:
            model: PyTorch model to train
            config: Configuration dict
            device: torch.device (cuda or cpu)
            detector: WaterRegionDetector instance
            output_dir: Directory for model checkpoints
        """
        self.model = model
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Water detector
        self.detector = detector or WaterRegionDetector(
            use_hsv=True,
            use_rgb=True,
            use_contrast=True
        )
        
        # Optimizer & Loss
        train_cfg = config.get("training", {})
        opt_cfg = train_cfg.get("optimizer", {})
        
        self.optimizer = AdamW(
            model.parameters(),
            lr=opt_cfg.get("learning_rate", 0.001),
            weight_decay=opt_cfg.get("weight_decay", 0.0001)
        )
        
        # HuberLoss for robustness
        self.criterion = nn.HuberLoss(delta=5.0)
        
        # Scheduler
        scheduler_cfg = train_cfg.get("lr_scheduler", {})
        self.scheduler_name = scheduler_cfg.get("name", "ReduceLROnPlateau")
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode=scheduler_cfg.get("mode", "min"),
            factor=scheduler_cfg.get("factor", 0.5),
            patience=scheduler_cfg.get("patience", 3),
            min_lr=scheduler_cfg.get("min_lr", 0.00001),
            verbose=True
        )
        
        self.onecycle_scheduler = None
        self.max_lr = opt_cfg.get("learning_rate", 0.001)
        
        # Early stopping
        es_cfg = train_cfg.get("early_stopping", {})
        self.early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 5),
            min_delta=es_cfg.get("min_delta", 0.001),
            metric=es_cfg.get("metric", "val_loss")
        )
        
        # Tracking
        self.best_val_loss = float("inf")
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "water_coverage": []
        }
    
    def detect_water_regions(self, images: torch.Tensor) -> tuple:
        """
        Detect water regions for a batch of images.
        
        Args:
            images: Tensor of shape (B, 3, H, W) with normalized values
        
        Returns:
            water_masks: Binary masks (B, 1, H, W)
            water_coverages: List of water percentages (B,)
        """
        batch_size = images.shape[0]
        water_masks = []
        water_coverages = []
        
        for i in range(batch_size):
            # Denormalize tensor to image
            img_tensor = images[i].cpu().numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = (img_tensor * std + mean) * 255
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            
            # Detect water
            mask, coverage = self.detector.detect(img_np)
            water_masks.append(torch.from_numpy(mask).float() / 255.0)
            water_coverages.append(coverage)
        
        # Stack masks
        water_masks = torch.stack(water_masks).unsqueeze(1).to(self.device)  # (B, 1, H, W)
        
        return water_masks, water_coverages
    
    def compute_masked_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss only on water regions.
        
        This prevents:
        - Learning from non-water pixels
        - Biasing from low-water-coverage images
        
        Args:
            outputs: Model predictions (B, 1, H, W)
            targets: Ground truth depths (B, 1, H, W)
            masks: Water region masks (B, 1, H, W)
        
        Returns:
            Masked loss value
        """
        # Count valid water pixels
        valid_pixels = masks.sum()
        
        if valid_pixels < 10:
            # Too few water pixels, skip this image
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Apply mask
        masked_outputs = outputs * masks
        masked_targets = targets * masks
        
        # Calculate loss
        masked_loss = self.criterion(masked_outputs, masked_targets)
        
        # Normalize by number of valid pixels
        # Ensures consistent loss scale across different water coverages
        total_pixels = outputs.numel()
        normalized_loss = masked_loss * (total_pixels / valid_pixels.clamp(min=1))
        
        return normalized_loss
    
    def train_epoch(self, train_loader) -> tuple:
        """
        Train for one epoch with water-awareness.
        
        Returns:
            avg_loss: Average training loss
            coverage_stats: Water coverage distribution
        """
        self.model.train()
        total_loss = 0.0
        coverage_stats = {"high": 0, "medium": 0, "low": 0}
        
        pbar = tqdm(train_loader, desc="Training (Water-Aware)")
        
        for images, depths in pbar:
            images = images.to(self.device)
            depths = depths.to(self.device).unsqueeze(1)
            
            # Detect water regions
            water_masks, water_coverages = self.detect_water_regions(images)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Compute masked loss
            loss = self.compute_masked_loss(outputs, depths, water_masks)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # OneCycleLR step (per batch)
            if self.onecycle_scheduler is not None:
                self.onecycle_scheduler.step()
            
            total_loss += loss.item()
            
            # Track water coverage distribution
            for coverage in water_coverages:
                if coverage > 70:
                    coverage_stats["high"] += 1
                elif coverage > 30:
                    coverage_stats["medium"] += 1
                else:
                    coverage_stats["low"] += 1
            
            pbar.set_postfix({"loss": loss.item(), "high_water": coverage_stats["high"]})
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss, coverage_stats
    
    def validate(self, val_loader) -> tuple:
        """
        Validate model on validation set with water-awareness.
        
        Returns:
            avg_loss: Average validation loss
            coverage_stats: Water coverage distribution
        """
        self.model.eval()
        total_loss = 0.0
        coverage_stats = {"high": 0, "medium": 0, "low": 0}
        
        pbar = tqdm(val_loader, desc="Validation")
        
        with torch.no_grad():
            for images, depths in pbar:
                images = images.to(self.device)
                depths = depths.to(self.device).unsqueeze(1)
                
                # Detect water regions
                water_masks, water_coverages = self.detect_water_regions(images)
                
                # Forward pass
                outputs = self.model(images)
                
                # Compute masked loss
                loss = self.compute_masked_loss(outputs, depths, water_masks)
                
                total_loss += loss.item()
                
                # Track coverage
                for coverage in water_coverages:
                    if coverage > 70:
                        coverage_stats["high"] += 1
                    elif coverage > 30:
                        coverage_stats["medium"] += 1
                    else:
                        coverage_stats["low"] += 1
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss, coverage_stats
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history
        }
        
        if is_best:
            best_path = self.output_dir / "best_flood_model_water_aware.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"✅ Best water-aware model saved to {best_path}")
        
        ckpt_path = self.output_dir / f"checkpoint_water_aware_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, ckpt_path)
    
    def train(self, train_loader, val_loader, epochs: int):
        """
        Run full water-aware training loop.
        """
        logger.info(f"Starting water-aware training for {epochs} epochs")
        
        # Initialize OneCycleLR if selected
        if self.scheduler_name == "OneCycleLR":
            steps_per_epoch = len(train_loader)
            total_steps = steps_per_epoch * epochs
            logger.info(f"Using OneCycleLR: {total_steps} total steps")
            
            self.onecycle_scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.max_lr,
                total_steps=total_steps,
                pct_start=0.3,
                anneal_strategy='cos',
                cycle_momentum=True
            )
        
        for epoch in range(1, epochs + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"EPOCH {epoch}/{epochs} (Water-Aware)")
            logger.info(f"{'='*60}")
            
            # Train
            train_loss, train_coverage = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_coverage = self.validate(val_loader)
            
            # Track metrics
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["learning_rate"].append(
                self.optimizer.param_groups[0]["lr"]
            )
            self.training_history["water_coverage"].append(train_coverage)
            
            logger.info(
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )
            
            logger.info(
                f"Water Coverage - High: {train_coverage['high']}, "
                f"Medium: {train_coverage['medium']}, Low: {train_coverage['low']}"
            )
            
            # Check if best
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                logger.info(f"🎯 New best validation loss: {val_loss:.6f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best=is_best)
            
            # LR scheduler step
            if self.scheduler_name != "OneCycleLR":
                self.scheduler.step(val_loss)
            
            # Early stopping
            if self.early_stopping(val_loss, epoch):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Water-aware training complete!")
        logger.info(f"Best val loss: {self.best_val_loss:.6f}")
        logger.info(f"Model saved: models/best_flood_model_water_aware.pth")
        logger.info(f"{'='*60}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN EXECUTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main(args):
    """Main water-aware training entry point."""
    
    # Load configuration
    config = load_config(args.config)
    train_cfg = config.get("training", {})
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Build model
    model = build_model(config, device)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    
    # Create water detector
    detector = WaterRegionDetector(
        use_hsv=True,
        use_rgb=True,
        use_contrast=True
    )
    
    # Create water-aware trainer
    trainer = WaterAwareTrainer(
        model=model,
        config=config,
        device=device,
        detector=detector,
        output_dir="models"
    )
    
    # Train
    epochs = train_cfg.get("epochs", 20)
    trainer.train(train_loader, val_loader, epochs)
    
    logger.info("✨ Water-aware training complete!")
    logger.info("📊 Check models/best_flood_model_water_aware.pth for best model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Water-aware training for flood depth estimation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file"
    )
    
    args = parser.parse_args()
    main(args)
