"""
OPTIMIZED TRAINING ENGINE - Amazon EC2 / AWS Trainium
EfficientNet-B0 backbone with transfer learning, AdamW optimizer, MSELoss,
ReduceLROnPlateau scheduler, and custom early stopping guardrail.
"""
import os
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from torchvision import models
from tqdm import tqdm
import numpy as np

# Local imports
from src.dataset import load_config, create_dataloaders

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EARLY STOPPING GUARDRAIL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EarlyStopping:
    """Monitor validation loss and halt training if no improvement."""
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        metric: str = "val_loss",
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.best_epoch = 0
    
    def __call__(self, current_score: float, epoch: int) -> bool:
        """Check if training should stop. Returns True if should stop."""
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
        elif self._is_improvement(current_score):
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(
                    f"Early stopping triggered after {self.patience} epochs "
                    f"without improvement. Best {self.metric}: {self.best_score:.4f} "
                    f"at epoch {self.best_epoch}"
                )
                return True
        
        return False
    
    def _is_improvement(self, current_score: float) -> bool:
        """Check if current score is an improvement."""
        if self.mode == "min":
            return current_score < (self.best_score - self.min_delta)
        else:
            return current_score > (self.best_score + self.min_delta)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL BUILDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_model(config: dict, device: torch.device) -> nn.Module:
    """Build EfficientNet-B0 backbone for flood depth regression."""
    
    train_cfg = config.get("training", {})
    model_type = train_cfg.get("model_type", "efficientnet_b0")
    weights = train_cfg.get("backbone_weights", "imagenet")
    
    logger.info(f"Building {model_type} with {weights} weights...")
    
    if model_type == "efficientnet_b0":
        if weights == "imagenet":
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            model = models.efficientnet_b0(weights=None)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Replace final classification layer with regression head
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
        nn.Sigmoid()  # Output in [0, 1] range
    )
    
    return model.to(device)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINING LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Trainer:
    """Production trainer with checkpointing and monitoring."""
    
    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: torch.device,
        output_dir: str = "models"
    ):
        self.model = model
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        train_cfg = config.get("training", {})
        opt_cfg = train_cfg.get("optimizer", {})
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=opt_cfg.get("learning_rate", 0.001),
            weight_decay=opt_cfg.get("weight_decay", 0.0001)
        )
        
        # Loss function: HuberLoss for robustness against outliers
        # MSELoss penalizes large errors too heavily; HuberLoss is more balanced
        # delta=5.0: threshold where loss transitions from quadratic to linear
        self.criterion = nn.HuberLoss(delta=5.0)
        
        # Learning rate scheduler
        scheduler_cfg = train_cfg.get("lr_scheduler", {})
        self.scheduler_name = scheduler_cfg.get("name", "ReduceLROnPlateau")
        
        # Initialize ReduceLROnPlateau (will be used with validation)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode=scheduler_cfg.get("mode", "min"),
            factor=scheduler_cfg.get("factor", 0.5),
            patience=scheduler_cfg.get("patience", 3),
            min_lr=scheduler_cfg.get("min_lr", 0.00001),
        )
        
        # OneCycleLR will be initialized later when we know epochs and steps
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
            "learning_rate": []
        }
    
    def train_epoch(self, train_loader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc="Training")
        for images, depths in pbar:
            images = images.to(self.device)
            depths = depths.to(self.device).unsqueeze(1)  # Add channel dimension
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, depths)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # OneCycleLR scheduler step (per batch)
            if self.onecycle_scheduler is not None:
                self.onecycle_scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader) -> float:
        """Validate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating")
            for images, depths in pbar:
                images = images.to(self.device)
                depths = depths.to(self.device).unsqueeze(1)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, depths)
                
                total_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
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
            best_path = self.output_dir / "best_flood_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"✅ Best model saved to {best_path}")
        
        # Save periodic checkpoint
        ckpt_path = self.output_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, ckpt_path)
    
    def train(self, train_loader, val_loader, epochs: int):
        """Run full training loop."""
        logger.info(f"Starting training for {epochs} epochs")
        
        # Initialize OneCycleLR if selected
        if self.scheduler_name == "OneCycleLR":
            steps_per_epoch = len(train_loader)
            total_steps = steps_per_epoch * epochs
            logger.info(f"Using OneCycleLR: {total_steps} total steps ({steps_per_epoch} per epoch)")
            
            self.onecycle_scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.max_lr,
                total_steps=total_steps,
                pct_start=0.3,  # First 30% is warm-up
                anneal_strategy='cos',
                cycle_momentum=True
            )
        
        for epoch in range(1, epochs + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"EPOCH {epoch}/{epochs}")
            logger.info(f"{'='*60}")
            
            # Train and validate
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            # Log metrics
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["learning_rate"].append(
                self.optimizer.param_groups[0]["lr"]
            )
            
            logger.info(
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )
            
            # Check if best
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                logger.info(f"🎯 New best validation loss: {val_loss:.6f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best=is_best)
            
            # LR scheduler step (only for ReduceLROnPlateau)
            if self.scheduler_name != "OneCycleLR":
                self.scheduler.step(val_loss)
            
            # Early stopping check
            if self.early_stopping(val_loss, epoch):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training complete! Best val loss: {self.best_val_loss:.6f}")
        logger.info(f"{'='*60}\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN EXECUTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main(args):
    """Main training entry point."""
    
    # Load configuration
    config = load_config(args.config)
    train_cfg = config.get("training", {})
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        config,
        use_s3=args.use_s3,
        s3_bucket=args.s3_bucket,
        s3_region=args.s3_region
    )
    
    # Build model
    model = build_model(config, device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = Trainer(model, config, device, output_dir=args.output_dir)
    
    # Train
    epochs = train_cfg.get("epochs", 20)
    trainer.train(train_loader, val_loader, epochs=epochs)
    
    logger.info(f"✅ Training complete! Best model: {args.output_dir}/best_flood_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train flood depth estimator model")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file path")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory for models")
    parser.add_argument("--use-s3", action="store_true", help="Use AWS S3 for data")
    parser.add_argument("--s3-bucket", type=str, default="bengaluru-flood-datasets", help="S3 bucket name")
    parser.add_argument("--s3-region", type=str, default="ap-south-1", help="AWS region")
    
    args = parser.parse_args()
    main(args)
