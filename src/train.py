"""
OPTIMIZED TRAINING ENGINE - Amazon EC2 / AWS Trainium
EfficientNet-B0 backbone with transfer learning, AdamW optimizer, MSELoss,
ReduceLROnPlateau scheduler, and custom early stopping guardrail.
"""
import os
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from torchvision import models
from tqdm import tqdm
import numpy as np

# Local imports
from src.dataset import load_config, create_dataloaders

try:
    import boto3
except Exception:
    boto3 = None

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


def setup_distributed(enabled: bool) -> tuple[bool, int, int, int]:
    """Initialize torch.distributed when launched with torchrun."""
    if not enabled:
        return False, 0, 1, 0

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if world_size <= 1:
        return False, rank, world_size, local_rank

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank

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
        output_dir: str = "models",
        is_primary: bool = True,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.is_primary = is_primary
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
        
        loss_name = str(train_cfg.get("loss_function", "HuberLoss"))
        huber_delta = float(train_cfg.get("huber_delta", 5.0))
        if loss_name == "MSELoss":
            self.criterion = nn.MSELoss()
        elif loss_name == "HuberLoss":
            self.criterion = nn.HuberLoss(delta=huber_delta)
        else:
            raise ValueError(f"Unsupported loss_function: {loss_name}")
        
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
        self.registry_cfg = train_cfg.get("model_registry", {})
    
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
        if not self.is_primary:
            return

        model_state = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history
        }
        
        if is_best:
            best_path = self.output_dir / "best_flood_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"✅ Best model saved to {best_path}")
            self._register_best_model(best_path, epoch=epoch, val_loss=self.best_val_loss)
        
        # Save periodic checkpoint
        ckpt_path = self.output_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, ckpt_path)

    def _register_best_model(self, model_path: Path, epoch: int, val_loss: float) -> None:
        """Optional model registry publish (S3) for lineage and governance."""
        if not self.registry_cfg.get("enabled", False):
            return

        if boto3 is None:
            logger.warning("Model registry enabled but boto3 is unavailable.")
            return

        bucket = self.registry_cfg.get("bucket")
        prefix = self.registry_cfg.get("prefix", "model-registry")
        if not bucket:
            logger.warning("Model registry enabled but bucket is not configured.")
            return

        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        model_key = f"{prefix}/best_flood_model_{timestamp}.pth"
        metadata_key = f"{prefix}/best_flood_model_{timestamp}.json"
        metadata = {
            "artifact": model_key,
            "created_at_utc": timestamp,
            "epoch": epoch,
            "best_val_loss": float(val_loss),
            "model_type": self.config.get("training", {}).get("model_type", "efficientnet_b0"),
            "loss_function": self.config.get("training", {}).get("loss_function", "HuberLoss"),
        }

        client = boto3.client("s3")
        client.upload_file(str(model_path), bucket, model_key)
        client.put_object(Bucket=bucket, Key=metadata_key, Body=json.dumps(metadata, indent=2).encode("utf-8"))
        logger.info("✅ Registered model artifact to s3://%s/%s", bucket, model_key)
    
    def train(self, train_loader, val_loader, epochs: int):
        """Run full training loop."""
        if self.is_primary:
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
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            if self.is_primary:
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
            
            if self.is_primary:
                logger.info(
                    f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )
            
            # Check if best
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                if self.is_primary:
                    logger.info(f"🎯 New best validation loss: {val_loss:.6f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best=is_best)
            
            # LR scheduler step (only for ReduceLROnPlateau)
            if self.scheduler_name != "OneCycleLR":
                self.scheduler.step(val_loss)
            
            # Early stopping check
            if self.early_stopping(val_loss, epoch):
                if self.is_primary:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        if self.is_primary:
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

    distributed, rank, world_size, local_rank = setup_distributed(args.distributed)
    
    # Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if distributed else "cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        config,
        use_s3=args.use_s3,
        s3_bucket=args.s3_bucket,
        s3_region=args.s3_region,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )
    
    # Build model
    model = build_model(config, device)
    if distributed:
        ddp_device_ids = [local_rank] if torch.cuda.is_available() else None
        model = nn.parallel.DistributedDataParallel(model, device_ids=ddp_device_ids)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = Trainer(model, config, device, output_dir=args.output_dir, is_primary=(rank == 0))
    
    # Train
    epochs = train_cfg.get("epochs", 20)
    trainer.train(train_loader, val_loader, epochs=epochs)
    
    if rank == 0:
        logger.info(f"✅ Training complete! Best model: {args.output_dir}/best_flood_model.pth")
    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train flood depth estimator model")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file path")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory for models")
    parser.add_argument("--use-s3", action="store_true", help="Use AWS S3 for data")
    parser.add_argument("--s3-bucket", type=str, default="bengaluru-flood-datasets", help="S3 bucket name")
    parser.add_argument("--s3-region", type=str, default="ap-south-1", help="AWS region")
    parser.add_argument("--distributed", action="store_true", help="Enable DistributedDataParallel (torchrun)")
    
    args = parser.parse_args()
    main(args)
