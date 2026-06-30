"""
COMPUTE DATASET STATISTICS
Calculates mean and std from training images for optimal normalization.
These values are dataset-specific and provide better performance than ImageNet defaults.
"""
import os
import yaml
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

logger = None

def setup_logging():
    """Configure logging."""
    import logging
    global logger
    if logger is None:
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class ImageStatsDataset:
    """Simple dataset to load images from directory."""
    
    def __init__(self, image_dir: str, transform=None):
        self.image_dir = Path(image_dir)
        self.image_files = list(self.image_dir.glob("*.jpg")) + \
                          list(self.image_dir.glob("*.png")) + \
                          list(self.image_dir.glob("*.jpeg"))
        self.transform = transform
        
        print(f"Found {len(self.image_files)} images in {image_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        
        return img

def compute_dataset_statistics(
    image_dir: str,
    num_workers: int = 4,
    batch_size: int = 32,
    max_images: int = None
) -> dict:
    """
    Compute mean and std for training images.
    
    Args:
        image_dir: Path to training images
        num_workers: DataLoader workers
        batch_size: Batch size for processing
        max_images: Limit number of images (None = all)
    
    Returns:
        Dictionary with 'mean' and 'std' (list of 3 values for RGB)
    """
    setup_logging()
    
    # Create dataset with no normalization yet
    dataset = ImageStatsDataset(image_dir, transform=transforms.ToTensor())
    
    # Limit to max_images if specified
    if max_images is not None and len(dataset) > max_images:
        indices = torch.randperm(len(dataset))[:max_images].tolist()
        dataset.image_files = [dataset.image_files[i] for i in indices]
        logger.info(f"Limited to {max_images} random images for statistics")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True
    )
    
    # Compute statistics
    logger.info("Computing dataset statistics...")
    mean = torch.zeros(3)
    std = torch.zeros(3)
    num_batches = len(dataloader)
    
    # First pass: compute mean
    for batch_idx, batch in enumerate(dataloader):
        # batch shape: (B, 3, H, W)
        for i in range(3):
            mean[i] += batch[:, i, :, :].mean()
        
        if (batch_idx + 1) % max(1, num_batches // 4) == 0:
            logger.info(f"  Mean computation: {batch_idx + 1}/{num_batches} batches")
    
    mean.div_(num_batches)
    logger.info(f"Mean: {mean.tolist()}")
    
    # Second pass: compute std
    for batch_idx, batch in enumerate(dataloader):
        # batch shape: (B, 3, H, W)
        for i in range(3):
            std[i] += ((batch[:, i, :, :] - mean[i]).pow(2)).mean()
        
        if (batch_idx + 1) % max(1, num_batches // 4) == 0:
            logger.info(f"  Std computation: {batch_idx + 1}/{num_batches} batches")
    
    std = (std / num_batches).sqrt()
    logger.info(f"Std: {std.tolist()}")
    
    return {
        "mean": mean.tolist(),
        "std": std.tolist()
    }

def update_config_with_stats(config_path: str, stats: dict):
    """Update config.yaml with computed statistics."""
    setup_logging()
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Update data normalization section
    if "data" not in config:
        config["data"] = {}
    
    config["data"]["normalization"] = {
        "mean": stats["mean"],
        "std": stats["std"]
    }
    
    # Write back to config
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"Updated {config_path} with dataset-specific normalization")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compute dataset statistics for optimal normalization"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/train/images",
        help="Path to training images"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit number of images (default: all)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers"
    )
    
    args = parser.parse_args()
    
    # Check if image directory exists
    if not Path(args.image_dir).exists():
        setup_logging()
        logger.warning(f"Image directory not found: {args.image_dir}")
        logger.info("Using ImageNet defaults instead")
        print("\n⚠️  Image directory not found. Using ImageNet defaults:")
        print("  Mean: [0.485, 0.456, 0.406]")
        print("  Std:  [0.229, 0.224, 0.225]")
    else:
        # Compute statistics
        stats = compute_dataset_statistics(
            args.image_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_images=args.max_images
        )
        
        # Update config
        update_config_with_stats(args.config, stats)
        
        print("\n✅ Dataset statistics computed successfully!")
        print(f"\n📊 Results:")
        print(f"  Mean: {stats['mean']}")
        print(f"  Std:  {stats['std']}")
        print(f"\n✨ config/config.yaml has been updated with dataset-specific normalization")
