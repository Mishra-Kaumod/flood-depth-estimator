"""
DATASET LAYER - Amazon S3 Data Processing & PyTorch Custom Dataset
Modular data ingestion with support for local filesystem and AWS S3.
"""
import os
import yaml
import csv
import hashlib
import shutil
from pathlib import Path
from typing import Optional, Tuple, List
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image
import logging

try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    boto3 = None
    ClientError = Exception
    BOTO3_AVAILABLE = False

logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION LOADER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load validated configuration with environment overlays."""
    try:
        from src.settings import load_settings_dict
        return load_settings_dict(config_path=config_path)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# S3 STORAGE HANDLER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class S3DataHandler:
    """Interface for reading images from AWS S3."""
    
    def __init__(self, bucket: str, region: str = "ap-south-1"):
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for use_s3=True. Install boto3 and botocore "
                "or run with local filesystem data."
            )
        self.bucket = bucket
        self.region = region
        self.s3_client = boto3.client("s3", region_name=region)
    
    def list_objects(self, prefix: str) -> List[str]:
        """List all objects under S3 prefix."""
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket, Prefix=prefix)
            
            objects = []
            for page in pages:
                if "Contents" in page:
                    objects.extend([obj["Key"] for obj in page["Contents"]])
            return objects
        except ClientError as e:
            logger.error(f"S3 list error: {e}")
            return []
    
    def get_image(self, s3_key: str) -> Optional[Image.Image]:
        """Fetch image from S3 and return PIL Image object."""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
            img = Image.open(response["Body"])
            return img.convert("RGB")
        except ClientError as e:
            logger.error(f"Failed to fetch {s3_key}: {e}")
            return None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PYTORCH CUSTOM DATASET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FloodDataset(Dataset):
    """
    Production-grade PyTorch Dataset for flood depth estimation.
    
    Supports:
    - Local filesystem loading
    - AWS S3 dynamic loading
    - Configurable augmentation (training mode only)
    - Normalized image preprocessing
    """
    
    def __init__(
        self,
        config: dict,
        dataset_type: str = "train",
        use_s3: bool = False,
        s3_bucket: Optional[str] = None,
        s3_region: str = "ap-south-1",
    ):
        """
        Initialize FloodDataset.
        
        Args:
            config: YAML config dict
            dataset_type: "train", "val", or "test"
            use_s3: Use AWS S3 or local filesystem
            s3_bucket: S3 bucket name (if use_s3=True)
            s3_region: AWS region
        """
        self.config = config
        self.dataset_type = dataset_type
        self.use_s3 = use_s3
        
        # Get configuration
        data_cfg = config.get("data", {})
        train_cfg = config.get("training", {})
        self.data_cfg = data_cfg
        self.strict_loading = bool(data_cfg.get("strict_loading", True))
        self.label_source = str(data_cfg.get("label_source", "manifest")).lower()
        self.manifest_file = str(data_cfg.get("manifest_file", "labels.csv"))
        self.checksum_validation = bool(data_cfg.get("checksum_validation", True))
        self.quarantine_dir = Path(data_cfg.get("quarantine_dir", "data/quarantine"))
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        
        self.image_size = tuple(train_cfg.get("image_size", [224, 224]))
        
        # Try to load dataset-specific normalization first (from compute_stats.py)
        # Fall back to ImageNet defaults if not found
        self.normalization = data_cfg.get("normalization") or train_cfg.get("normalization", {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        })
        
        # Build transform pipeline
        self.transform = self._build_transforms(train_cfg)
        
        # Load image paths
        if use_s3:
            self.s3_handler = S3DataHandler(s3_bucket, s3_region)
            prefix = data_cfg.get(f"s3_{dataset_type}_prefix", f"{dataset_type}/")
            self.image_paths = self._load_s3_images(prefix)
        else:
            dataset_dir = data_cfg.get(f"{dataset_type}_dir", f"flood_dataset/{dataset_type}")
            self.image_paths = self._load_local_images(dataset_dir)
        
        logger.info(f"Loaded {len(self.image_paths)} {dataset_type} images")
    
    def _build_transforms(self, train_cfg: dict) -> transforms.Compose:
        """Build augmentation pipeline based on dataset type."""
        base_transforms = [
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.normalization["mean"],
                std=self.normalization["std"]
            )
        ]
        
        # Add augmentation ONLY for training
        if self.dataset_type == "train":
            aug_cfg = train_cfg.get("augmentation", {})
            augmentations = [
                # Geometric augmentations
                transforms.RandomHorizontalFlip(aug_cfg.get("horizontal_flip_prob", 0.5)),
                transforms.RandomRotation(aug_cfg.get("rotation_degrees", 15)),
                transforms.RandomPerspective(
                    distortion_scale=aug_cfg.get("perspective_distortion", 0.3),
                    p=0.5
                ),
                
                # Color augmentations
                transforms.ColorJitter(
                    brightness=aug_cfg.get("color_jitter", {}).get("brightness", 0.2),
                    contrast=aug_cfg.get("color_jitter", {}).get("contrast", 0.2),
                    saturation=aug_cfg.get("color_jitter", {}).get("saturation", 0.2),
                    hue=aug_cfg.get("color_jitter", {}).get("hue", 0.1)
                ),
                
                # Blur augmentation
                transforms.GaussianBlur(
                    kernel_size=aug_cfg.get("gaussian_blur_kernel", 3),
                    sigma=aug_cfg.get("gaussian_blur_sigma", (0.1, 2.0))
                ),
                
                # Noise-like augmentation: random erasing
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1)
                )
            ]
            # Insert augmentations before tensor conversion
            return transforms.Compose(
                augmentations +
                [
                    transforms.Resize(self.image_size),
                    transforms.ToTensor(),
                    # RandomErasing after tensor conversion
                    transforms.RandomErasing(
                        p=0.2,
                        scale=(0.02, 0.1),
                        ratio=(0.3, 3.0)
                    ),
                    transforms.Normalize(
                        mean=self.normalization["mean"],
                        std=self.normalization["std"]
                    )
                ]
            )
        
        return transforms.Compose(base_transforms)

    def _file_sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _quarantine_local_file(self, file_path: Path, reason: str) -> None:
        try:
            target = self.quarantine_dir / f"{reason}_{file_path.name}"
            if target.exists():
                target = self.quarantine_dir / f"{reason}_{file_path.stem}_{os.getpid()}{file_path.suffix}"
            shutil.move(str(file_path), str(target))
            logger.warning(f"Quarantined file: {file_path} -> {target}")
        except Exception as e:
            logger.error(f"Failed to quarantine {file_path}: {e}")

    def _load_manifest(self, dataset_path: Path) -> dict:
        manifest_path = dataset_path / self.manifest_file
        if not manifest_path.exists():
            if self.label_source == "manifest":
                raise FileNotFoundError(
                    f"Manifest required but not found: {manifest_path}. "
                    f"Set data.label_source=hybrid to allow filename fallback."
                )
            return {}

        labels = {}
        with open(manifest_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"filename", "depth_cm"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(
                    f"Manifest {manifest_path} must include columns: filename,depth_cm "
                    f"(optional: sha256)"
                )
            for row in reader:
                file_name = (row.get("filename") or "").strip()
                if not file_name:
                    continue
                labels[file_name] = {
                    "depth_cm": int(float(row["depth_cm"])),
                    "sha256": (row.get("sha256") or "").strip().lower() or None,
                }
        logger.info(f"Loaded {len(labels)} labels from manifest {manifest_path}")
        return labels

    def _validate_image_integrity(self, path: Path, expected_sha256: Optional[str]) -> bool:
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as e:
            logger.error(f"Corrupt image file: {path} ({e})")
            self._quarantine_local_file(path, "corrupt")
            return False

        if self.checksum_validation and expected_sha256:
            try:
                actual = self._file_sha256(path)
                if actual.lower() != expected_sha256.lower():
                    logger.error(f"Checksum mismatch for {path}")
                    self._quarantine_local_file(path, "checksum_mismatch")
                    return False
            except Exception as e:
                logger.error(f"Checksum verification failed for {path}: {e}")
                self._quarantine_local_file(path, "checksum_error")
                return False
        return True
    
    def _load_local_images(self, dataset_dir: str) -> List[Tuple[str, int]]:
        """Load local dataset with manifest labels and corruption quarantine."""
        supported_formats = self.config.get("data", {}).get("supported_formats", [".jpg", ".png"])
        image_paths: List[Tuple[str, int]] = []

        dataset_path = Path(dataset_dir)
        candidate_dirs = [dataset_path]
        if dataset_path.parts and dataset_path.parts[0] == "flood_dataset":
            split_name = dataset_path.name
            candidate_dirs = [Path("data") / split_name / "images", Path("data") / split_name, dataset_path]

        for candidate in candidate_dirs:
            if candidate.exists():
                dataset_path = candidate
                break
        else:
            message = f"Dataset directory not found: {dataset_dir}"
            logger.error(message)
            if self.strict_loading:
                raise FileNotFoundError(message)
            return []

        manifest_labels = self._load_manifest(dataset_path)
        zero_label_count = 0

        for img_file in sorted(dataset_path.rglob("*")):
            if not img_file.is_file() or img_file.suffix.lower() not in supported_formats:
                continue
            if img_file.name == self.manifest_file:
                continue

            label_entry = manifest_labels.get(img_file.name)
            depth_cm: Optional[int] = None
            expected_sha: Optional[str] = None
            if label_entry:
                depth_cm = label_entry["depth_cm"]
                expected_sha = label_entry.get("sha256")
            elif self.label_source in ("hybrid", "filename"):
                depth_cm = self._extract_depth_from_filename(img_file.name)
            else:
                self._quarantine_local_file(img_file, "missing_manifest_label")
                continue

            if depth_cm is None:
                self._quarantine_local_file(img_file, "missing_depth")
                continue

            if not self._validate_image_integrity(img_file, expected_sha):
                continue

            if depth_cm == 0:
                zero_label_count += 1
            image_paths.append((str(img_file), depth_cm))

        total = len(image_paths)
        if total == 0 and self.strict_loading:
            raise RuntimeError(f"No valid images loaded from {dataset_path}.")

        if total > 0:
            pct_zero = zero_label_count / total * 100
            if pct_zero > 80:
                logger.error(
                    "=" * 70 + "\n"
                    f"  ⚠️  LABEL COLLAPSE WARNING: {zero_label_count}/{total} images "
                    f"({pct_zero:.0f}%) have depth_cm = 0!\n"
                    "  Training on this data will produce a model that always predicts 0.\n"
                    "  Fix: use manifest labels with calibrated depth_cm values.\n"
                    + "=" * 70
                )

        return image_paths
    
    def _load_s3_images(self, prefix: str) -> List[Tuple[str, int]]:
        """Load image paths from AWS S3."""
        objects = self.s3_handler.list_objects(prefix)
        supported_formats = self.config.get("data", {}).get("supported_formats", [".jpg", ".png"])
        
        image_paths = []
        for obj_key in objects:
            if any(obj_key.lower().endswith(fmt) for fmt in supported_formats):
                depth_cm = self._extract_depth_from_filename(obj_key)
                image_paths.append((obj_key, depth_cm))
        
        return image_paths
    
    def _extract_depth_from_filename(self, filename: str) -> int:
        """Extract depth value from filename. Format: 'image_depth25cm.jpg' -> 25."""
        import re
        match = re.search(r'depth(\d+)cm', filename.lower())
        return int(match.group(1)) if match else 0
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch image and depth label, apply transforms."""
        image_path, depth_cm = self.image_paths[idx]
        
        try:
            # Load image
            if self.use_s3:
                img = self.s3_handler.get_image(image_path)
            else:
                img = Image.open(image_path).convert("RGB")
            
            if img is None:
                msg = f"Failed to load image: {image_path}"
                logger.error(msg)
                if self.strict_loading:
                    raise RuntimeError(msg)
                return torch.zeros((3, *self.image_size)), torch.tensor(0.0)
            
            # Apply transforms
            img_tensor = self.transform(img)
            
            # Convert depth to float tensor (normalized 0-1 where 100cm = 1.0)
            depth_tensor = torch.tensor(min(depth_cm / 100.0, 1.0), dtype=torch.float32)
            
            return img_tensor, depth_tensor
        
        except Exception as e:
            logger.error(f"Error loading {image_path}: {e}")
            if not self.use_s3 and isinstance(image_path, str):
                p = Path(image_path)
                if p.exists():
                    self._quarantine_local_file(p, "runtime_read_error")
            if self.strict_loading:
                raise
            return torch.zeros((3, *self.image_size)), torch.tensor(0.0)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATALOADER FACTORY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_dataloaders(
    config: dict,
    use_s3: bool = False,
    s3_bucket: Optional[str] = None,
    s3_region: str = "ap-south-1",
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Factory function to create train and validation dataloaders."""
    
    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 32)
    num_workers = train_cfg.get("num_workers", 4)
    pin_memory = train_cfg.get("pin_memory", True)
    
    # Create datasets
    train_dataset = FloodDataset(
        config, dataset_type="train", use_s3=use_s3, s3_bucket=s3_bucket, s3_region=s3_region
    )
    val_dataset = FloodDataset(
        config, dataset_type="val", use_s3=use_s3, s3_bucket=s3_bucket, s3_region=s3_region
    )

    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    train_loader, val_loader = create_dataloaders(cfg, use_s3=False)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Inspect one batch
    for images, depths in train_loader:
        print(f"Image batch shape: {images.shape}")
        print(f"Depth batch shape: {depths.shape}")
        print(f"Depth range: [{depths.min():.3f}, {depths.max():.3f}]")
        break
