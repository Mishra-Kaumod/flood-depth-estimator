"""
Google Colab Notebook: Retrain Existing Model with Gemini Pro Labeling
========================================================================
Complete workflow to fine-tune the existing flood depth model using GitHub.

Steps:
1. Clone GitHub repo
2. Load existing model checkpoint
3. Accept user image uploads
4. Label with Gemini Pro AI
5. Fine-tune with GPU
6. Save and prepare for GitHub push
"""

# ============================================================================
# SETUP & IMPORTS
# ============================================================================

# !pip install -q torch torchvision efficientnet_pytorch google-genai pillow tqdm

import os
import sys
import json
import csv
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import tempfile
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import cv2

# Google Colab specific
try:
    from google.colab import files, drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    print("Not running in Google Colab")

# Gemini Pro
import google.genai as genai

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# STEP 1: GITHUB SETUP
# ============================================================================

def clone_github_repo(repo_url: str, repo_dir: str = "flood-depth-estimator") -> Path:
    """Clone GitHub repo with Git LFS support."""
    logger.info(f"Cloning {repo_url}...")
    
    os.system(f"git clone {repo_url} {repo_dir}")
    
    # Initialize Git LFS
    os.system(f"cd {repo_dir} && git lfs install")
    os.system(f"cd {repo_dir} && git lfs pull")
    
    repo_path = Path(repo_dir)
    logger.info(f"Repository cloned to {repo_path}")
    
    return repo_path


# ============================================================================
# STEP 2: MODEL ARCHITECTURE
# ============================================================================

class FloodDepthRegressor(nn.Module):
    """Existing model architecture (same as production)."""
    
    def __init__(self, max_depth: float = 100.0):
        super().__init__()
        self.max_depth = max_depth
        
        # EfficientNet-B0 backbone
        self.backbone = models.efficientnet_b0(pretrained=False)
        backbone_dim = 1280
        
        # Remove classification head
        self.backbone.classifier = nn.Identity()
        
        # Regression head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(backbone_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(128, 1),
        )
        self.output_fn = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        pooled = self.global_pool(features).view(features.size(0), -1)
        logits = self.head(pooled)
        output = self.output_fn(logits) * self.max_depth
        return output


# ============================================================================
# STEP 3: LOAD EXISTING MODEL FROM GITHUB
# ============================================================================

def load_existing_model(
    repo_path: Path,
    model_path: str = "models/best_flood_model_water_aware.pth",
    device: str = "cuda"
) -> Tuple[FloodDepthRegressor, Dict]:
    """Load existing model checkpoint from GitHub."""
    
    checkpoint_path = repo_path / model_path
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model not found at {checkpoint_path}")
    
    logger.info(f"Loading existing model from {checkpoint_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = FloodDepthRegressor(max_depth=100.0)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    logger.info("✓ Existing model loaded successfully")
    logger.info(f"  - Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"  - Val loss: {checkpoint.get('val_loss', 'unknown')}")
    logger.info(f"  - Val MAE: {checkpoint.get('val_mae', 'unknown')}cm")
    
    return model, checkpoint


# ============================================================================
# STEP 4: GEMINI PRO LABELING
# ============================================================================

def setup_gemini_pro(api_key: str):
    """Configure Gemini Pro API."""
    genai.configure(api_key=api_key)
    logger.info("✓ Gemini Pro configured")


def label_image_with_gemini(
    image_path: Path,
    model_name: str = "gemini-1.5-flash"
) -> float:
    """
    Use Gemini Pro vision to estimate flood depth from image.
    
    Returns depth in cm.
    """
    try:
        # Read image
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # Upload to Gemini
        file = genai.upload_file(image_data, mime_type="image/jpeg")
        
        # Create prompt
        prompt = """Analyze this image for flood depth estimation. 
        
        Look for:
        1. Water level relative to known objects (vehicles, people, buildings, poles)
        2. Visibility of reference objects (tires, bumpers, doors, wheels)
        3. Water color and clarity (indicates depth)
        4. Surrounding context (street level, buildings, vegetation)
        
        Based on visual cues, estimate the FLOOD DEPTH IN CENTIMETERS.
        
        Respond with ONLY a single number in centimeters (e.g., "45" for 45cm).
        If no flood is visible, respond with "0".
        If depth cannot be estimated, respond with "30" (default estimate).
        """
        
        # Call Gemini
        model = genai.GenerativeModel(model_name)
        response = model.generate_content([prompt, file])
        
        # Parse response
        depth_str = response.text.strip().split('\n')[0]
        depth_cm = float(''.join(filter(str.isdigit, depth_str[:5])) or '30')
        depth_cm = max(0, min(200, depth_cm))  # Clamp to [0, 200]
        
        logger.info(f"  {image_path.name}: {depth_cm:.1f}cm")
        return depth_cm
        
    except Exception as e:
        logger.warning(f"  {image_path.name}: Gemini failed ({e}), using default 30cm")
        return 30.0


def label_images_batch(
    image_dir: Path,
    output_csv: Path,
    gemini_api_key: str
) -> List[Dict]:
    """Label all images using Gemini Pro, save to CSV."""
    
    setup_gemini_pro(gemini_api_key)
    
    logger.info(f"Labeling images from {image_dir}...")
    
    labels = []
    image_files = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    
    for img_path in image_files:
        depth = label_image_with_gemini(img_path)
        labels.append({
            "filename": img_path.name,
            "depth_cm": depth,
            "timestamp": datetime.now().isoformat(),
        })
    
    # Save to CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "depth_cm", "timestamp"])
        writer.writeheader()
        writer.writerows(labels)
    
    logger.info(f"✓ Labeled {len(labels)} images, saved to {output_csv}")
    
    return labels


# ============================================================================
# STEP 5: DATASET
# ============================================================================

class FloodDepthDataset(Dataset):
    """Dataset for training."""
    
    def __init__(
        self,
        image_dir: Path,
        labels_csv: Path,
        image_size: Tuple[int, int] = (512, 512),
        transform=None
    ):
        self.image_dir = image_dir
        self.image_size = image_size
        self.transform = transform
        
        # Load labels
        self.data = []
        with open(labels_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = image_dir / row["filename"]
                if img_path.exists():
                    self.data.append({
                        "path": img_path,
                        "depth": float(row["depth_cm"]),
                    })
        
        logger.info(f"Dataset loaded: {len(self.data)} samples")
        
        # Stats
        depths = [d["depth"] for d in self.data]
        logger.info(f"  Depth range: {min(depths):.1f} - {max(depths):.1f}cm")
        logger.info(f"  Mean: {np.mean(depths):.1f}cm, Std: {np.std(depths):.1f}cm")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image = Image.open(item["path"]).convert("RGB")
        image = image.resize(self.image_size, Image.BILINEAR)
        
        # Transform
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0)
            image = image.permute(2, 0, 1)
        
        return {
            "image": image,
            "depth": torch.tensor(item["depth"], dtype=torch.float32),
        }


# ============================================================================
# STEP 6: TRAINING
# ============================================================================

class HuberLoss(nn.Module):
    """Huber loss - robust to outliers."""
    def __init__(self, delta: float = 5.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred, target):
        residual = torch.abs(pred - target)
        condition = residual < self.delta
        small_residual = 0.5 * residual ** 2
        large_residual = self.delta * (residual - 0.5 * self.delta)
        return torch.where(condition, small_residual, large_residual).mean()


def train_model(
    model: FloodDepthRegressor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 15,
    learning_rate: float = 1e-4,
    freeze_backbone: bool = True,
    device: str = "cuda"
) -> Dict:
    """Fine-tune existing model on new data."""
    
    # Freeze backbone for transfer learning
    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
        logger.info("✓ Backbone frozen (transfer learning)")
    else:
        for param in model.backbone.parameters():
            param.requires_grad = True
        logger.info("✓ Backbone unfrozen (fine-tuning)")
    
    # Optimizer (only train head if frozen)
    if freeze_backbone:
        params = list(model.head.parameters())
    else:
        params = model.parameters()
    
    optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)
    criterion = HuberLoss(delta=5.0)
    
    # Training loop
    best_val_loss = float("inf")
    metrics = {
        "train_loss": [],
        "train_mae": [],
        "val_loss": [],
        "val_mae": [],
    }
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        
        for batch in train_loader:
            images = batch["image"].to(device)
            targets = batch["depth"].to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_mae += torch.abs(predictions - targets).mean().item()
        
        train_loss /= len(train_loader)
        train_mae /= len(train_loader)
        metrics["train_loss"].append(train_loss)
        metrics["train_mae"].append(train_mae)
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                targets = batch["depth"].to(device).unsqueeze(1)
                
                predictions = model(images)
                loss = criterion(predictions, targets)
                
                val_loss += loss.item()
                val_mae += torch.abs(predictions - targets).mean().item()
        
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        metrics["val_loss"].append(val_loss)
        metrics["val_mae"].append(val_mae)
        
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.6f}, MAE: {train_mae:.2f}cm | "
            f"Val Loss: {val_loss:.6f}, MAE: {val_mae:.2f}cm"
        )
        
        scheduler.step(val_loss)
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_mae": val_mae,
                "metrics": metrics,
                "training_config": {
                    "learning_rate": learning_rate,
                    "freeze_backbone": freeze_backbone,
                    "num_epochs": num_epochs,
                }
            }
            best_checkpoint = checkpoint
    
    logger.info(f"✓ Training completed. Best val_loss: {best_val_loss:.6f}")
    
    return best_checkpoint, metrics


# ============================================================================
# STEP 7: UPLOAD & PREPARE FOR GITHUB
# ============================================================================

def prepare_for_github_push(
    repo_path: Path,
    trained_model_checkpoint: Dict,
    training_data_dir: Path,
    labels_csv: Path,
    version: str = "v2"
) -> Dict:
    """
    Organize trained model and dataset for GitHub push.
    Returns instructions for user.
    """
    
    logger.info("Preparing for GitHub push...")
    
    # Save model checkpoint
    model_output = repo_path / "models" / f"best_flood_model_{version}.pth"
    model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trained_model_checkpoint, model_output)
    logger.info(f"✓ Model saved to {model_output}")
    
    # Archive dataset
    dataset_output = repo_path / "datasets" / f"training_{version}"
    dataset_output.mkdir(parents=True, exist_ok=True)
    
    # Copy images
    for img_file in training_data_dir.glob("*"):
        if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            shutil.copy2(img_file, dataset_output / img_file.name)
    
    # Copy labels
    shutil.copy2(labels_csv, dataset_output / "labels.csv")
    
    logger.info(f"✓ Dataset saved to {dataset_output}")
    
    # Create metadata
    metadata = {
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "epoch": trained_model_checkpoint.get("epoch"),
        "val_loss": trained_model_checkpoint.get("val_loss"),
        "val_mae": trained_model_checkpoint.get("val_mae"),
        "images_count": len(list(dataset_output.glob("*.jpg"))) + len(list(dataset_output.glob("*.png"))),
    }
    
    with open(repo_path / "model_registry.json", "a") as f:
        json.dump({version: metadata}, f)
    
    logger.info(f"✓ Metadata saved")
    
    # Return instructions
    instructions = f"""
    ╔════════════════════════════════════════════════════════════════╗
    ║  MODEL TRAINED & READY FOR GITHUB                             ║
    ╚════════════════════════════════════════════════════════════════╝
    
    Download these folders locally:
    1. models/best_flood_model_{version}.pth
    2. datasets/training_{version}/
    
    Then in your local repo:
    
    $ cd flood-depth-estimator
    $ git pull origin main
    $ git add models/best_flood_model_{version}.pth
    $ git add datasets/training_{version}/
    $ git add model_registry.json
    $ git commit -m "Trained model {version}: {len(list(dataset_output.glob('*.*')))} images, val_loss={trained_model_checkpoint.get('val_loss'):.6f}"
    $ git push origin main
    
    Then on your server:
    $ git pull origin main
    $ python app.py
    $ # Update config.yaml inference.model_path to models/best_flood_model_{version}.pth
    
    Server will auto-load the new model!
    """
    
    print(instructions)
    return metadata


# ============================================================================
# MAIN COLAB WORKFLOW
# ============================================================================

def main():
    """Complete Colab training workflow."""
    
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║  FLOOD DEPTH ESTIMATOR - RETRAIN WITH GEMINI PRO             ║
    ║  Google Colab + GitHub + Transfer Learning                   ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    # ========== STEP 1: GitHub Setup ==========
    print("\n[1/7] Cloning GitHub Repository...")
    REPO_URL = "https://github.com/Mishra-Kaumod/flood-depth-estimator.git"
    repo_path = clone_github_repo(REPO_URL)
    print("✓ GitHub repo cloned\n")
    
    # ========== STEP 2: Load Existing Model ==========
    print("[2/7] Loading Existing Model from GitHub...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = load_existing_model(
        repo_path,
        model_path="models/best_flood_model_water_aware.pth",
        device=device
    )
    print("✓ Model loaded\n")
    
    # ========== STEP 3: Upload Images ==========
    print("[3/7] Upload Your Flood Images...")
    if IN_COLAB:
        print("Click 'Choose Files' button to upload your images (50-200 recommended)")
        uploaded = files.upload()
        upload_dir = Path("uploaded_images")
        upload_dir.mkdir(exist_ok=True)
        
        for filename, data in uploaded.items():
            with open(upload_dir / filename, "wb") as f:
                f.write(data)
            print(f"  ✓ {filename}")
    else:
        upload_dir = Path("sample_images")  # For local testing
    
    print(f"✓ {len(list(upload_dir.glob('*')))} images uploaded\n")
    
    # ========== STEP 4: Gemini Pro Labeling ==========
    print("[4/7] Labeling Images with Gemini Pro...")
    GEMINI_API_KEY = input("Enter your Gemini Pro API key: ").strip()
    
    labels_csv = Path("gemini_labels.csv")
    labels = label_images_batch(upload_dir, labels_csv, GEMINI_API_KEY)
    print(f"✓ Labeled {len(labels)} images\n")
    
    # ========== STEP 5: Create Dataset & Split ==========
    print("[5/7] Creating Training/Validation Split...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = FloodDepthDataset(upload_dir, labels_csv, transform=transform)
    
    # 80/20 train/val split
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    
    train_data, val_data = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )
    
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    
    print(f"✓ Train: {n_train}, Val: {n_val}\n")
    
    # ========== STEP 6: Fine-tune Model ==========
    print("[6/7] Fine-tuning Existing Model with GPU...")
    best_checkpoint, metrics = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=15,
        learning_rate=1e-4,
        freeze_backbone=True,  # Transfer learning
        device=device
    )
    print("✓ Training completed\n")
    
    # ========== STEP 7: Download & GitHub Instructions ==========
    print("[7/7] Preparing for GitHub Push...")
    metadata = prepare_for_github_push(
        repo_path,
        best_checkpoint,
        upload_dir,
        labels_csv,
        version="v2"
    )
    
    # Download trained model
    if IN_COLAB:
        print("\nDownloading trained model...")
        files.download(str(repo_path / "models" / "best_flood_model_v2.pth"))
    
    print("\n✓ All done! Follow the instructions above to push to GitHub.\n")


if __name__ == "__main__":
    main()
