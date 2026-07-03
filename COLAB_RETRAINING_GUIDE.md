# Google Colab: Retrain Flood Model with Gemini Pro Labeling

**Complete Step-by-Step Guide** for retraining your existing model using GitHub + Gemini Pro + GPU.

---

## **Quick Start (Copy-Paste Into Colab)**

Open a new **Google Colab** notebook at https://colab.research.google.com and follow these cells:

---

## **CELL 1: Install Dependencies**

```python
# Install required packages
!pip install -q torch torchvision efficientnet-pytorch google-genai pillow tqdm

# Check GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

---

## **CELL 2: Clone GitHub Repository**

```python
import os
from pathlib import Path

# Clone repo with Git LFS
REPO_URL = "https://github.com/Mishra-Kaumod/flood-depth-estimator.git"
REPO_DIR = "flood-depth-estimator"

os.system(f"git clone {REPO_URL} {REPO_DIR}")
os.chdir(REPO_DIR)

# Initialize Git LFS and pull model
os.system("git lfs install")
os.system("git lfs pull")

print("✓ Repository cloned and Git LFS initialized")
print(f"✓ Files ready in {REPO_DIR}/")
```

---

## **CELL 3: Load Existing Model from GitHub**

```python
import torch
import torch.nn as nn
import torchvision.models as models
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FloodDepthRegressor(nn.Module):
    """Existing model architecture."""
    
    def __init__(self, max_depth: float = 100.0):
        super().__init__()
        self.max_depth = max_depth
        
        # EfficientNet-B0 backbone
        self.backbone = models.efficientnet_b0(pretrained=False)
        backbone_dim = 1280
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


# Load checkpoint
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "models/best_flood_model_water_aware.pth"

checkpoint = torch.load(checkpoint_path, map_location=device)
model = FloodDepthRegressor(max_depth=100.0)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

logger.info("✓ Existing model loaded successfully")
logger.info(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
logger.info(f"  Val loss: {checkpoint.get('val_loss', 'unknown')}")
logger.info(f"  Val MAE: {checkpoint.get('val_mae', 'unknown')}cm")
```

---

## **CELL 4: Upload Your Flood Images**

```python
from google.colab import files
from pathlib import Path

# Create upload directory
upload_dir = Path("uploaded_images")
upload_dir.mkdir(exist_ok=True)

print("Click 'Choose Files' button to upload your images")
print("Recommended: 50-200 flood images from different depths")
print("Supported: .jpg, .jpeg, .png")

uploaded = files.upload()

for filename, data in uploaded.items():
    with open(upload_dir / filename, "wb") as f:
        f.write(data)
    print(f"✓ {filename}")

print(f"\n✓ {len(list(upload_dir.glob('*')))} images uploaded")
```

---

## **CELL 5: Configure Gemini Pro API**

```python
import google.genai as genai

# Get API key
print("Go to: https://aistudio.google.com/app/apikey")
print("Copy your API key and paste below")

GEMINI_API_KEY = input("Enter Gemini Pro API key: ").strip()
genai.configure(api_key=GEMINI_API_KEY)

print("✓ Gemini Pro configured")
```

---

## **CELL 6: Label Images with Gemini Pro**

```python
import csv
from datetime import datetime
from PIL import Image

def label_image_with_gemini(image_path: Path) -> float:
    """Use Gemini Pro vision to estimate flood depth."""
    
    try:
        # Read image
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # Upload to Gemini
        file = genai.upload_file(image_data, mime_type="image/jpeg")
        
        # Prompt
        prompt = """Analyze this image for flood depth estimation.
        
Look for:
1. Water level relative to known objects (vehicles, people, buildings)
2. Vehicle submersion (tires, bumpers, doors, hood)
3. Person submersion (ankle, knee, waist, chest)
4. Water color and clarity

Estimate the FLOOD DEPTH IN CENTIMETERS.

Respond with ONLY a single number in centimeters (e.g., "45" for 45cm).
If no flood: respond "0"
If uncertain: respond "30" (default)
"""
        
        # Call Gemini
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([prompt, file])
        
        # Parse response
        depth_str = response.text.strip().split('\n')[0]
        depth_cm = float(''.join(filter(str.isdigit, depth_str[:5])) or '30')
        depth_cm = max(0, min(200, depth_cm))  # Clamp to [0, 200]
        
        return depth_cm
        
    except Exception as e:
        print(f"  Error: {e}, using default 30cm")
        return 30.0


# Label all images
labels = []
image_files = sorted(upload_dir.glob("*.jpg")) + sorted(upload_dir.glob("*.png"))

print(f"Labeling {len(image_files)} images with Gemini Pro...")
print("This may take a few minutes...\n")

for img_path in image_files:
    depth = label_image_with_gemini(img_path)
    labels.append({
        "filename": img_path.name,
        "depth_cm": depth,
        "timestamp": datetime.now().isoformat(),
    })
    print(f"✓ {img_path.name}: {depth:.1f}cm")

# Save to CSV
labels_csv = Path("gemini_labels.csv")
with open(labels_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "depth_cm", "timestamp"])
    writer.writeheader()
    writer.writerows(labels)

print(f"\n✓ Saved {len(labels)} labels to {labels_csv}")
```

---

## **CELL 7: Create Dataset & Split**

```python
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class FloodDepthDataset(Dataset):
    """Dataset for training."""
    
    def __init__(self, image_dir: Path, labels_csv: Path, image_size=(512, 512)):
        self.image_dir = image_dir
        self.image_size = image_size
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
        
        depths = [d["depth"] for d in self.data]
        print(f"Dataset loaded: {len(self.data)} samples")
        print(f"  Depth range: {min(depths):.1f} - {max(depths):.1f}cm")
        print(f"  Mean: {np.mean(depths):.1f}cm, Std: {np.std(depths):.1f}cm")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["path"]).convert("RGB")
        image = image.resize(self.image_size, Image.BILINEAR)
        
        # Normalize
        image = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0)
        image = image.permute(2, 0, 1)
        
        return {
            "image": image,
            "depth": torch.tensor(item["depth"], dtype=torch.float32),
        }


# Create dataset
dataset = FloodDepthDataset(upload_dir, labels_csv)

# 80/20 train/val split
n_train = int(0.8 * len(dataset))
n_val = len(dataset) - n_train

train_data, val_data = torch.utils.data.random_split(
    dataset, [n_train, n_val]
)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=0)

print(f"\n✓ Train split: {n_train} samples")
print(f"✓ Val split: {n_val} samples")
```

---

## **CELL 8: Fine-tune with GPU**

```python
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

class HuberLoss(nn.Module):
    """Robust to outliers."""
    def __init__(self, delta=5.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred, target):
        residual = torch.abs(pred - target)
        condition = residual < self.delta
        small_residual = 0.5 * residual ** 2
        large_residual = self.delta * (residual - 0.5 * self.delta)
        return torch.where(condition, small_residual, large_residual).mean()


# Freeze backbone for transfer learning
for param in model.backbone.parameters():
    param.requires_grad = False

print("✓ Backbone frozen (transfer learning)")

# Optimizer - only train regression head
optimizer = optim.AdamW(
    model.head.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=False)
criterion = HuberLoss(delta=5.0)

# Training
num_epochs = 15
best_val_loss = float("inf")
metrics = {"train_loss": [], "train_mae": [], "val_loss": [], "val_mae": []}

print(f"\nStarting fine-tuning for {num_epochs} epochs...\n")

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
    
    print(f"Epoch {epoch+1:2d}/{num_epochs} | Train Loss: {train_loss:.6f} MAE: {train_mae:.2f}cm | Val Loss: {val_loss:.6f} MAE: {val_mae:.2f}cm")
    
    scheduler.step(val_loss)
    
    # Save best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_mae": val_mae,
            "metrics": metrics,
        }

print(f"\n✓ Fine-tuning completed! Best val_loss: {best_val_loss:.6f}")
```

---

## **CELL 9: Save Model & Prepare for GitHub**

```python
from datetime import datetime

# Save model
output_dir = Path("models")
output_dir.mkdir(exist_ok=True)

model_path = output_dir / "best_flood_model_v2.pth"
torch.save(best_checkpoint, model_path)

print(f"✓ Model saved to {model_path}")
print(f"  Size: {model_path.stat().st_size / 1e6:.1f}MB")

# Create metadata
metadata = {
    "version": "v2",
    "timestamp": datetime.now().isoformat(),
    "epoch": best_checkpoint["epoch"],
    "val_loss": best_checkpoint["val_loss"],
    "val_mae": best_checkpoint["val_mae"],
    "images_count": len(dataset),
}

print("\nMetadata:")
for key, value in metadata.items():
    print(f"  {key}: {value}")
```

---

## **CELL 10: Download & GitHub Instructions**

```python
# Download model
from google.colab import files

print("\nDownloading trained model...")
files.download(str(model_path))

# Print instructions
instructions = """
╔════════════════════════════════════════════════════════════════╗
║  MODEL TRAINED & READY FOR GITHUB                             ║
╚════════════════════════════════════════════════════════════════╝

NEXT STEPS:

1. Download the model file:
   - best_flood_model_v2.pth (already downloaded above)

2. On your LOCAL MACHINE (your Windows PC):
   
   $ cd flood-depth-estimator
   $ git pull origin main
   
   # Copy the downloaded model
   $ cp ~/Downloads/best_flood_model_v2.pth models/

   # Commit and push
   $ git add models/best_flood_model_v2.pth
   $ git commit -m "Trained model v2: {len(dataset)} images, val_loss={best_checkpoint['val_loss']:.6f}, val_mae={best_checkpoint['val_mae']:.2f}cm"
   $ git push origin main

3. Update config.yaml:
   
   inference:
     model_path: models/best_flood_model_v2.pth
   
   training:
     best_model_path: models/best_flood_model_v2.pth

4. Restart the server:
   
   $ python app.py
   
   # Visit http://localhost:5000
   # Server will auto-load the new model!

5. Test the new model:
   - Upload test images
   - Check predictions
   - Verify depth values are no longer stuck at 0cm
"""

print(instructions)
```

---

## **How It Works (Overview)**

```
1. CLONE GITHUB
   ↓
2. LOAD EXISTING MODEL (transfer learning, not scratch)
   ↓
3. UPLOAD IMAGES
   ↓
4. GEMINI PRO LABELS IMAGES automatically (no manual work!)
   ↓
5. CREATE TRAIN/VAL SPLIT (80/20)
   ↓
6. FINE-TUNE ONLY REGRESSION HEAD (backbone frozen) - FAST!
   ↓
7. SAVE & DOWNLOAD
   ↓
8. GIT PUSH TO MAIN
   ↓
9. SERVER AUTO-LOADS NEW MODEL
```

---

## **Expected Results**

After training:
- ✅ `val_loss` should drop (< 0.05 if labels are good)
- ✅ `val_mae` should be 5-15cm (depends on image quality)
- ✅ Model should **NOT** predict 0cm for everything
- ✅ Reference CV fallback should deactivate once model is healthy

---

## **Troubleshooting**

| Problem | Solution |
|---------|----------|
| **Gemini API fails** | Check API key at https://aistudio.google.com/app/apikey |
| **Out of memory** | Reduce batch_size to 8, reduce image_size to 384 |
| **Training too slow** | Ensure GPU is enabled (Runtime → Change Runtime Type → GPU) |
| **Model still predicts 0** | Images/labels may be poor; collect more diverse flood photos |
| **Git LFS error** | Run `git lfs install` in Colab before cloning |

---

## **Questions?**

Check the repo docs:
- `MODEL_DATASET_MANAGEMENT.md` - Asset versioning
- `AFTER_TRAINING_CHECKLIST.md` - Deployment steps
- `enterprise_flood_model.py` - Complete model code

Happy training! 🚀
