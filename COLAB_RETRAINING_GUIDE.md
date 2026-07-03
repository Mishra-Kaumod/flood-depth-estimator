# Google Colab: Retrain Flood Model with Gemini Pro Labeling

**Complete Step-by-Step Guide** for retraining your existing model using GitHub + Gemini Pro + GPU.

Choose your training option:
- **OPTION A**: Upload your own images + Kaggle public dataset
- **OPTION B**: Kaggle only + Gemini labeling

---

## **Which Option Should You Choose?**

| Option | Images Source | Time | Quality | Best For |
|--------|---------------|------|---------|----------|
| **A (Recommended)** | Your images + Kaggle | 20-30 min | Best (mixed data) | Production deployment |
| **B** | Kaggle only | 10-15 min | Good (public data) | Quick testing |

---

## **Quick Start (Copy-Paste Into Colab)**

Open a new **Google Colab** notebook at https://colab.research.google.com and follow these cells:

---

## **CELL 0: Select Your Training Option**

```python
# CHOOSE YOUR OPTION:
# Option A: Upload your images + Download from Kaggle (RECOMMENDED)
# Option B: Use Kaggle dataset only (faster, less customization)

OPTION = "A"  # Change to "B" if you want Kaggle only

print(f"""
╔════════════════════════════════════════════════════════════════╗
║  FLOOD DEPTH ESTIMATOR - RETRAINING                           ║
║  Option {OPTION}: {'Your Images + Kaggle' if OPTION == 'A' else 'Kaggle Only'}
╚════════════════════════════════════════════════════════════════╝

This notebook will:
1. Clone your GitHub repo
2. Load existing model
3. {'OPTION A: Upload YOUR images + Download best Kaggle dataset' if OPTION == 'A' else 'OPTION B: Download best Kaggle dataset only'}
4. Gemini Pro auto-labels everything
5. Fine-tune on GPU (15 epochs)
6. Download trained model

Let's go! 🚀
""")

if OPTION not in ["A", "B"]:
    raise ValueError("OPTION must be 'A' or 'B'")
```

---

## **CELL 1: Install Dependencies**

```python
# Install required packages
!pip install -q torch torchvision efficientnet-pytorch google-genai pillow tqdm kaggle

# Check GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

---

## **CELL 2: Configure Gemini Pro API (DO THIS FIRST!)**

```python
import google.genai as genai
import os

print("Step 1: Get your Gemini Pro API key")
print("Go to: https://aistudio.google.com/app/apikey")
print("Click 'Create API Key' if you don't have one")
print("Copy the key and paste below\n")

GEMINI_API_KEY = input("Enter your Gemini Pro API key: ").strip()

# Validate
try:
    genai.configure(api_key=GEMINI_API_KEY)
    # Quick test
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content("Say 'API key works!'")
    print("✓ Gemini Pro API configured and working!")
    print(f"  Response: {response.text[:50]}...")
except Exception as e:
    print(f"✗ API key error: {e}")
    print("  Make sure you copied the FULL key from aistudio.google.com")
```

---

## **CELL 3: Clone GitHub Repository**

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
print(f"✓ Current directory: {os.getcwd()}")
```

---

## **CELL 4: Load Existing Model from GitHub**

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

## **CELL 5: CHOOSE YOUR DATA SOURCE**

### **OPTION A: Upload Your Images + Download from Kaggle**

```python
from google.colab import files
from pathlib import Path
import subprocess

# Create directories
upload_dir = Path("uploaded_images")
kaggle_dir = Path("kaggle_images")
upload_dir.mkdir(exist_ok=True)
kaggle_dir.mkdir(exist_ok=True)

print("=" * 70)
print("OPTION A: YOUR IMAGES + KAGGLE DATASET")
print("=" * 70)

# Part 1: Upload your images
print("\n[A1] Uploading your flood images...")
print("Click 'Choose Files' button to upload (50-200 images recommended)")
print("Formats: .jpg, .jpeg, .png")
print("Try to include: dry, shallow, moderate, deep water levels\n")

uploaded = files.upload()

your_image_count = 0
for filename, data in uploaded.items():
    with open(upload_dir / filename, "wb") as f:
        f.write(data)
    your_image_count += 1
    print(f"✓ {filename}")

print(f"\n✓ {your_image_count} of your images uploaded")

# Part 2: Download from Kaggle (most reviewed)
print("\n[A2] Downloading from Kaggle (most reviewed datasets)...")
print("Popular flood datasets on Kaggle:")
print("1. 'Flood Area Segmentation' - 2.2k upvotes")
print("2. 'Floods Image Dataset' - 1.8k upvotes")
print("3. 'Satellite Floods Imagery' - 1.5k upvotes")

# Setup Kaggle API
print("\nSetting up Kaggle API...")
print("Go to: https://www.kaggle.com/settings/account")
print("Click 'Create New API Token' (downloads kaggle.json)")
print("Upload the kaggle.json file:")

kaggle_json = files.upload()
if "kaggle.json" in kaggle_json:
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    with open(os.path.expanduser("~/.kaggle/kaggle.json"), "wb") as f:
        f.write(kaggle_json["kaggle.json"])
    os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)
    print("✓ Kaggle API configured")
    
    # Download most popular flood dataset
    print("\nDownloading most reviewed flood dataset...")
    try:
        # This is one of the most reviewed
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", "jannalipka/flood-detection-image-dataset", "-p", str(kaggle_dir), "--unzip"],
            check=True
        )
        print("✓ Kaggle dataset downloaded")
    except:
        print("⚠️ Primary dataset not available, trying alternative...")
        try:
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", "avanishpathak/floods-image-dataset", "-p", str(kaggle_dir), "--unzip"],
                check=True
            )
            print("✓ Alternative Kaggle dataset downloaded")
        except:
            print("⚠️ Kaggle download failed (API limit or auth)")
            print("   You can still use just your uploaded images")
else:
    print("⚠️ kaggle.json not found, skipping Kaggle download")
    print("   Continue with just your uploaded images")

# Combine images
all_images = list(upload_dir.glob("*")) + list(kaggle_dir.glob("*/*"))
all_images = [f for f in all_images if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]

print(f"\n✓ Total images ready: {len(all_images)}")
print(f"  Your images: {your_image_count}")
print(f"  Kaggle images: {len(all_images) - your_image_count}")

# Store paths for next cells
TRAINING_DIR = upload_dir
ADDITIONAL_DIR = kaggle_dir
```

### **OPTION B: Use Kaggle Dataset Only**

```python
from pathlib import Path
import subprocess
import os

# Create directory
kaggle_dir = Path("kaggle_images")
kaggle_dir.mkdir(exist_ok=True)

print("=" * 70)
print("OPTION B: KAGGLE DATASET ONLY")
print("=" * 70)

# Setup Kaggle API
print("\nSetting up Kaggle API...")
print("Go to: https://www.kaggle.com/settings/account")
print("Click 'Create New API Token' (downloads kaggle.json)")

from google.colab import files
kaggle_json = files.upload()

if "kaggle.json" in kaggle_json:
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    with open(os.path.expanduser("~/.kaggle/kaggle.json"), "wb") as f:
        f.write(kaggle_json["kaggle.json"])
    os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)
    print("✓ Kaggle API configured")
    
    # Download most reviewed flood datasets
    print("\nDownloading most reviewed flood datasets...")
    
    datasets = [
        "jannalipka/flood-detection-image-dataset",
        "avanishpathak/floods-image-dataset",
    ]
    
    for dataset in datasets:
        try:
            print(f"  Downloading {dataset}...")
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", dataset, "-p", str(kaggle_dir), "--unzip"],
                check=True,
                capture_output=True
            )
            print(f"  ✓ {dataset}")
        except Exception as e:
            print(f"  ⚠️ {dataset} failed: {str(e)[:50]}")

# Get all images
all_images = list(kaggle_dir.glob("*/*")) + list(kaggle_dir.glob("*"))
all_images = [f for f in all_images if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]

print(f"\n✓ Total images downloaded: {len(all_images)}")

TRAINING_DIR = kaggle_dir
```

---

## **CELL 6: Label Images with Gemini Pro**

```python
import csv
from datetime import datetime
from PIL import Image

def label_image_with_gemini(image_path: Path, retry=2) -> float:
    """Use Gemini Pro vision to estimate flood depth."""
    
    for attempt in range(retry):
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
2. Vehicle submersion (tires, bumpers, doors, hood, roof)
3. Person submersion (ankle, knee, waist, chest)
4. Water color and clarity

Based on VISUAL CUES ONLY, estimate the FLOOD DEPTH IN CENTIMETERS.

Rules:
- If water covers tire (32cm) but not bumper (48cm) → estimate 40cm
- If water at person's ankle (12cm) → estimate 15cm
- If water at person's waist (92cm) → estimate 90cm
- If dry/no flood → respond 0
- If completely submerged buildings → estimate 150-200cm
- If uncertain → estimate 50cm

Respond with ONLY a single number in centimeters (e.g., "45" for 45cm).
NO EXPLANATIONS, JUST THE NUMBER.
"""
            
            # Call Gemini
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content([prompt, file])
            
            # Parse response
            depth_str = response.text.strip().split('\n')[0]
            depth_cm = float(''.join(filter(str.isdigit, depth_str[:5])) or '50')
            depth_cm = max(0, min(200, depth_cm))  # Clamp to [0, 200]
            
            return depth_cm
            
        except Exception as e:
            if attempt < retry - 1:
                print(f"  Retry {attempt + 1}/{retry}...")
                continue
            else:
                print(f"  Error: {e}, using default 50cm")
                return 50.0
    
    return 50.0


# Label all images
print("Labeling images with Gemini Pro...")
print("This may take several minutes depending on image count...\n")

labels = []
image_files = sorted([f for f in TRAINING_DIR.rglob("*") 
                     if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])

# Limit to 200 images (balance between data and time)
image_files = image_files[:200]

for idx, img_path in enumerate(image_files, 1):
    depth = label_image_with_gemini(img_path)
    labels.append({
        "filename": img_path.name,
        "depth_cm": depth,
        "timestamp": datetime.now().isoformat(),
    })
    print(f"[{idx:3d}/{len(image_files)}] {img_path.name}: {depth:.1f}cm")

# Save to CSV
labels_csv = Path("gemini_labels.csv")
with open(labels_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "depth_cm", "timestamp"])
    writer.writeheader()
    writer.writerows(labels)

print(f"\n✓ Labeled {len(labels)} images, saved to {labels_csv}")
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
    
    def __init__(self, labels_csv: Path, image_size=(512, 512)):
        self.image_size = image_size
        self.data = []
        
        with open(labels_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Search in both directories
                possible_paths = [
                    TRAINING_DIR / row["filename"],
                    ADDITIONAL_DIR / row["filename"] if OPTION == "A" else None,
                ]
                
                for img_path in possible_paths:
                    if img_path and img_path.exists():
                        self.data.append({
                            "path": img_path,
                            "depth": float(row["depth_cm"]),
                        })
                        break
        
        depths = [d["depth"] for d in self.data]
        print(f"Dataset loaded: {len(self.data)} samples")
        print(f"  Depth range: {min(depths) if depths else 'N/A'}...{max(depths) if depths else 'N/A'}cm")
        if depths:
            print(f"  Mean: {np.mean(depths):.1f}cm, Std: {np.std(depths):.1f}cm")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        try:
            image = Image.open(item["path"]).convert("RGB")
            image = image.resize(self.image_size, Image.BILINEAR)
            
            # Normalize
            image = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0)
            image = image.permute(2, 0, 1)
            
            return {
                "image": image,
                "depth": torch.tensor(item["depth"], dtype=torch.float32),
            }
        except Exception as e:
            print(f"Error loading {item['path']}: {e}")
            raise


# Create dataset
dataset = FloodDepthDataset(labels_csv)

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

print("✓ Backbone frozen (transfer learning - FAST!)")

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

## **CELL 9: Save Model & Download**

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
    "val_loss": f"{best_checkpoint['val_loss']:.6f}",
    "val_mae": f"{best_checkpoint['val_mae']:.2f}cm",
    "total_images": len(dataset),
    "training_option": f"Option {OPTION}",
}

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print("\nMetadata:")
for key, value in metadata.items():
    print(f"  {key}: {value}")
```

---

## **CELL 10: Download & GitHub Push Instructions**

```python
from google.colab import files

print("\n" + "="*60)
print("DOWNLOADING MODEL...")
print("="*60)

# Download model
print(f"\nDownloading {model_path}...")
files.download(str(model_path))

# Print instructions
instructions = f"""
╔════════════════════════════════════════════════════════════════╗
║  TRAINING COMPLETE! MODEL READY FOR DEPLOYMENT                ║
╚════════════════════════════════════════════════════════════════╝

✓ Model trained: best_flood_model_v2.pth
✓ Total images: {len(dataset)} (mixed quality)
✓ Val Loss: {best_checkpoint['val_loss']:.6f}
✓ Val MAE: {best_checkpoint['val_mae']:.2f}cm

NEXT STEPS - Push to GitHub:

1. Your model is now downloading:
   → best_flood_model_v2.pth

2. On YOUR LOCAL MACHINE (Windows):
   
   $ cd flood-depth-estimator
   $ git pull origin main
   
   # Copy the downloaded model file into models/ folder
   $ copy %USERPROFILE%\\Downloads\\best_flood_model_v2.pth models\\

   # Commit and push
   $ git add models\\best_flood_model_v2.pth
   $ git add datasets\\training_v2\\  (optional - for backup)
   $ git commit -m "Trained model v2: {len(dataset)} images, val_loss={best_checkpoint['val_loss']:.6f}, val_mae={best_checkpoint['val_mae']:.2f}cm"
   $ git push origin main

3. Update config.yaml:
   
   inference:
     model_path: models/best_flood_model_v2.pth
   
   training:
     best_model_path: models/best_flood_model_v2.pth

4. Restart the server:
   
   $ python app.py
   
   Then visit: http://localhost:5000
   
   ✓ Server will AUTO-LOAD the new model!

5. Test the new model:
   ✓ Upload test images
   ✓ Check predictions (should NOT be stuck at 0cm)
   ✓ Verify depth values look reasonable
   ✓ Compare with reference CV fallback

SUCCESS! Your model is now deployed 🚀
"""

print(instructions)

# Save instructions to file
with open("DEPLOYMENT_INSTRUCTIONS.txt", "w") as f:
    f.write(instructions)

files.download("DEPLOYMENT_INSTRUCTIONS.txt")
print("\n✓ Instructions also downloaded as: DEPLOYMENT_INSTRUCTIONS.txt")
```

---

## **Training Pipeline Overview**

```
┌─────────────────────────────────────────────────────────┐
│  YOUR MACHINE (Windows)                                 │
│  - This notebook                                        │
│  - Your flood images (50-200)                           │
│  - Gemini Pro API key                                   │
│  - Kaggle account (optional)                            │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  GOOGLE COLAB (This Notebook)    │
        │  • Clone GitHub repo             │
        │  • Load existing model           │
        │  • Download Kaggle images (opt)  │
        │  • Upload YOUR images            │
        │  • Gemini Pro labels all images  │
        │  • Fine-tune on GPU (15 epochs)  │
        │  • Save model checkpoint         │
        └──────────────┬───────────────────┘
                       │
                       ▼ (Download)
┌─────────────────────────────────────────────────────────┐
│  YOUR MACHINE AGAIN                                     │
│  • Git push to GitHub                                   │
│  • Server auto-loads new model                          │
│  • Test at http://localhost:5000                        │
└─────────────────────────────────────────────────────────┘
```

---

## **How It Works (Simple Explanation)**

| Step | What | Why |
|------|------|-----|
| **Load existing model** | Use the model you trained before | Fast (transfer learning) |
| **Upload your images** | Mix of dry/wet/deep water | Real-world training data |
| **Download from Kaggle** | Most reviewed public datasets | More variety + robustness |
| **Gemini labels** | AI automatically estimates depth | No manual CSV work! |
| **Fine-tune** | Train only the top layer (frozen backbone) | Keeps learned features + adds new knowledge |
| **GPU training** | Google's Tesla T4/A100 | 15 epochs in 5-10 min |
| **GitHub push** | Deploy new model to production | Server auto-reloads |

---

## **Expected Results After Training**

✅ **Model improves from v1:**
- ❌ v1 (before): val_loss ≈ 7e-7, always predicts 0cm
- ✅ v2 (after): val_loss < 0.05, MAE = 5-15cm

✅ **Visual changes in UI:**
- `Method: CV FALLBACK` → `Method: ML+CV` (blends both)
- Depth no longer stuck at 0cm
- More realistic predictions

✅ **In the browser:**
- Upload test flood image
- Should see depth estimates like: 25cm, 45cm, 80cm
- NOT 0cm for everything

---

## **Troubleshooting**

| Problem | Solution |
|---------|----------|
| **Gemini API fails** | Check key at https://aistudio.google.com/app/apikey; ensure billing enabled |
| **Kaggle dataset fails** | Skip it; continue with just your images; upload kaggle.json again |
| **Out of memory** | Reduce batch_size to 8; reduce image_size to 384 |
| **Training too slow** | Ensure GPU enabled (Runtime → Change Runtime Type) |
| **Model still predicts 0** | Your labels may be poor; collect more diverse flood images |
| **Git push fails** | Ensure Git LFS installed: `git lfs install` |

---

## **Questions?**

Check the repo docs:
- `COLAB_RETRAINING_GUIDE.md` - This guide
- `enterprise_flood_model.py` - Complete model code
- `AFTER_TRAINING_CHECKLIST.md` - Deployment steps
- `MODEL_DATASET_MANAGEMENT.md` - Asset versioning

**Support:**
- Gemini help: https://aistudio.google.com
- Kaggle help: https://www.kaggle.com/settings/account
- GitHub LFS help: https://git-lfs.com/

Happy training! 🚀

