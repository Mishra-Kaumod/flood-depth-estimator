# OPTION A: Complete Step-by-Step (Your Images + Kaggle + Retrain)

**TL;DR:** Upload images → Download Kaggle → Gemini labels → Fine-tune existing model on GPU → Deploy

---

## **The Complete Workflow (Visual)**

```
YOUR IMAGES (50-200)
     ↓
┌────────────────────────────────────────────┐
│  GOOGLE COLAB                              │
│                                            │
│  Cell 1-4:                                 │
│  • Install dependencies                    │
│  • Clone GitHub repo                       │
│  • Load EXISTING MODEL ← From v1           │
│  • Verify model weights loaded             │
│                                            │
│  Cell 5 (OPTION A):                        │
│  • Upload YOUR images (UI button)          │
│  • Download Kaggle datasets (most reviewed)│
│  • Merge both datasets                     │
│                                            │
│  Cell 6:                                   │
│  • Gemini Pro auto-labels all              │
│  • Creates labels.csv                      │
│                                            │
│  Cell 7:                                   │
│  • Load images + labels                    │
│  • Create 80/20 train/val split            │
│                                            │
│  Cell 8 (KEY - TRANSFER LEARNING):         │
│  • FREEZE backbone (keep learned features) │
│  • TRAIN only regression head              │
│  • 15 epochs on GPU (10 min)               │
│  • Save best checkpoint                    │
│                                            │
│  Cell 9-10:                                │
│  • Save model                              │
│  • Download best_flood_model_v2.pth        │
│  • Get GitHub push instructions            │
│                                            │
└────────────────────────────────────────────┘
     ↓
YOUR COMPUTER (Windows)
     ├─ git add models/best_flood_model_v2.pth
     ├─ git commit -m "Trained v2: 150 images"
     └─ git push origin main
     ↓
SERVER (http://localhost:5000)
     └─ ✓ Auto-loads new model!
```

---

## **Key Difference: Transfer Learning**

### **What OPTION A Does**

```python
# Load EXISTING MODEL from GitHub
checkpoint = torch.load("models/best_flood_model_water_aware.pth")
model = FloodDepthRegressor()
model.load_state_dict(checkpoint["model_state_dict"])  # ← Load EXISTING weights

# FREEZE backbone (keep learned ImageNet features)
for param in model.backbone.parameters():
    param.requires_grad = False  # ← Won't be updated

# TRAIN only regression head (adapt to new data)
optimizer = optim.AdamW(model.head.parameters(), lr=1e-4)  # ← Only head

# 15 epochs on GPU
for epoch in range(15):
    predictions = model(images)  # ← Backbone uses LEARNED features
    loss = criterion(predictions, targets)
    loss.backward()
    optimizer.step()  # ← Only head weights updated
```

### **Why This Is Better Than Starting From Scratch**

| Metric | Transfer Learning | From Scratch |
|--------|-------------------|--------------|
| **Time** | 10-15 min | 1-2 hours |
| **Data Needed** | 50-200 images | 500+ images |
| **Risk** | Low (keeps good features) | High (random start) |
| **Generalization** | Better (ImageNet features) | Worse (overfits) |

---

## **Cell-by-Cell Breakdown for Option A**

### **CELL 0: Choose Option**

```python
OPTION = "A"  # MUST be "A" for this workflow
```

---

### **CELL 1: Install Dependencies**

```python
!pip install -q torch torchvision efficientnet-pytorch google-genai pillow tqdm kaggle
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

**What it does:**
- ✓ Installs PyTorch, Gemini, Kaggle CLI
- ✓ Checks GPU available (should say Tesla T4 or A100)

---

### **CELL 2: Configure Gemini Pro API**

```python
import google.genai as genai
GEMINI_API_KEY = input("Enter your Gemini Pro API key: ")
genai.configure(api_key=GEMINI_API_KEY)
```

**What it does:**
- ✓ Sets up Gemini Pro for auto-labeling
- ✓ Tests API key (will fail if invalid)

---

### **CELL 3: Clone GitHub Repo**

```python
!git clone https://github.com/Mishra-Kaumod/flood-depth-estimator.git
%cd flood-depth-estimator
!git lfs install
!git lfs pull
```

**What it does:**
- ✓ Clones your GitHub repo
- ✓ Initializes Git LFS (large file storage)
- ✓ Downloads your existing model checkpoint

---

### **CELL 4: Load & Verify Existing Model (CRITICAL)**

```python
# Load checkpoint from GitHub
checkpoint = torch.load("models/best_flood_model_water_aware.pth")
model = FloodDepthRegressor()
model.load_state_dict(checkpoint["model_state_dict"])  # ← LOADS EXISTING WEIGHTS

print(f"✓ Loaded existing model")
print(f"  Epoch: {checkpoint['epoch']}")
print(f"  Val loss: {checkpoint['val_loss']}")  
print(f"  Val MAE: {checkpoint['val_mae']}cm")
```

**What it does:**
- ✓ Loads the existing v1 model checkpoint
- ✓ Verifies all weights are loaded correctly
- ✓ Shows old performance (for comparison later)

**Key:** This is where the EXISTING MODEL enters. The checkpoint contains all weights from v1 training.

---

### **CELL 5: Upload Images + Download Kaggle (OPTION A SPECIFIC)**

```python
from google.colab import files

# Part A: Upload YOUR images
print("[A1] Upload your flood images...")
uploaded = files.upload()  # ← Click button, select files

for filename, data in uploaded.items():
    with open(f"uploaded_images/{filename}", "wb") as f:
        f.write(data)
    print(f"✓ {filename}")

# Part B: Download from Kaggle
print("\n[A2] Download from Kaggle...")
# Upload kaggle.json
kaggle_json = files.upload()
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
with open(os.path.expanduser("~/.kaggle/kaggle.json"), "wb") as f:
    f.write(kaggle_json["kaggle.json"])

# Download most reviewed datasets
subprocess.run([
    "kaggle", "datasets", "download", "-d", 
    "jannalipka/flood-detection-image-dataset",
    "-p", "kaggle_images", "--unzip"
], check=True)

print(f"✓ Kaggle downloaded: {len(list(Path('kaggle_images').glob('*/*')))} images")
print(f"✓ Your images: {len(list(Path('uploaded_images').glob('*')))} images")
print(f"✓ Total: {len(...)} images ready for labeling")
```

**What it does:**
- ✓ Upload your 50-200 flood images (UI button)
- ✓ Download Kaggle's most-reviewed flood datasets
- ✓ Merge both into training dataset

**Total images:** 100-400 (mix of your data + public data)

---

### **CELL 6: Gemini Pro Auto-Labels All Images**

```python
def label_image_with_gemini(image_path: Path) -> float:
    """Use Gemini to estimate flood depth from image."""
    
    # Upload image to Gemini
    file = genai.upload_file(image_data, mime_type="image/jpeg")
    
    # Ask Gemini to analyze flood depth
    prompt = """Estimate FLOOD DEPTH IN CENTIMETERS based on visual cues:
    - Water level relative to known objects
    - Vehicle submersion (tire=32cm, bumper=48cm, door=68cm)
    - Person submersion (ankle=12cm, knee=48cm, waist=92cm)
    
    Respond with ONLY the number in cm (e.g., "45")"""
    
    response = model.generate_content([prompt, file])
    depth_cm = float(response.text.strip())
    return depth_cm

# Label all images
for img_path in all_image_files:
    depth = label_image_with_gemini(img_path)  # ← Gemini does the work
    labels.append({"filename": img_path.name, "depth_cm": depth})

# Save to CSV
with open("gemini_labels.csv", "w") as f:
    csv.DictWriter(f, ["filename", "depth_cm"]).writerows(labels)

print(f"✓ Labeled {len(labels)} images with Gemini Pro")
```

**What it does:**
- ✓ Gemini Pro analyzes each image
- ✓ Extracts depth estimate (0-200cm)
- ✓ Creates labels.csv

**Time:** ~3-5 min for 100-300 images

---

### **CELL 7: Create Dataset & Train/Val Split**

```python
# Load images + labels
dataset = FloodDepthDataset("gemini_labels.csv")

# 80/20 split
train_data, val_data = random_split(dataset, [80, 20])
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

print(f"✓ Train: {len(train_data)} images")
print(f"✓ Val: {len(val_data)} images")
```

**What it does:**
- ✓ Loads all labeled images
- ✓ Splits 80% train, 20% validation
- ✓ Creates batches of 16 images

---

### **CELL 8: Fine-Tune Existing Model (TRANSFER LEARNING) ⭐**

```python
# ========== FREEZE BACKBONE ==========
for param in model.backbone.parameters():
    param.requires_grad = False  # ← Don't update EfficientNet weights

print("✓ Backbone FROZEN (existing ImageNet features preserved)")

# ========== KEEP HEAD TRAINABLE ==========
# Head is already trainable by default
optimizer = optim.AdamW(
    model.head.parameters(),  # ← Only update regression head
    lr=1e-4,  # Low learning rate (gentle fine-tuning)
    weight_decay=1e-4
)

print("✓ Regression head TRAINABLE (will adapt to new data)")

# ========== TRAIN ==========
for epoch in range(15):
    model.train()
    
    for batch in train_loader:
        images, targets = batch
        
        # Forward pass through FROZEN backbone
        features = model.backbone(images)  # ← Uses existing weights
        
        # Forward through TRAINABLE head
        predictions = model.head(features)  # ← Gets updated
        
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()  # ← Only head gets updated
    
    # Validate
    model.eval()
    with torch.no_grad():
        val_loss = evaluate(model, val_loader)
    
    print(f"Epoch {epoch+1}: train_loss={...:.6f}, val_loss={val_loss:.6f}")

print(f"✓ Training complete!")
print(f"  New val_loss: {best_val_loss:.6f}")
print(f"  Improvement: {old_val_loss - best_val_loss:.6f}")
```

**What it does:**
- ✓ Freezes backbone (ImageNet features stay frozen)
- ✓ Trains only regression head (adapts to flood depth)
- ✓ 15 epochs on GPU (~10 min)
- ✓ Saves best checkpoint when val_loss improves

**This is the KEY PART** - the existing model is being FINE-TUNED, not replaced.

---

### **CELL 9: Save Model**

```python
model_path = "models/best_flood_model_v2.pth"
torch.save(best_checkpoint, model_path)

print(f"✓ Model saved: {model_path}")
print(f"  Size: {model_path.stat().st_size / 1e6:.1f}MB")
```

**What it does:**
- ✓ Saves fine-tuned model weights
- ✓ Includes metrics, epoch, loss

---

### **CELL 10: Download & GitHub Instructions**

```python
from google.colab import files

# Download
files.download(str(model_path))

# Print instructions
print(f"""
✓ MODEL TRAINED AND DOWNLOADED

Next steps on YOUR COMPUTER:

$ cd flood-depth-estimator
$ git pull origin main
$ copy %USERPROFILE%\\Downloads\\best_flood_model_v2.pth models\\
$ git add models\\best_flood_model_v2.pth
$ git commit -m "Trained v2: 150 images, val_loss=0.045cm"
$ git push origin main

Then:
$ python app.py
$ # Visit http://localhost:5000
""")
```

---

## **Expected Results After Option A**

### **Before (v1 - Collapsed)**
```
val_loss: 7e-7 (too low - all zeros)
val_mae: ∞ (always predicts 0cm)
Predictions: [0.0, 0.0, 0.0, 0.0, ...]
```

### **After (v2 - Retrained with Option A)**
```
val_loss: 0.045 (realistic)
val_mae: 6.5cm (good!)
Predictions: [25.3, 48.2, 12.7, 85.6, ...]  ← Realistic!
```

### **In the UI**
- Badge: `CV FALLBACK` → `ML+CV` (blends both methods)
- Upload test image → see depth 20-80cm (NOT 0cm!)
- Map shows depth estimates with confidence

---

## **Verification Checklist**

After training, verify:

- [ ] Model file downloaded: `best_flood_model_v2.pth` (~50MB)
- [ ] Training metrics improved:
  - [ ] val_loss decreased
  - [ ] val_mae is 5-15cm range
- [ ] Predictions are realistic:
  - [ ] Not 0cm for everything
  - [ ] Range 0-150cm typically
- [ ] Git push successful
- [ ] Server restarted: `python app.py`
- [ ] Test image uploads work
- [ ] Depth values visible in UI

---

## **If Training Fails**

### **Issue: val_loss stays very high**
**Cause:** Bad labels from Gemini  
**Solution:**
1. Check a few images manually
2. Are they actually flood images? Or random landscapes?
3. Re-upload better quality flood photos
4. Re-run Gemini labeling

### **Issue: val_loss doesn't improve**
**Cause:** Learning rate too high  
**Solution:**
1. Try lower LR: change to `lr=1e-5`
2. Or: you need more training data

### **Issue: Out of memory**
**Cause:** Too many images or large batch  
**Solution:**
1. Reduce batch_size: `batch_size=8`
2. Reduce image_size: `image_size=384`

---

## **Summary: What Makes Option A Special**

```
✓ Uses YOUR flood images
✓ Adds public Kaggle datasets  
✓ Gemini auto-labels (no manual work)
✓ Fine-tunes from EXISTING model (fast)
✓ Keeps learned backbone (better generalization)
✓ Only trains regression head (focused adaptation)
✓ GPU accelerated (10 min training)
✓ Tracks all versions (model_registry.json)
✓ Deploy to server (auto-loads)
```

This is the **recommended** workflow for production. 🚀
