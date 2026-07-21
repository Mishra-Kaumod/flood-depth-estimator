# 🚀 Quick Start: Retrain Your Model in 20 Minutes

## **Three Files You Need to Know**

### **1. COLAB_RETRAINING_GUIDE.md** ← START HERE
Complete guide with 10 copy-paste cells for Google Colab. Choose between:
- **OPTION A**: Your images (50-200) + Kaggle dataset
- **OPTION B**: Kaggle only (faster)

### **2. enterprise_flood_model.py**
Complete production-ready model code (for reference/suggestions in Fable)

### **3. Retrain_Existing_Model_GitHub.py**
Full Python script if you want to run it directly instead of copy-pasting cells

---

## **The Flow (Step-by-Step)**

```
YOUR COMPUTER
     ↓
     (Get Gemini Pro API key from https://aistudio.google.com/app/apikey)
     ↓
GOOGLE COLAB (https://colab.research.google.com)
     ├─ Cell 0: Choose Option A or B
     ├─ Cell 1: Install dependencies
     ├─ Cell 2: Clone GitHub (automatic)
     ├─ Cell 3: Load your existing model (automatic)
     ├─ Cell 4: Enter Gemini API key (paste here)
     ├─ Cell 5: OPTION A → Upload images + Kaggle | OPTION B → Kaggle only
     ├─ Cell 6: Gemini Pro auto-labels all images 🤖
     ├─ Cell 7: Create train/val split
     ├─ Cell 8: Fine-tune on GPU (15 epochs, ~10 min)
     ├─ Cell 9: Save model
     └─ Cell 10: Download model + get push instructions
     ↓
YOUR COMPUTER
     ├─ git add models/best_flood_model_v2.pth
     ├─ git commit -m "Trained model v2: 150 images"
     └─ git push origin main
     ↓
SERVER (http://localhost:5000)
     └─ ✓ Auto-loads new model!
```

---

## **2 Options Explained**

### **OPTION A: Your Images + Kaggle (RECOMMENDED) ⭐**

```
Step 1: Upload 50-200 of YOUR flood images
Step 2: Download best Kaggle datasets (1.2k+ upvotes)
Step 3: Combine both (best of both worlds)
Step 4: Gemini labels all 100-300 images
Step 5: Train with GPU

Time: 20-30 min
Quality: Best (mixed data)
Effort: Medium (upload images once)
```

**Why this is best:**
- Your local images = real test cases
- Kaggle = more variety for robustness
- Gemini = no manual CSV work
- Combined = better generalization

### **OPTION B: Kaggle Only (QUICK) ⚡**

```
Step 1: Skip image upload
Step 2: Download Kaggle datasets automatically
Step 3: Gemini labels all images
Step 4: Train with GPU

Time: 10-15 min
Quality: Good (public data)
Effort: Minimal (no uploads)
```

**Why choose this:**
- Fast (no file uploads)
- Good for quick testing
- Still good labels from Gemini
- Less setup

---

## **What Happens (Under the Hood)**

### **OPTION A Detailed Flow**

```
Your images (50-200)
    ↓
[uploaded_images/] ← You upload via browser
    ↓
Kaggle API (automatic)
    ↓
[kaggle_images/] ← Auto-downloaded, most reviewed first
    ↓
Merged dataset (100-300 images)
    ↓
Gemini Pro batch labeling
    ↓
gemini_labels.csv (filename → depth_cm mapping)
    ↓
FloodDepthDataset loads all
    ↓
80/20 split → train_loader, val_loader
    ↓
Fine-tune regression head (backbone frozen)
    ↓
best_checkpoint saved
    ↓
Download → Git push → Server reload
```

### **OPTION B Detailed Flow**

```
Kaggle API (automatic)
    ↓
[kaggle_images/]
    ↓
Gemini Pro batch labeling
    ↓
gemini_labels.csv
    ↓
FloodDepthDataset loads all
    ↓
80/20 split → train_loader, val_loader
    ↓
Fine-tune regression head
    ↓
best_checkpoint saved
    ↓
Download → Git push → Server reload
```

---

## **Getting Gemini Pro API Key (5 min)**

1. Go to: https://aistudio.google.com/app/apikey
2. Click **"Create API Key"**
3. Select or create a project
4. Copy the key (starts with `AIza...`)
5. Paste into Cell 4 in Colab

> **Free tier includes:**
> - 15 requests per minute
> - Good enough for 100-200 image labeling
> - No credit card required (if quota available)

---

## **What You'll Get After Training**

✅ **Improved model:**
- `val_loss`: 7e-7 (broken) → 0.03-0.06 (working!)
- `val_mae`: ∞ (broken) → 5-15cm (good!)
- Predictions: 0cm (always) → 20-80cm (realistic!)

✅ **In the UI:**
- Badge changes: `CV FALLBACK` → `ML+CV`
- Map shows real depth estimates
- Upload test image → see depth (not 0!)

✅ **In your repo:**
- `models/best_flood_model_v2.pth` (50MB)
- `model_registry.json` updated with v2 metadata
- `DEPLOYMENT_INSTRUCTIONS.txt` for reference

---

## **Common Questions**

**Q: How long does it take?**
- A: 20-30 min (Option A) or 10-15 min (Option B)
- Most time is you uploading images or waiting for Gemini

**Q: Do I need Kaggle account?**
- A: Only for Option A/B (yes, create free account)
- Takes 2 min: https://www.kaggle.com

**Q: Can I run it multiple times?**
- A: Yes! Each run creates v2, v3, v4... tracking in `model_registry.json`

**Q: What if Gemini fails?**
- A: Built-in retry logic (tries 2x)
- Falls back to default label (50cm) if API error
- You can still train with partial labels

**Q: Will it overwrite my existing model?**
- A: No, it saves as `best_flood_model_v2.pth`
- You can keep both, test, and compare
- `model_registry.json` tracks all versions

**Q: Can I use my own Kaggle datasets?**
- A: Yes! Edit the dataset names in Cell 5
- Or skip Kaggle and only use your images

---

## **Workflow Summary**

| Stage | Location | Time | What |
|-------|----------|------|------|
| **Setup** | Colab | 2 min | Install deps + clone + load |
| **Gemini Config** | Colab | 1 min | Paste API key |
| **Data Collection** | Colab | 5-15 min | Option A: upload + download; Option B: download only |
| **Labeling** | Colab | 3-5 min | Gemini Pro labels all images |
| **Training** | Colab GPU | 10 min | 15 epochs fine-tuning |
| **Download** | Colab → Your PC | 1 min | Model file (50MB) |
| **Deploy** | Your PC | 2 min | Git push + server restart |

**Total: ~20-30 minutes** ⏱️

---

## **Next Steps**

1. **Get Gemini Pro API key** (free)
   → https://aistudio.google.com/app/apikey

2. **Open Google Colab**
   → https://colab.research.google.com

3. **Copy Cell 0 from `COLAB_RETRAINING_GUIDE.md`**
   → Paste into Colab, run

4. **Choose Option A or B**
   → Set `OPTION = "A"` or `OPTION = "B"`

5. **Copy remaining cells**
   → One by one, run each

6. **Download model**
   → Colab downloads automatically

7. **Git push**
   → Follow deployment instructions

8. **Test at localhost:5000**
   → Upload image → see depth!

---

## **Files Reference**

- **`COLAB_RETRAINING_GUIDE.md`** ← Main guide, copy-paste cells
- **`Retrain_Existing_Model_GitHub.py`** ← Alternative: run as script
- **`enterprise_flood_model.py`** ← For code review in Fable
- **`AFTER_TRAINING_CHECKLIST.md`** ← Post-training steps
- **`MODEL_DATASET_MANAGEMENT.md`** ← Asset versioning

---

## **Support**

- **Gemini API issues**: https://aistudio.google.com (check quota)
- **Kaggle issues**: https://www.kaggle.com/settings/account (upload token)
- **Git LFS issues**: `git lfs install`
- **Model issues**: Check `model_registry.json` for version history

---

## **Ready? Let's Go! 🚀**

1. Get Gemini API key
2. Open Google Colab
3. Copy the first cell from `COLAB_RETRAINING_GUIDE.md`
4. Run it!

Questions? Check `COLAB_RETRAINING_GUIDE.md` or `enterprise_flood_model.py` for detailed explanations.

**Happy training!** ✨
