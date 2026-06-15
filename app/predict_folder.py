from pathlib import Path
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms, models

# -----------------------------
# Config
# -----------------------------
IMAGE_FOLDER = "test_images/team"
OUTPUT_FILE = "results/predictions.csv"

LABELS = {
    0: "No / Very Low Flood",
    1: "Minor Flooding",
    2: "Moderate Flooding",
    3: "High Flooding",
    4: "Severe Flooding"
}

# -----------------------------
# Load Model
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 5)

model.load_state_dict(
    torch.load("severity_model.pth", map_location=device)
)

model.to(device)
model.eval()

# -----------------------------
# Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------------
# Create output folder
# -----------------------------
Path("results").mkdir(exist_ok=True)

# -----------------------------
# Predict
# -----------------------------
rows = []

image_extensions = ["*.jpg", "*.jpeg", "*.png"]

images = []

for ext in image_extensions:
    images.extend(Path(IMAGE_FOLDER).glob(ext))

print(f"\nFound {len(images)} images\n")

for img_path in images:

    try:
        img = Image.open(img_path).convert("RGB")

        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)

            pred = int(torch.argmax(probs, dim=1).item())
            confidence = float(torch.max(probs).item())

        rows.append({
            "image": img_path.name,
            "severity": pred,
            "severity_name": LABELS[pred],
            "confidence": round(confidence, 4)
        })

        print(
            f"{img_path.name} -> "
            f"{LABELS[pred]} "
            f"(confidence={confidence:.2f})"
        )

    except Exception as e:
        print(f"Failed: {img_path.name} -> {e}")

# -----------------------------
# Save CSV
# -----------------------------
df = pd.DataFrame(rows)

df.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved results to {OUTPUT_FILE}")
print(f"Processed {len(df)} images")