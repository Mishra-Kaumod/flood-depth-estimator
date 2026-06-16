from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import argparse

from depth_band_estimator import estimate_depth

LABELS = {
    0: "No / Very Low Flood",
    1: "Minor Flooding",
    2: "Moderate Flooding",
    3: "High Flooding",
    4: "Severe Flooding"
}

device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 5)

model.load_state_dict(
    torch.load("severity_model.pth", map_location=device)
)

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def predict(image_path):

    img = Image.open(image_path).convert("RGB")

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        severity = int(torch.argmax(probs, dim=1).item())
        confidence = float(torch.max(probs).item())

    depth = estimate_depth(severity)

    print("\nPrediction")
    print("-" * 40)
    print(f"Severity      : {severity}")
    print(f"Severity Name : {LABELS[severity]}")
    print(f"Confidence    : {confidence:.4f}")
    print(f"Depth Band    : {depth['depth_band']}")
    print(f"Depth (cm)    : {depth['depth_cm']}")
    print("-" * 40)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("image")

    args = parser.parse_args()

    predict(args.image)