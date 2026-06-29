import sys

import torch
import torchvision

from PIL import Image

from model_paths import get_flood_model_path, get_severity_model_path
from severity_model_loader import load_severity_model
from torchvision import transforms


def _load_model():
    model = torchvision.models.mobilenet_v3_small(weights=None)
    model.classifier[0] = torch.nn.Linear(model.classifier[0].in_features, 256)
    model.classifier[3] = torch.nn.Linear(256, 5)
    ckpt = torch.load(get_flood_model_path(), map_location="cpu")
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    return model


_MODEL = _load_model()
_REAL_MODEL = True

device = "cuda" if torch.cuda.is_available() else "cpu"

severity_names = {
    0: "No / Very Low Flood",
    1: "Minor Flood",
    2: "Moderate Flood",
    3: "High Flood",
    4: "Severe Flood"
}

if len(sys.argv) < 2:
    print("Usage:")
    print("python app/predict_image_v2.py <image_path>")
    sys.exit()

image_path = sys.argv[1]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

model = load_severity_model(get_severity_model_path(), device)

image = Image.open(image_path).convert("RGB")
image = transform(image)
image = image.unsqueeze(0).to(device)

with torch.no_grad():

    outputs = model(image)

    probs = torch.softmax(outputs, dim=1)

    print("\nClass Probabilities")
    print("----------------------------------------")

    for i in range(5):
        print(
            f"Severity {i}: "
            f"{probs[0][i].item():.4f}"
        )

    pred = probs.argmax(1).item()

    confidence = probs[0][pred].item()
    
depth_map = {
    0: ("0-5 cm", 5),
    1: ("5-20 cm", 15),
    2: ("20-50 cm", 35),
    3: ("50-80 cm", 65),
    4: ("80+ cm", 100)
}

band, depth = depth_map[pred]

print("\nPrediction")
print("----------------------------------------")
print("Severity      :", pred)
print("Severity Name :", severity_names[pred])
print("Confidence    :", round(confidence, 4))
print("Depth Band    :", band)
print("Depth (cm)    :", depth)
print("----------------------------------------")