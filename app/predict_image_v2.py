import sys
import torch
import torch.nn as nn

from PIL import Image

from torchvision import transforms
from torchvision.models import efficientnet_b0

MODEL_PATH = "models/severity_efficientnet.pth"

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

model = efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    5
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)
model.eval()

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