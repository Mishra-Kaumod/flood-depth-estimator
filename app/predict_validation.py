from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 5)

model.load_state_dict(
    torch.load("severity_model.pth", map_location=device)
)

model.to(device)
model.eval()

tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

val_folder = Path(
    "datasets/flooddet/extracted/FloodDET/val"
)

for img_path in list(val_folder.glob("*.jpg"))[:20]:

    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x).argmax(1).item()

    print(
        f"{img_path.name} -> Severity {pred}"
    )