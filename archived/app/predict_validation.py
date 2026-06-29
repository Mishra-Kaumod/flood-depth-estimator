from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms

from model_paths import get_severity_model_path
from severity_model_loader import load_severity_model

device = "cuda" if torch.cuda.is_available() else "cpu"

model = load_severity_model(get_severity_model_path(), device)

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
