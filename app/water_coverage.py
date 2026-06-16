from PIL import Image
import numpy as np
import torch
from torchvision import models, transforms

# -----------------------------
# Device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# -----------------------------
# Load DeepLabV3
# -----------------------------
model = models.segmentation.deeplabv3_resnet50(
    weights="DEFAULT"
)

model.eval()
model.to(device)

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((520, 520)),
    transforms.ToTensor(),
])

# -----------------------------
# Water Coverage Function
# -----------------------------
def get_water_coverage(image_path):

    img = Image.open(image_path).convert("RGB")

    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)["out"][0]

    prediction = output.argmax(0).cpu().numpy()

    # Anything not background
    mask = (prediction > 0).astype(np.uint8)

    coverage = (
        mask.sum() /
        mask.size
    ) * 100

    return round(coverage, 2)

# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":

    image_path = (
        "datasets/flooddet/extracted/"
        "FloodDET/test/Flood_10855.jpg"
    )

    coverage = get_water_coverage(image_path)

    print("\n--------------------------------")
    print(f"Water Coverage: {coverage}%")
    print("--------------------------------")