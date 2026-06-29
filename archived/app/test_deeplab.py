from PIL import Image
import numpy as np
from pathlib import Path
import torch
from torchvision import models, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.segmentation.deeplabv3_resnet50(
    weights="DEFAULT"
)

model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.Resize((520, 520)),
    transforms.ToTensor(),
])

image_path = "datasets/flooddet/extracted/FloodDET/test/Flood_10855.jpg"

img = Image.open(image_path).convert("RGB")

input_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)["out"][0]

prediction = output.argmax(0).cpu().numpy()

print("Unique Classes:", np.unique(prediction))

mask = (prediction > 0).astype(np.uint8) * 255

coverage = mask.sum() / 255 / mask.size * 100

print(f"Coverage: {coverage:.2f}%")

Image.fromarray(mask).save("deeplab_mask.png")

print("Saved deeplab_mask.png")