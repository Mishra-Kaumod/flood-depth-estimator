from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt

image_path = "datasets/flooddet/extracted/FloodDET/val/Flood_9064.jpg"

depth_estimator = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf"
)

image = Image.open(image_path).convert("RGB")

result = depth_estimator(image)

depth = result["depth"]

depth.save("depth_map.png")

print("Saved depth_map.png")