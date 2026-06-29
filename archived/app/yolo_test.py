from ultralytics import YOLO
from pathlib import Path
import pandas as pd

model = YOLO("yolov8n.pt")

rows = []

imgs = list(Path("datasets/flooddet/extracted/FloodDET/train").glob("*.jpg"))

for img in imgs:
    result = model(str(img), verbose=False)

    objs = []

    for cls in result[0].boxes.cls:
        objs.append(model.names[int(cls)])

    rows.append({
        "image": img.name,
        "objects": ",".join(objs)
    })

pd.DataFrame(rows).to_csv("yolo_objects.csv", index=False)

print("Saved yolo_objects.csv")