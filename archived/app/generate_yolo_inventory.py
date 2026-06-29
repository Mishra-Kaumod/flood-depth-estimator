from ultralytics import YOLO
from pathlib import Path
import pandas as pd

model = YOLO("yolov8n.pt")

rows = []

for split in ["train", "val", "test"]:
    imgs = list(Path(f"datasets/flooddet/extracted/FloodDET/{split}").glob("*.jpg"))

    print(f"{split}: {len(imgs)} images")

    for img in imgs:
        result = model(str(img), verbose=False)

        objs = []

        if result[0].boxes is not None:
            for cls in result[0].boxes.cls:
                objs.append(model.names[int(cls)])

        rows.append({
            "split": split,
            "image": img.name,
            "objects": ",".join(objs),
            "object_count": len(objs)
        })

df = pd.DataFrame(rows)
df.to_csv("yolo_inventory.csv", index=False)

print(f"Saved {len(df)} rows to yolo_inventory.csv")
print(df.head())