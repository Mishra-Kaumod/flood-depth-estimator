import pandas as pd

df = pd.read_csv("yolo_inventory.csv")

def severity(objects):
    objs = str(objects).split(",")

    score = 0

    score += objs.count("boat") * 2
    score += objs.count("truck") * 1
    score += objs.count("car") * 1

    if objs.count("person") >= 3:
        score += 1

    if score <= 1:
        return 0
    elif score <= 3:
        return 1
    elif score <= 5:
        return 2
    elif score <= 8:
        return 3
    else:
        return 4

df["severity"] = df["objects"].apply(severity)

df.to_csv("severity_labels.csv", index=False)

print(df["severity"].value_counts().sort_index())
print("Saved severity_labels.csv")