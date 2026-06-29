import os
import pandas as pd

CSV_FILE = "/workspaces/flood-depth-estimator/data/floodnet/train/masks/benchmark_labels.csv"
IMAGE_DIR = "/workspaces/flood-depth-estimator/data/floodnet/train/images"

print("=" * 60)
print("DATASET VALIDATION")
print("=" * 60)

# -----------------------------
# LOAD CSV
# -----------------------------

df = pd.read_csv("/workspaces/flood-depth-estimator/data/floodnet/train/masks/benchmark_labels.csv")

print(f"\nCSV Rows: {len(df)}")

# Clean filenames
df["image_name"] = (
    df["image_name"]
    .astype(str)
    .str.strip()
)

# -----------------------------
# IMAGE FILES
# -----------------------------

image_files = {
    f.strip()
    for f in os.listdir(IMAGE_DIR)
    if os.path.isfile(os.path.join(IMAGE_DIR, f))
}

print(f"Images Found: {len(image_files)}")

# -----------------------------
# MATCH CSV -> IMAGE
# -----------------------------

matched_df = df[
    df["image_name"].isin(image_files)
].copy()

print(f"Matched Rows: {len(matched_df)}")

# -----------------------------
# FIND MISSING IMAGES
# -----------------------------

missing_images = sorted(
    set(df["image_name"]) - image_files
)

if missing_images:

    print("\nMISSING IMAGE FILES")
    print("-" * 40)

    for img in missing_images[:50]:
        print(img)

    if len(missing_images) > 50:
        print(
            f"... and {len(missing_images)-50} more"
        )

else:

    print("\nNo missing image files.")

# -----------------------------
# FIND EXTRA FILES
# -----------------------------

extra_files = sorted(
    image_files - set(df["image_name"])
)

if extra_files:

    print("\nIMAGES WITHOUT LABELS")
    print("-" * 40)

    for img in extra_files[:50]:
        print(img)

    if len(extra_files) > 50:
        print(
            f"... and {len(extra_files)-50} more"
        )

else:

    print("\nAll images have labels.")

# -----------------------------
# DUPLICATES
# -----------------------------

duplicates = df[
    df["image_name"].duplicated()
]

print(
    f"\nDuplicate Labels: "
    f"{len(duplicates)}"
)

# -----------------------------
# SEVERITY CHECK
# -----------------------------

print("\nSeverity Distribution")
print("-" * 40)

if "severity" in matched_df.columns:

    print(
        matched_df["severity"]
        .value_counts()
        .sort_index()
    )

# -----------------------------
# SAVE CLEAN CSV
# -----------------------------

clean_csv = (
    "/workspaces/flood-depth-estimator/data/"
    "benchmark_labels_clean.csv"
)

matched_df.to_csv(
    clean_csv,
    index=False
)

print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)

print(
    f"\nClean CSV saved to:\n"
    f"{clean_csv}"
)

print(
    f"\nReady for training on "
    f"{len(matched_df)} images."
)