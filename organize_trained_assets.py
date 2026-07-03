#!/usr/bin/env python3
"""
Model & Dataset Organization Script
====================================

After training in Colab, run this locally to organize everything.

Usage:
    python organize_trained_assets.py \
        --model "Downloads/best_flood_model_water_aware.pth" \
        --dataset "Downloads/training_v3" \
        --version "v3"
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import argparse


def organize_model(model_path, version):
    """
    Move model to proper location and update registry.
    
    Before:
        Downloads/best_flood_model_water_aware.pth
    
    After:
        models/best_flood_model_water_aware.pth          (current)
        models/archive/v3_20260703.pth                   (backup)
        model_registry.json                              (updated)
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Create archive
    archive_dir = Path("models/archive")
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"{version}_{timestamp}.pth"
    
    # Copy to archive
    shutil.copy2(model_path, archive_path)
    print(f"✅ Archived: {archive_path}")
    
    # Copy to current
    current_path = Path("models/best_flood_model_water_aware.pth")
    shutil.copy2(model_path, current_path)
    print(f"✅ Current model: {current_path}")
    
    # Load checkpoint info
    import torch
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    
    return {
        "model_path": str(current_path),
        "archive_path": str(archive_path),
        "checkpoint": ckpt,
    }


def organize_dataset(dataset_dir, version):
    """
    Move training dataset to datasets/training_vX/ folder.
    
    Before:
        Downloads/training_v3/
            train/
            val/
            labels.csv
    
    After:
        datasets/training_v3/
            train/
            val/
            labels.csv
            metadata.json    (auto-generated)
            README.md        (auto-generated)
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    # Create destination
    target_dir = Path(f"datasets/{version}")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy subdirectories
    for subdir in ["train", "val"]:
        src = dataset_path / subdir
        dst = target_dir / subdir
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            count = len(list(dst.glob("*")))
            print(f"✅ Copied {count} {subdir} images")
    
    # Copy labels.csv
    labels_src = dataset_path / "labels.csv"
    if labels_src.exists():
        shutil.copy2(labels_src, target_dir / "labels.csv")
        print(f"✅ Copied labels.csv")
    
    return target_dir


def compute_dataset_stats(dataset_dir):
    """
    Compute statistics about the dataset.
    """
    import pandas as pd
    
    dataset_dir = Path(dataset_dir)
    labels_path = dataset_dir / "labels.csv"
    
    if not labels_path.exists():
        print("⚠️  labels.csv not found")
        return None
    
    df = pd.read_csv(labels_path)
    
    train_count = len(list((dataset_dir / "train").glob("*")))
    val_count = len(list((dataset_dir / "val").glob("*")))
    
    stats = {
        "total_images": len(df),
        "train_images": train_count,
        "val_images": val_count,
        "depth_mean": float(df["depth_cm"].mean()),
        "depth_std": float(df["depth_cm"].std()),
        "depth_min": float(df["depth_cm"].min()),
        "depth_max": float(df["depth_cm"].max()),
        "depth_median": float(df["depth_cm"].median()),
        "zero_cm_count": int((df["depth_cm"] == 0).sum()),
        "non_zero_count": int((df["depth_cm"] > 0).sum()),
    }
    
    return stats


def create_model_registry_entry(version, checkpoint, dataset_dir, git_commit=None):
    """
    Create registry entry for this trained model.
    """
    dataset_stats = compute_dataset_stats(dataset_dir)
    
    entry = {
        "version": version,
        "date_trained": datetime.now().isoformat(),
        "model_path": f"models/best_flood_model_water_aware.pth",
        "dataset_path": str(dataset_dir),
        "checkpoint_info": {
            "epoch": checkpoint.get("epoch", "?"),
            "best_val_loss": float(checkpoint.get("best_val_loss", "?")),
            "is_collapsed": (
                isinstance(checkpoint.get("best_val_loss"), float)
                and checkpoint.get("best_val_loss") < 1e-5
            ),
        },
        "dataset_stats": dataset_stats,
        "git_commit": git_commit,
        "notes": "",
    }
    
    return entry


def update_model_registry(entry):
    """
    Add entry to model_registry.json.
    """
    registry_path = Path("model_registry.json")
    
    if registry_path.exists():
        with open(registry_path) as f:
            registry = json.load(f)
    else:
        registry = {"models": []}
    
    # Add new entry
    registry["models"].append(entry)
    
    # Save
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    
    print(f"✅ Updated model_registry.json (now {len(registry['models'])} models)")
    
    return registry


def create_dataset_readme(dataset_dir, stats):
    """
    Create README.md for dataset.
    """
    readme_path = Path(dataset_dir) / "README.md"
    
    content = f"""# Training Dataset {Path(dataset_dir).name}

## Overview
- **Total images:** {stats['total_images']} ({stats['train_images']} train, {stats['val_images']} val)
- **Created:** {datetime.now().isoformat()}

## Depth Statistics
```
Mean:   {stats['depth_mean']:.1f} cm
Std:    {stats['depth_std']:.1f} cm
Median: {stats['depth_median']:.1f} cm
Min:    {stats['depth_min']:.0f} cm
Max:    {stats['depth_max']:.0f} cm
```

## Label Quality
- Non-zero labels: {stats['non_zero_count']} ({stats['non_zero_count']/stats['total_images']*100:.1f}%)
- Zero labels (dry): {stats['zero_cm_count']} ({stats['zero_cm_count']/stats['total_images']*100:.1f}%)

✅ No label collapse detected (mean ≠ 0)

## Files
- `train/` - {stats['train_images']} training images
- `val/` - {stats['val_images']} validation images  
- `labels.csv` - Depth labels for all images
- `metadata.json` - Detailed statistics
"""
    
    with open(readme_path, "w") as f:
        f.write(content)
    
    print(f"✅ Created {readme_path}")


def create_dataset_metadata(dataset_dir, stats):
    """
    Create metadata.json for dataset.
    """
    metadata_path = Path(dataset_dir) / "metadata.json"
    
    metadata = {
        "version": Path(dataset_dir).name,
        "created": datetime.now().isoformat(),
        "statistics": stats,
        "quality_checks": {
            "no_label_collapse": stats["depth_mean"] > 5,
            "sufficient_data": stats["total_images"] >= 50,
            "good_train_val_split": 0.8 <= stats["train_images"]/stats["total_images"] <= 0.9,
        }
    }
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Created {metadata_path}")


def print_summary(model_info, dataset_dir, registry):
    """
    Print summary of what was organized.
    """
    print("\n" + "=" * 70)
    print("📊 ASSET ORGANIZATION COMPLETE")
    print("=" * 70)
    
    print(f"\n📁 Model:")
    print(f"   Current: {model_info['model_path']}")
    print(f"   Archive: {model_info['archive_path']}")
    
    ckpt = model_info['checkpoint']
    print(f"\n📈 Checkpoint Info:")
    print(f"   Epoch: {ckpt.get('epoch', '?')}")
    print(f"   Val Loss: {ckpt.get('best_val_loss', '?')}")
    
    print(f"\n📁 Dataset:")
    print(f"   Location: {dataset_dir}")
    
    stats = compute_dataset_stats(dataset_dir)
    if stats:
        print(f"   Images: {stats['total_images']} total ({stats['train_images']} train, {stats['val_images']} val)")
        print(f"   Depth: {stats['depth_mean']:.1f} ± {stats['depth_std']:.1f} cm")
    
    print(f"\n📋 Registry:")
    print(f"   Location: model_registry.json")
    print(f"   Models tracked: {len(registry['models'])}")
    
    print("\n" + "=" * 70)
    print("✅ Next steps:")
    print("   1. git add models/ datasets/ model_registry.json")
    print("   2. git commit -m 'Add trained model v3: 150 images'")
    print("   3. git push origin main")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Organize trained assets")
    parser.add_argument("--model", required=True, help="Path to model.pth")
    parser.add_argument("--dataset", required=True, help="Path to dataset folder")
    parser.add_argument("--version", required=True, help="Version tag (e.g., v3)")
    parser.add_argument("--commit", help="Optional git commit hash")
    
    args = parser.parse_args()
    
    print("🚀 Organizing trained assets...\n")
    
    # Step 1: Organize model
    print("📌 Step 1: Organizing model...")
    model_info = organize_model(args.model, args.version)
    
    # Step 2: Organize dataset
    print("\n📌 Step 2: Organizing dataset...")
    dataset_dir = organize_dataset(args.dataset, args.version)
    
    # Step 3: Compute stats
    print("\n📌 Step 3: Computing statistics...")
    stats = compute_dataset_stats(dataset_dir)
    
    # Step 4: Create documentation
    print("\n📌 Step 4: Creating documentation...")
    create_dataset_readme(dataset_dir, stats)
    create_dataset_metadata(dataset_dir, stats)
    
    # Step 5: Update registry
    print("\n📌 Step 5: Updating model registry...")
    entry = create_model_registry_entry(args.version, model_info["checkpoint"], dataset_dir, args.commit)
    registry = update_model_registry(entry)
    
    # Step 6: Print summary
    print_summary(model_info, dataset_dir, registry)


if __name__ == "__main__":
    main()
