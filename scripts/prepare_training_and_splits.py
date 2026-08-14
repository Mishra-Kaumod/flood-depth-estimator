#!/usr/bin/env python3
"""
Prepare dataset splits from repo training images and labels for retraining.

Behavior:
 - Reads images from training_data/images and labels from training_data/labels.csv (defaults)
 - Selects the first N images (default 700) in alphabetical order for reproducibility
 - Creates deterministic splits: train_count (default 600), val_count (50), test_count (50)
 - Copies image files into out_dir/images/{train,val,test}
 - Writes labels CSVs: labels_train.csv, labels_val.csv, labels_test.csv and dataset_manifest.csv
 - Emits a small run_train.sh with instructions to run a retrain script/notebook

Usage:
  python scripts/prepare_training_and_splits.py [--images-dir PATH] [--labels CSV] [--n N] \
      [--train TRAIN] [--val VAL] [--test TEST] [--out OUTDIR] [--seed SEED]

This script only prepares the package and does not run model training.
"""

from pathlib import Path
import argparse
import csv
import shutil
import datetime
import sys


def read_labels(labels_csv: Path):
    labels = {}
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")
    with labels_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if 'filename' not in reader.fieldnames or 'depth_cm' not in reader.fieldnames:
            raise ValueError("labels CSV must contain headers 'filename' and 'depth_cm'")
        for row in reader:
            labels[row['filename']] = row['depth_cm']
    return labels


def write_csv(out_path: Path, rows):
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "depth_cm"])
        for fn, depth in rows:
            writer.writerow([fn, depth])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', default='training_data/images', help='Path to images folder (repo-relative)')
    parser.add_argument('--labels', default='training_data/labels.csv', help='CSV with filename,depth_cm')
    parser.add_argument('--n', type=int, default=700, help='Number of images to select (first N alphabetically)')
    parser.add_argument('--train', type=int, default=600, help='Number of training images')
    parser.add_argument('--val', type=int, default=50, help='Number of validation images')
    parser.add_argument('--test', type=int, default=50, help='Number of test images')
    parser.add_argument('--out', default=None, help='Output base dir (default: training_runs/<timestamp>)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (used only for deterministic shuffling if enabled)')
    parser.add_argument('--copy', action='store_true', help='Copy files instead of creating symlinks (Windows uses copy by default)')
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    labels_csv = Path(args.labels)

    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        sys.exit(1)

    try:
        labels_map = read_labels(labels_csv)
    except Exception as e:
        print(f"Failed to read labels: {e}")
        sys.exit(1)

    all_images = sorted([p.name for p in images_dir.iterdir() if p.is_file()])
    if len(all_images) == 0:
        print(f"No images found in {images_dir}")
        sys.exit(1)

    N = min(args.n, len(all_images))
    selected = all_images[:N]

    requested_total = args.train + args.val + args.test
    if requested_total > N:
        print(f"Requested splits ({requested_total}) > selected images ({N}). Adjusting test count to fit.")
        remaining = N - args.train - args.val
        if remaining < 0:
            print("Train+Val counts exceed N; reducing train to fit.")
            # reduce train first
            new_train = max(0, N - args.val - args.test)
            args.train = new_train
            remaining = N - args.train - args.val
        args.test = remaining

    train_list = selected[:args.train]
    val_list = selected[args.train:args.train+args.val]
    test_list = selected[args.train+args.val:args.train+args.val+args.test]

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_base = Path(args.out) if args.out else Path('training_runs') / timestamp
    out_base.mkdir(parents=True, exist_ok=True)

    images_out = out_base / 'images'
    (images_out / 'train').mkdir(parents=True, exist_ok=True)
    (images_out / 'val').mkdir(parents=True, exist_ok=True)
    (images_out / 'test').mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    def copy_or_link(src: Path, dst: Path):
        try:
            # On Windows, prefer copy to avoid permissions/symlink issues
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"Failed to copy {src} -> {dst}: {e}")

    # Helper to populate subset
    for subset_name, lst in [('train', train_list), ('val', val_list), ('test', test_list)]:
        dst_dir = images_out / subset_name
        for fn in lst:
            src = images_dir / fn
            dst = dst_dir / fn
            if not src.exists():
                print(f"Warning: image not found: {src}")
                depth = labels_map.get(fn, '')
            else:
                depth = labels_map.get(fn, '')
                copy_or_link(src, dst)
            manifest_rows.append((fn, depth, subset_name, str(src)))

    # Append any selected images that were skipped (shouldn't happen)
    # Also include any CSV rows referencing files not present (kept aside)

    # Write split CSVs
    train_rows = [(fn, labels_map.get(fn, '')) for fn in train_list]
    val_rows = [(fn, labels_map.get(fn, '')) for fn in val_list]
    test_rows = [(fn, labels_map.get(fn, '')) for fn in test_list]

    write_csv(out_base / 'labels_train.csv', train_rows)
    write_csv(out_base / 'labels_val.csv', val_rows)
    write_csv(out_base / 'labels_test.csv', test_rows)

    # Write dataset_manifest.csv
    with (out_base / 'dataset_manifest.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'depth_cm', 'split', 'src_path'])
        for r in manifest_rows:
            writer.writerow(r)

    # Create a small run_train.sh (and .ps1) with instructions
    run_sh = out_base / 'run_train.sh'
    run_ps1 = out_base / 'run_train.ps1'
    train_cmd = "# Replace with your preferred training command. Example: \n# python scripts/retrain_flood_classifier.py --train-images {train_dir} --train-labels {labels_train} --val-labels {labels_val} --model-path models/flood_model_final.pth --output models/candidate/\n".format(
        train_dir=str(images_out / 'train'), labels_train=str(out_base / 'labels_train.csv'), labels_val=str(out_base / 'labels_val.csv'))
    run_sh.write_text("""#!/bin/sh
# Training run placeholder
{cmd}
""".format(cmd=train_cmd))
    run_ps1.write_text("""# PowerShell training run placeholder
{cmd}
""".format(cmd=train_cmd))

    # Summary
    print("Prepared training package:")
    print(f"  out_dir: {out_base}")
    print(f"  train: {len(train_list)} images")
    print(f"  val: {len(val_list)} images")
    print(f"  test: {len(test_list)} images")
    print(f"  manifest: {out_base / 'dataset_manifest.csv'}")
    print('\nNext steps:')
    print(' - Transfer the out_dir to Colab or run your local training tool with the provided labels_train/labels_val/labels_test files.')
    print(' - Example training command has been written to run_train.sh and run_train.ps1. Edit to point to your retrain script or notebook.')


if __name__ == '__main__':
    main()
