#!/usr/bin/env python3
"""Inspect the FloodNet dataset schema without downloading the full dataset."""

import argparse
from pathlib import Path
from typing import Any

from datasets import DownloadConfig, load_dataset


def format_value(value: Any) -> str:
    try:
        return repr(value)
    except Exception:
        return str(type(value))


def print_sample_info(sample: dict) -> None:
    print("\n=== First sample summary ===")
    for key, value in sample.items():
        print(f"{key}: {type(value).__name__}")

    print("\nField names:")
    for key in sample.keys():
        print(f"- {key}")

    print("\nPython types:")
    for key, value in sample.items():
        print(f"- {key}: {type(value).__name__}")

    if "image" in sample:
        image_value = sample["image"]
        print("\nImage metadata:")
        try:
            from PIL import Image
            if hasattr(image_value, "size") and hasattr(image_value, "mode"):
                print(f"- PIL Image size: {image_value.size}")
                print(f"- PIL Image mode: {image_value.mode}")
                print(f"- PIL Image format: {image_value.format}")
            else:
                image_array = image_value
                try:
                    import numpy as np
                    arr = np.array(image_array)
                    print(f"- ndarray shape: {arr.shape}")
                    print(f"- ndarray dtype: {arr.dtype}")
                except Exception:
                    print(f"- image object type: {type(image_value)}")
        except ImportError:
            print("- PIL not available; unable to inspect image metadata")
    else:
        print("\nNo 'image' field present in the first sample.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect FloodNet schema in streaming mode")
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of samples to inspect (default: 1)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to inspect (default: train)",
    )
    args = parser.parse_args()

    print("Connecting to FloodNet in streaming mode...")
    dataset = load_dataset(
        "torchgeo/floodnet",
        split=args.split,
        streaming=True,
        download_config=DownloadConfig(use_etag=False),
    )

    print(f"Loaded streaming dataset for split '{args.split}'")
    sample = None
    for idx, item in enumerate(dataset):
        if idx >= args.limit:
            break
        sample = item
        break

    if sample is None:
        print("No samples were retrieved from the dataset.")
        return

    print_sample_info(sample)


if __name__ == "__main__":
    main()
