#!/usr/bin/env python3
"""
Download a FloodNet subset using streaming mode and generate an inventory report.

Requirements:
- Download only the first 200 FloodNet samples
- Do not cache the full dataset
- Save image-mask pairs to datasets/floodnet/raw
- Verify image-mask dimensions
- Generate inventory.csv
- Print total pairs downloaded
"""

import argparse
import csv
import logging
import tempfile
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
from datasets import DownloadConfig, load_dataset
from PIL import Image


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FloodNetSubsetDownloader:
    def __init__(self, output_base: Path = Path("datasets/floodnet"), limit: int = 200):
        self.output_base = Path(output_base)
        self.limit = limit
        self.raw_dir = self.output_base / "raw"
        self.images_dir = self.raw_dir / "images"
        self.masks_dir = self.raw_dir / "masks"
        self.inventory_file = self.output_base / "raw_inventory.csv"

        self.inventory: list[Dict] = []
        self.stats = {
            "total_downloaded": 0,
            "valid_pairs": 0,
            "size_mismatches": 0,
            "missing_masks": 0,
            "errors": 0,
        }

    def setup_directories(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Prepared directories under {self.raw_dir}")

    def download_subset(self) -> None:
        logger.info("Starting streaming download of FloodNet subset...")
        with tempfile.TemporaryDirectory() as temp_cache_dir:
            dataset = load_dataset(
                "torchgeo/floodnet",
                split="train",
                streaming=True,
                cache_dir=temp_cache_dir,
                download_config=DownloadConfig(use_etag=False)
            )
            for idx, sample in enumerate(dataset):
                if idx >= self.limit:
                    break
                try:
                    self._save_sample(idx, sample)
                    self.stats["total_downloaded"] += 1
                    if (idx + 1) % 50 == 0:
                        logger.info(f"Downloaded {idx + 1} samples")
                except Exception as exc:
                    logger.warning(f"Failed to process sample {idx}: {exc}")
                    self.stats["errors"] += 1

    def _save_sample(self, idx: int, sample: Dict) -> None:
        if "image" not in sample or "mask" not in sample:
            raise ValueError("Sample is missing 'image' or 'mask' fields")

        image = sample["image"]
        mask = sample["mask"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))
        if not isinstance(mask, Image.Image):
            mask = Image.fromarray(np.array(mask))

        image_path = self.images_dir / f"floodnet_{idx:04d}.png"
        mask_path = self.masks_dir / f"floodnet_{idx:04d}_mask.png"
        image.save(image_path)
        mask.save(mask_path)

        image_shape = image.size
        mask_shape = mask.size
        size_match = image_shape == mask_shape

        record = {
            "image_file": image_path.name,
            "mask_file": mask_path.name,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "image_shape": f"{image_shape[0]}x{image_shape[1]}",
            "mask_shape": f"{mask_shape[0]}x{mask_shape[1]}",
            "image_dtype": str(np.array(image).dtype),
            "mask_dtype": str(np.array(mask).dtype),
            "size_match": size_match,
            "status": "VALID" if size_match else "SIZE_MISMATCH",
            "notes": "" if size_match else f"Image {image_shape} != Mask {mask_shape}",
        }

        if size_match:
            self.stats["valid_pairs"] += 1
        else:
            self.stats["size_mismatches"] += 1

        self.inventory.append(record)

    def generate_inventory(self) -> None:
        logger.info(f"Writing inventory to {self.inventory_file}")
        fieldnames = [
            "image_file",
            "mask_file",
            "image_path",
            "mask_path",
            "image_shape",
            "mask_shape",
            "image_dtype",
            "mask_dtype",
            "size_match",
            "status",
            "notes",
        ]
        self.inventory_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.inventory_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.inventory:
                writer.writerow(row)

    def print_summary(self) -> None:
        logger.info("Download complete")
        logger.info(f"Total pairs downloaded: {self.stats['total_downloaded']}")
        logger.info(f"Valid pairs: {self.stats['valid_pairs']}")
        logger.info(f"Size mismatches: {self.stats['size_mismatches']}")
        logger.info(f"Errors: {self.stats['errors']}")
        print(f"Total pairs downloaded: {self.stats['total_downloaded']}")

    def run(self) -> None:
        self.setup_directories()
        self.download_subset()
        self.generate_inventory()
        self.print_summary()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the first FloodNet samples in streaming mode and generate inventory."
    )
    parser.add_argument(
        "--output-base",
        default="datasets/floodnet",
        help="Base output directory for raw files and inventory (default: datasets/floodnet)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Number of FloodNet samples to download (default: 200)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    downloader = FloodNetSubsetDownloader(output_base=Path(args.output_base), limit=args.limit)
    downloader.run()


if __name__ == "__main__":
    main()
