#!/usr/bin/env python3
"""
Download FloodNet dataset from Hugging Face and generate inventory report.

Requirements:
- Downloads FloodNet from torchgeo/floodnet on Hugging Face
- Saves raw files to datasets/floodnet/raw
- Verifies image-mask pairs
- Generates inventory report to datasets/floodnet/raw_inventory.csv
"""

import os
import sys
import csv
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
from PIL import Image
from datasets import load_dataset


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FloodNetDownloader:
    """Download and inventory FloodNet dataset from Hugging Face."""

    def __init__(self, output_base: Path = Path("datasets/floodnet")):
        """
        Initialize downloader.

        Args:
            output_base: Base output directory (default: datasets/floodnet)
        """
        self.output_base = Path(output_base)
        self.raw_dir = self.output_base / "raw"
        self.images_dir = self.raw_dir / "images"
        self.masks_dir = self.raw_dir / "masks"
        self.inventory_file = self.output_base / "raw_inventory.csv"
        
        self.inventory: List[Dict] = []
        self.stats = {
            "total_downloaded": 0,
            "valid_pairs": 0,
            "invalid_pairs": 0,
            "size_mismatches": 0,
            "errors": []
        }

    def setup_directories(self) -> None:
        """Create output directories."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directories ready: {self.raw_dir}")

    def download_dataset(self) -> None:
        """Download FloodNet dataset from Hugging Face."""
        logger.info("Loading FloodNet dataset from Hugging Face (torchgeo/floodnet)...")
        try:
            dataset = load_dataset("torchgeo/floodnet", split="train")
            logger.info(f"Loaded dataset with {len(dataset)} samples")
            self._process_dataset(dataset)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def _process_dataset(self, dataset) -> None:
        """
        Process and save dataset samples.

        Args:
            dataset: Loaded Hugging Face dataset
        """
        logger.info("Processing and saving samples...")
        for idx, sample in enumerate(dataset):
            try:
                self._save_sample(idx, sample)
                self.stats["total_downloaded"] += 1
                if (idx + 1) % 100 == 0:
                    logger.info(f"Processed {idx + 1} samples...")
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                self.stats["errors"].append({"sample_idx": idx, "error": str(e)})

    def _save_sample(self, idx: int, sample: Dict) -> None:
        """
        Save a single sample (image and mask).

        Args:
            idx: Sample index
            sample: Dataset sample dictionary
        """
        # Extract image and mask from sample
        if "image" not in sample or "mask" not in sample:
            raise ValueError(f"Sample {idx} missing 'image' or 'mask' key")

        image_pil = sample["image"]
        mask_pil = sample["mask"]

        if not isinstance(image_pil, Image.Image):
            image_pil = Image.fromarray(image_pil)
        if not isinstance(mask_pil, Image.Image):
            mask_pil = Image.fromarray(mask_pil)

        # Save image
        image_name = f"floodnet_{idx:04d}.png"
        image_path = self.images_dir / image_name
        image_pil.save(image_path)

        # Save mask
        mask_name = f"floodnet_{idx:04d}_mask.png"
        mask_path = self.masks_dir / mask_name
        mask_pil.save(mask_path)

    def verify_pairs(self) -> None:
        """Verify that all image-mask pairs exist and are compatible."""
        logger.info("Verifying image-mask pairs...")
        
        image_files = sorted(self.images_dir.glob("*.png"))
        logger.info(f"Found {len(image_files)} image files")
        
        for image_file in image_files:
            try:
                # Extract base name (remove .png)
                base_name = image_file.stem
                mask_file = self.masks_dir / f"{base_name}_mask.png"
                
                # Check if mask exists
                if not mask_file.exists():
                    self.stats["invalid_pairs"] += 1
                    record = {
                        "image_file": image_file.name,
                        "mask_file": mask_file.name,
                        "image_path": str(image_file),
                        "mask_path": str(mask_file),
                        "image_shape": None,
                        "mask_shape": None,
                        "image_dtype": None,
                        "mask_dtype": None,
                        "size_match": False,
                        "status": "MISSING_MASK",
                        "notes": f"Mask file not found: {mask_file.name}"
                    }
                    self.inventory.append(record)
                    continue
                
                # Load and verify compatibility
                image = Image.open(image_file)
                mask = Image.open(mask_file)
                
                image_shape = image.size  # (width, height)
                mask_shape = mask.size    # (width, height)
                
                image_arr = np.array(image)
                mask_arr = np.array(mask)
                
                # Check size match
                size_match = image_shape == mask_shape
                
                record = {
                    "image_file": image_file.name,
                    "mask_file": mask_file.name,
                    "image_path": str(image_file),
                    "mask_path": str(mask_file),
                    "image_shape": f"{image_shape[0]}x{image_shape[1]}",
                    "mask_shape": f"{mask_shape[0]}x{mask_shape[1]}",
                    "image_dtype": str(image_arr.dtype),
                    "mask_dtype": str(mask_arr.dtype),
                    "image_channels": image_arr.ndim,
                    "mask_channels": mask_arr.ndim,
                    "size_match": size_match,
                    "status": "VALID" if size_match else "SIZE_MISMATCH",
                    "notes": ""
                }
                
                if size_match:
                    self.stats["valid_pairs"] += 1
                else:
                    self.stats["size_mismatches"] += 1
                    record["notes"] = f"Image {image_shape} != Mask {mask_shape}"
                
                self.inventory.append(record)
                
            except Exception as e:
                self.stats["errors"].append({
                    "image_file": image_file.name,
                    "error": str(e)
                })
                record = {
                    "image_file": image_file.name,
                    "mask_file": "unknown",
                    "image_path": str(image_file),
                    "mask_path": "unknown",
                    "image_shape": None,
                    "mask_shape": None,
                    "image_dtype": None,
                    "mask_dtype": None,
                    "size_match": False,
                    "status": "ERROR",
                    "notes": str(e)
                }
                self.inventory.append(record)

    def generate_inventory_report(self) -> None:
        """Generate and save inventory CSV report."""
        logger.info(f"Generating inventory report: {self.inventory_file}")
        
        if not self.inventory:
            logger.warning("No inventory data to write")
            return
        
        # Prepare output directory
        self.inventory_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write CSV
        fieldnames = [
            "image_file",
            "mask_file",
            "image_path",
            "mask_path",
            "image_shape",
            "mask_shape",
            "image_dtype",
            "mask_dtype",
            "image_channels",
            "mask_channels",
            "size_match",
            "status",
            "notes"
        ]
        
        with open(self.inventory_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.inventory:
                writer.writerow(record)
        
        logger.info(f"Inventory saved to {self.inventory_file}")

    def print_summary(self) -> None:
        """Print download and verification summary."""
        logger.info("\n" + "="*60)
        logger.info("FLOODNET DOWNLOAD SUMMARY")
        logger.info("="*60)
        logger.info(f"Output directory:    {self.raw_dir}")
        logger.info(f"Total downloaded:    {self.stats['total_downloaded']}")
        logger.info(f"Valid pairs:         {self.stats['valid_pairs']}")
        logger.info(f"Invalid pairs:       {self.stats['invalid_pairs']}")
        logger.info(f"Size mismatches:     {self.stats['size_mismatches']}")
        logger.info(f"Processing errors:   {len(self.stats['errors'])}")
        logger.info(f"Inventory report:    {self.inventory_file}")
        logger.info("="*60 + "\n")

    def run(self, verify_only: bool = False) -> None:
        """
        Execute full download and verification pipeline.

        Args:
            verify_only: If True, only verify existing pairs (skip download)
        """
        try:
            self.setup_directories()
            
            if not verify_only:
                self.download_dataset()
            else:
                logger.info("Skipping download (verify_only=True)")
            
            self.verify_pairs()
            self.generate_inventory_report()
            self.print_summary()
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download FloodNet dataset from Hugging Face and generate inventory"
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="datasets/floodnet",
        help="Base output directory (default: datasets/floodnet)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing pairs, skip download"
    )
    
    args = parser.parse_args()
    
    downloader = FloodNetDownloader(output_base=Path(args.output_base))
    downloader.run(verify_only=args.verify_only)


if __name__ == "__main__":
    main()
