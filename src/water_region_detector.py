"""
WATER REGION DETECTION & PREPROCESSING
Identifies water regions in flood images before depth estimation.
Essential for handling partially flooded images (one side flooded, other dry).
"""
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class WaterRegionDetector:
    """
    Detects water regions in flood images using multiple methods:
    1. Color-based detection (blue/cyan channels)
    2. Contrast-based detection (water has different reflectance)
    3. Morphological operations (clean up noise)
    """
    
    def __init__(
        self,
        use_hsv: bool = True,
        use_rgb: bool = True,
        use_contrast: bool = True,
        min_water_area_ratio: float = 0.01,  # At least 1% of image is water
    ):
        """
        Initialize water detector.
        
        Args:
            use_hsv: Use HSV color space detection
            use_rgb: Use RGB channel analysis
            use_contrast: Use contrast-based detection
            min_water_area_ratio: Minimum percentage of image that must be water
        """
        self.use_hsv = use_hsv
        self.use_rgb = use_rgb
        self.use_contrast = use_contrast
        self.min_water_area_ratio = min_water_area_ratio
    
    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect water regions in image.
        
        Args:
            image: RGB image (H, W, 3) with values 0-255
        
        Returns:
            water_mask: Binary mask (H, W) where 1=water, 0=not water
            water_coverage: Percentage of image that is water (0-100)
        """
        h, w = image.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Method 1: HSV-based detection
        if self.use_hsv:
            hsv_mask = self._detect_hsv(image)
            combined_mask = np.maximum(combined_mask, hsv_mask)
        
        # Method 2: RGB channel detection
        if self.use_rgb:
            rgb_mask = self._detect_rgb(image)
            combined_mask = np.maximum(combined_mask, rgb_mask)
        
        # Method 3: Contrast-based detection
        if self.use_contrast:
            contrast_mask = self._detect_contrast(image)
            combined_mask = np.maximum(combined_mask, contrast_mask)
        
        # Morphological cleanup
        combined_mask = self._morphological_cleanup(combined_mask)
        
        # Calculate coverage
        water_pixels = np.sum(combined_mask > 0)
        total_pixels = h * w
        water_coverage = (water_pixels / total_pixels) * 100
        
        return combined_mask, water_coverage
    
    def _detect_hsv(self, image: np.ndarray) -> np.ndarray:
        """Detect water using HSV color space."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Water typically has:
        # - Hue: 90-180 (cyan to blue range)
        # - Saturation: 50-255 (saturated)
        # - Value: 50-255 (visible)
        
        lower_water = np.array([80, 50, 50])
        upper_water = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_water, upper_water)
        return mask
    
    def _detect_rgb(self, image: np.ndarray) -> np.ndarray:
        """Detect water using RGB channels."""
        r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
        
        # Water typically has:
        # - Blue > Green > Red (blue-dominant)
        # - High blue, medium green, low red
        
        # Condition 1: Blue channel dominant
        blue_dominant = (b > g) & (b > r)
        
        # Condition 2: Not too dark (avoid shadows)
        not_dark = (b > 30)
        
        # Condition 3: Blue-green difference (water has strong blue channel)
        blue_green_diff = (b - g) > 20
        
        mask = (blue_dominant & not_dark & blue_green_diff).astype(np.uint8) * 255
        return mask
    
    def _detect_contrast(self, image: np.ndarray) -> np.ndarray:
        """Detect water using contrast characteristics."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Water typically has lower contrast (smooth reflections)
        # Calculate local contrast
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        
        # Local mean
        local_mean = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Local std (approximated)
        contrast = np.abs(gray.astype(float) - local_mean.astype(float))
        
        # Low contrast regions are likely water
        threshold = np.percentile(contrast, 25)  # Bottom 25% contrast
        mask = (contrast < threshold).astype(np.uint8) * 255
        
        return mask
    
    def _morphological_cleanup(self, mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Clean up mask using morphological operations."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Close small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Open small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return mask
    
    def get_water_bounding_boxes(self, mask: np.ndarray) -> list:
        """
        Get bounding boxes of water regions.
        Useful for visualizing and processing distinct water areas.
        
        Returns:
            List of (x, y, w, h) tuples for each water region
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Only keep significant regions
            if area > 100:  # At least 100 pixels
                bboxes.append((x, y, w, h))
        
        return bboxes


class RegionBasedDataLoader:
    """
    Wraps a standard DataLoader to provide:
    1. Water mask alongside image
    2. Only calculate depth for water regions
    3. Masking of non-water regions during loss calculation
    """
    
    def __init__(self, dataloader, detector: Optional[WaterRegionDetector] = None):
        self.dataloader = dataloader
        self.detector = detector or WaterRegionDetector()
    
    def __iter__(self):
        for batch in self.dataloader:
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                images, depths = batch
                
                # Detect water regions
                water_masks = []
                water_coverages = []
                
                for i in range(images.shape[0]):
                    # Convert from tensor to numpy (undo normalization for detection)
                    img_np = self._tensor_to_image(images[i])
                    
                    # Detect water
                    mask, coverage = self.detector.detect(img_np)
                    water_masks.append(torch.from_numpy(mask).float())
                    water_coverages.append(coverage)
                
                # Stack masks
                water_masks = torch.stack(water_masks).unsqueeze(1)  # (B, 1, H, W)
                
                # Return augmented batch
                yield {
                    'images': images,
                    'depths': depths,
                    'water_masks': water_masks,
                    'water_coverages': water_coverages
                }
            else:
                yield batch
    
    def __len__(self):
        return len(self.dataloader)
    
    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert normalized tensor to uint8 RGB image."""
        # Assume normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        img = tensor.cpu().numpy().transpose(1, 2, 0)
        
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img * std + mean) * 255
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        return img


class RegionAwareTrainer:
    """
    Training wrapper that:
    1. Detects water regions
    2. Only calculates loss on water regions
    3. Focuses model learning on actual flooded areas
    """
    
    def __init__(self, base_trainer, detector: Optional[WaterRegionDetector] = None):
        self.base_trainer = base_trainer
        self.detector = detector or WaterRegionDetector()
    
    def train_epoch_with_region_awareness(self, train_loader):
        """
        Train for one epoch, focusing on water regions only.
        
        This ensures:
        - Model learns depth ONLY from flooded areas
        - Partially flooded images (one side water, one dry) work correctly
        - No training bias from non-water regions
        """
        self.base_trainer.model.train()
        total_loss = 0.0
        total_water_samples = 0
        region_stats = {'high_coverage': 0, 'medium_coverage': 0, 'low_coverage': 0}
        
        from tqdm import tqdm
        pbar = tqdm(train_loader, desc="Training (Water-Aware)")
        
        for batch in pbar:
            images = batch['images'].to(self.base_trainer.device)
            depths = batch['depths'].to(self.base_trainer.device).unsqueeze(1)
            water_masks = batch['water_masks'].to(self.base_trainer.device)
            water_coverages = batch['water_coverages']
            
            # Forward pass
            self.base_trainer.optimizer.zero_grad()
            outputs = self.base_trainer.model(images)
            
            # Apply water mask: only calculate loss for water regions
            # This prevents model from learning meaningless patterns in dry areas
            masked_outputs = outputs * water_masks
            masked_depths = depths * water_masks
            
            # Custom loss that accounts for masking
            loss = self._masked_loss(
                masked_outputs, 
                masked_depths, 
                water_masks,
                depths  # For statistical tracking
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.base_trainer.model.parameters(), max_norm=1.0)
            self.base_trainer.optimizer.step()
            
            if self.base_trainer.onecycle_scheduler is not None:
                self.base_trainer.onecycle_scheduler.step()
            
            total_loss += loss.item()
            
            # Track water coverage distribution
            for coverage in water_coverages:
                if coverage > 70:
                    region_stats['high_coverage'] += 1
                elif coverage > 30:
                    region_stats['medium_coverage'] += 1
                else:
                    region_stats['low_coverage'] += 1
            
            total_water_samples += len(water_coverages)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'high_water': f"{region_stats['high_coverage']}/{total_water_samples}"
            })
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Water Coverage Distribution:")
        logger.info(f"  High (>70%): {region_stats['high_coverage']}")
        logger.info(f"  Medium (30-70%): {region_stats['medium_coverage']}")
        logger.info(f"  Low (<30%): {region_stats['low_coverage']}")
        
        return avg_loss
    
    def _masked_loss(self, outputs, depths, masks, full_depths):
        """
        Calculate loss only on water regions.
        
        This prevents:
        - Learning from non-water areas
        - Biasing toward images with low water coverage
        """
        # Count valid (water) pixels
        valid_pixels = masks.sum()
        
        if valid_pixels == 0:
            # No water in batch - use small penalty
            return torch.tensor(0.01, device=outputs.device, requires_grad=True)
        
        # Calculate loss only where mask > 0
        masked_loss = self.base_trainer.criterion(outputs * masks, depths * masks)
        
        # Normalize by number of water pixels
        # This prevents batches with little water from having disproportionate loss
        normalized_loss = masked_loss * (masks.shape[0] * masks.shape[1] * masks.shape[2] * masks.shape[3]) / valid_pixels
        
        return normalized_loss


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXAMPLE USAGE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    """
    Example: Using water region detection for training
    """
    
    # Create detector
    detector = WaterRegionDetector(
        use_hsv=True,
        use_rgb=True,
        use_contrast=True
    )
    
    # Example: Load an image and detect water
    # image = cv2.imread("path/to/flood_image.jpg")
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # water_mask, coverage = detector.detect(image)
    # print(f"Water coverage: {coverage:.1f}%")
    
    # Visualize
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # axes[0].imshow(image)
    # axes[0].set_title("Original Image")
    # axes[1].imshow(water_mask, cmap='gray')
    # axes[1].set_title(f"Water Mask ({coverage:.1f}%)")
    # axes[2].imshow(image * water_mask[:,:,None] / 255)
    # axes[2].set_title("Water Region Only")
    # plt.show()
    
    logger.info("Water region detection ready for integration with training!")
