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
    def __init__(self, use_hsv=True, use_rgb=True, use_contrast=True, min_water_area_ratio=0.01):
        self.use_hsv = use_hsv
        self.use_rgb = use_rgb
        self.use_contrast = use_contrast
        self.min_water_area_ratio = min_water_area_ratio

    def detect(self, image):
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if self.use_hsv:
            mask = np.maximum(mask, self._detect_hsv(image))
        if self.use_rgb:
            mask = np.maximum(mask, self._detect_rgb(image))
        if self.use_contrast:
            mask = np.maximum(mask, self._detect_flatness(image))

        mask = self._morphological_cleanup(mask)

        # Position prior (applied AFTER morphology so close/open can't undo it):
        # Hard-zero the top 30% of the image — that region is nearly always sky/rooftop,
        # not flood water. Ramp-fade rows 30-45% to soften the boundary.
        hard_zero = int(h * 0.30)
        ramp_end  = int(h * 0.45)
        mask[:hard_zero, :] = 0
        if ramp_end > hard_zero:
            ramp = np.linspace(0.0, 1.0, ramp_end - hard_zero, dtype=np.float32)
            mask[hard_zero:ramp_end, :] = np.clip(
                mask[hard_zero:ramp_end, :].astype(np.float32) * ramp[:, np.newaxis], 0, 255
            ).astype(np.uint8)
        water_pct = float(np.sum(mask > 0)) / float(h * w) * 100.0
        return mask, water_pct

    def _detect_hsv(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h_, s_, v_ = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        # Clear / blue-green water
        clear = ((h_ >= 85) & (h_ <= 140) & (s_ >= 40) & (v_ >= 40)).astype(np.uint8) * 255
        # Murky brown (most Bengaluru floods): hue 0-25 or 160-180
        muddy_h = (h_ <= 25) | (h_ >= 160)
        muddy = (muddy_h & (s_ >= 15) & (s_ <= 190) & (v_ >= 45) & (v_ <= 225)).astype(np.uint8) * 255
        # Grey/dark stagnant water: only when saturation is very low AND
        # value is mid-range (distinguishes from sky and very bright surfaces)
        grey = ((s_ < 30) & (v_ >= 50) & (v_ <= 160)).astype(np.uint8) * 255
        return np.maximum(np.maximum(clear, muddy), grey)

    def _detect_rgb(self, image):
        r = image[:,:,0].astype(np.float32)
        g = image[:,:,1].astype(np.float32)
        b = image[:,:,2].astype(np.float32)
        bright = (r + g + b) / 3.0
        # Clear blue water
        blue = ((b > g) & (b > r) & ((b - r) > 15) & (b > 30)).astype(np.uint8) * 255
        # Turbid brown/orange water
        brown = ((r > b) & (g > b) & (r > 40) & (r < 215) & ((r - b) > 10) & ((r - b) < 110) & (bright < 205) & (bright > 28)).astype(np.uint8) * 255
        # Dark stagnant water: neutral dark pixels that are NOT too uniform
        # (add minimum brightness floor to skip near-black non-water)
        dark = ((bright > 35) & (bright < 110) & (np.abs(r - g) < 25) & (np.abs(g - b) < 25) & (np.abs(r - b) < 25)).astype(np.uint8) * 255
        return np.maximum(np.maximum(blue, brown), dark)

    def _detect_flatness(self, image):
        """Low-texture regions in the bottom half are likely water surface.
        Requires the image to have meaningful texture variation before flagging."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        h = gray.shape[0]
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        tex = cv2.GaussianBlur(lap, (11, 11), 0)
        global_tex_std = float(np.std(tex))
        # Only apply when image has meaningful texture variation
        # (pure-colour synthetic images would otherwise flag everywhere)
        if global_tex_std < 1.5:
            return np.zeros(gray.shape, dtype=np.uint8)
        thresh = float(np.percentile(tex, 30))
        # Require minimum absolute texture to be counted as flat-water
        min_abs_thresh = max(thresh, 1.0)
        flat = (tex <= min_abs_thresh).astype(np.float32)
        # Only keep bottom 60% of image (water on ground, not sky / upper scene)
        flat[:int(h * 0.40), :] = 0.0
        return (flat * 255).astype(np.uint8)

    def _morphological_cleanup(self, mask, kernel_size=7):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        return mask

    def get_water_bounding_boxes(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3] > 100]


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
