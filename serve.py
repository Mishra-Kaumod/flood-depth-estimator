"""
HIGH-THROUGHPUT INFERENCE ENGINE - AWS ECS on Fargate / AWS Lambda
Production-ready LitServe deployment with dynamic batching, optimized decode/encode,
and complete decoupling from training logic.
"""
import os
import yaml
import logging
import base64
from pathlib import Path
from typing import Dict, Any, List, Tuple
from io import BytesIO

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

try:
    from litserve import LitServer, Request, Response
except ImportError:
    LitServer = None
    Request = None
    Response = None
    logging.warning("LitServe not installed. Install: pip install litserve")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION LOADER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load YAML configuration."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL LOADER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_inference_model(device: torch.device) -> nn.Module:
    """Build clean EfficientNet-B0 backbone (no training artifacts)."""
    model = models.efficientnet_b0(weights=None)
    
    # Replace final layer with regression head
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    
    return model.to(device)

def load_model_weights(model: nn.Module, checkpoint_path: str, device: torch.device):
    """Load pre-trained weights from checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle both direct state_dict and checkpoint dict
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=True)
        logger.info(f"✅ Loaded model weights from {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")
        raise

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LITSERVE API HANDLER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FloodDepthPredictor(LitServer if LitServer else object):
    """
    Production inference server for flood depth estimation.
    Handles image batching, preprocessing, inference, and response encoding.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"🚀 Initializing FloodDepthPredictor on {self.device}")
        
        # Load model
        self.model = build_inference_model(self.device)
        inference_cfg = self.config.get("inference", {})
        model_path = inference_cfg.get("model_path", "models/best_flood_model.pth")
        
        if Path(model_path).exists():
            load_model_weights(self.model, model_path, self.device)
        else:
            logger.warning(f"Model file not found: {model_path}. Using random initialization.")
        
        self.model.eval()
        
        # Image preprocessing
        train_cfg = self.config.get("training", {})
        self.image_size = tuple(train_cfg.get("image_size", [224, 224]))
        norm_cfg = train_cfg.get("normalization", {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        })
        
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=norm_cfg["mean"],
                std=norm_cfg["std"]
            )
        ])
        
        self.inference_cfg = inference_cfg
        logger.info("✅ FloodDepthPredictor ready for inference")
    
    def decode_request(self, request: Any) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Decode incoming batch request to list of (image_id, metadata) tuples.
        
        Expected input format:
        {
            "images": [
                {
                    "id": "image_001",
                    "data": "base64_encoded_image_bytes",
                    "format": "jpg"
                },
                ...
            ]
        }
        """
        try:
            payload = request.json()
            images_list = payload.get("images", [])
            
            decoded = []
            for img_obj in images_list:
                image_id = img_obj.get("id", "unknown")
                image_data = img_obj.get("data", "")
                image_format = img_obj.get("format", "jpg")
                
                try:
                    # Decode base64 image data
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(BytesIO(image_bytes)).convert("RGB")
                    
                    decoded.append((image_id, {
                        "image": image,
                        "format": image_format,
                        "timestamp": img_obj.get("timestamp", "unknown")
                    }))
                except Exception as e:
                    logger.error(f"Failed to decode image {image_id}: {e}")
                    # Return error placeholder
                    decoded.append((image_id, {"error": str(e)}))
            
            return decoded
        except Exception as e:
            logger.error(f"Request decode failed: {e}")
            raise
    
    def predict(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run inference on batch of images.
        
        Args:
            images: List of image dictionaries from decode_request
        
        Returns:
            List of predictions (one per image)
        """
        predictions = []
        
        with torch.no_grad():
            for img_meta in images:
                if "error" in img_meta:
                    predictions.append({
                        "error": img_meta["error"],
                        "depth_cm": None,
                        "confidence": None
                    })
                    continue
                
                try:
                    # Preprocess image
                    image = img_meta["image"]
                    img_tensor = self.transform(image).unsqueeze(0).to(self.device)
                    
                    # Forward pass
                    output = self.model(img_tensor)
                    depth_normalized = output.squeeze().item()
                    
                    # Denormalize depth (0-1) back to cm (0-100)
                    depth_cm = round(depth_normalized * 100.0, 2)
                    
                    # Calculate confidence (higher normalized output = higher confidence)
                    confidence = round(min(depth_normalized * 1.2, 1.0), 3)
                    
                    predictions.append({
                        "depth_cm": depth_cm,
                        "confidence": confidence,
                        "intensity": self._depth_to_intensity(depth_cm),
                        "is_flooded": depth_cm >= 5.0
                    })
                except Exception as e:
                    logger.error(f"Prediction failed: {e}")
                    predictions.append({
                        "error": str(e),
                        "depth_cm": None,
                        "confidence": None
                    })
        
        return predictions
    
    def encode_response(self, predictions: List[Dict[str, Any]], image_ids: List[str]) -> Dict[str, Any]:
        """
        Encode prediction results into JSON response.
        
        Args:
            predictions: List of prediction dicts
            image_ids: List of corresponding image IDs
        
        Returns:
            JSON-serializable response
        """
        results = []
        for img_id, pred in zip(image_ids, predictions):
            result = {
                "image_id": img_id,
                "prediction": pred,
                "status": "error" if "error" in pred else "success"
            }
            results.append(result)
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "summary": {
                "total_images": len(results),
                "successful": sum(1 for r in results if r["status"] == "success"),
                "failed": sum(1 for r in results if r["status"] == "error"),
                "avg_depth_cm": round(
                    np.mean([r["prediction"]["depth_cm"] for r in results 
                            if r["status"] == "success" and r["prediction"]["depth_cm"] is not None]),
                    2
                ) if any(r["status"] == "success" for r in results) else None
            }
        }
    
    @staticmethod
    def _depth_to_intensity(depth_cm: float) -> str:
        """Convert depth to intensity classification."""
        if depth_cm <= 5:
            return "SAFE"
        elif depth_cm <= 20:
            return "MEDIUM"
        elif depth_cm <= 50:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def predict_batch(self, batch: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Main batch prediction method called by LitServe.
        
        Args:
            batch: List of (image_id, metadata) tuples from decode_request
        
        Returns:
            JSON response
        """
        image_ids = [img_id for img_id, _ in batch]
        images_meta = [meta for _, meta in batch]
        
        # Run inference
        predictions = self.predict(images_meta)
        
        # Encode response
        response = self.encode_response(predictions, image_ids)
        return response

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SERVER STARTUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    """Main entry point for LitServe inference server."""
    
    config = load_config("config/config.yaml")
    inference_cfg = config.get("inference", {})
    litserve_cfg = inference_cfg.get("litserve", {})
    
    # Initialize predictor
    predictor = FloodDepthPredictor("config/config.yaml")
    
    # Start LitServe
    if LitServer:
        server = LitServer(
            predictor,
            port=litserve_cfg.get("port", 8000),
            host=litserve_cfg.get("host", "0.0.0.0"),
            max_batch_size=litserve_cfg.get("max_batch_size", 8),
            batch_timeout=litserve_cfg.get("batch_timeout", 0.05),
            workers=litserve_cfg.get("workers", 4),
        )
        server.run()
    else:
        logger.error("LitServe not available. Install: pip install litserve")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HEALTH CHECK & READY PROBE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def health_check() -> Dict[str, Any]:
    """Health check endpoint (for Kubernetes/ECS)."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "flood-depth-estimator",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    from datetime import datetime
    main()
