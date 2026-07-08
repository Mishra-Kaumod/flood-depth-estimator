"""
Ensemble Predictions: Combine multiple model checkpoints for better accuracy
Reduces variance and improves robustness without any training.

Usage:
    python ensemble_predict.py \\
        --models models/checkpoint_1.pth models/checkpoint_2.pth \\
        --image test_image.jpg \\
        --method average
        
Expected Improvement: -5% to -15% MAE (no training!)
"""

import argparse
import json
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
import numpy as np
from glob import glob


class EnsemblePredictor:
    """Combine predictions from multiple models."""
    
    def __init__(self, model_paths, device=None):
        """
        Initialize ensemble predictor.
        
        Args:
            model_paths: List of paths to model checkpoints
            device: torch device (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = []
        
        print(f"\n📦 Loading {len(model_paths)} models...")
        print(f"{'='*70}")
        
        for i, path in enumerate(model_paths, 1):
            try:
                model = self._load_model(path)
                self.models.append(model)
                model_size_mb = Path(path).stat().st_size / (1024*1024)
                print(f"{i}. ✅ {Path(path).name:<40} ({model_size_mb:.1f} MB)")
            except Exception as e:
                print(f"{i}. ❌ {Path(path).name:<40} Error: {e}")
        
        print(f"{'='*70}")
        print(f"Successfully loaded: {len(self.models)} models")
        
        if not self.models:
            raise ValueError("No models loaded successfully!")
    
    def _load_model(self, model_path):
        """Load a single model checkpoint."""
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[-1] = torch.nn.Linear(1280, 1)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def _preprocess(self, image_path):
        """Load and preprocess image."""
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        tensor = transform(img)
        return tensor.unsqueeze(0).to(self.device)
    
    @torch.no_grad()
    def predict_average(self, image_path):
        """Average predictions from all models."""
        tensor = self._preprocess(image_path)
        
        predictions = []
        print(f"\n{'='*70}")
        print(f"Model Predictions (Average Ensemble)")
        print(f"{'='*70}")
        print(f"Image: {Path(image_path).name}\n")
        print(f"{'Model':<15} {'Prediction (cm)':<20} {'Status':<10}")
        print(f"{'-'*70}")
        
        for i, model in enumerate(self.models, 1):
            with torch.no_grad():
                output = model(tensor)
                depth = output.item()
            predictions.append(depth)
            print(f"Model {i:<10} {depth:>15.2f} cm      ✓")
        
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        print(f"{'-'*70}")
        print(f"{'Ensemble (avg)':<15} {mean_pred:>15.2f} cm")
        print(f"Std Deviation: {std_pred:>20.2f} cm")
        print(f"{'='*70}")
        
        return {
            'method': 'average',
            'predictions': predictions,
            'mean': mean_pred,
            'std': std_pred,
            'confidence_lower': mean_pred - std_pred,
            'confidence_upper': mean_pred + std_pred,
            'min': np.min(predictions),
            'max': np.max(predictions)
        }
    
    @torch.no_grad()
    def predict_median(self, image_path):
        """Median predictions from all models (robust to outliers)."""
        tensor = self._preprocess(image_path)
        
        predictions = []
        print(f"\n{'='*70}")
        print(f"Model Predictions (Median Ensemble)")
        print(f"{'='*70}")
        print(f"Image: {Path(image_path).name}\n")
        print(f"{'Model':<15} {'Prediction (cm)':<20} {'Status':<10}")
        print(f"{'-'*70}")
        
        for i, model in enumerate(self.models, 1):
            with torch.no_grad():
                output = model(tensor)
                depth = output.item()
            predictions.append(depth)
            print(f"Model {i:<10} {depth:>15.2f} cm      ✓")
        
        predictions = np.array(predictions)
        median_pred = np.median(predictions)
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        print(f"{'-'*70}")
        print(f"{'Ensemble (median)':<15} {median_pred:>15.2f} cm")
        print(f"Ensemble (mean):  {mean_pred:>20.2f} cm")
        print(f"Std Deviation:    {std_pred:>20.2f} cm")
        print(f"{'='*70}")
        
        return {
            'method': 'median',
            'predictions': predictions.tolist(),
            'median': median_pred,
            'mean': mean_pred,
            'std': std_pred,
            'confidence_lower': median_pred - std_pred,
            'confidence_upper': median_pred + std_pred,
            'min': np.min(predictions),
            'max': np.max(predictions)
        }
    
    @torch.no_grad()
    def predict_weighted(self, image_path, weights=None):
        """Weighted ensemble (good models get higher weight)."""
        tensor = self._preprocess(image_path)
        
        if weights is None:
            # Equal weights by default
            weights = np.ones(len(self.models)) / len(self.models)
        
        if len(weights) != len(self.models):
            raise ValueError(f"Weights length ({len(weights)}) != models ({len(self.models)})")
        
        predictions = []
        print(f"\n{'='*70}")
        print(f"Model Predictions (Weighted Ensemble)")
        print(f"{'='*70}")
        print(f"Image: {Path(image_path).name}\n")
        print(f"{'Model':<15} {'Weight':<12} {'Prediction (cm)':<20} {'Status':<10}")
        print(f"{'-'*70}")
        
        for i, model in enumerate(self.models, 1):
            with torch.no_grad():
                output = model(tensor)
                depth = output.item()
            predictions.append(depth)
            print(f"Model {i:<10} {weights[i-1]:>10.2%}  {depth:>15.2f} cm      ✓")
        
        predictions = np.array(predictions)
        weighted_pred = np.average(predictions, weights=weights)
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        print(f"{'-'*70}")
        print(f"{'Ensemble (weighted)':<15} {weighted_pred:>15.2f} cm")
        print(f"Ensemble (mean):   {mean_pred:>20.2f} cm")
        print(f"Std Deviation:     {std_pred:>20.2f} cm")
        print(f"{'='*70}")
        
        return {
            'method': 'weighted',
            'predictions': predictions.tolist(),
            'weights': weights.tolist(),
            'weighted': weighted_pred,
            'mean': mean_pred,
            'std': std_pred,
            'confidence_lower': weighted_pred - std_pred,
            'confidence_upper': weighted_pred + std_pred,
            'min': np.min(predictions),
            'max': np.max(predictions)
        }
    
    def predict(self, image_path, method='average', weights=None):
        """
        Make ensemble prediction.
        
        Args:
            image_path: Path to test image
            method: 'average', 'median', or 'weighted'
            weights: Weights for weighted ensemble
        """
        if method == 'average':
            return self.predict_average(image_path)
        elif method == 'median':
            return self.predict_median(image_path)
        elif method == 'weighted':
            return self.predict_weighted(image_path, weights)
        else:
            raise ValueError(f"Unknown method: {method}")


def main():
    parser = argparse.ArgumentParser(
        description='Ensemble Predictions from Multiple Models'
    )
    parser.add_argument('--models', type=str, nargs='+', required=True,
                        help='Paths to model checkpoints (glob patterns supported)')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to test image')
    parser.add_argument('--method', type=str, default='average',
                        choices=['average', 'median', 'weighted'],
                        help='Ensemble method')
    parser.add_argument('--weights', type=float, nargs='+', default=None,
                        help='Weights for weighted ensemble')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Expand glob patterns
    all_models = []
    for pattern in args.models:
        matches = glob(pattern)
        if not matches:
            print(f"⚠️  No models found matching: {pattern}")
        else:
            all_models.extend(matches)
    
    if not all_models:
        print("❌ No models found!")
        return
    
    # Validate image
    if not Path(args.image).exists():
        print(f"❌ Image not found: {args.image}")
        return
    
    # Setup device
    device = torch.device(args.device if args.device != 'auto'
                         else ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    print(f"\n🌊 Flood Depth - Ensemble Predictions")
    print(f"Device: {device}")
    print(f"Method: {args.method}")
    
    # Create ensemble
    ensemble = EnsemblePredictor(all_models, device)
    
    # Make prediction
    result = ensemble.predict(args.image, method=args.method, weights=args.weights)
    
    # Save results
    output_file = Path(args.image).stem + f'_ensemble_{args.method}.json'
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    print(f"\n📊 Summary:")
    print(f"  Method: {result['method']}")
    if 'weighted' in result:
        print(f"  Weighted Prediction: {result['weighted']:.2f} cm")
    elif 'median' in result:
        print(f"  Median Prediction: {result['median']:.2f} cm")
    else:
        print(f"  Mean Prediction: {result['mean']:.2f} cm")
    print(f"  Std Deviation: {result['std']:.2f} cm")
    print(f"  Confidence Range: {result['confidence_lower']:.2f} - {result['confidence_upper']:.2f} cm")


if __name__ == '__main__':
    main()
