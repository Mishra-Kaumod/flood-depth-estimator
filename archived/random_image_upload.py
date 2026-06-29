"""
Random Image Upload System with HTML Report Generation
Supports uploading random images and generating 11-page HTML report:
- Page 1: Summary with overall statistics
- Pages 2-11: Side-by-side image + analysis (10 images)
"""

import os
import json
import base64
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any
import hashlib

from PIL import Image
import numpy as np

# Model imports (adjust based on your setup)
try:
    import torch
    import torchvision
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class RandomImageUploadProcessor:
    """Process random uploaded images and generate analysis report"""
    
    def __init__(self, upload_dir: str = "uploaded_images", model_path: str = None):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
        
        self.model = None
        self.model_path = model_path or Path(__file__).parent / "models" / "flood_model_final.pth"
        
        if TORCH_AVAILABLE and self.model_path.exists():
            self._load_model()
    
    def _load_model(self):
        """Load flood detection model"""
        try:
            self.model = torchvision.models.segmentation.deeplabv3_resnet101(num_classes=2)
            ckpt = torch.load(self.model_path, map_location="cpu")
            self.model.load_state_dict(ckpt.get("model_state_dict", ckpt))
            self.model.eval()
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            self.model = None
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process single image and generate analysis"""
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        
        # Generate mock analysis (replace with actual model inference)
        analysis = self._generate_analysis(image_array)
        
        return {
            'filename': Path(image_path).name,
            'size_mb': os.path.getsize(image_path) / (1024 * 1024),
            'dimensions': image.size,
            'is_flood': analysis['is_flood'],
            'confidence': analysis['confidence'],
            'depth_cm': analysis['depth_cm'],
            'intensity': analysis['intensity'],
            'color_code': analysis['color_code'],
            'brightness': analysis['brightness'],
            'contrast': analysis['contrast'],
            'edge_count': analysis['edge_count'],
            'water_pixels': analysis['water_pixels'],
        }
    
    def _generate_analysis(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Generate mock analysis based on image properties"""
        # Calculate image properties
        gray = np.mean(image_array, axis=2)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Edge detection for water surface estimation
        edges = np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()
        edge_count = int(edges)
        
        # Color analysis for water detection
        blue_channel = image_array[:, :, 2]
        green_channel = image_array[:, :, 1]
        red_channel = image_array[:, :, 0]
        
        # Water detection heuristic
        water_ratio = np.mean((blue_channel > green_channel) & (blue_channel > red_channel * 0.8))
        water_pixels = int(water_ratio * 100)
        
        # Determine flood status based on image properties
        is_flood = water_pixels > 30 and brightness < 200
        
        # Confidence based on water detection strength
        confidence = min(99, max(50, 70 + (water_pixels - 30) * 0.5)) if is_flood else min(99, max(50, 90 - water_pixels))
        
        # Depth estimation
        if is_flood:
            depth_cm = int(water_pixels * 2)
        else:
            depth_cm = 0
        
        # Intensity level
        if is_flood:
            if confidence > 90:
                intensity = "CRITICAL"
                color_code = "#8B0000"
            elif confidence > 80:
                intensity = "HIGH"
                color_code = "#FF4500"
            else:
                intensity = "MEDIUM"
                color_code = "#FFD700"
        else:
            if confidence > 85:
                intensity = "SAFE"
                color_code = "#228B22"
            else:
                intensity = "UNCERTAIN"
                color_code = "#FFD700"
        
        return {
            'is_flood': is_flood,
            'confidence': confidence,
            'depth_cm': depth_cm,
            'intensity': intensity,
            'color_code': color_code,
            'brightness': brightness,
            'contrast': contrast,
            'edge_count': edge_count,
            'water_pixels': water_pixels,
        }
    
    def process_batch(self, image_paths: List[str]) -> Tuple[List[Dict], str]:
        """Process batch of images and generate HTML report"""
        if len(image_paths) > 10:
            image_paths = image_paths[:10]
        
        results = []
        for img_path in image_paths:
            try:
                analysis = self.process_image(img_path)
                results.append(analysis)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # Generate HTML report
        report_html = self._generate_html_report(results)
        
        # Save report
        batch_id = str(uuid.uuid4())[:8]
        report_dir = self.upload_dir / batch_id
        report_dir.mkdir(exist_ok=True)
        
        report_path = report_dir / "report.html"
        with open(report_path, 'w') as f:
            f.write(report_html)
        
        # Copy images to report directory
        for i, img_path in enumerate(image_paths, 1):
            import shutil
            dest = report_dir / f"image_{i:02d}.jpg"
            shutil.copy(img_path, dest)
        
        return results, str(report_path)
    
    def _generate_html_report(self, results: List[Dict]) -> str:
        """Generate 11-page HTML report with side-by-side layout"""
        
        # Calculate statistics
        total = len(results)
        flooded = sum(1 for r in results if r['is_flood'])
        dry = total - flooded
        avg_confidence = np.mean([r['confidence'] for r in results])
        avg_depth = np.mean([r['depth_cm'] for r in results]) if flooded > 0 else 0
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Flood Detection Analysis Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f5f5;
            line-height: 1.6;
        }}
        
        .page {{
            width: 100%;
            height: 100vh;
            page-break-after: always;
            display: flex;
            flex-direction: column;
            background: white;
            overflow: hidden;
        }}
        
        /* PAGE 1: SUMMARY PAGE */
        .summary-page {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 60px 40px;
            justify-content: center;
            align-items: center;
            text-align: center;
        }}
        
        .summary-page h1 {{
            font-size: 48px;
            margin-bottom: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .summary-page .subtitle {{
            font-size: 20px;
            opacity: 0.9;
            margin-bottom: 40px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin: 40px 0;
            padding: 0 40px;
        }}
        
        .stat-card {{
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }}
        
        .stat-card .number {{
            font-size: 36px;
            font-weight: bold;
            color: #fff;
        }}
        
        .stat-card .label {{
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 10px;
            opacity: 0.8;
        }}
        
        .timestamp {{
            margin-top: 50px;
            font-size: 12px;
            opacity: 0.7;
        }}
        
        /* IMAGE + OUTPUT PAGES */
        .image-output-page {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0;
            padding: 0;
        }}
        
        .image-section {{
            display: flex;
            align-items: center;
            justify-content: center;
            background: #f8f9fa;
            border-right: 2px solid #e0e0e0;
            padding: 20px;
            overflow: hidden;
        }}
        
        .image-section img {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            border-radius: 5px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        
        .analysis-section {{
            display: flex;
            flex-direction: column;
            padding: 40px;
            background: white;
            overflow-y: auto;
        }}
        
        .image-header {{
            border-bottom: 2px solid #667eea;
            padding-bottom: 20px;
            margin-bottom: 25px;
        }}
        
        .image-number {{
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 10px;
        }}
        
        .filename {{
            color: #666;
            font-size: 14px;
            word-break: break-all;
        }}
        
        .analysis-content {{
            flex: 1;
        }}
        
        .result-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid #e0e0e0;
            font-size: 14px;
        }}
        
        .result-label {{
            font-weight: 600;
            color: #333;
            flex: 0 0 35%;
        }}
        
        .result-value {{
            flex: 1;
            text-align: right;
            color: #666;
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 12px;
            color: white;
        }}
        
        .status-flood {{
            background-color: #ff6b6b;
        }}
        
        .status-dry {{
            background-color: #51cf66;
        }}
        
        .intensity-badge {{
            padding: 6px 14px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 12px;
            color: white;
            text-align: center;
        }}
        
        .intensity-critical {{
            background-color: #8B0000;
        }}
        
        .intensity-high {{
            background-color: #FF4500;
        }}
        
        .intensity-medium {{
            background-color: #FFD700;
            color: #333;
        }}
        
        .intensity-safe {{
            background-color: #228B22;
        }}
        
        .intensity-uncertain {{
            background-color: #FFD700;
            color: #333;
        }}
        
        .confidence-bar {{
            width: 100%;
            height: 20px;
            background: #e0e0e0;
            border-radius: 10px;
            margin-top: 5px;
            overflow: hidden;
        }}
        
        .confidence-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            transition: width 0.3s ease;
        }}
        
        .details-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
        }}
        
        .detail-item {{
            background: #f8f9fa;
            padding: 12px;
            border-radius: 5px;
        }}
        
        .detail-label {{
            font-size: 11px;
            text-transform: uppercase;
            color: #999;
            margin-bottom: 5px;
            letter-spacing: 0.5px;
        }}
        
        .detail-value {{
            font-size: 14px;
            font-weight: 600;
            color: #333;
        }}
        
        @media print {{
            body {{ margin: 0; padding: 0; }}
            .page {{ page-break-after: always; height: auto; }}
            .image-output-page {{ display: grid; grid-template-columns: 1fr 1fr; }}
        }}
    </style>
</head>
<body>

<!-- PAGE 1: SUMMARY -->
<div class="page summary-page">
    <div>
        <h1>🌊 Flood Detection Analysis Report</h1>
        <p class="subtitle">Comprehensive Image Analysis Results</p>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="number">{total}</div>
                <div class="label">Total Images</div>
            </div>
            <div class="stat-card">
                <div class="number">{flooded}</div>
                <div class="label">Flooded</div>
            </div>
            <div class="stat-card">
                <div class="number">{dry}</div>
                <div class="label">Dry</div>
            </div>
            <div class="stat-card">
                <div class="number">{avg_confidence:.1f}%</div>
                <div class="label">Avg Confidence</div>
            </div>
        </div>
        
        <div style="font-size: 16px; margin-top: 30px;">
            Average Depth: <strong>{avg_depth:.1f} cm</strong>
        </div>
        
        <div class="timestamp">
            Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
            Report Version: 1.0 | Bengaluru Flood Detection System
        </div>
    </div>
</div>
"""
        
        # Pages 2-11: Image + Analysis
        for idx, result in enumerate(results, 1):
            intensity_class = f"intensity-{result['intensity'].lower()}"
            status_class = "status-flood" if result['is_flood'] else "status-dry"
            status_text = "🌊 FLOOD DETECTED" if result['is_flood'] else "✅ DRY"
            
            html += f"""
<!-- PAGE {idx+1}: IMAGE {idx} + ANALYSIS -->
<div class="page image-output-page">
    <!-- LEFT: IMAGE -->
    <div class="image-section">
        <img src="image_{idx:02d}.jpg" alt="Image {idx}">
    </div>
    
    <!-- RIGHT: ANALYSIS -->
    <div class="analysis-section">
        <div class="image-header">
            <div class="image-number">Image #{idx:02d}</div>
            <div class="filename">{result['filename']}</div>
        </div>
        
        <div class="analysis-content">
            <div class="result-row">
                <div class="result-label">Status</div>
                <div class="result-value">
                    <span class="status-badge {status_class}">{status_text}</span>
                </div>
            </div>
            
            <div class="result-row">
                <div class="result-label">Confidence</div>
                <div class="result-value">{result['confidence']:.1f}%</div>
            </div>
            <div style="padding: 0 0 12px 0;">
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {result['confidence']}%"></div>
                </div>
            </div>
            
            <div class="result-row">
                <div class="result-label">Intensity Level</div>
                <div class="result-value">
                    <span class="intensity-badge {intensity_class}">{result['intensity']}</span>
                </div>
            </div>
            
            <div class="result-row">
                <div class="result-label">Estimated Depth</div>
                <div class="result-value">{result['depth_cm']:.0f} cm</div>
            </div>
            
            <div class="result-row">
                <div class="result-label">Image Size</div>
                <div class="result-value">{result['dimensions'][0]}x{result['dimensions'][1]}px</div>
            </div>
            
            <div class="result-row">
                <div class="result-label">File Size</div>
                <div class="result-value">{result['size_mb']:.2f} MB</div>
            </div>
            
            <div class="details-grid">
                <div class="detail-item">
                    <div class="detail-label">Brightness</div>
                    <div class="detail-value">{result['brightness']:.1f}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Contrast</div>
                    <div class="detail-value">{result['contrast']:.1f}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Water Pixels</div>
                    <div class="detail-value">{result['water_pixels']}%</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Edge Count</div>
                    <div class="detail-value">{result['edge_count']}</div>
                </div>
            </div>
        </div>
    </div>
</div>
"""
        
        html += """
</body>
</html>
"""
        
        return html


def main():
    """Example usage"""
    processor = RandomImageUploadProcessor()
    
    # Example: process images from a directory
    test_dir = Path("test_images/batch_upload")
    if test_dir.exists():
        image_paths = list(test_dir.glob("*.jpg"))[:10]
        if image_paths:
            results, report_path = processor.process_batch([str(p) for p in image_paths])
            print(f"Report generated: {report_path}")
            print(f"Processed {len(results)} images")


if __name__ == "__main__":
    main()
