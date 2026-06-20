"""
VIDEO FLOOD DETECTION & DEPTH ESTIMATION PIPELINE

Processes video frames to:
1. Detect water presence (WaterDetectionAnalyzer)
2. Classify flood severity (ResNet18)
3. Estimate water depth (DepthBandEstimator)
4. Save results per frame to CSV
"""

import cv2
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from torchvision import transforms, models
from PIL import Image

from water_detection import WaterDetectionAnalyzer
from depth_band_estimator import estimate_depth
from hybrid_depth_estimator import HybridDepthEstimator


class VideoFloodAnalyzer:
    """
    Process video frames for flood detection and depth estimation.
    Uses multi-method ensemble for robust depth estimation.
    """
    
    def __init__(self, model_path="severity_model.pth", use_hybrid=True):
        """
        Initialize video analyzer.
        
        Args:
            model_path: Path to trained severity model
            use_hybrid: Whether to use hybrid depth estimation (with YOLO)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load severity classification model (ResNet18)
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 5)
        
        try:
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Water detection analyzer
        self.water_detector = WaterDetectionAnalyzer()
        
        # Hybrid depth estimator (with YOLO)
        self.use_hybrid = use_hybrid
        if use_hybrid:
            try:
                self.depth_estimator = HybridDepthEstimator()
                print("✓ Hybrid depth estimator initialized (with YOLO)")
            except Exception as e:
                print(f"⚠ Hybrid depth estimator failed to initialize: {e}")
                print("  Falling back to simple depth estimation")
                self.use_hybrid = False
        
        # Severity labels
        self.severity_names = {
            0: "No / Very Low Flood",
            1: "Minor Flood",
            2: "Moderate Flood",
            3: "High Flood",
            4: "Severe Flood"
        }
    
    def process_video(self, video_path, output_csv="video_analysis.csv", 
                     skip_frames=1, save_frames_dir=None):
        """
        Process entire video frame by frame.
        
        Args:
            video_path: Path to video file
            output_csv: Output CSV filename
            skip_frames: Process every Nth frame (1 = all frames)
            save_frames_dir: Optional directory to save analyzed frames
            
        Returns:
            DataFrame with analysis results
        """
        
        # Setup video capture
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\n{'='*60}")
        print(f"Video Info:")
        print(f"  FPS: {fps}")
        print(f"  Total Frames: {frame_count}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Skip Rate: {skip_frames}")
        print(f"{'='*60}\n")
        
        # Setup output directory for frames if requested
        if save_frames_dir:
            save_frames_path = Path(save_frames_dir)
            save_frames_path.mkdir(parents=True, exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(
                str(save_frames_path / "output_video.mp4"),
                fourcc, fps, (width, height)
            )
        
        # Results storage
        results = []
        frame_num = 0
        processed_frames = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Skip frames
                if frame_num % skip_frames != 0:
                    frame_num += 1
                    continue
                
                processed_frames += 1
                
                # Convert BGR to RGB for PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # ===== STEP 1: Water Detection =====
                water_result = self.water_detector.detect_water_surface(frame_rgb)
                water_detected = water_result['water_detected']
                water_confidence = water_result['confidence']
                water_percentage = water_result['water_percentage']
                
                # Initialize severity/depth as None
                severity = None
                severity_name = "N/A"
                severity_confidence = None
                depth_band = "N/A"
                depth_cm = None
                depth_method = "No water"
                
                # ===== STEP 2: Severity Classification (only if water detected) =====
                if water_detected:
                    try:
                        pil_image = Image.fromarray(frame_rgb)
                        x = self.transform(pil_image).unsqueeze(0).to(self.device)
                        
                        with torch.no_grad():
                            logits = self.model(x)
                            probs = torch.softmax(logits, dim=1)
                            severity = int(torch.argmax(probs, dim=1).item())
                            severity_confidence = float(torch.max(probs).item())
                        
                        severity_name = self.severity_names[severity]
                        
                        # ===== STEP 3: Depth Estimation =====
                        if self.use_hybrid:
                            # Use hybrid estimator (with YOLO)
                            depth_result = self.depth_estimator.estimate_depth(
                                image=frame,  # Use BGR frame for YOLO
                                water_detected=water_detected,
                                water_percentage=water_percentage,
                                severity_class=severity,
                                severity_confidence=severity_confidence
                            )
                            depth_cm = depth_result['depth_cm']
                            depth_band = depth_result['depth_band']
                            depth_method = depth_result['method']
                        else:
                            # Fallback to simple severity-based depth
                            depth_info = estimate_depth(severity)
                            depth_band = depth_info['depth_band']
                            depth_cm = depth_info['depth_cm']
                            depth_method = "Severity-based"
                        
                    except Exception as e:
                        print(f"Error processing frame {frame_num}: {e}")
                
                # Store results
                results.append({
                    'frame_number': frame_num,
                    'time_seconds': frame_num / fps if fps > 0 else 0,
                    'water_detected': water_detected,
                    'water_confidence': round(water_confidence, 4),
                    'water_percentage': round(water_percentage * 100, 2),
                    'severity_class': severity,
                    'severity_name': severity_name,
                    'severity_confidence': round(severity_confidence, 4) if severity_confidence else None,
                    'depth_band': depth_band,
                    'depth_cm': depth_cm,
                    'depth_method': depth_method
                })
                
                # Print progress
                if processed_frames % 30 == 0:
                    print(f"Processed {processed_frames} frames (video frame {frame_num}/{frame_count})")
                    print(f"  Frame {frame_num}: Water={water_detected} ({water_percentage*100:.1f}%), "
                          f"Severity={severity_name}, Depth={depth_cm}cm")
                
                # Save annotated frame if requested
                if save_frames_dir and processed_frames % 5 == 0:
                    annotated_frame = self._annotate_frame(
                        frame.copy(), water_detected, water_percentage,
                        severity_name, depth_cm
                    )
                    cv2.imwrite(
                        str(save_frames_path / f"frame_{processed_frames:06d}.jpg"),
                        annotated_frame
                    )
                    out_video.write(annotated_frame)
                
                frame_num += 1
        
        finally:
            cap.release()
            if save_frames_dir:
                out_video.release()
        
        # Save results to CSV
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        
        print(f"\n{'='*60}")
        print(f"Analysis Complete!")
        print(f"  Total frames processed: {processed_frames}")
        print(f"  Results saved to: {output_csv}")
        if save_frames_dir:
            print(f"  Annotated frames saved to: {save_frames_dir}")
        print(f"{'='*60}\n")
        
        return df
    
    def _annotate_frame(self, frame, water_detected, water_pct, 
                        severity_name, depth_cm):
        """
        Add text annotations to frame.
        """
        h, w = frame.shape[:2]
        
        # Background color: Green if water, Red if no water
        color = (0, 255, 0) if water_detected else (255, 0, 0)
        
        # Add text
        y_offset = 30
        texts = [
            f"Water Detected: {water_detected}",
            f"Water: {water_pct*100:.1f}%",
            f"Severity: {severity_name}",
            f"Depth: {depth_cm}cm" if depth_cm else "Depth: N/A"
        ]
        
        for i, text in enumerate(texts):
            cv2.putText(
                frame, text,
                (10, y_offset + i*30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2
            )
        
        return frame


def main():
    """
    Example usage of video processor.
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python process_video.py <video_path> [output_csv] [skip_frames]")
        print("\nExample:")
        print("  python process_video.py my_video.mp4")
        print("  python process_video.py my_video.mp4 results.csv 2")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "video_analysis.csv"
    skip_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    
    analyzer = VideoFloodAnalyzer()
    df = analyzer.process_video(
        video_path,
        output_csv=output_csv,
        skip_frames=skip_frames,
        save_frames_dir="output_frames"
    )
    
    if df is not None:
        print("\nSummary Statistics:")
        print(f"  Frames with water: {df['water_detected'].sum()}")
        print(f"  Average water %: {df['water_percentage'].mean():.2f}%")
        print(f"  Severity distribution: {df['severity_name'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
