"""
FLOOD DETECTION & DEPTH ESTIMATION SYSTEM
Main Entry Point

This system provides four main capabilities:
1. Detect water presence in images (WaterDetectionAnalyzer)
2. Classify flood severity in images (SeverityPredictor)
3. Process video frames for continuous monitoring (VideoFloodAnalyzer)
4. Detect objects (vehicles, people) for anchor-based depth estimation (ObjectDetector)

Storage options:
- Local (DEFAULT): Read/write from local folders
- AWS S3: Read/write from S3 bucket
"""

import sys
from unittest import result
import cv2
from pathlib import Path
import tempfile
import os

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / "modules"))

from modules.flood_analyzer import FloodAnalyzer
from modules.process_video import VideoFloodAnalyzer
from modules.object_detection import ObjectDetector


def get_storage_mode(args):
    """
    Extract storage mode from command line args.
    
    Args:
        args: Command line arguments list
        
    Returns:
        str: "local" or "aws" (default: "local")
    """
    for arg in args:
        if arg.startswith("--storage="):
            mode = arg.split("=")[1].lower()
            if mode in ["local", "aws"]:
                return mode
    return "local"


def get_s3_handler(storage_mode, bucket_name=None):
    """
    Get S3 handler if storage mode is AWS.
    
    Args:
        storage_mode: "local" or "aws"
        bucket_name: S3 bucket name (optional)
        
    Returns:
        S3Handler or None
    """
    if storage_mode == "aws":
        try:
            from modules.s3_handler import S3Handler
            return S3Handler(bucket_name=bucket_name)
        except ImportError:
            print("Error: boto3 not installed. Install with: pip install boto3")
            sys.exit(1)
        except Exception as e:
            print(f"Error initializing S3: {e}")
            sys.exit(1)
    return None


def process_single_image(image_path, model_path="severity_model.pth", storage_mode="local", s3_handler=None):
    """
    Analyze a single image for flood severity.
    
    Args:
        image_path: Path to image file (local path or S3 key)
        model_path: Path to trained model
        storage_mode: "local" or "aws"
        s3_handler: S3Handler instance (required if storage_mode is "aws")
    """
    print("\n" + "="*60)
    print("SINGLE IMAGE ANALYSIS")
    print(f"Storage Mode: {storage_mode.upper()}")
    print("="*60)
    
    # Load the image as BGR for the shared still-image/video-frame pipeline.
    if storage_mode == "aws":
        if s3_handler is None:
            print("Error: S3 handler not initialized")
            return
        
        try:
            image = s3_handler.read_image_from_s3(image_path)
        except Exception as e:
            print(f"Error reading image from S3: {e}")
            return
    else:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Cannot read image from {image_path}")
            return

    result = FloodAnalyzer(model_path=model_path).analyze_bgr(image, image_path)
    
    if "error" in result:
        print(f"Error: {result['error']}")

    print(f"Image: {result['image_path']}")

    if not result['water_detected']:
        print("Result: No water_detected")
    elif "error" not in result:
        print("water_detected: Yes")
        print(f"Water  Level: {result['final_flood_level']}")
        print(f"Estimated Depth: {result['depth_cm']} cm")
        print("\nDEBUG")
        print("Water %:", result["water_percentage"])
        print("Water Confidence:", result["water_confidence"])
        print("\nDepth Details:")
        print(result.get("depth_details", {}))
        print("\nMethod Votes:")
        print(result["method_votes"])
        
    
    print("="*60 + "\n")


def process_video_file(video_path, output_csv="video_analysis.csv", 
                       skip_frames=1, model_path="severity_model.pth",
                       storage_mode="local", s3_handler=None):
    """
    Process a video file frame by frame.
    
    Args:
        video_path: Path to video file (local path or S3 key)
        output_csv: Output CSV filename
        skip_frames: Process every Nth frame
        model_path: Path to trained model
        storage_mode: "local" or "aws"
        s3_handler: S3Handler instance (required if storage_mode is "aws")
    """
    print("\n" + "="*60)
    print("VIDEO PROCESSING")
    print(f"Storage Mode: {storage_mode.upper()}")
    print("="*60)
    
    # Download video if using S3
    local_video_path = video_path
    if storage_mode == "aws":
        if s3_handler is None:
            print("Error: S3 handler not initialized")
            return
        
        try:
            temp_dir = tempfile.gettempdir()
            temp_video_path = os.path.join(temp_dir, "temp_video.mp4")
            local_video_path = s3_handler.read_video_from_s3(video_path, temp_video_path)
        except Exception as e:
            print(f"Error downloading video from S3: {e}")
            return
    
    try:
        analyzer = VideoFloodAnalyzer(model_path=model_path)
        df = analyzer.process_video(
            local_video_path,
            output_csv=output_csv if storage_mode == "local" else "temp_video_analysis.csv",
            skip_frames=skip_frames,
            save_frames_dir="output_frames"
        )
        
        if df is not None:
            print("\nSummary Statistics:")
            print(f"  Total frames processed: {len(df)}")
            print(f"  Frames with water detected: {df['water_detected'].sum()}")
            print(f"  Average water percentage: {df['water_percentage'].mean():.2f}%")
            
            severity_counts = df['severity_name'].value_counts()
            if len(severity_counts) > 0:
                print(f"  Severity distribution:")
                for severity, count in severity_counts.items():
                    print(f"    {severity}: {count} frames")
            
            # Upload results to S3 if needed
            if storage_mode == "aws":
                try:
                    s3_handler.write_csv_to_s3(df, output_csv)
                    os.remove("temp_video_analysis.csv")
                except Exception as e:
                    print(f"Warning: Could not upload CSV to S3: {e}")
            
            print(f"\n  CSV Results saved to: {output_csv}")
    
    finally:
        # Clean up temporary video file
        if storage_mode == "aws" and local_video_path != video_path:
            try:
                s3_handler.cleanup_temp_file(local_video_path)
            except Exception as e:
                print(f"Warning: Could not clean up temp file: {e}")


def process_object_detection(image_path, output_image="objects_detected.jpg", 
                             storage_mode="local", s3_handler=None):
    """
    Detect and visualize objects in an image using YOLO.
    
    Args:
        image_path: Path to image file (local path or S3 key)
        output_image: Path to save annotated image
        storage_mode: "local" or "aws"
        s3_handler: S3Handler instance (required if storage_mode is "aws")
    """
    print("\n" + "="*60)
    print("OBJECT DETECTION")
    print(f"Storage Mode: {storage_mode.upper()}")
    print("="*60)
    
    try:
        detector = ObjectDetector()
        
        # Load image based on storage mode
        if storage_mode == "aws":
            if s3_handler is None:
                print("Error: S3 handler not initialized")
                return
            
            try:
                image = s3_handler.read_image_from_s3(image_path)
            except Exception as e:
                print(f"Error reading image from S3: {e}")
                return
        else:
            # Read local image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Cannot read image from {image_path}")
                return
        
        print(f"Image: {image_path}")
        print(f"Resolution: {image.shape[1]}x{image.shape[0]}")
        
        # Detect objects
        detections = detector.detect_objects(image)
        
        if detections:
            print(f"\nDetected {len(detections)} objects:")
            for i, det in enumerate(detections, 1):
                print(f"  {i}. {det['class']}: {det['confidence']:.2%} confidence")
            
            # Get inventory
            inventory = detector.create_object_inventory(detections)
            print(f"\nObject Inventory:")
            for class_name, data in inventory['object_types'].items():
                count = data['count']
                avg_conf = data['avg_confidence']
                print(f"  {class_name}: {count} ({avg_conf:.2%} avg confidence)")
            
            # Get largest object and estimate depth
            largest = detector.get_largest_object(detections)
            if largest:
                depth_result = detector.estimate_depth_from_object(
                    largest, image.shape[0]
                )
                if depth_result['depth_cm'] is not None:
                    print(f"\nLargest Object Depth Estimate:")
                    print(f"  Object: {largest['class']}")
                    print(f"  Estimated Depth: {depth_result['depth_cm']} cm")
                    print(f"  Reference: {depth_result['method']}")
        else:
            print("\nNo objects detected in image")
        
        # Draw and save/upload annotated image
        annotated = detector.draw_detections(image, detections)
        
        if storage_mode == "aws":
            try:
                s3_handler.write_image_to_s3(annotated, output_image)
            except Exception as e:
                print(f"Error uploading annotated image to S3: {e}")
        else:
            cv2.imwrite(output_image, annotated)
            print(f"\nAnnotated image saved to: {output_image}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()



def main():
    """
    Main entry point with command-line interface.
    """
    if len(sys.argv) < 2:
        print("""
╔════════════════════════════════════════════════════════════╗
║   FLOOD DETECTION & DEPTH ESTIMATION SYSTEM v2.1          ║
║   With YOLO & S3 Support                                  ║
╚════════════════════════════════════════════════════════════╝

Usage:
  1. Analyze single image:
     python main.py image <image_path> [--storage=local|aws]
  
  2. Process video:
     python main.py video <video_path> [output_csv] [skip_frames] [--storage=local|aws]
  
  3. Detect objects (YOLO):
     python main.py object <image_path> [output_image] [--storage=local|aws]

Storage Options:
  --storage=local  (DEFAULT) - Use local files
  --storage=aws    - Use AWS S3 bucket

Examples (Local):
  python main.py image test_images/flood_image.jpg
  python main.py video test_videos/flood_video.mp4
  python main.py object test_images/flood_image.jpg

Examples (AWS S3):
  python main.py image images/flood_image.jpg --storage=aws
  python main.py video videos/flood_video.mp4 results.csv 2 --storage=aws
  python main.py object images/flood_image.jpg objects_output.jpg --storage=aws

Features:
  - Water surface detection (6-method ensemble)
  - Flood severity classification (ResNet18)
  - Depth estimation (3-method hybrid with YOLO)
  - Object detection & anchor-based depth (YOLO)
  - Multi-frame video processing with CSV export
  - Local or AWS S3 storage
  - GPU acceleration (optional)

AWS Setup:
  - Set AWS credentials: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
  - Set S3 bucket: S3_BUCKET environment variable (or use default: flood-analysis)
  - Requires: pip install boto3

Requirements:
  - severity_model.pth (trained model file)
  - test_images/ folder (local mode) or S3 bucket (AWS mode)
  - CUDA capable GPU (optional, will use CPU otherwise)

Output:
  - Single image: Console output
  - Video: CSV file + annotated frames in output_frames/
  - Objects: Annotated image with bounding boxes
        """)
        sys.exit(1)
    
    # Parse storage mode
    storage_mode = get_storage_mode(sys.argv)
    s3_handler = get_s3_handler(storage_mode)
    
    print(f"\n✓ Storage Mode: {storage_mode.upper()}")
    if storage_mode == "aws" and s3_handler:
        print(f"✓ S3 Bucket: {s3_handler.bucket_name}")
    
    mode = sys.argv[1].lower()
    
    if mode == "image" or mode == "img":
        if len(sys.argv) < 3:
            print("Error: Please provide image path")
            print("Usage: python main.py image <image_path> [--storage=local|aws]")
            sys.exit(1)
        
        image_path = sys.argv[2]
        process_single_image(image_path, storage_mode=storage_mode, s3_handler=s3_handler)
    
    elif mode == "video" or mode == "vid":
        if len(sys.argv) < 3:
            print("Error: Please provide video path")
            print("Usage: python main.py video <video_path> [output_csv] [skip_frames] [--storage=local|aws]")
            sys.exit(1)
        
        video_path = sys.argv[2]
        output_csv = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else "video_analysis.csv"
        skip_frames_arg = sys.argv[4] if len(sys.argv) > 4 and not sys.argv[4].startswith("--") else "1"
        
        try:
            skip_frames = int(skip_frames_arg)
        except ValueError:
            skip_frames = 1
        
        process_video_file(video_path, output_csv, skip_frames, 
                          storage_mode=storage_mode, s3_handler=s3_handler)
    
    elif mode == "object" or mode == "obj":
        if len(sys.argv) < 3:
            print("Error: Please provide image path")
            print("Usage: python main.py object <image_path> [output_image] [--storage=local|aws]")
            sys.exit(1)
        
        image_path = sys.argv[2]
        output_image = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else "objects_detected.jpg"
        
        process_object_detection(image_path, output_image,
                               storage_mode=storage_mode, s3_handler=s3_handler)
    
    else:
        print(f"Error: Unknown mode '{mode}'")
        print("Use 'image', 'video', or 'object'")
        sys.exit(1)


if __name__ == "__main__":
    main()
