"""Video processing built on the shared image/frame flood-analysis pipeline."""

import cv2
import pandas as pd
from pathlib import Path
from collections import deque

from flood_analyzer import FloodAnalyzer


class VideoFloodAnalyzer:
    """Analyze each selected video frame using the same pipeline as a photo."""

    def __init__(self, model_path="severity_model.pth", use_hybrid=True):
        self.analyzer = FloodAnalyzer(model_path=model_path, use_hybrid=use_hybrid)
        self.previous_depth = None
        self.water_history = deque(maxlen=5)
        

    def process_video(self, video_path, output_csv="video_analysis.csv",
                      skip_frames=1, save_frames_dir=None):
        """Process a video and preserve only frames where water is detected."""
        if skip_frames < 1:
            raise ValueError("skip_frames must be at least 1")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\n{'=' * 60}")
        print("Video Info:")
        print(f"  FPS: {fps}")
        print(f"  Total Frames: {frame_count}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Skip Rate: {skip_frames}")
        print(f"{'=' * 60}\n")

        out_video = None
        if save_frames_dir:
            save_frames_path = Path(save_frames_dir)
            save_frames_path.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(
                str(save_frames_path / "water_detected_video.mp4"),
                fourcc, fps, (width, height),
            )

        results = []
        frame_num = 0
        processed_frames = 0
        water_frames_saved = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_num % skip_frames != 0:
                    frame_num += 1
                    continue

                processed_frames += 1
                analysis = self.analyzer.analyze_bgr(frame, f"frame_{frame_num}")
                self.water_history.append(
                int(analysis["water_detected"])
                )

                if len(self.water_history) >= 3:

                    majority_vote = (
                        sum(self.water_history)
                        >= len(self.water_history) // 2 + 1
                    )

                    analysis["water_detected"] = majority_vote
               
                if analysis["water_detected"]:

                    current_depth = analysis["depth_cm"]

                    if self.previous_depth is not None:

                        current_depth = int(
                            0.7 * self.previous_depth +
                            0.3 * current_depth
                        )

                    analysis["depth_cm"] = current_depth

                    self.previous_depth = current_depth    

                results.append({
                    'frame_number': frame_num,
                    'time_seconds': frame_num / fps if fps > 0 else 0,
                    'water_detected': analysis['water_detected'],
                    'water_confidence': round(analysis['water_confidence'], 4),
                    'water_percentage': round(analysis['water_percentage'] * 100, 2),
                    'severity_class': analysis['severity_class'],
                    'severity_name': analysis['severity_name'],
                    'severity_confidence': analysis['severity_confidence'],
                    'depth_band': analysis['depth_band'],
                    'depth_cm': analysis['depth_cm'],
                    'final_flood_level': analysis['final_flood_level'],
                    'depth_method': analysis['depth_method'],
                    'error': analysis.get('error'),
                })

                if 'error' in analysis:
                    print(f"Error processing frame {frame_num}: {analysis['error']}")

                if processed_frames % 30 == 0:
                    print(
                        f"Processed {processed_frames} frames "
                        f"(video frame {frame_num}/{frame_count})"
                    )

                # Dry frames are recorded in the CSV but are not exported as
                # images or added to the derived video.
                if save_frames_dir and analysis['water_detected']:
                    annotated_frame = self._annotate_frame(
                        frame.copy(),
                        analysis['water_percentage'],
                        analysis['final_flood_level'],
                        analysis['depth_cm'],
                    )
                    cv2.imwrite(
                        str(save_frames_path / f"frame_{frame_num:06d}.jpg"),
                        annotated_frame,
                    )
                    out_video.write(annotated_frame)
                    water_frames_saved += 1

                frame_num += 1
        finally:
            cap.release()
            if out_video is not None:
                out_video.release()

        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)

        print(f"\n{'=' * 60}")
        print("Analysis Complete!")
        print(f"  Total frames processed: {processed_frames}")
        print(f"  Water frames found: {int(df['water_detected'].sum())}")
        print(f"  Results saved to: {output_csv}")
        if save_frames_dir:
            print(f"  Water frames saved to: {save_frames_dir} ({water_frames_saved} frames)")
        print(f"{'=' * 60}\n")

        return df

    def _annotate_frame(self, frame, water_pct, flood_level, depth_cm):
        """Add water, severity, and depth labels to a confirmed water frame."""
        color = (0, 255, 0)
        texts = [
            "Water Detected: True",
            f"Water: {water_pct * 100:.1f}%",
            f"Flood Level: {flood_level}",
            f"Depth: {depth_cm}cm" if depth_cm is not None else "Depth: N/A",
        ]

        for index, text in enumerate(texts):
            cv2.putText(
                frame,
                text,
                (10, 30 + index * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )
        return frame


def main():
    """Example command-line usage for standalone video processing."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python process_video.py <video_path> [output_csv] [skip_frames]")
        sys.exit(1)

    video_path = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "video_analysis.csv"
    skip_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    analyzer = VideoFloodAnalyzer()
    analyzer.process_video(video_path, output_csv, skip_frames, "output_frames")


if __name__ == "__main__":
    main()
