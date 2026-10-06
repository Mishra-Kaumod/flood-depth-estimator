import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.run_v6_shadow_video import CSV_FIELDS, process_saved_frames
from src.v6_video_input import VideoFrame


class Reader:
    backend, fps, diagnostics = "opencv", 15.0, []
    def iter_frames(self, _):
        for index in range(2):
            yield VideoFrame(index, index / 15.0, np.zeros((4, 5, 3), dtype=np.uint8), "opencv")


class Result:
    numerical_owner = "efficientnet_primary_anchor"
    class uncertainty: flags = ()
    def __init__(self, depth): self.primary_depth_cm = depth; self.final_shadow_depth_cm = depth


class Pipeline:
    def __init__(self): self.shapes = []; self.index = 0
    def predict(self, image):
        self.shapes.append(image.shape)
        depth = 20.0 if self.index == 0 else None
        self.index += 1
        return Result(depth)


class V6VideoRunnerTests(unittest.TestCase):
    def test_saved_frames_are_reloaded_for_v6_and_none_depth_is_safe(self):
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = Pipeline()
            rows, summary = process_saved_frames(Reader(), pipeline, "sample.mp4", Path(temporary), max_frames=2, skip_frames=1)
            self.assertEqual(summary["decoded_frame_count"], 2)
            self.assertEqual(pipeline.shapes, [(4, 5, 3), (4, 5, 3)])
            self.assertTrue((Path(temporary) / "frames" / "frame_000000.jpg").is_file())
            self.assertTrue((Path(temporary) / "frames" / "frame_000001.jpg").is_file())
            self.assertEqual(rows[0]["status"], "processed")
            self.assertEqual(rows[1]["status"], "unavailable_valid_frame_no_depth")
            self.assertIsNone(rows[1]["final_shadow_depth_cm"])
            self.assertEqual(tuple(rows[0]), CSV_FIELDS)
