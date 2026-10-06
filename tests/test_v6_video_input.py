import io
import json
import subprocess
import unittest

import numpy as np

from scripts.run_v6_shadow_video import finite_depth, record_from_v6_result
from src.v6_video_input import V6VideoInput


class Capture:
    def __init__(self, frames, opened=True): self.frames, self.opened, self.index = frames, opened, 0
    def isOpened(self): return self.opened
    def get(self, _): return 25.0
    def read(self):
        if self.index >= len(self.frames): return False, None
        frame = self.frames[self.index]; self.index += 1; return True, frame
    def release(self): pass


class CV2Stub:
    CAP_PROP_FPS = 5
    def __init__(self, frames, opened=True): self.frames, self.opened = frames, opened
    def VideoCapture(self, _): return Capture(self.frames, self.opened)


class Process:
    def __init__(self, raw, code=0): self.stdout, self.stderr, self.code = io.BytesIO(raw), io.BytesIO(), code
    def wait(self): return self.code


def run_ok(command, **_):
    if "-demuxers" in command: return subprocess.CompletedProcess(command, 0, " D dav\n", "")
    return subprocess.CompletedProcess(command, 0, json.dumps({"streams": [{"width": 2, "height": 1, "avg_frame_rate": "25/1"}]}), "")


class Result:
    primary_depth_cm = 22.5
    final_shadow_depth_cm = 22.5
    numerical_owner = "efficientnet_primary_anchor"
    class uncertainty: flags = ("semantic_context_conflict",)


class NoDepthResult:
    primary_depth_cm = None
    final_shadow_depth_cm = float("nan")
    numerical_owner = "efficientnet_primary_anchor"
    class uncertainty: flags = ()


class V6VideoInputTests(unittest.TestCase):
    def test_normal_opencv_path(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        reader = V6VideoInput(cv2_module=CV2Stub([frame]), which_fn=lambda _: None)
        frames = list(reader.iter_frames("normal.mp4"))
        self.assertEqual(len(frames), 1); self.assertEqual(frames[0].backend, "opencv")

    def test_zero_frame_opencv_falls_back_to_ffmpeg(self):
        raw = bytes(range(6))
        reader = V6VideoInput(cv2_module=CV2Stub([]), which_fn=lambda name: f"/{name}", run_fn=run_ok, popen_factory=lambda *_, **__: Process(raw))
        frames = list(reader.iter_frames("camera.dav"))
        self.assertEqual(len(frames), 1); self.assertEqual(frames[0].backend, "ffmpeg")
        self.assertIn("opencv_no_decodable_frame", [item.code for item in reader.diagnostics])
        self.assertIn("dav_demuxer_status", [item.code for item in reader.diagnostics])

    def test_ffmpeg_unavailable_is_clean(self):
        reader = V6VideoInput(cv2_module=CV2Stub([], opened=False), which_fn=lambda _: None)
        self.assertEqual(list(reader.iter_frames("camera.dav")), [])
        self.assertIn("ffmpeg_unavailable", [item.code for item in reader.diagnostics])

    def test_bad_opencv_frame_is_skipped(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        reader = V6VideoInput(cv2_module=CV2Stub([None, frame]), which_fn=lambda _: None)
        frames = list(reader.iter_frames("normal.mp4"))
        self.assertEqual(len(frames), 1); self.assertIn("opencv_bad_frame", [item.code for item in reader.diagnostics])

    def test_v6_depth_contract_handles_none_and_nonfinite(self):
        self.assertIsNone(finite_depth(None)); self.assertIsNone(finite_depth(float("nan"))); self.assertIsNone(finite_depth(float("inf")))
        row = record_from_v6_result(1, 0.04, "opencv", Result())
        self.assertEqual(row["final_shadow_depth_cm"], 22.5); self.assertEqual(row["status"], "processed")
        unavailable = record_from_v6_result(2, 0.08, "opencv", NoDepthResult())
        self.assertIsNone(unavailable["final_shadow_depth_cm"])
        self.assertEqual(unavailable["status"], "unavailable_valid_frame_no_depth")
