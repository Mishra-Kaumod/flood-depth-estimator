"""V6-only, OpenCV-first video decoding with an FFmpeg fallback.

This module owns input robustness only. It never makes flood-depth decisions and
does not import or modify the V5 CLI video path.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoDiagnostic:
    code: str
    message: str
    frame_index: Optional[int] = None


@dataclass(frozen=True)
class VideoFrame:
    frame_index: int
    timestamp_seconds: Optional[float]
    frame_bgr: np.ndarray
    backend: str


@dataclass(frozen=True)
class FFmpegAvailability:
    ffmpeg_path: Optional[str]
    ffprobe_path: Optional[str]
    dav_demuxer_status: str

    @property
    def available_for_raw_frames(self) -> bool:
        return self.ffmpeg_path is not None and self.ffprobe_path is not None


class V6VideoInput:
    """Decodes ordered frames without changing V6's numerical contract."""

    def __init__(
        self,
        cv2_module: Any = cv2,
        which_fn: Callable[[str], Optional[str]] = shutil.which,
        run_fn: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        popen_factory: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        self._cv2 = cv2_module
        self._which = which_fn
        self._run = run_fn
        self._popen = popen_factory
        self.diagnostics: list[VideoDiagnostic] = []
        self.backend: Optional[str] = None
        self.fps: Optional[float] = None

    def ffmpeg_availability(self) -> FFmpegAvailability:
        ffmpeg_path, ffprobe_path = self._which("ffmpeg"), self._which("ffprobe")
        if not ffmpeg_path:
            return FFmpegAvailability(None, ffprobe_path, "not_checked_ffmpeg_unavailable")
        try:
            completed = self._run([ffmpeg_path, "-hide_banner", "-demuxers"], capture_output=True, text=True, check=False)
            text = f"{completed.stdout}\n{completed.stderr}".lower()
            status = "reported" if ("dav" in text or "dhav" in text) else "not_reported"
        except OSError:
            status = "not_checked_command_failed"
        return FFmpegAvailability(ffmpeg_path, ffprobe_path, status)

    @staticmethod
    def _valid_frame(frame: Any) -> bool:
        return isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.size > 0

    def iter_frames(self, video_path: str | Path) -> Iterator[VideoFrame]:
        """Yield OpenCV frames, falling back only if OpenCV yields none."""
        self.diagnostics.clear()
        self.backend, self.fps = None, None
        path = str(video_path)
        cap = self._cv2.VideoCapture(path)
        yielded = False
        try:
            if cap.isOpened():
                fps = float(cap.get(self._cv2.CAP_PROP_FPS) or 0.0)
                self.fps = fps if math.isfinite(fps) and fps > 0 else None
                frame_index = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if not self._valid_frame(frame):
                        self.diagnostics.append(VideoDiagnostic("opencv_bad_frame", "OpenCV returned an invalid frame; skipped.", frame_index))
                        frame_index += 1
                        continue
                    yielded = True
                    self.backend = "opencv"
                    timestamp = frame_index / self.fps if self.fps else None
                    yield VideoFrame(frame_index, timestamp, frame, "opencv")
                    frame_index += 1
            else:
                self.diagnostics.append(VideoDiagnostic("opencv_open_failed", "OpenCV could not open the video."))
        finally:
            cap.release()

        if yielded:
            return
        self.diagnostics.append(VideoDiagnostic("opencv_no_decodable_frame", "OpenCV produced no valid frame; attempting FFmpeg fallback."))
        yield from self._iter_ffmpeg(path)

    def _probe(self, ffprobe_path: str, path: str) -> tuple[int, int, Optional[float]]:
        command = [
            ffprobe_path, "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate", "-of", "json", path,
        ]
        completed = self._run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError("ffprobe failed to inspect a video stream")
        streams = json.loads(completed.stdout).get("streams", [])
        if not streams:
            raise RuntimeError("ffprobe found no video stream")
        stream = streams[0]
        width, height = int(stream["width"]), int(stream["height"])
        value = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/0"
        numerator, denominator = value.split("/", 1)
        fps = float(numerator) / float(denominator) if float(denominator) else None
        return width, height, fps if fps and math.isfinite(fps) and fps > 0 else None

    @staticmethod
    def _read_exact(stream: Any, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            block = stream.read(remaining)
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        return b"".join(chunks)

    def _iter_ffmpeg(self, path: str) -> Iterator[VideoFrame]:
        availability = self.ffmpeg_availability()
        if not availability.ffmpeg_path:
            self.diagnostics.append(VideoDiagnostic("ffmpeg_unavailable", "FFmpeg is not available on PATH."))
            return
        if not availability.ffprobe_path:
            self.diagnostics.append(VideoDiagnostic("ffprobe_unavailable", "FFprobe is required for raw-frame fallback and is not available on PATH."))
            return
        if Path(path).suffix.lower() in {".dav", ".dhav"}:
            self.diagnostics.append(VideoDiagnostic("dav_demuxer_status", f"FFmpeg DAV/DHAV demuxer status: {availability.dav_demuxer_status}."))
        try:
            width, height, fps = self._probe(availability.ffprobe_path, path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
            self.diagnostics.append(VideoDiagnostic("ffmpeg_probe_failed", str(exc)))
            return
        frame_size = width * height * 3
        command = [
            availability.ffmpeg_path, "-hide_banner", "-loglevel", "error", "-i", path,
            "-map", "0:v:0", "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
        ]
        try:
            process = self._popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as exc:
            self.diagnostics.append(VideoDiagnostic("ffmpeg_start_failed", str(exc)))
            return
        self.backend, self.fps = "ffmpeg", fps
        index = 0
        try:
            while True:
                raw = self._read_exact(process.stdout, frame_size)
                if not raw:
                    break
                if len(raw) != frame_size:
                    self.diagnostics.append(VideoDiagnostic("ffmpeg_bad_frame", "FFmpeg produced a truncated raw frame; skipped.", index))
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()
                timestamp = index / fps if fps else None
                yield VideoFrame(index, timestamp, frame, "ffmpeg")
                index += 1
        finally:
            stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
            return_code = process.wait()
            if return_code != 0:
                detail = stderr.strip() or f"exit code {return_code}"
                self.diagnostics.append(VideoDiagnostic("ffmpeg_decode_failed", detail))
        if index == 0:
            self.diagnostics.append(VideoDiagnostic("ffmpeg_no_decodable_frame", "FFmpeg produced no valid video frame."))
