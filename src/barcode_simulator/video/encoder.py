"""
Video encoding engine supporting FFmpeg H.264 pipeline and OpenCV VideoWriter fallback.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

from barcode_simulator.utils.logging import setup_logger

logger = setup_logger("video_encoder")


class VideoEncoder:
    """
    Encodes video frames to MP4.
    Prefers FFmpeg H.264 (libx264, yuv420p) for maximum quality and compatibility.
    Falls back gracefully to OpenCV VideoWriter if FFmpeg is unavailable.
    """

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: int = 30,
        codec: str = "h264",
        crf: int = 18,
        bitrate: Optional[str] = "4M",
    ):
        self.output_path = str(Path(output_path).resolve())
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.crf = crf
        self.bitrate = bitrate

        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

        self._ffmpeg_proc: Optional[subprocess.Popen] = None
        self._cv2_writer: Optional[cv2.VideoWriter] = None
        self._frame_count: int = 0

        self._initialize_encoder()

    def _initialize_encoder(self) -> None:
        # Check if FFmpeg binary is available on PATH
        ffmpeg_bin = shutil.which("ffmpeg")

        if ffmpeg_bin and self.codec.lower() in ("h264", "libx264", "mp4"):
            try:
                cmd = [
                    ffmpeg_bin,
                    "-y",  # Overwrite output file
                    "-f", "rawvideo",
                    "-vcodec", "rawvideo",
                    "-s", f"{self.width}x{self.height}",
                    "-pix_fmt", "rgb24",
                    "-r", str(self.fps),
                    "-i", "-",  # Read from stdin
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-preset", "medium",
                    "-crf", str(self.crf),
                    self.output_path,
                ]
                self._ffmpeg_proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                logger.info(f"Initialized FFmpeg H.264 video encoder -> {self.output_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to start FFmpeg process ({e}), falling back to OpenCV VideoWriter.")
                self._ffmpeg_proc = None

        # Fallback to OpenCV VideoWriter
        fourcc_code = "mp4v"
        fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
        self._cv2_writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            float(self.fps),
            (self.width, self.height),
            isColor=True,
        )
        logger.info(f"Initialized OpenCV VideoWriter ({fourcc_code}) -> {self.output_path}")

    def write_frame(self, frame_rgb: np.ndarray) -> None:
        """Write a single RGB frame (H, W, 3) to the video."""
        if frame_rgb.shape[:2] != (self.height, self.width):
            frame_rgb = cv2.resize(frame_rgb, (self.width, self.height))

        if self._ffmpeg_proc and self._ffmpeg_proc.stdin:
            try:
                # Direct raw byte write
                self._ffmpeg_proc.stdin.write(frame_rgb.tobytes())
            except (BrokenPipeError, OSError) as e:
                logger.error(f"FFmpeg pipe broken: {e}")
                self._ffmpeg_proc = None
        elif self._cv2_writer:
            # OpenCV expects BGR format
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            self._cv2_writer.write(frame_bgr)

        self._frame_count += 1

    def close(self) -> None:
        """Finish encoding and close file handles."""
        if self._ffmpeg_proc:
            if self._ffmpeg_proc.stdin:
                self._ffmpeg_proc.stdin.close()
            stderr_out = self._ffmpeg_proc.stderr.read() if self._ffmpeg_proc.stderr else b""
            self._ffmpeg_proc.wait()
            if self._ffmpeg_proc.returncode != 0:
                logger.warning(f"FFmpeg process exited with code {self._ffmpeg_proc.returncode}: {stderr_out.decode('utf-8', errors='ignore')}")
            self._ffmpeg_proc = None

        if self._cv2_writer:
            self._cv2_writer.release()
            self._cv2_writer = None

        logger.info(f"Video encoded successfully: {self._frame_count} frames -> {self.output_path}")

    def __enter__(self) -> VideoEncoder:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
