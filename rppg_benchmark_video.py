"""
Benchmark video iterator for prerecorded video files.

Provides deterministic frame iteration with reliable timestamp derivation
from video timing metadata, not wall-clock time.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Iterator, Tuple
import os


@dataclass
class VideoFrame:
    """Single frame from a prerecorded video."""
    frame: np.ndarray
    frame_index: int
    timestamp_seconds: float
    fps: float
    width: int
    height: int


class VideoIterator:
    """
    Deterministic prerecorded-video iterator.

    IMPORTANT: Uses video timing metadata for timestamps, NOT wall-clock time.
    This ensures the benchmark timeline is derived from the recording itself.
    """

    def __init__(self, video_path: str):
        """
        Initialize iterator with a video file.

        Parameters
        ----------
        video_path : str
            Path to video file (mp4, avi, etc. supported by OpenCV).

        Raises
        ------
        FileNotFoundError
            If video file does not exist.
        ValueError
            If video cannot be opened or has invalid FPS.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        # Obtain video metadata
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._duration = self._frame_count / self._fps if self._fps > 0 else 0.0

        # Validate FPS
        if self._fps <= 0 or not np.isfinite(self._fps):
            raise ValueError(
                f"Invalid video FPS: {self._fps}. "
                "Cannot derive reliable timestamps."
            )

        # Current position
        self._frame_index = 0
        self._timestamp_offset = 0.0  # Optional offset for alignment

    @property
    def fps(self) -> float:
        """Frames per second from video metadata."""
        return self._fps

    @property
    def frame_count(self) -> int:
        """Total number of frames in video."""
        return self._frame_count

    @property
    def duration_seconds(self) -> float:
        """Video duration in seconds."""
        return self._duration

    @property
    def width(self) -> int:
        """Video frame width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Video frame height in pixels."""
        return self._height

    @property
    def metadata(self) -> dict:
        """Video metadata dictionary."""
        return {
            "fps": self._fps,
            "frame_count": self._frame_count,
            "width": self._width,
            "height": self._height,
            "duration_seconds": self._duration,
        }

    def _compute_timestamp(self, frame_index: int) -> float:
        """
        Compute timestamp from frame index and FPS.

        This is the critical method: timestamp is derived from video timing,
        not from wall-clock time.time().
        """
        return float(frame_index) / self._fps

    def __iter__(self) -> Iterator[VideoFrame]:
        """Iterate through video frames in deterministic order."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._frame_index = 0

        while True:
            ret, frame = self._cap.read()
            if not ret:
                break

            timestamp = self._compute_timestamp(self._frame_index)
            yield VideoFrame(
                frame=frame.copy(),
                frame_index=self._frame_index,
                timestamp_seconds=timestamp,
                fps=self._fps,
                width=self._width,
                height=self._height,
            )
            self._frame_index += 1

        self._cap.release()

    def read_frame(self) -> Optional[VideoFrame]:
        """
        Read next single frame (alternative to iterator).

        Returns None when video ends.
        """
        ret, frame = self._cap.read()
        if not ret:
            self._cap.release()
            return None

        timestamp = self._compute_timestamp(self._frame_index)
        video_frame = VideoFrame(
            frame=frame.copy(),
            frame_index=self._frame_index,
            timestamp_seconds=timestamp,
            fps=self._fps,
            width=self._width,
            height=self._height,
        )
        self._frame_index += 1
        return video_frame

    def reset(self) -> None:
        """Reset to beginning of video."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._frame_index = 0

    def seek_frame(self, frame_index: int) -> None:
        """Seek to specific frame index."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self._frame_index = frame_index

    def close(self) -> None:
        """Release video capture resources."""
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
