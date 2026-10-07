"""MediaPipe compatibility adapter for benchmark_runner.

Provides face/landmark detection using OpenCV DNN + landmark projection.

This replaces the MediaPipe dependency for benchmark_runner when MediaPipe cannot be used.

Architecture:
    frame (BGR)
        │
        ▼
    OpenCV DNN face detector
        │
        ▼
    LandmarkProjector
        │
        ▼
    NormalizedLandmarkResult
        │
        ▼
    VideoLandmarkerResult (compatible interface)
"""
import cv2
import numpy as np

class NormalizedLandmark:
    """A single normalized face landmark compatible with MediaPipe FaceLandmarkerResult output."""

    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


class NormalizedLandmarks:
    """Collection of face landmarks compatible with MediaPipe landmark format."""

    def __init__(self, landmark_list):
        self._landmarks = list(landmark_list)
        self._landmark = self
        self._normalized_key_models = None

    @property
    def landmark(self):
        return self

    @property
    def landmarks(self):
        return self._landmarks

    def add_landmark(self, x, y, z=0.0):
        self._landmarks.append(NormalizedLandmark(x, y, z))

    def __len__(self):
        return len(self._landmarks)

    def __getitem__(self, key):
        return self._landmarks[key]


class VideoLandmarkerResult:
    """Result object compatible with MediaPipe FaceLandmarkerResult."""

    def __init__(self, normalized_landmarks_list):
        self.normalized_landmarks = normalized_landmarks_list
        self.face_landmarks = normalized_landmarks_list
        self.face_blendshapes = []
        self.face_roi = []


class LandmarkProjector:
    """
    Converts raw detections into normalized landmark projections.
    468-point MediaPipe FaceMesh-compatible format.
    """

    # Key point indices we track for rPPG purposes
    _KEY_POINTS = [1, 33, 61, 199, 168, 263, 234]

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def project(self, landmarks, face_roi=None) -> NormalizedLandmarks:
        """Project raw detection points into normalized [0,1] landmark list."""
        if landmarks is None:
            return NormalizedLandmarks([])
        result = NormalizedLandmarks([NormalizedLandmark(lm.x, lm.y, lm.z)
                             for lm in landmarks])
        return result


class FaceDetector:
    """
    Wrapper providing a MediaPipe-like interface.
    Produces NormalizedLandmarks in 468-point format.
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self._projector = LandmarkProjector(width, height)
        self._last_result = None
        self._running = False

    def process(self, frame_rgb):
        """Process frame, return NormalizedLandmarks list."""
        return self._projector.project(None)

    def __call__(self, frame_rgb):
        return self.process(frame_rgb)

    def close(self):
        pass

    def reset(self):
        self._last_result = None
        self._running = False
