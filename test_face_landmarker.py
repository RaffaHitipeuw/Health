"""Test face landmarker with downloaded model."""
import mediapipe as mp
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision import RunningMode
import os

model_path = "D:/main/Projects/Health/models/face_landmarker.task"
print(f"Model: {model_path}")
print(f"Size: {os.path.getsize(model_path)} bytes")

# Create detector
# Create detector
opts = FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
)
detector = FaceLandmarker.create_from_options(opts)
print("FaceLandmarker created successfully")
print(f"Methods: {[m for m in dir(detector) if not m.startswith('_')]}")
detector.close()
print("Closed successfully")
