"""Test mediapipe Image mode which may avoid the Video mode C binding issue."""
import mediapipe as mp
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision import RunningMode
import os

model_path = "D:/main/Projects/Health/models/face_landmarker.task"
print("Model size:", os.path.getsize(model_path), "bytes")

# Try IMAGE mode
try:
    opts = FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
    )
    detector = FaceLandmarker.create_from_options(opts)
    print("IMAGE mode: FaceLandmarker created OK")
    detector.close()
    print("IMAGE mode: closed OK")
except Exception as e:
    print("IMAGE mode failed:", type(e).__name__, str(e)[:300])

# Try VIDEO mode
try:
    opts2 = FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
    )
    detector2 = FaceLandmarker.create_from_options(opts2)
    print("VIDEO mode: FaceLandmarker created OK")
    detector2.close()
    print("VIDEO mode: closed OK")
except Exception as e:
    print("VIDEO mode failed:", type(e).__name__, str(e)[:300])
