"""
Quick integration test for FaceLandmarker adapter.
"""
import sys
import os
sys.path.insert(0, 'D:/main/Projects/Health')

import cv2
import numpy as np

# Test 1: Direct adapter usage
print("=" * 60)
print("TEST 1: Direct FaceLandmarkerAdapter usage")
print("=" * 60)

from rppg_benchmark_face_landmarker import FaceMeshWrapper, FaceLandmarkerAdapter

# Create test face image
test_img = np.ones((480, 640, 3), dtype=np.uint8) * 200
cv2.ellipse(test_img, (320, 240), (140, 180), 0, 0, 360, (180, 140, 120), -1)
cv2.circle(test_img, (270, 210), 25, (255, 255, 255), -1)
cv2.circle(test_img, (270, 210), 12, (50, 80, 120), -1)
cv2.circle(test_img, (270, 210), 5, (0, 0, 0), -1)
cv2.circle(test_img, (370, 210), 25, (255, 255, 255), -1)
cv2.circle(test_img, (370, 210), 12, (50, 80, 120), -1)
cv2.circle(test_img, (370, 210), 5, (0, 0, 0), -1)

adapter = FaceMeshWrapper()
rgb_frame = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
result = adapter.process(rgb_frame)

if result.multi_face_landmarks:
    landmarks = result.multi_face_landmarks[0]
    print(f"✅ SUCCESS: Detected {len(landmarks)} landmarks")

    # Verify V2 indices
    forehead = [109, 67, 108, 151, 337, 297, 338]
    for idx in forehead:
        if idx < len(landmarks):
            lm = landmarks[idx]
            print(f"   Forehead index {idx}: ({lm.x:.4f}, {lm.y:.4f})")
else:
    print("❌ FAIL: No face detected")

adapter.close()

# Test 2: Benchmark runner initialization
print()
print("=" * 60)
print("TEST 2: Benchmark runner FaceMesh initialization")
print("=" * 60)

# This tests that the _initialize_face_mesh function works
from rppg_benchmark_run import BenchmarkRunner

print("Testing benchmark runner face mesh initialization...")
# Note: We can't fully test without video, but we can check the import works

try:
    from rppg_benchmark_face_landmarker import get_face_landmarker
    fm = get_face_landmarker(max_num_faces=1)
    print("✅ FaceLandmarker via get_face_landmarker() works")
    fm.close()
except Exception as e:
    print(f"❌ FaceLandmarker initialization failed: {e}")

# Test 3: Verify V2 ROI compatibility
print()
print("=" * 60)
print("TEST 3: V2 ROI compatibility verification")
print("=" * 60)

from rppg_benchmark_face_landmarker import FaceLandmarkerAdapter

adapter = FaceLandmarkerAdapter()
print(f"✅ FaceLandmarkerAdapter initialized")
print(f"   V2 Forehead landmarks: {adapter.V2_FOREHEAD_LANDMARKS}")
print(f"   V2 Left cheek landmarks: {adapter.V2_CHEEK_LEFT_LANDMARKS[:5]}...")
print(f"   V2 Right cheek landmarks: {adapter.V2_CHEEK_RIGHT_LANDMARKS[:5]}...")
adapter.close()

print()
print("=" * 60)
print("ALL TESTS COMPLETED")
print("=" * 60)
