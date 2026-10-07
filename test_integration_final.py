"""
Quick integration test for FaceLandmarker adapter - improved test.
"""
import sys
sys.path.insert(0, 'D:/main/Projects/Health')

import cv2
import numpy as np

print("=" * 60)
print("FaceLandmarker Adapter Integration Test")
print("=" * 60)

from rppg_benchmark_face_landmarker import FaceMeshWrapper, FaceLandmarkerAdapter

# Create a more realistic test face image with proper features
test_img = np.ones((480, 640, 3), dtype=np.uint8) * 220  # Light background

# Face oval
cv2.ellipse(test_img, (320, 240), (150, 180), 0, 0, 360, (195, 155, 135), -1)

# Hair
cv2.ellipse(test_img, (320, 160), (140, 60), 0, 180, 360, (80, 60, 50), -1)

# Eyebrows
cv2.ellipse(test_img, (265, 195), (40, 8), -10, 180, 360, (100, 80, 60), 5)
cv2.ellipse(test_img, (375, 195), (40, 8), 10, 180, 360, (100, 80, 60), 5)

# Eyes - white sclera
cv2.ellipse(test_img, (265, 215), (28, 18), 0, 0, 360, (250, 250, 250), -1)
cv2.ellipse(test_img, (375, 215), (28, 18), 0, 0, 360, (250, 250, 250), -1)

# Eyes - iris
cv2.circle(test_img, (265, 215), 14, (80, 120, 160), -1)
cv2.circle(test_img, (375, 215), 14, (80, 120, 160), -1)

# Eyes - pupil
cv2.circle(test_img, (265, 215), 6, (20, 20, 20), -1)
cv2.circle(test_img, (375, 215), 6, (20, 20, 20), -1)

# Eyes - highlight
cv2.circle(test_img, (260, 210), 4, (255, 255, 255), -1)
cv2.circle(test_img, (370, 210), 4, (255, 255, 255), -1)

# Nose
cv2.line(test_img, (320, 220), (312, 265), (165, 125, 110), 4)
cv2.line(test_img, (312, 265), (305, 285), (165, 125, 110), 3)
cv2.line(test_img, (312, 265), (328, 285), (165, 125, 110), 3)

# Mouth - upper lip
cv2.ellipse(test_img, (320, 315), (55, 12), 0, 0, 180, (180, 100, 100), 3)

# Mouth - lower lip
cv2.ellipse(test_img, (320, 325), (45, 15), 0, 180, 360, (200, 120, 120), -1)

# Neck
cv2.rectangle(test_img, (280, 400), (360, 480), (195, 155, 135), -1)

# Save test image
cv2.imwrite('D:/main/Projects/Health/validation_test_face_improved.png', test_img)
print("Saved improved test face image")

# Initialize adapter
adapter = FaceMeshWrapper()
rgb_frame = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
result = adapter.process(rgb_frame)
h, w = test_img.shape[:2]

if result.multi_face_landmarks:
    landmarks = result.multi_face_landmarks[0]
    print(f"\n✅ SUCCESS: Detected {len(landmarks)} landmarks")

    # V2 ROI verification
    forehead = [109, 67, 108, 151, 337, 297, 338]
    cheek_left = [50, 101, 118, 117, 116, 123, 147, 187, 207, 206, 205, 36, 142, 126, 100, 203]
    cheek_right = [280, 330, 347, 346, 345, 352, 376, 411, 427, 426, 425, 266, 371, 355, 329, 423]

    # Analyze positions
    forehead_ys = [landmarks[i].y for i in forehead if i < len(landmarks)]
    left_xs = [landmarks[i].x for i in cheek_left if i < len(landmarks)]
    right_xs = [landmarks[i].x for i in cheek_right if i < len(landmarks)]

    print(f"\nV2 ROI Analysis:")
    print(f"  Forehead: Mean Y = {np.mean(forehead_ys):.4f} (expected ~0.3)")
    print(f"  Left cheek: Mean X = {np.mean(left_xs):.4f} (expected < 0.5)")
    print(f"  Right cheek: Mean X = {np.mean(right_xs):.4f} (expected > 0.5)")

    # Create visualization
    vis_img = test_img.copy()

    # Draw landmarks
    for idx in forehead:
        if idx < len(landmarks):
            lm = landmarks[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(vis_img, (x, y), 5, (0, 255, 0), -1)

    for idx in cheek_left[:8]:
        if idx < len(landmarks):
            lm = landmarks[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(vis_img, (x, y), 4, (255, 0, 0), -1)

    for idx in cheek_right[:8]:
        if idx < len(landmarks):
            lm = landmarks[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(vis_img, (x, y), 4, (0, 0, 255), -1)

    # Draw ROI polygons
    forehead_pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)]
                              for i in forehead if i < len(landmarks)], np.int32)
    left_pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)]
                          for i in cheek_left if i < len(landmarks)], np.int32)
    right_pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)]
                           for i in cheek_right if i < len(landmarks)], np.int32)

    if len(forehead_pts) >= 3:
        cv2.polylines(vis_img, [forehead_pts], True, (0, 255, 0), 2)
    if len(left_pts) >= 3:
        cv2.polylines(vis_img, [left_pts], True, (255, 0, 0), 2)
    if len(right_pts) >= 3:
        cv2.polylines(vis_img, [right_pts], True, (0, 0, 255), 2)

    cv2.imwrite('D:/main/Projects/Health/validation_test_final.png', vis_img)
    print("\nSaved visualization to validation_test_final.png")

else:
    print("\n❌ No face detected - MediaPipe needs a more realistic face")

adapter.close()

# Test V2 integration
print("\n" + "=" * 60)
print("V2 INTEGRATION TEST")
print("=" * 60)

try:
    from rppg_core import MultiROIFusionEngineV2, ROI_CONFIGS

    print(f"V2 ROI configs loaded:")
    for name, cfg in ROI_CONFIGS.items():
        print(f"  {name}: {len(cfg['landmarks'])} landmarks")

    # Create a mock landmarks result
    class MockLandmarks:
        def __init__(self, points):
            self._points = points

        @property
        def landmark(self):
            return self

        def __len__(self):
            return len(self._points)

        def __getitem__(self, key):
            return self._points[key]

    class MockResult:
        def __init__(self, landmarks):
            self._landmarks = landmarks

        @property
        def multi_face_landmarks(self):
            return self._landmarks

        def __len__(self):
            return len(self._landmarks)

    # Create dummy landmarks for V2 test
    class DummyPoint:
        def __init__(self, x, y, z=0):
            self.x = x
            self.y = y
            self.z = z

    dummy_landmarks = [DummyPoint(0.5, 0.3) for _ in range(478)]

    mock_result = MockResult([MockLandmarks(dummy_landmarks)])

    # Test V2 fusion engine with mock landmarks
    engine = MultiROIFusionEngineV2()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # This should work without errors
    result = engine.update(frame, mock_result.multi_face_landmarks[0], 480, 640)
    print(f"\n✅ V2 FusionEngineV2 accepts landmark format: BPM={result.fused_bpm:.1f}, SQI={result.fused_sqi:.1f}")

except Exception as e:
    print(f"\n❌ V2 integration test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("INTEGRATION TEST COMPLETE")
print("=" * 60)
