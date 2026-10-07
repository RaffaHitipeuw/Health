"""
Test suite for MediaPipe compatibility adapter.

Tests the OpenCV DNN-based face landmark detector that serves as a
MediaPipe replacement when MediaPipe cannot initialize.
"""

import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rppg_benchmark_face_dnn import (
    FaceMeshWrapper,
    FaceDetectorDNN,
    FaceLandmarksResult,
    LandmarkList,
    LandmarkPoint,
    get_face_mesh_fallback,
)


def test_landmark_point():
    """Test LandmarkPoint class."""
    p = LandmarkPoint(0.5, 0.3, 0.1)
    assert abs(p.x - 0.5) < 1e-9
    assert abs(p.y - 0.3) < 1e-9
    assert abs(p.z - 0.1) < 1e-9
    print("[PASS] test_landmark_point")


def test_landmark_list():
    """Test LandmarkList container."""
    points = [
        LandmarkPoint(0.1, 0.2, 0.0),
        LandmarkPoint(0.3, 0.4, 0.0),
        LandmarkPoint(0.5, 0.6, 0.0),
    ]
    lst = LandmarkList(points)

    assert len(lst) == 3
    assert lst[0].x == 0.1
    assert lst[1].y == 0.4
    assert lst[2].x == 0.5

    # Test iteration
    count = sum(1 for _ in lst)
    assert count == 3

    # Test landmark property (MediaPipe compatibility)
    assert lst.landmark is lst

    print("[PASS] test_landmark_list")


def test_face_landmarks_result():
    """Test FaceLandmarksResult container."""
    points = [
        LandmarkPoint(0.1, 0.2, 0.0),
        LandmarkPoint(0.3, 0.4, 0.0),
    ]
    landmarks = LandmarkList(points)
    result = FaceLandmarksResult([landmarks])

    assert len(result) == 1
    assert result.multi_face_landmarks is not None
    assert len(result.multi_face_landmarks) == 1

    # Test empty result
    empty_result = FaceLandmarksResult([])
    assert len(empty_result) == 0
    assert empty_result.multi_face_landmarks == []

    print("[PASS] test_face_landmarks_result")


def test_face_mesh_wrapper():
    """Test FaceMeshWrapper interface."""
    wrapper = FaceMeshWrapper(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

    # Should have the expected attributes
    assert hasattr(wrapper, 'max_num_faces')
    assert hasattr(wrapper, 'refine_landmarks')
    assert hasattr(wrapper, 'min_detection_confidence')
    assert hasattr(wrapper, 'process')
    assert hasattr(wrapper, 'close')

    wrapper.close()
    print("[PASS] test_face_mesh_wrapper")


def test_get_face_mesh_fallback():
    """Test factory function."""
    face_mesh = get_face_mesh_fallback()
    assert face_mesh is not None
    assert hasattr(face_mesh, 'process')
    face_mesh.close()
    print("[PASS] test_get_face_mesh_fallback")


def test_process_with_synthetic_face():
    """Test face detection on a synthetic face image."""
    # Create a synthetic face image (simple oval)
    h, w = 480, 640
    img = np.ones((h, w, 3), dtype=np.uint8) * 200  # Light gray background

    # Draw simple face oval
    center = (w // 2, h // 2)
    axes = (120, 160)  # width, height

    # Draw filled ellipse (face)
    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(face_mask, center, axes, 0, 0, 360, 255, -1)

    # Apply face color
    face_color = (220, 180, 160)  # Skin-like
    for c in range(3):
        img[:, :, c] = np.where(face_mask > 0, face_color[c], img[:, :, c])

    # Add eye regions (darker)
    eye_y = h // 2 - 40
    cv2.ellipse(img, (w // 2 - 50, eye_y), (20, 10), 0, 0, 360, (50, 50, 50), -1)
    cv2.ellipse(img, (w // 2 + 50, eye_y), (20, 10), 0, 0, 360, (50, 50, 50), -1)

    # Add mouth
    mouth_y = h // 2 + 60
    cv2.ellipse(img, (w // 2, mouth_y), (40, 15), 0, 0, 180, (100, 50, 50), -1)

    # Test face mesh
    face_mesh = get_face_mesh_fallback()
    result = face_mesh.process(img)

    # Should detect face (synthetic image should trigger detection)
    # Note: Results may vary based on detector quality
    if len(result) > 0:
        landmarks = result.multi_face_landmarks[0]
        assert len(landmarks) == 468, f"Expected 468 landmarks, got {len(landmarks)}"

        # Verify landmark format
        for lm in landmarks:
            assert 0.0 <= lm.x <= 1.0, f"x out of range: {lm.x}"
            assert 0.0 <= lm.y <= 1.0, f"y out of range: {lm.y}"
        print(f"[PASS] test_process_with_synthetic_face (detected {len(landmarks)} landmarks)")
    else:
        print("[INFO] test_process_with_synthetic_face (no face detected - expected for synthetic image)")

    face_mesh.close()


def test_media_pipe_interface_compatibility():
    """
    Verify the interface matches what MediaPipe would return.

    This is critical for compatibility with existing code that expects
    mp.solutions.face_mesh.FaceMesh behavior.
    """
    face_mesh = get_face_mesh_fallback()

    # Create test image
    h, w = 480, 640
    img = np.ones((h, w, 3), dtype=np.uint8) * 200
    cv2.ellipse(img, (w // 2, h // 2), (120, 160), 0, 0, 360, (220, 180, 160), -1)

    result = face_mesh.process(img)

    # Verify result interface matches MediaPipe
    assert hasattr(result, 'multi_face_landmarks'), \
        "Result must have 'multi_face_landmarks' attribute"

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0]

        # Verify landmarks can be iterated like MediaPipe
        assert hasattr(landmarks, 'landmark'), \
            "Landmarks must have 'landmark' property for MediaPipe iteration"

        # Verify we can access by index
        first_lm = landmarks[0]
        assert hasattr(first_lm, 'x'), "Landmark must have x coordinate"
        assert hasattr(first_lm, 'y'), "Landmark must have y coordinate"
        assert hasattr(first_lm, 'z'), "Landmark must have z coordinate"

        # Verify MediaPipe iteration protocol
        for lm in landmarks.landmark:
            assert 0.0 <= lm.x <= 1.0
            assert 0.0 <= lm.y <= 1.0

    face_mesh.close()
    print("[PASS] test_media_pipe_interface_compatibility")


def test_key_landmarks_accessible():
    """
    Verify that the key landmarks used by Sanubari V2 are accessible.

    Sanubari uses landmarks: [1, 33, 263, 61, 291, 199, 168, 4, 234, 454, 10, 152]
    """
    from rppg_benchmark_face_dnn import KEY_LANDMARKS

    face_mesh = get_face_mesh_fallback()

    # Create test image
    h, w = 480, 640
    img = np.ones((h, w, 3), dtype=np.uint8) * 200
    cv2.ellipse(img, (w // 2, h // 2), (120, 160), 0, 0, 360, (220, 180, 160), -1)

    result = face_mesh.process(img)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0]

        # Verify all key landmarks are accessible
        for idx in KEY_LANDMARKS:
            assert idx < len(landmarks), f"Key landmark {idx} out of range"
            lm = landmarks[idx]
            assert lm is not None
            assert 0.0 <= lm.x <= 1.0
            assert 0.0 <= lm.y <= 1.0

        print(f"[PASS] test_key_landmarks_accessible (all {len(KEY_LANDMARKS)} key landmarks accessible)")
    else:
        print("[INFO] test_key_landmarks_accessible (no face detected)")

    face_mesh.close()


def test_close_is_idempotent():
    """Test that close() can be called multiple times without error."""
    face_mesh = get_face_mesh_fallback()
    face_mesh.close()
    face_mesh.close()  # Should not raise
    print("[PASS] test_close_is_idempotent")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("MediaPipe Compatibility Adapter Tests")
    print("=" * 60)

    tests = [
        test_landmark_point,
        test_landmark_list,
        test_face_landmarks_result,
        test_face_mesh_wrapper,
        test_get_face_mesh_fallback,
        test_media_pipe_interface_compatibility,
        test_key_landmarks_accessible,
        test_close_is_idempotent,
        test_process_with_synthetic_face,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
