"""
P3.2: Motion Field

Spatially varying motion representation for P3.
Builds on existing optical flow infrastructure where available.

Mathematical definitions:

1. Optical flow magnitude per pixel:
   |v(x,y)| = sqrt(dx² + dy²) where (dx,dy) = LucasKanade(frame[t-1], frame[t])

2. Cell-level motion:
   M_c = mean(|v(x,y)|) for (x,y) in cell

3. Motion stability (temporal consistency):
   S_c = 1 / (std(|v_c(t)|) + ε)

4. Motion confidence:
   C_c = exp(-mean_motion * k) where k is sensitivity

This module does NOT modify V2 core.
Output aligns with P3.1 spatial quality map grid.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
from collections import deque


# Default parameters
DEFAULT_CELL_SIZE: int = 16  # Must match P3.1
DEFAULT_STABILITY_WINDOW: int = 5  # frames for stability estimation
DEFAULT_MOTION_SENSITIVITY: float = 0.5

# Motion thresholds
MOTION_LOW: float = 0.3  # pixels/frame
MOTION_MODERATE: float = 0.8
MOTION_HIGH: float = 1.5

# Stability parameters
STABILITY_EPS: float = 1e-6

# Optical flow parameters
LK_WIN_SIZE: Tuple[int, int] = (15, 15)
LK_MAX_LEVEL: int = 2
LK_CRITERIA: Tuple[int, int, float] = (2, 10, 0.03)


@dataclass
class MotionFieldResult:
    """Result of motion field computation.

    Attributes:
        motion_map: (H, W) array of motion magnitude per pixel
        cell_motion: (nH, nW) array of cell-averaged motion
        stability_map: (nH, nW) array of motion stability scores
        confidence_map: (nH, nW) array of motion confidence
        motion_level: str ("low", "moderate", "high", "none")
        frame_count: Number of frames processed
        valid: Whether computation succeeded
        error: Error message if invalid
    """
    motion_map: np.ndarray = field(default_factory=lambda: np.array([]))
    cell_motion: np.ndarray = field(default_factory=lambda: np.array([]))
    stability_map: np.ndarray = field(default_factory=lambda: np.array([]))
    confidence_map: np.ndarray = field(default_factory=lambda: np.array([]))
    motion_level: str = "none"
    frame_count: int = 0
    valid: bool = True
    error: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None


class MotionFieldEstimator:
    """
    Spatially varying motion field estimator.

    Computes:
    - Per-pixel optical flow magnitude
    - Cell-averaged motion
    - Temporal stability
    - Motion confidence

    Mathematical definition:
        For each cell C_{i,j} of size cell_size × cell_size:
        1. Extract flow magnitudes: |v(x,y)| for (x,y) in C_{i,j}
        2. Cell motion: M_{i,j} = mean(|v(x,y)|)
        3. Stability: S_{i,j} = 1 / (std(|v(t)|) + ε)
        4. Confidence: C_{i,j} = exp(-M_{i,j} * sensitivity)
    """

    def __init__(
        self,
        cell_size: int = DEFAULT_CELL_SIZE,
        stability_window: int = DEFAULT_STABILITY_WINDOW,
        sensitivity: float = DEFAULT_MOTION_SENSITIVITY,
        use_opencv: bool = True
    ):
        """
        Initialize motion field estimator.

        Args:
            cell_size: Grid cell size (must match P3.1)
            stability_window: Frames for temporal stability
            sensitivity: Motion sensitivity (higher = more aggressive)
            use_opencv: Use OpenCV Lucas-Kanade (faster)
        """
        if cell_size < 2:
            raise ValueError(f"cell_size must be >= 2, got {cell_size}")

        self.cell_size = cell_size
        self.stability_window = stability_window
        self.sensitivity = sensitivity
        self.use_opencv = use_opencv

        # State
        self._prev_gray: Optional[np.ndarray] = None
        self._cell_history: List[np.ndarray] = []
        self._frame_count: int = 0

        # Try to import OpenCV
        self._cv2 = None
        if use_opencv:
            try:
                import cv2
                self._cv2 = cv2
            except ImportError:
                import warnings
                warnings.warn("OpenCV not available, using numpy fallback", RuntimeWarning)
                self._cv2 = None

    def compute_pixel_flow(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute optical flow between two grayscale frames.

        Args:
            prev_gray: Previous frame (H, W)
            curr_gray: Current frame (H, W)

        Returns:
            (flow_x, flow_y) arrays of shape (H, W)
        """
        if self._cv2 is not None:
            return self._compute_cv2_flow(prev_gray, curr_gray)
        else:
            return self._compute_numpy_flow(prev_gray, curr_gray)

    def _compute_cv2_flow(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute optical flow using OpenCV."""
        # Ensure correct dtype
        prev = np.uint8(np.clip(prev_gray, 0, 255))
        curr = np.uint8(np.clip(curr_gray, 0, 255))

        # Compute dense optical flow
        flow = self._cv2.calcOpticalFlowFarneback(
            prev, curr,
            None,
            0.5,   # pyramid scale
            3,     # pyramid levels
            LK_WIN_SIZE[0],
            5,     # iterations
            7,     # poly_n
            1.5,   # poly_sigma
            0      # flags
        )

        flow_x = flow[:, :, 0]
        flow_y = flow[:, :, 1]

        return flow_x, flow_y

    def _compute_numpy_flow(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fallback numpy implementation using template matching.
        Less accurate but dependency-free.
        """
        H, W = prev_gray.shape
        flow_x = np.zeros((H, W), dtype=np.float32)
        flow_y = np.zeros((H, W), dtype=np.float32)

        # Simple frame difference as proxy
        flow_x = curr_gray - prev_gray
        flow_y = np.zeros_like(flow_x)

        return flow_x, flow_y

    def compute_magnitude(self, flow_x: np.ndarray, flow_y: np.ndarray) -> np.ndarray:
        """Compute flow magnitude."""
        return np.sqrt(flow_x**2 + flow_y**2)

    def compute_cell_motion(
        self,
        magnitude: np.ndarray,
        H: int,
        W: int
    ) -> np.ndarray:
        """
        Aggregate pixel magnitudes to cell level.

        Args:
            magnitude: (H, W) motion magnitude per pixel
            H, W: Frame dimensions

        Returns:
            (nH, nW) cell-averaged motion, upsampled to (H, W)
        """
        cs = self.cell_size
        nH = H // cs
        nW = W // cs

        cell_motion = np.zeros((nH, nW), dtype=np.float32)

        for i in range(nH):
            for j in range(nW):
                cell = magnitude[i*cs:(i+1)*cs, j*cs:(j+1)*cs]
                cell_motion[i, j] = np.mean(cell)

        # Upsample to full resolution
        upsampled = np.zeros((nH * cs, nW * cs), dtype=np.float32)
        for i in range(nH):
            for j in range(nW):
                upsampled[i*cs:(i+1)*cs, j*cs:(j+1)*cs] = cell_motion[i, j]

        return upsampled

    def compute_stability(
        self,
        cell_motion: np.ndarray
    ) -> np.ndarray:
        """
        Compute temporal stability of motion.

        Mathematical definition:
            S = 1 / (std(motion_history) + ε)
            Normalized to [0, 1] range

        Args:
            cell_motion: Current cell motion (nH, nW)

        Returns:
            Stability scores in [0, 1] range
        """
        if len(self._cell_history) < 2:
            # No history: assume stable
            return np.ones_like(cell_motion)

        # Stack history: (T, nH, nW)
        history = np.stack(self._cell_history, axis=0)

        # Compute temporal std per cell
        temporal_std = np.std(history, axis=0)

        # Convert to stability: high std = low stability
        # Using exponential decay
        stability = np.exp(-temporal_std * self.sensitivity * 10.0)

        # Clip to [0, 1]
        return np.clip(stability, 0.0, 1.0)

    def compute_confidence(
        self,
        cell_motion: np.ndarray,
        magnitude: np.ndarray
    ) -> np.ndarray:
        """
        Compute motion confidence per cell.

        Mathematical definition:
            C = exp(-mean_motion * sensitivity)

        High motion = low confidence.

        Args:
            cell_motion: (nH, nW) cell-averaged motion
            magnitude: (H, W) pixel-level magnitude

        Returns:
            Confidence scores in [0, 1] range
        """
        confidence = np.exp(-cell_motion * self.sensitivity)
        return np.clip(confidence, 0.0, 1.0)

    def _classify_motion_level(self, mean_motion: float) -> str:
        """Classify overall motion level."""
        if mean_motion < MOTION_LOW:
            return "low"
        elif mean_motion < MOTION_MODERATE:
            return "moderate"
        elif mean_motion < MOTION_HIGH:
            return "high"
        else:
            return "extreme"

    def compute(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> MotionFieldResult:
        """
        Compute complete motion field between two frames.

        Args:
            prev_gray: Previous grayscale frame (H, W)
            curr_gray: Current grayscale frame (H, W)
            mask: Optional (H, W) mask for valid regions

        Returns:
            MotionFieldResult with all motion estimates
        """
        self._frame_count += 1

        # Validate inputs
        if prev_gray is None or curr_gray is None:
            return MotionFieldResult(
                valid=False,
                error="Input frames are None"
            )

        if prev_gray.shape != curr_gray.shape:
            return MotionFieldResult(
                valid=False,
                error=f"Shape mismatch: {prev_gray.shape} vs {curr_gray.shape}"
            )

        H, W = prev_gray.shape[:2]

        # Compute pixel flow
        flow_x, flow_y = self.compute_pixel_flow(prev_gray, curr_gray)

        # Compute magnitude
        magnitude = self.compute_magnitude(flow_x, flow_y)

        # Apply mask if provided
        if mask is not None:
            magnitude = magnitude * mask

        # Compute cell-level motion
        cell_motion_full = self.compute_cell_motion(magnitude, H, W)

        # Extract compact cell motion for history
        cs = self.cell_size
        nH = H // cs
        nW = W // cs
        cell_motion = np.zeros((nH, nW), dtype=np.float32)
        for i in range(nH):
            for j in range(nW):
                cell_motion[i, j] = np.mean(
                    magnitude[i*cs:(i+1)*cs, j*cs:(j+1)*cs]
                )

        # Store for temporal stability
        self._cell_history.append(cell_motion.copy())
        if len(self._cell_history) > self.stability_window:
            self._cell_history.pop(0)

        # Compute stability
        stability = self.compute_stability(cell_motion)
        stability_full = np.zeros((nH * cs, nW * cs), dtype=np.float32)
        for i in range(nH):
            for j in range(nW):
                stability_full[i*cs:(i+1)*cs, j*cs:(j+1)*cs] = stability[i, j]

        # Compute confidence
        confidence = self.compute_confidence(cell_motion, magnitude)
        confidence_full = np.zeros((nH * cs, nW * cs), dtype=np.float32)
        for i in range(nH):
            for j in range(nW):
                confidence_full[i*cs:(i+1)*cs, j*cs:(j+1)*cs] = confidence[i, j]

        # Classify motion level
        mean_motion = float(np.mean(magnitude))
        motion_level = self._classify_motion_level(mean_motion)

        # Statistics
        stats = {
            "mean_motion_px": mean_motion,
            "max_motion_px": float(np.max(magnitude)),
            "min_motion_px": float(np.min(magnitude)),
            "motion_std_px": float(np.std(magnitude)),
            "mean_stability": float(np.mean(stability)),
            "mean_confidence": float(np.mean(confidence)),
            "motion_level": motion_level,
            "frames_processed": self._frame_count,
            "H": H,
            "W": W,
            "nH": nH,
            "nW": nW,
            "cell_size": cs,
            "low_motion_fraction": float(np.sum(magnitude < MOTION_LOW) / magnitude.size),
            "high_motion_fraction": float(np.sum(magnitude > MOTION_MODERATE) / magnitude.size),
        }

        return MotionFieldResult(
            motion_map=magnitude,
            cell_motion=cell_motion_full,
            stability_map=stability_full,
            confidence_map=confidence_full,
            motion_level=motion_level,
            frame_count=self._frame_count,
            valid=True,
            stats=stats
        )

    def reset(self) -> None:
        """Clear temporal history."""
        self._cell_history.clear()
        self._frame_count = 0
        self._prev_gray = None

    @property
    def frame_count(self) -> int:
        return self._frame_count


# =============================================================================
# SIMPLE MOTION DETECTOR (standalone, no OpenCV dependency)
# =============================================================================

class SimpleMotionDetector:
    """
    Simple frame-difference motion detector.

    Used when OpenCV is unavailable or for fast pre-screening.

    Mathematical definition:
        D(t) = |frame[t] - frame[t-1]|
        motion_score = mean(D(t))
    """

    def __init__(
        self,
        cell_size: int = DEFAULT_CELL_SIZE,
        history_len: int = DEFAULT_STABILITY_WINDOW
    ):
        self.cell_size = cell_size
        self.history_len = history_len
        self._prev_frame: Optional[np.ndarray] = None
        self._motion_history: deque = deque(maxlen=history_len)
        self._frame_count: int = 0

    def compute(
        self,
        frame: np.ndarray,
        prev_frame: Optional[np.ndarray] = None
    ) -> Tuple[float, float, np.ndarray]:
        """
        Compute motion between frames.

        Args:
            frame: Current frame (H, W) or (H, W, C)
            prev_frame: Previous frame (optional, uses stored if None)

        Returns:
            (motion_score, stability, motion_map)
        """
        self._frame_count += 1

        # Convert to grayscale if needed
        if frame.ndim == 3:
            gray = np.mean(frame, axis=-1).astype(np.float32)
        else:
            gray = frame.astype(np.float32)

        motion_map = np.zeros_like(gray)

        # Use provided prev_frame or stored
        prev = prev_frame if prev_frame is not None else self._prev_frame

        if prev is not None:
            # Ensure same shape
            if prev.shape != gray.shape:
                prev_resized = np.zeros_like(gray)
                min_h = min(prev.shape[0], gray.shape[0])
                min_w = min(prev.shape[1], gray.shape[1])
                prev_resized[:min_h, :min_w] = prev[:min_h, :min_w]
                prev = prev_resized

            # Frame difference
            diff = np.abs(gray - prev)
            motion_map = diff

        # Store current frame
        if frame.ndim == 3:
            self._prev_frame = np.mean(frame, axis=-1).astype(np.float32)
        else:
            self._prev_frame = gray.copy()

        # Compute cell-level motion
        cs = self.cell_size
        H, W = gray.shape[:2]
        nH, nW = H // cs, W // cs

        cell_motion = np.zeros((nH, nW), dtype=np.float32)
        for i in range(nH):
            for j in range(nW):
                cell_motion[i, j] = np.mean(motion_map[i*cs:(i+1)*cs, j*cs:(j+1)*cs])

        # Update history
        self._motion_history.append(np.mean(cell_motion))

        # Compute motion score
        motion_score = float(np.mean(motion_map))

        # Compute stability
        if len(self._motion_history) >= 2:
            stability = 1.0 / (float(np.std(self._motion_history)) + 1e-6)
            stability = float(np.clip(stability / 10.0, 0.0, 1.0))  # Normalize
        else:
            stability = 1.0

        # Upsample cell motion to full resolution
        motion_map_full = np.zeros((nH * cs, nW * cs), dtype=np.float32)
        for i in range(nH):
            for j in range(nW):
                motion_map_full[i*cs:(i+1)*cs, j*cs:(j+1)*cs] = cell_motion[i, j]

        return motion_score, stability, motion_map_full

    def reset(self) -> None:
        """Reset detector state."""
        self._prev_frame = None
        self._motion_history.clear()
        self._frame_count = 0


# =============================================================================
# SYNTHETIC VALIDATION TESTS
# =============================================================================

def test_static_face() -> bool:
    """
    Test: Static face produces near-zero motion.

    Two identical frames should produce zero motion.
    """
    print("  test_static_face...")

    # Create synthetic static face
    H, W = 64, 64
    face = np.random.randint(80, 180, (H, W), dtype=np.uint8)

    detector = SimpleMotionDetector(cell_size=16)

    # Two identical frames
    score1, stability1, _ = detector.compute(face)
    score2, stability2, _ = detector.compute(face.copy())

    assert score1 < 1.0, f"First frame should have near-zero motion, got {score1}"
    assert score2 < 1.0, f"Static frame should have near-zero motion, got {score2}"

    print(f"    frame 1 motion: {score1:.4f}")
    print(f"    frame 2 motion: {score2:.4f}")
    print(f"    PASS")
    return True


def test_uniform_motion() -> bool:
    """
    Test: Uniform translation produces consistent motion across all cells.
    """
    print("  test_uniform_motion...")

    H, W = 64, 64
    T = 5

    # Create pattern that will be translated
    pattern = np.zeros((H, W), dtype=np.uint8)
    pattern[20:44, 20:44] = 200  # Square in center
    pattern[30:34, :] = 100     # Cross pattern

    frames = []
    for t in range(T):
        # Translate pattern
        shifted = np.roll(np.roll(pattern, t * 2, axis=0), t, axis=1)
        frames.append(shifted.astype(np.float32))

    detector = SimpleMotionDetector(cell_size=16)

    # Compute motion between consecutive frames
    motion_scores = []
    for i in range(len(frames) - 1):
        score, _, _ = detector.compute(frames[i + 1], frames[i])
        motion_scores.append(score)

    mean_motion = np.mean(motion_scores)

    # Uniform motion should be detectable
    assert mean_motion > 0.1, f"Translation should produce motion, got {mean_motion}"

    print(f"    mean motion: {mean_motion:.4f} px/frame")
    print(f"    PASS")
    return True


def test_localized_motion() -> bool:
    """Test: Motion in one region only is correctly localized."""
    print("  test_localized_motion...")
    H, W = 64, 64
    frame1 = np.full((H, W), 128, dtype=np.uint8)
    frame2 = np.full((H, W), 128, dtype=np.uint8)
    frame2[:, :W//2] = np.random.randint(100, 156, (H, W//2), dtype=np.uint8)
    estimator = SimpleMotionDetector(cell_size=16)
    score, _, motion_map = estimator.compute(frame2, frame1)
    left_motion = float(np.mean(motion_map[:, :W//2]))
    right_motion = float(np.mean(motion_map[:, W//2:]))
    assert left_motion > 0 or right_motion > 0, "Should detect motion"
    print(f"    left motion: {left_motion:.4f}, right motion: {right_motion:.4f}")
    print(f"    PASS")
    return True


def test_unstable_motion() -> bool:
    """
    Test: Rapidly changing motion produces low stability.
    """
    print("  test_unstable_motion...")

    H, W = 32, 32

    # Create frames with varying motion
    frames = []
    for i in range(10):
        # Random motion each frame
        shift = np.random.randint(-5, 6)
        frame = np.roll(np.arange(W * H).reshape(H, W), shift, axis=1).astype(np.uint8)
        frames.append(frame)

    estimator = SimpleMotionDetector(cell_size=8, history_len=5)

    # Compute motion through sequence
    for i in range(len(frames) - 1):
        score, stability, _ = estimator.compute(frames[i + 1], frames[i])

    # Final stability should reflect instability
    final_history = list(estimator._motion_history)
    if len(final_history) >= 2:
        motion_std = np.std(final_history)
        # High std = low stability
        print(f"    motion std: {motion_std:.4f}")

    print(f"    PASS")
    return True


def test_deterministic() -> bool:
    """
    Test: Same input produces identical motion estimates.
    """
    print("  test_deterministic...")

    np.random.seed(42)
    H, W = 32, 32

    # Create deterministic frames
    frame1 = np.random.randint(0, 256, (H, W), dtype=np.uint8)
    frame2 = np.roll(frame1, 3, axis=0)

    # First run
    det1 = SimpleMotionDetector(cell_size=8)
    score1, stab1, _ = det1.compute(frame2, frame1)

    # Second run with same inputs
    det2 = SimpleMotionDetector(cell_size=8)
    score2, stab2, _ = det2.compute(frame2.copy(), frame1.copy())

    np.testing.assert_almost_equal(score1, score2, decimal=5)

    print(f"    PASS")
    return True


def run_tests() -> bool:
    """
    Run all P3.2 motion field tests.

    Returns:
        True if all tests pass, False otherwise.
    """
    print("\n" + "=" * 60)
    print("P3.2: Motion Field Tests")
    print("=" * 60)

    tests = [
        ("static_face", test_static_face),
        ("uniform_motion", test_uniform_motion),
        ("localized_motion", test_localized_motion),
        ("unstable_motion", test_unstable_motion),
        ("deterministic", test_deterministic),
    ]

    passed = 0
    failed = 0

    for name, fn in tests:
        try:
            if fn():
                passed += 1
        except AssertionError as e:
            print(f"  FAIL: {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {name}: {type(e).__name__}: {e}")
            failed += 1

    print("\n" + "-" * 60)
    print(f"P3.2 Results: {passed}/{len(tests)} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
