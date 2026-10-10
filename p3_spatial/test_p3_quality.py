"""
P3.1 Spatial Quality Map - Synthetics tests."""

import numpy as np


def test_uniform():
    print("  uniform_quality...")
    fps = 30.0
    T, H, W = 300, 64, 64
    t = np.arange(T) / fps
    sig = 100 + 20 * np.sin(2 * np.pi * 72.0 / 60.0 * t)
    buf = np.zeros((T, H, W), dtype=np.float32)
    for i in range(T):
        buf[i] = sig[i]
    from p3_spatial.p3_quality_map import SpatialQualityMap
    r = SpatialQualityMap(cell_size=16).compute(buf, fps)
    assert r.valid, f"Result invalid: {r.error}"
    assert r.quality_map.shape == (H, W)
    assert np.all(np.isfinite(r.quality_map))
    print(f"    shape: {r.quality_map.shape}")
    return True


def test_high_low():
    print("  high_vs_low...")
    fps = 30.0
    T, H, W = 300, 32, 32
    t = np.arange(T) / fps
    buf = np.zeros((T, H, W), dtype=np.float32)
    for i in range(T):
        buf[i, :H//2] = 100 + 20 * np.sin(2 * np.pi * 72.0 / 60.0 * t[i])
        buf[i, H//2:] = 100 + 2 * np.random.randn(H//2, W)
    from p3_spatial.p3_quality_map import SpatialQualityMap
    r = SpatialQualityMap(cell_size=8).compute(buf, fps)
    top = float(np.mean(r.quality_map[:H//2]))
    bot = float(np.mean(r.quality_map[H//2:]))
    print(f"    top={top:.2f} dB bot={bot:.2f} dB")
    assert top > bot
    return True


def test_invalid():
    print("  invalid_inputs...")
    fps = 30.0
    from p3_spatial.p3_quality_map import SpatialQualityMap
    r = SpatialQualityMap().compute(np.zeros((0, 10, 10)), fps)
    assert not r.valid
    r = SpatialQualityMap().compute(np.zeros((10, 32, 32)), fps)
    assert not r.valid
    print("    handled correctly")
    return True


def test_deterministic():
    print("  deterministic...")
    np.random.seed(42)
    buf = np.random.randn(120, 16, 16).astype(np.float32)
    from p3_spatial.p3_quality_map import SpatialQualityMap
    r1 = SpatialQualityMap().compute(buf.copy(), 30.0)
    np.random.seed(42)
    r2 = SpatialQualityMap().compute(buf.copy(), 30.0)
    np.testing.assert_array_equal(r1.quality_map, r2.quality_map)
    print("    PASS")
    return True


def run():
    print("P3.1 Spatial Quality Map")
    print("-" * 40)
    tests = [
        ("uniform", test_uniform),
        ("high_vs_low", test_high_low),
        ("invalid_inputs", test_invalid),
        ("deterministic", test_deterministic),
    ]
    passed = 0
    for name, fn in tests:
        try:
            if fn():
                print(f"  {name}: PASS")
                passed += 1
            else:
                print(f"  {name}: FAIL")
        except Exception as e:
            print(f"  {name}: ERROR {e}")
    print(f"\nP3.1: {passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
