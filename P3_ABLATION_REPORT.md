# P3 ABLATION REPORT

## PURPOSE

This report documents the ablation framework for P3 spatial/motion evolution, enabling isolated measurement of each component's contribution.

## ABLATION FRAMEWORK

P3 is designed for independent component ablation:

```
P3 Components:
├── P3.1: Spatial Quality Map
├── P3.2: Motion Field
├── P3.3: Motion-Aware Quality
├── P3.4: Dynamic Candidates
├── P3.5: Spatial-Temporal Consistency
└── P3.6: Pipeline Integration
```

## ISOLATION MECHANISMS

### P3.3 Weighting Modes

Each mode isolates a different combination:

```python
class WeightingMode(Enum):
    QUALITY_ONLY = "quality_only"     # Q only (ignores motion)
    MOTION_ONLY = "motion_only"       # C × S only (ignores quality)
    FULL = "full"                     # Q × C × S (all factors)
    ADDITIVE = "additive"             # Q × (w_c*C + w_s*S)
    HARMONIC = "harmonic"             # Q × C*S / (C + S + ε)
```

### P3.4 Component Enablement

```python
# Enable/disable candidate generation
selector = DynamicCandidateSelector(...)
# vs
# Use fixed candidate set from V2
```

### P3.5 Temporal Tracking

```python
# Enable temporal tracking
tracker = SpatialTemporalConsistency(window_size=10)
# vs
# Use instantaneous quality only
```

## EXPECTED ABLATION RESULTS

### Hypothesized Component Contributions

| Component | Expected Impact | Metric |
|----------|----------------|--------|
| P3.1 Quality Map | MAE reduction in high-quality regions | MAE |
| P3.2 Motion Field | MAE reduction in motion conditions | Motion-MAE |
| P3.3 Weighting | Improved robustness | Overall MAE |
| P3.4 Candidates | Better spatial selection | Spatial stability |
| P3.5 Temporal | Reduced noise | Temporal stability |

### Synthetic Validation

Since real-data is blocked, synthetic experiments demonstrate ablation isolation:

```python
# Test: Motion contamination downweighting
quality = high_value  # 0.9
confidence = low_value  # 0.1 (high motion)

# QUALITY_ONLY: ignores motion → high quality
# FULL: considers motion → low quality
```

## TEST RESULTS

All ablation isolation tests passed (47/47 P3 tests):

```
P3.3 Motion-Aware Quality:
- test_quality_only_mode: PASS (Q only = normalized Q)
- test_motion_only_mode: PASS (C × S only = C*S)
- test_full_mode: PASS (Q × C × S = normalized(Q) * C * S)
- test_high_motion_downweighting: PASS
- test_clean_region_preserved: PASS

P3.6 Pipeline Integration:
- test_pipeline_initialization: PASS (all 7 experiments)
- test_e0_baseline: PASS (no P3 processing)
- test_e1_quality_only: PASS
- test_e6_full_pipeline: PASS
```

## ABLATION ISOLATION VERIFICATION

### QUALITY_ONLY Mode
- Input: Q=10dB, C=0.1, S=0.1
- Expected: normalized(Q) = 0.75
- Actual: 0.75
- Status: ✓ VERIFIED

### MOTION_ONLY Mode
- Input: Q=0dB, C=0.8, S=0.9
- Expected: C*S = 0.72
- Actual: 0.72
- Status: ✓ VERIFIED

### FULL Mode
- Input: Q=10dB, C=0.8, S=0.9
- Expected: normalized(Q) × C × S = 0.75 × 0.72 = 0.54
- Actual: 0.54
- Status: ✓ VERIFIED

---

## LIMITATIONS

1. **Synthetic only** — Real-data ablation not possible
2. **Component coupling** — Some interactions unavoidable
3. **Metric selection** — MAE may not capture all improvements

---

*Ablation framework documentation for P3*
