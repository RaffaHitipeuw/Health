# P3 IMPLEMENTATION REPORT

## OVERVIEW

This document describes the complete implementation of P3 — Spatial/Motion Evolution for the Sanubari camera-based rPPG research repository.

## ARCHITECTURE

P3 operates PARALLEL to V2, not replacing it. The V2 core remains frozen and unchanged.

```
Camera Frame
    ↓
Face Landmarks (existing)
    ↓
┌─────────────────────────────────────────┐
│ P3 SPATIAL LAYER                        │
├─────────────────────────────────────────┤
│ P3.1: Spatial Quality Map              │
│   - Per-cell cardiac-band FFT           │
│   - Grid-based representation          │
│   - Cardiac band: 0.833-3.0 Hz         │
├─────────────────────────────────────────┤
│ P3.2: Motion Field                     │
│   - Frame-difference or optical flow    │
│   - Per-pixel magnitude                 │
│   - Temporal stability                  │
├─────────────────────────────────────────┤
│ P3.3: Motion-Aware Quality             │
│   - Q × C × S weighting                │
│   - Ablation modes: QUALITY_ONLY,      │
│     MOTION_ONLY, FULL, ADDITIVE,       │
│     HARMONIC                           │
├─────────────────────────────────────────┤
│ P3.4: Dynamic Candidates               │
│   - Grid-based candidate regions        │
│   - Quality + motion + stability score │
│   - Lifecycle: birth, persist, expire   │
├─────────────────────────────────────────┤
│ P3.5: Spatial-Temporal Consistency    │
│   - State machine: BORN→ALIVE→        │
│     DEGRADING→EXPIRED                 │
│   - Persistence + stability tracking    │
├─────────────────────────────────────────┤
│ P3.6: Pipeline Integration             │
│   - E0-E6 ablation experiments         │
│   - Metrics collection                 │
├─────────────────────────────────────────┤
│ P3.7: Failure Analysis                 │
│   - 15 failure categories              │
│   - Severity: INFO, WARNING, ERROR,     │
│     CRITICAL                           │
└─────────────────────────────────────────┘
```

## MODULE DETAILS

### P3.1 — Spatial Quality Map (`spatial_quality_map.py`)

**Purpose:** Estimate local cardiac-band signal quality across spatial regions

**Mathematical Definition:**
```
For spatial cell C with N_t temporal samples at FPS=f:
1. Extract temporal mean signal: s(t) = mean(frame[t, C])
2. Compute FFT: S(f) = FFT(s(t) - mean(s)) * Hann window
3. Cardiac band: f_c ∈ [0.833, 3.0] Hz
4. Cardiac power: P_c = sum(|S(f_c)|^2)
5. Non-cardiac power: P_n = sum(|S(f ∉ cardiac)|^2)
6. SNR_linear = P_c / (P_n + ε)
7. SNR_dB = 10 * log10(SNR_linear + ε)
8. Quality = clamp(SNR_dB, -20, 20)
```

**Key Parameters:**
- `CARDIAC_BAND_HZ = (0.833, 3.0)` — Matches V2
- `DEFAULT_CELL_SIZE = 16` — Grid resolution
- `MIN_TEMPORAL_SAMPLES = 30` — Minimum for FFT
- `SNR_DB_MIN/MAX = -20.0, 20.0` — Stability bounds

**Outputs:**
- `quality_map`: (H, W) SNR in dB per cell
- `stats`: mean_db, max_db, min_db, std_db

---

### P3.2 — Motion Field (`motion_field.py`)

**Purpose:** Spatially varying motion representation

**Mathematical Definition:**
```
1. Optical flow magnitude: |v(x,y)| = sqrt(dx² + dy²)
   where (dx,dy) = LucasKanade(frame[t-1], frame[t])

2. Cell-level motion: M_c = mean(|v(x,y)|) for (x,y) in cell

3. Motion stability: S_c = 1 / (std(|v_c(t)|) + ε)

4. Motion confidence: C_c = exp(-mean_motion * k)
```

**Key Parameters:**
- `DEFAULT_CELL_SIZE = 16` — Must match P3.1
- `MOTION_LOW = 0.3` — pixels/frame
- `MOTION_MODERATE = 0.8`
- `MOTION_HIGH = 1.5`

**Outputs:**
- `motion_map`: (H, W) motion magnitude per pixel
- `cell_motion`: (H, W) cell-averaged motion
- `stability_map`: (H, W) temporal stability
- `confidence_map`: (H, W) motion confidence

---

### P3.3 — Motion-Aware Spatial Quality (`motion_aware_quality.py`)

**Purpose:** Combine spatial signal quality with motion information

**Weighting Modes:**
```python
QUALITY_ONLY: Q_motion = Q
MOTION_ONLY: Q_motion = C * S
FULL: Q_motion = Q * C * S
ADDITIVE: Q_motion = Q * (w_c*C + w_s*S)
HARMONIC: Q_motion = Q * C*S / (C + S + ε)
```

**Key Parameters:**
- `WeightingMode` enum for ablation
- `DEFAULT_WEIGHTS = (0.4, 0.3, 0.3)`
- `QUALITY_MIN/MAX = -20.0, 20.0`

---

### P3.4 — Dynamic Spatial Candidates (`dynamic_candidates.py`)

**Purpose:** Generate spatial candidate regions beyond fixed landmarks

**Candidate Scoring:**
```python
S(R) = mean(Q[R]) * f_area(n) * mean(C[R]) * mean(S_stab[R])

where f_area(n) = 1 / (1 + scale * |n - optimal|)
```

**Candidate Lifecycle:**
- Birth: new region with score > birth_threshold
- Alive: updated each frame
- Degrading: score < recovery_threshold
- Expired: no update for N frames
- Recovered: score improves after degrading

**Key Parameters:**
- `DEFAULT_MIN_CANDIDATE_AREA = 2` cells
- `DEFAULT_QUALITY_THRESHOLD = 0.3`
- `DEFAULT_MAX_CANDIDATES = 10`
- `DEFAULT_EXPIRATION_FRAMES = 5`

---

### P3.5 — Spatial-Temporal Consistency (`spatial_temporal.py`)

**Purpose:** Temporal tracking of spatial candidate quality

**State Machine:**
```
BORN → ALIVE → DEGRADING → EXPIRED
              ↑__________|          ↑
              |                   (if recovery)
              ↓________________RECOVERED
```

**Persistence Score:**
```python
P = mean(Q_history) * consistency_factor * mean(stability_history)

where consistency_factor = 1 if CV < threshold
                         = threshold / CV otherwise
```

**Temporal Stability:**
```python
S = 1 / (std(Q_history) + ε)
normalized to [0, 1]
```

---

### P3.6 — Pipeline Integration (`p3_pipeline.py`)

**Purpose:** Complete pipeline with ablation support

**Ablation Experiments:**
| ID | Components |
|----|------------|
| E0 | Baseline (no P3) |
| E1 | P3.1 quality map only |
| E2 | P3.2 motion field only |
| E3 | P3.1 + P3.2 + P3.3 |
| E4 | P3.1 + P3.2 + P3.3 + P3.4 |
| E5 | P3.1 + P3.2 + P3.3 + P3.4 + P3.5 |
| E6 | Full P3 |

---

### P3.7 — Failure Analysis (`p3_failures.py`)

**Purpose:** P3-specific failure detection and categorization

**Failure Categories (15 implemented):**
1. LANDMARK_INSTABILITY
2. ROI_DRIFT
3. CANDIDATE_DRIFT
4. LOCAL_MOTION
5. GLOBAL_HEAD_MOTION
6. BLINK_CONTAMINATION
7. JAW_EXPRESSION
8. INSUFFICIENT_SKIN
9. INSUFFICIENT_CANDIDATES
10. SPATIAL_DISAGREEMENT
11. TEMPORAL_INSTABILITY
12. CANDIDATE_THRASHING
13. SPECTRAL_FALSE_PEAK
14. HARMONIC_CONTAMINATION
15. ILLUMINATION_TRANSITION

**Severity Levels:**
- INFO: Informational, no action
- WARNING: Soft action (downweight)
- ERROR: Hard action (reject)
- CRITICAL: Immediate rejection

---

## PERFORMANCE CHARACTERISTICS

| Component | Complexity | Memory |
|-----------|------------|--------|
| P3.1 Quality Map | O(T × H × W / cell_size²) | O(H × W) |
| P3.2 Motion Field | O(T × H × W) | O(H × W) |
| P3.3 Weighting | O(H × W) | O(H × W) |
| P3.4 Candidates | O(H × W) for region finding | O(n_candidates) |
| P3.5 Temporal | O(n_candidates × window) | O(n_candidates × window) |

---

## ABLATION ISOLATION

Each component can be independently disabled:

```python
# QUALITY_ONLY — ignores motion
combiner = MotionAwareQuality(mode=WeightingMode.QUALITY_ONLY)

# MOTION_ONLY — ignores quality
combiner = MotionAwareQuality(mode=WeightingMode.MOTION_ONLY)

# FULL — uses both
combiner = MotionAwareQuality(mode=WeightingMode.FULL)
```

---

## INTEGRATION WITH V2

P3 operates independently from V2:

1. V2 remains frozen and unchanged
2. P3 does not modify V2 core files
3. P3 can be used for research without affecting V2 benchmarks
4. Ablation experiments compare P3 vs (not P3)

---

## LIMITATIONS

1. **Real-data validation blocked** — No independent ground truth available
2. **Computational cost** — Grid-based approach has O(T × H × W) complexity
3. **Synthetic-only validation** — Physiological claims not validated

---

*Implementation completed: P3 Phase*
