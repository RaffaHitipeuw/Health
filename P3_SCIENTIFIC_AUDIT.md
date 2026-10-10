# P3.0 SCIENTIFIC AND ARCHITECTURE AUDIT

## EXECUTIVE SUMMARY

**Purpose:** Define spatial/motion evolution path without modifying frozen V2 core
**Status:** AUDIT PHASE - Implementation NOT STARTED

---

## 1. REPOSITORY ARCHITECTURE FINDINGS

### 1.1 V2 Core (FROZEN)

| File | Status | Purpose |
|------|--------|---------|
| rppg_core.py | FROZEN | MultiROIFusionEngineV2 |
| rppg_signal.py | FROZEN | Optical flow, pulse selector |
| rppg_roi.py | EXISTS | AdaptiveROIManager (skin-based only) |
| rppg_sqi.py | FROZEN | SQI computation |
| rppg_temporal.py | FROZEN | Kalman, probabilistic fusion |
| rppg_physio_state.py | FROZEN | Physiological state |
| rppg_uncertainty.py | EXISTS | Uncertainty weighting |

### 1.2 Existing Spatial Components

| Component | File | Actual Function | Scientific Purpose | Limitation |
|-----------|------|-----------------|-------------------|------------|
| AdaptiveROIManager | rppg_roi.py | Skin mask (YCbCr+HSV rules) | Skin presence detection | Uses luminance thresholds only |
| segment_skin() | rppg_roi.py | YCbCr + HSV union | Skin region mask | NOT perfusion map |
| get_adaptive_roi() | rppg_roi.py | Mean RGB within skin+polygon | Weighted mean extraction | No local quality |
| SkinToneCalibrator | rppg_core.py | LUM-A/B/C luminance grouping | Diagnostic label only | Not used in extraction |
| per_pixel_quality (in filter) | rppg_signal.py | Per-pixel pulse-richness | Candidate weighting | Within existing ROI only |

### 1.3 Existing Motion Components

| Component | File | Actual Function | Used For | Limitation |
|-----------|------|-----------------|----------|-------------|
| MotionArtifactDetector | rppg_core.py | Landmark + pixel diff | SQI penalty | Frame-level only |
| OpticalFlowROIStabilizer | rppg_signal.py | Lucas-Kanade within ROI | ROI stabilization | Within fixed ROI only |
| HeadPoseGate | rppg_core.py | Yaw/pitch rejection | Frame rejection | Binary gate |
| JawBlinkSuppressor | rppg_core.py | Jaw/blink detection | Cheek suppression | Binary weighting |
| ROIMicroMotionDetector | rppg_signal.py | Per-ROI motion penalty | Dynamic weighting | Within ROI only |

---

## 2. EXISTING SPATIAL MECHANISMS ANALYSIS

### 2.1 What Is Implemented vs What Is Claimed

| Claim | Actual Implementation | Gap |
|-------|---------------------|-----|
| "perfusion map" | **Skin mask only** (Cr/Cb range + H/S range) | No local pulse content |
| "skin segmentation" | Simple YCbCr + HSV union | No texture/quality |
| "adaptive ROI" | Fixed landmarks + skin intersection | No spatial candidate search |
| "spatial quality" | Mean RGB within mask | No local variance/quality |
| "LUM-A/B/C" | Luminance-only grouping | Not Fitzpatrick classification |

### 2.2 Terminology Audit

```
Current implementation uses:
  LUM-A: mean(RGB) > 180 (bright appearance)
  LUM-B: mean(RGB) > 120 (medium appearance)  
  LUM-C: otherwise (dark appearance)

This is NOT Fitzpatrick skin type classification.
This is luminance-based grouping only.

Terminology PRESERVED as LUM-A/B/C per audit requirement.
```

---

## 3. EXISTING MOTION MECHANISMS ANALYSIS

### 3.1 Motion Detection vs Motion Compensation vs Motion Weighting

| Capability | Implemented | Scope |
|------------|-------------|--------|
| Detecting motion | YES | MotionArtifactDetector |
| Rejecting motion frames | YES | HeadPoseGate |
| Compensating for motion | PARTIAL | OpticalFlowROIStabilizer (within ROI only) |
| Weighting by motion | YES | SQI + dynamic ROI weight penalties |
| Spatially relocating ROI | NO | Fixed landmark polygons |
| Local motion field | NO | Frame-level only |
| Per-ROI motion tracking | NO | Aggregate score only |

### 3.2 Critical Finding: Optical Flow Is ROI-Bounded

The `OpticalFlowROIStabilizer` stabilizes pixels WITHIN the existing ROI polygon.
It does NOT:
- Generate new spatial candidates
- Search for better ROI locations
- Adapt to local motion patterns outside fixed polygons

---

## 4. EXISTING DYNAMIC ROI ANALYSIS

### 4.1 Current ROI Selection Logic

```
V2 uses:
  1. Fixed landmark indices (FOREHEAD_LANDMARKS, CHEEK_LEFT_LANDMARKS, CHEEK_RIGHT_LANDMARKS)
  2. Polygon intersection with skin mask
  3. Mean RGB within intersection
  4. SQI-based dynamic weighting
  5. Probabilistic fusion
  6. Hierarchical clustering
```

### 4.2 What Is NOT Dynamic

| Component | Current Behavior | P3 Opportunity |
|------------|-------------------|----------------|
| ROI location | Fixed landmark polygons | Search for better candidates |
| ROI size | Fixed polygon | Adaptive to local quality |
| Candidate selection | Landmark intersection only | Per-pixel quality map |
| Spatial weighting | Uniform mean | Quality-weighted mean |
| Motion adaptation | SQI penalty | Spatially-varying motion field |

---

## 5. SCIENTIFIC GAP ANALYSIS

### 5.1 Genuine Gaps in Current Implementation

| Gap | Current State | Scientific Impact | P3 Value |
|-----|----------------|-------------------|----------|
| No local pulse quality map | Uniform mean extraction | Cannot prefer high-quality pixels | HIGH |
| No motion field | Binary motion score | Cannot spatially weight | HIGH |
| No spatial candidate search | Fixed landmarks only | May miss better regions | HIGH |
| No ROI trajectory tracking | Instantaneous only | Cannot detect drift | MEDIUM |
| No spatial-temporal consistency | Per-frame only | Cannot verify persistence | MEDIUM |
| No illumination robustness | Single brightness metric | Cannot adapt to gradients | MEDIUM |
| No local pulse agreement | ROI-level only | Cannot verify spatial coherence | MEDIUM |

### 5.2 Proposed P3 Capabilities Classification

| Capability | Classification | Justification |
|------------|----------------|----------------|
| Dense spatial quality map | MUST HAVE | Enables candidate selection |
| Motion field estimation | MUST HAVE | Enables spatial weighting |
| Motion-aware spatial weighting | MUST HAVE | Uses motion field |
| Dynamic candidate search | HIGH VALUE | Extends fixed ROIs |
| ROI trajectory tracking | HIGH VALUE | Detects drift |
| Spatial-temporal consistency | HIGH VALUE | Validates measurements |
| Local pulse agreement | OPTIONAL | May be premature |
| Illumination robustness | OPTIONAL | Requires validation first |
| Adaptive spatial selection | OPTIONAL | Depends on other components |

---

## 6. PROPOSED P3 ARCHITECTURE

### 6.1 Conceptual Pipeline (Non-Binding)

```
Camera Frame
    ↓
Face Landmarks (existing)
    ↓
┌─────────────────────────────────────────┐
│ P3 SPATIAL LAYER (NEW MODULE)          │
├─────────────────────────────────────────┤
│ 1. Skin/Quality Candidate Map           │
│    - Skin probability from pixels        │
│    - Local pulse richness from FFT       │
│    - No V2 modification                │
│ 2. Motion Field                       │
│    - Per-pixel LK optical flow         │
│    - Aggregated to patch level         │
│ 3. Spatial-Temporal Quality            │
│    - Quality history for patches        │
│    - Consistency scoring               │
│ 4. Dynamic Candidate Selection         │
│    - Patch-based instead of landmark   │
│    - Quality + motion weighted         │
│    - Bounded by skin probability      │
│ 5. Candidate → ROI Projection          │
│    - Project quality-selected regions   │
│    - Generate candidate polygons         │
└─────────────────────────────────────────┘
    ↓
Signal Extraction (NEW: per-candidate, P3 only)
    ↓
Classical Methods (NEW: P3 signal path)
    ↓
Fusion (NEW: P3 fusion, NOT V2 fusion)
    ↓
Benchmark

Note: P3 operates PARALLEL to V2, not replacing it.
      V2 remains frozen and unchanged.
```

### 6.2 Module Boundaries

```
P3.1: spatial_quality_map.py
  - Per-pixel cardiac band FFT
  - Skin probability integration
  - Output: quality_map[h, w]

P3.2: motion_field.py
  - Lucas-Kanade per pixel
  - Patch aggregation
  - Output: motion_field[h, w]

P3.3: spatial_weighting.py
  - Quality * motion weighting
  - Output: weight_map[h, w]

P3.4: candidate_selector.py
  - N-best patches by quality
  - Spatial coherence check
  - Output: candidate_patches[]

P3.5: candidate_extractor.py
  - Extract candidate signals
  - P3 signal path (not V2)

P3.6: spatial_fusion.py
  - Candidate-level fusion
  - Not V2 fusion

P3.7: p3_benchmark.py
  - Benchmark integration
  - Ablation support
```

---

## 7. EXPERIMENT DESIGN

### 7.1 Experiment Matrix

| ID | Name | Independent Variable | Baseline | Status |
|----|------|---------------------|----------|--------|
| P3-E0 | V2 Baseline | N/A | V2 frozen | READY |
| P3-E1 | Spatial Quality Only | Quality map enabled | E0 | PROPOSED |
| P3-E2 | Motion Field Only | Motion weighting enabled | E0 | PROPOSED |
| P3-E3 | Full P3 Candidates | Dynamic selection | E0 | PROPOSED |
| P3-E4 | Spatial-Temporal | Consistency weighting | E0 | PROPOSED |
| P3-E5 | Full P3 System | All P3 components | E0 | PROPOSED |

### 7.2 Per-Experiment Metrics

| Metric | E0 | E1 | E2 | E3 | E4 | E5 |
|--------|----|----|----|----|----|----|
| MAE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| RMSE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Pearson | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Motion-conditioned MAE | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Candidate stability | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ |
| Spatial consistency | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |

---

## 8. MOTION STRATIFICATION

### 8.1 Motion Levels

| Level | Definition | Measurement | Metric Required |
|-------|------------|-------------|----------------|
| Low | motion_score < 0.3 | MotionArtifactDetector | Separate MAE |
| Moderate | 0.3 ≤ motion_score < 0.8 | MotionArtifactDetector | Separate MAE |
| High | motion_score ≥ 0.8 | MotionArtifactDetector | Separate MAE |

### 8.2 Motion Categories

| Category | Detection | Separate Reporting |
|----------|-----------|-------------------|
| Head rotation | HeadPoseGate | YES |
| Local motion | ROIMicroMotionDetector | YES |
| Blink/jaw | JawBlinkSuppressor | YES |
| Illumination change | ExposureCompensator | YES |
| Global motion | MotionArtifactDetector | YES |

---

## 9. ABLATION MATRIX

### 9.1 Component Ablation

| Component | Ablation | Metric Impact | Test Required |
|-----------|----------|--------------|--------------|
| Quality map | Zero weights | MAE, RMSE | Test zero weights |
| Motion field | Uniform motion | MAE | Test uniform motion |
| Candidate selection | Landmark-only | MAE, RMSE | Test landmark subset |
| Temporal tracking | Instantaneous | Candidate stability | Test stability |
| Spatial coherence | Uncoordinated | Spatial consistency | Test coherence |

### 9.2 Full Ablation Sequence

```
Baseline (V2 frozen)
    ↓
+ Spatial Quality Map
    ↓
+ Motion Field
    ↓
+ Spatial-Temporal Weighting
    ↓
+ Candidate Selection
    ↓
Full P3 System
```

---

## 10. FAILURE TAXONOMY

### 10.1 P3-Specific Failures

| Category | Current Support | P3 Extension |
|----------|----------------|--------------|
| Landmark instability | HeadPoseGate | ROI drift tracking |
| ROI disappearance | Valid ROI check | Candidate coverage |
| Local motion contamination | MotionArtifactDetector | Per-patch motion field |
| Global head motion | HeadPoseGate | Trajectory tracking |
| Blink contamination | JawBlinkSuppressor | Blink-aware candidate weighting |
| Illumination transition | ExposureCompensator | Illumination map |
| Insufficient skin coverage | Skin confidence | Quality threshold |
| Spatial disagreement | ROI agreement | Patch-level agreement |
| Temporal instability | BPM history | Quality history |
| No valid candidates | N/A | Candidate coverage check |

---

## 11. TEST STRATEGY

### 11.1 Unit Tests Required

| Component | Test | Deterministic |
|-----------|------|--------------|
| spatial_quality_map | FFT correctness on synthetic signal | YES |
| spatial_quality_map | Skin mask intersection | YES |
| motion_field | Flow vector correctness | YES |
| motion_field | Patch aggregation | YES |
| spatial_weighting | Weight bounded [0,1] | YES |
| candidate_selector | N-best ordering | YES |
| candidate_selector | Coverage check | YES |
| candidate_extractor | Signal extraction | PARTIAL |
| spatial_fusion | Output range | YES |

### 11.2 Integration Tests

| Test | Target | Synthetic Data | Real Data |
|------|---------|----------------|-----------|
| V2 unchanged | rppg_core.py | N/A | N/A |
| P3 parallel path | spatial_fusion | Synthetic pulse | BLOCKED |
| Benchmark isolation | p3_benchmark | Synthetic pulse | BLOCKED |
| Ablation isolation | Each component | Synthetic pulse | BLOCKED |

---

## 12. IMPLEMENTATION STAGES

### Stage P3.1: Spatial Representation
- **Files:** spatial_quality_map.py
- **Files untouched:** V2, rppg_core.py
- **Dependencies:** None (standalone)
- **Tests:** Unit tests for FFT correctness
- **Exit criteria:** Quality map output bounded [0,1]

### Stage P3.2: Motion Field
- **Files:** motion_field.py
- **Files untouched:** V2, rppg_core.py
- **Dependencies:** P3.1 (optional)
- **Tests:** Motion vector validation
- **Exit criteria:** Motion field exists for all patches

### Stage P3.3: Motion-Aware Weighting
- **Files:** spatial_weighting.py
- **Files untouched:** V2, rppg_core.py
- **Dependencies:** P3.1, P3.2
- **Tests:** Weight bounds, ablation isolation
- **Exit criteria:** Quality * motion weight valid

### Stage P3.4: Dynamic Candidate Selection
- **Files:** candidate_selector.py, candidate_extractor.py
- **Files untouched:** V2, rppg_core.py
- **Dependencies:** P3.1, P3.2, P3.3
- **Tests:** Candidate ordering, coverage
- **Exit criteria:** N-best candidates extractable

### Stage P3.5: Spatial-Temporal Consistency
- **Files:** temporal_quality_tracker.py
- **Files untouched:** V2, rppg_core.py
- **Dependencies:** P3.1, P3.3
- **Tests:** Consistency score bounded [0,1]
- **Exit criteria:** Temporal consistency tracking works

### Stage P3.6: Ablation Validation
- **Files:** p3_ablation_runner.py
- **Files untouched:** V2, all V2 files
- **Dependencies:** P3.1-P3.5
- **Tests:** Component isolation tests
- **Exit criteria:** Each component ablates independently

### Stage P3.7: Failure Analysis
- **Files:** p3_failure_analyzer.py
- **Files untouched:** V2, rppg_core.py
- **Dependencies:** All P3 stages
- **Tests:** Failure categorization
- **Exit criteria:** All failure categories tracked

### Stage P3.8: Scientific Closure
- **Files:** P3_SCIENTIFIC_AUDIT.md (updated), P3_CLOSURE_REPORT.md
- **Dependencies:** All stages + real data (BLOCKED)
- **Exit criteria:** Real data acquired, benchmarks run

---

## 13. V2 FREEZE VERIFICATION

### Files That Must Remain UNCHANGED

| File | Constraint |
|------|------------|
| rppg_core.py | FROZEN - No modifications |
| rppg_signal.py | FROZEN - Signal processing unchanged |
| rppg_sqi.py | FROZEN - SQI computation unchanged |
| rppg_roi.py | EXISTS - May add P3 module, existing functions unchanged |
| rppg_temporal.py | FROZEN - Kalman, fusion unchanged |
| rppg_physio_state.py | FROZEN - Physiological state unchanged |
| rppg_uncertainty.py | EXISTS - May extend, existing interface unchanged |
| rppg_config.py | Configuration only - No algorithm changes |

---

## 14. REGRESSION TESTS REQUIRED

### Existing Tests (Must Pass)

```
test_benchmark_infrastructure.py: 28 tests
test_benchmark_alignment.py: 20 tests
test_ground_truth.py: 6 tests
benchmark_synthetic_example.py: 5 tests
rppg_phase2_signal_lab.py: 1 validation
─────────────────────────────────────────────────
TOTAL: 60 tests, 0 failures
```

### P3-Specific Tests (To Be Added)

| Stage | Tests Required | Synthetic | Real Data |
|-------|----------------|-----------|-----------|
| P3.1 | Quality map correctness | YES | BLOCKED |
| P3.2 | Motion field correctness | YES | BLOCKED |
| P3.3 | Weight bounds | YES | BLOCKED |
| P3.4 | Candidate ordering | YES | BLOCKED |
| P3.5 | Temporal tracking | YES | BLOCKED |
| P3.6 | Ablation isolation | YES | BLOCKED |
| P3.7 | Failure categorization | YES | BLOCKED |
| P3.8 | V2 unchanged | N/A | BLOCKED |

---

## 15. P3.0 VERDICT

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║   P3.0 AUDIT: COMPLETE                                             ║
║   P3 IMPLEMENTATION: NOT STARTED                                   ║
║   REAL-DATA BENCHMARK: BLOCKED (same as P2)                      ║
║   V2 FROZEN: VERIFIED - Unchanged                                 ║
║                                                                       ║
║   ARCHITECTURE: Defined (non-binding)                             ║
║   EXPERIMENTS: Designed                                          ║
║   TESTS: Specified                                              ║
║                                                                       ║
║   READY FOR: P3.1 Spatial Representation implementation          ║
║   BLOCKED BY: P2 real data (same blocker as P2)                   ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

### Recommended Next Action

Implement P3.1 (Spatial Representation) as a standalone module:
- Create `spatial_quality_map.py`
- Add unit tests with synthetic data
- Verify V2 unchanged
- Proceed to P3.2 only after P3.1 tests pass

---

*Audit completed: P3.0 Scientific and Architecture Audit*
*Status: READY FOR P3.1 IMPLEMENTATION*
*Real-data validation: BLOCKED (same as P2)*