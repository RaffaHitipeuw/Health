# P3 FAILURE ANALYSIS

## OVERVIEW

This document catalogs P3-specific failure modes, detection mechanisms, and response strategies.

## FAILURE TAXONOMY

### 1. LANDMARK_INSTABILITY
- **Detection:** landmark_variance > 5.0 pixels
- **Affected:** face_landmarker
- **Response:** REJECT
- **Confidence:** min(1.0, variance / 10.0)
- **Testable:** YES (synthetic landmark jitter)

### 2. ROI_DRIFT
- **Detection:** candidate_center_displacement > threshold
- **Affected:** candidate_selector
- **Response:** DOWNWEIGHT
- **Testable:** YES (synthetic drift patterns)

### 3. CANDIDATE_DRIFT
- **Detection:** candidate_trajectory_unstable
- **Affected:** candidate_selector
- **Response:** DOWNWEIGHT
- **Testable:** YES (synthetic candidate movement)

### 4. LOCAL_MOTION
- **Detection:** motion_magnitude > 0.8 pixels/frame
- **Affected:** motion_field
- **Response:** REJECT
- **Confidence:** min(1.0, magnitude / 1.6)
- **Testable:** YES (synthetic motion)

### 5. GLOBAL_HEAD_MOTION
- **Detection:** global_motion > threshold
- **Affected:** motion_field
- **Response:** REJECT
- **Testable:** YES (synthetic head movement)

### 6. BLINK_CONTAMINATION
- **Detection:** eye_region_motion_pattern
- **Affected:** spatial_temporal
- **Response:** DOWNWEIGHT
- **Testable:** PARTIAL (synthetic blink patterns)

### 7. JAW_EXPRESSION
- **Detection:** jaw_region_variance
- **Affected:** spatial_temporal
- **Response:** DOWNWEIGHT
- **Testable:** PARTIAL (synthetic expression)

### 8. INSUFFICIENT_SKIN
- **Detection:** skin_coverage < 0.2
- **Affected:** candidate_selector
- **Response:** REJECT
- **Testable:** YES (synthetic skin mask)

### 9. INSUFFICIENT_CANDIDATES
- **Detection:** n_candidates < 1
- **Affected:** candidate_selector
- **Response:** REJECT
- **Confidence:** 1.0
- **Testable:** YES (synthetic quality maps)

### 10. SPATIAL_DISAGREEMENT
- **Detection:** candidate_score_CV > 0.4
- **Affected:** candidate_selector
- **Response:** DOWNWEIGHT
- **Testable:** YES (synthetic disagreement)

### 11. TEMPORAL_INSTABILITY
- **Detection:** quality_history_CV > 0.5 AND std > 0.3
- **Affected:** spatial_temporal
- **Response:** DOWNWEIGHT
- **Confidence:** CV
- **Testable:** YES (synthetic noise)

### 12. CANDIDATE_THRASHING
- **Detection:** candidate_count_variance > threshold
- **Affected:** candidate_selector
- **Response:** DOWNWEIGHT
- **Testable:** YES (synthetic instability)

### 13. SPECTRAL_FALSE_PEAK
- **Detection:** peak_shape_anomaly
- **Affected:** spatial_quality_map
- **Response:** DOWNWEIGHT
- **Testable:** PARTIAL (synthetic spectra)

### 14. HARMONIC_CONTAMINATION
- **Detection:** harmonic_ratio_anomaly
- **Affected:** spatial_quality_map
- **Response:** DOWNWEIGHT
- **Testable:** PARTIAL (synthetic harmonics)

### 15. ILLUMINATION_TRANSITION
- **Detection:** illumination_change_rate
- **Affected:** motion_field
- **Response:** LOG
- **Testable:** PARTIAL (synthetic illumination)

## RESPONSE ACTIONS

| Response | Behavior |
|----------|----------|
| NONE | No action taken |
| DOWNWEIGHT | Reduce candidate weight by confidence |
| REJECT | Reject frame/candidate |
| RECOVER | Attempt to recover from failure |
| LOG | Record failure in log only |

## SEVERITY LEVELS

| Level | Definition | Action |
|-------|------------|--------|
| INFO | Informational | LOG |
| WARNING | Soft issue | DOWNWEIGHT |
| ERROR | Hard issue | REJECT |
| CRITICAL | Immediate failure | REJECT |

## DETECTION THRESHOLDS

| Failure Type | Warning | Error |
|------------|---------|-------|
| LOCAL_MOTION | > 0.4 px/f | > 0.8 px/f |
| TEMPORAL_INSTABILITY | CV > 0.3 | CV > 0.5 |
| SPATIAL_DISAGREEMENT | CV > 0.3 | CV > 0.4 |
| INSUFFICIENT_SKIN | < 0.3 | < 0.2 |
| LANDMARK_INSTABILITY | > 2.0 px | > 5.0 px |

## SYNTHETIC VALIDATION

All failure detection tests pass:

```
P3.7 Failure Analysis Tests:
- test_motion_failure_detection: PASS
- test_low_motion_no_failure: PASS
- test_insufficient_candidates: PASS
- test_temporal_instability: PASS
- test_spatial_disagreement: PASS
- test_candidate_thrashing: PASS
- test_failure_response: PASS
- test_reset: PASS

TOTAL: 8/8 passed
```

## RESPONSE MATRIX

| Failure | Detection | Response | Confidence |
|---------|-----------|---------|-----------|
| Motion > 0.8 | MotionArtifactDetector | REJECT | magnitude / 1.6 |
| n_candidates < 1 | CandidateSet | REJECT | 1.0 |
| Temporal CV > 0.5 | QualityHistory | DOWNWEIGHT | CV |
| Spatial CV > 0.4 | CandidateScores | DOWNWEIGHT | CV |
| Landmark var > 5.0 | FaceLandmarker | REJECT | variance / 10.0 |

---

*Failure taxonomy for P3 spatial/motion evolution*
