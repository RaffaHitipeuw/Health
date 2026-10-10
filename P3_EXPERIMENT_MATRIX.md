# P3 EXPERIMENT MATRIX

## ABLATION EXPERIMENTS

| ID | Name | Components Enabled | Independent Variable | Baseline |
|----|------|-------------------|---------------------|----------|
| E0 | V2 Baseline | None (no P3) | N/A | N/A |
| E1 | Quality Only | P3.1 | Quality map enabled | E0 |
| E2 | Motion Only | P3.2 | Motion field enabled | E0 |
| E3 | Quality+Motion | P3.1 + P3.2 + P3.3 | Motion-aware weighting | E0 |
| E4 | +Candidates | P3.1 + P3.2 + P3.3 + P3.4 | Dynamic candidate selection | E0 |
| E5 | +Temporal | P3.1-5 | Spatial-temporal consistency | E0 |
| E6 | Full P3 | All P3 | Full P3 system | E0 |

## METRICS PER EXPERIMENT

| Metric | E0 | E1 | E2 | E3 | E4 | E5 | E6 |
|--------|----|----|----|----|----|----|----|
| MAE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| RMSE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Pearson | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Bias | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Median AE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Valid frame ratio | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Rejection rate | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Motion-conditioned MAE | - | - | ✓ | ✓ | ✓ | ✓ | ✓ |
| Spatial stability | - | - | - | - | - | ✓ | ✓ |
| Candidate turnover | - | - | - | - | ✓ | ✓ | ✓ |
| Failure rate | - | - | - | - | - | - | ✓ |

## MOTION STRATIFICATION

| Level | Definition | Measurement | Metric Required |
|-------|------------|-------------|----------------|
| Low | motion_score < 0.3 | MotionArtifactDetector | Separate MAE |
| Moderate | 0.3 ≤ motion_score < 0.8 | MotionArtifactDetector | Separate MAE |
| High | motion_score ≥ 0.8 | MotionArtifactDetector | Separate MAE |

## COMPONENT ABLATION

| Component | Disable Method | Metric Impact |
|----------|----------------|---------------|
| Quality map | Use QUALITY_ONLY mode | Compare Q × C × S vs C × S |
| Motion field | Use MOTION_ONLY mode | Compare Q × C × S vs Q |
| Quality × motion | Use QUALITY_ONLY or MOTION_ONLY | Compare combined vs individual |
| Dynamic candidates | Use fixed candidate set | Compare dynamic vs fixed |
| Temporal consistency | Use instantaneous only | Compare tracked vs instantaneous |

## FAILURE MODE EXPERIMENTS

| Failure Type | Detection | Affected Component | Response |
|--------------|-----------|-------------------|----------|
| Landmark instability | landmark_variance > 5.0 | face_landmarker | REJECT |
| Local motion | motion_magnitude > 0.8 | motion_field | REJECT |
| Insufficient candidates | n_candidates < 1 | candidate_selector | REJECT |
| Temporal instability | CV > 0.5, std > 0.3 | spatial_temporal | DOWNWEIGHT |
| Candidate thrashing | variance > threshold | candidate_selector | DOWNWEIGHT |
| Spatial disagreement | CV > 0.4 | candidate_selector | DOWNWEIGHT |

## RUNNING EXPERIMENTS

```python
from p3_spatial import P3Pipeline, ExperimentID

# Run specific experiment
pipeline = P3Pipeline(experiment_id=ExperimentID.E6_FULL)
result = pipeline.process_buffer(buffer, fps)

# Run all ablations
from p3_spatial import run_all_ablations
results = run_all_ablations(buffer, fps)
```

---

*Experiment matrix for P3 validation*
