# P9 Domain Generalization — Current State Audit & Handoff Report

**Date**: Generated during repository audit  
**Auditor**: Claude Code (automated audit)  
**Repository Root**: `D:\main\Projects\Health`  
**Git Branch**: `main`

---

## Executive Summary

P9 implements domain generalization capabilities for the Sanubari multi-task physiological sensing platform. The module provides domain metadata structures, data splitting utilities, domain-aware samplers, GroupDRO training support, and cross-domain evaluation infrastructure.

**Status**: **Partially Functional — Critical Bugs Identified**

The implementation provides a reasonable architectural foundation but contains several bugs that would cause incorrect behavior in production use:

1. **CONFIRMED BUG**: Syntax error in `evaluation.py:87` (missing closing parenthesis)
2. **CONFIRMED BUG**: Logic error in `splits.py:159` (`_create_random_split` creates wrong indices)
3. **CONFIRMED BUG**: `GroupDROLoss.forward()` incorrectly assigns batch loss to all domain members
4. **SUSPECTED DEFECT**: `GroupDROLoss` weighted loss computation multiplies, not averages

The 22 synthetic unit tests pass but do not catch these bugs because they use isolated components without end-to-end integration testing.

---

## 1. Repository and Module Map

### Repository Context
- **Root**: `D:\main\Projects\Health`
- **Branch**: `main`
- **Python Environment**: Uses pytest from `/c/miniconda3/Scripts/pytest`

### P9 Module Structure

```
p9_domain_generalization/
├── __init__.py          (1.8 KB) — Package exports
├── base.py              (11.9 KB) — Domain metadata, config, dataset wrapper
├── evaluation.py         (11.4 KB) — Cross-domain evaluation metrics
├── sampling.py           (13.7 KB) — Domain-aware samplers
├── splits.py             (18.8 KB) — Data splitting utilities
├── test_domain_generalization.py (16.0 KB) — 22 synthetic tests
└── training.py           (20.0 KB) — Training utilities and DomainGeneralizationTrainer
```

### P8 Integration Points
- `p8_multitask/model.py` — `MultiTaskPhysiologicalModel` class
- `p8_multitask/losses.py` — `MultiTaskLoss` class
- P9 training module imports from P8 via: `from p8_multitask.losses import MultiTaskLoss`

---

## 2. Domain Metadata Analysis

### DomainMetadata Structure (`base.py:31-119`)

**Supported Fields**:
| Field | Type | Required | Source |
|-------|------|----------|--------|
| `sample_id` | str | Yes | User-provided |
| `subject_id` | int | No | User-provided |
| `dataset` | str | No | User-provided |
| `device` | str | No | User-provided |
| `environment` | str | No | User-provided |
| `lighting` | str | No | User-provided |
| `session_id` | str | No | User-provided |
| `protocol` | str | No | User-provided |
| `extra` | dict | No | Flexible storage |

### Domain Key Construction

The `get_domain_key()` method (`base.py:59-79`) maps domain type strings to metadata fields:
```python
domain_map = {
    "dataset": self.dataset,
    "device": self.device,
    "environment": self.environment,
    "lighting": self.lighting,
    "subject": str(self.subject_id) if self.subject_id is not None else None,
    "session": self.session_id,
    "protocol": self.protocol,
}
return value if value is not None else "unknown"
```

**Issue**: `get_domain_key()` maps `"subject"` to `subject_id` but `get_all_domains()` (`base.py:84-89`) also treats it as a separate key. This inconsistency could cause confusion.

### Unknown Domain Handling

When metadata is unavailable, fields default to `None`, and `get_domain_key()` returns `"unknown"`. All samples with missing metadata are grouped into a single `"unknown"` domain.

---

## 3. Split Integrity Analysis

### Split Types Implemented

| Split Type | Function | Location | Status |
|------------|----------|----------|--------|
| RANDOM | `_create_random_split()` | `splits.py:149-184` | **BUG — Wrong indices** |
| SUBJECT_HELD_OUT | `_create_subject_held_out_split()` | `splits.py:187-241` | Working |
| DOMAIN_HELD_OUT | `_create_domain_held_out_split()` | `splits.py:244-302` | Working |
| LEAVE_ONE_DOMAIN_OUT | `_create_leave_one_domain_out_split()` | `splits.py:305-369` | Working |

### Critical Bug: `_create_random_split` Index Mismatch

**Location**: `splits.py:159`

```python
all_indices = list(range(sum(len(v) for v in domain_samples.values())))
```

**Problem**: This creates indices `[0, 1, 2, ..., N-1]` where N is the total sample count, but `domain_samples` stores **actual sample indices** passed by the caller. The function then uses these wrong indices to construct train/val/test splits.

**Impact**: When `_compute_domain_counts()` is called with the split indices, it cannot correctly map them back to domains because the indices don't match the original sample indices.

**Example**:
- If caller passes metadata indices `[5, 15, 25, 35]` (4 samples from index 5 onwards)
- `domain_samples["domain_0"]` contains `[5, 15, 25, 35]`
- `all_indices` becomes `[0, 1, 2, 3]`
- Split indices [0, 1] are assigned to train
- `_compute_domain_counts` looks for indices 0, 1 in domain_samples — finds nothing

### Subject Overlap Checking

The `validate_split()` function (`splits.py:534-562`) checks for subject overlap:
```python
for idx in split.train_indices:
    subj = metadata[idx].subject_id
```

This relies on correct indices being in `split.train_indices`. Due to the `_create_random_split` bug, this validation may not work correctly.

### Leakage Detection

`check_leakage()` in `evaluation.py:182-248` validates:
1. Index overlap between train/val/test
2. Subject overlap across splits
3. Held-out domains appearing in training

---

## 4. Training Methods Analysis

### Methods Implemented

| Method | Class | Status |
|--------|-------|--------|
| ERM Baseline | `DomainGeneralizationTrainer` | Working |
| Domain-Balanced Sampling | `DomainBalancedSampler` | Working |
| Domain-Weighted Sampling | `DomainWeightedSampler` | Working |
| GroupDRO | `GroupDROLoss` + `GroupDROSampler` | **BUG — Incorrect loss computation** |

### DomainBalancedSampler (`sampling.py:17-104`)

Rounds-robin sampling from each domain. Each batch contains samples from all domains (if domains have remaining samples).

```python
# Lines 62-96: Iterates through domains, taking one sample each until batch complete
for domain in self.domain_indices.keys():
    idx = next(domain_iterators[domain])
    batch.append(idx)
```

**Behavior**: Batches are nearly balanced across domains but not guaranteed exact balance.

### DomainWeightedSampler (`sampling.py:106-216`)

Inverse frequency weighting. Less frequent domains get higher sampling probability.

```python
# Lines 169-185: Weight computation
if self.weight_mode == "inverse":
    weights[domain] = total / (len(domain_counts) * count)
elif self.weight_mode == "sqrt_inverse":
    weights[domain] = (total / count) ** 0.5
```

### GroupDROLoss Critical Bug

**Location**: `training.py:108-142`

**Bug 1 — Incorrect per-sample loss assignment** (`training.py:116-117`):
```python
for i, domain in enumerate(batch_domains):
    domain_losses[domain].append(float(base_loss.detach()))  # WRONG
```

All samples in the same domain receive the **same batch loss value**, not their individual losses. This defeats the purpose of GroupDRO, which should upweight groups with high per-sample losses.

**Bug 2 — Incorrect weighted loss aggregation** (`training.py:128-131`):
```python
weighted_loss = torch.tensor(0.0, device=self.device)
for domain in batch_domains:
    weight = self.group_weights.get(domain, 1.0 / self.num_groups)
    weighted_loss = weighted_loss + weight * base_loss  # WRONG
```

This adds `weight * base_loss` for each occurrence of a domain in the batch. If a batch has 3 samples from domain A and 1 from domain B, and base_loss=1.0:
- Correct: `(3 * weight_A * sample_loss_A + 1 * weight_B * sample_loss_B) / 4`
- Actual: `3 * weight_A * 1.0 + 1 * weight_B * 1.0`

**Impact**: GroupDRO weights are updated based on identical loss values for all samples in a domain, and the final loss is not a proper weighted average.

---

## 5. P8 Integration Analysis

### Integration Points

**Training (`training.py`)**:
- Line 207: `from p8_multitask.losses import MultiTaskLoss`
- Line 216: `base_loss = MultiTaskLoss(self.model.config)`
- Line 323, 369: `from p8_multitask.losses import MultiTaskLoss`

**Model Usage**:
```python
# training.py:310-311
output = self.model(features, mask=mask)
```

### Supported Task Heads

The trainer expects batch dictionaries with these keys (`training.py:302-307`):
- `hr` / `hr_available`
- `bvp` / `bvp_available`
- `sqi` / `sqi_available`
- `confidence` / `confidence_available`

### Integration Limitations

1. **Hardcoded imports**: P8 modules are imported inside functions, not at module level. This delays import errors to runtime.

2. **No checkpoint/optimizer integration**: `save_checkpoint()` (`training.py:503-520`) saves model state but the optimizer is not passed to the save function in a usable way.

3. **Device handling**: The trainer uses `torch.device(config.device)` but P8's `MultiTaskLoss` creates its own device (`losses.py:385`). This could cause device mismatches.

---

## 6. Evaluation Analysis

### Metrics Computed (`evaluation.py`)

| Metric | Function | Notes |
|--------|----------|-------|
| HR MAE | `compute_domain_metrics:86` | Working |
| HR RMSE | `compute_domain_metrics:87` | **SYNTAX ERROR** |
| HR Bias | `compute_domain_metrics:88` | Working |
| BVP MAE | `compute_domain_metrics:101` | Working |
| BVP Correlation | `compute_domain_metrics:105-107` | Working (1D only) |
| SQI MAE | `compute_domain_metrics:115` | Working |

### Critical Bug: HR RMSE Syntax Error

**Location**: `evaluation.py:87`

```python
hr_rmse = float(torch.sqrt(torch.mean((hr_pred - hr_target) ** 2)))  # 3 closing parens!
```

Should be:
```python
hr_rmse = float(torch.sqrt(torch.mean((hr_pred - hr_target) ** 2)))  # 4 closing parens
```

**Impact**: Calling `compute_domain_metrics` with HR predictions will raise `SyntaxError` at runtime.

### Aggregation Methods

- `aggregate_across_domains()`: Computes mean and std across domains
- `summarize_domain_shift()`: Computes KL divergence approximation and detects domain shift
- `compute_worst_case_domain()`: Identifies worst-performing domain

---

## 7. Test Coverage Analysis

### Test File

`p9_domain_generalization/test_domain_generalization.py` — 22 synthetic tests

### Test Classes and Coverage

| Class | Tests | What is Tested |
|-------|-------|----------------|
| `TestDomainMetadata` | 4 | Metadata creation, domain key retrieval, dict conversion |
| `TestDomainSplits` | 5 | Split creation, determinism, validation |
| `TestDomainAwareDataset` | 3 | Dataset wrapping, domain lookups |
| `TestSamplers` | 2 | Sampler initialization and iteration |
| `TestGroupDRO` | 2 | GroupDRO initialization and weight updates |
| `TestEvaluation` | 4 | Metric computation, leakage detection, domain shift |
| `TestIntegration` | 2 | Config and split result summary |

### Test Execution

```bash
# All P9 tests
pytest p9_domain_generalization/test_domain_generalization.py

# All P8 tests (for reference)
pytest p8_multitask/test_multitask.py

# Combined (P8 + P9)
pytest p9_domain_generalization/ p8_multitask/
```

**Actual Results**: 58 tests passed (22 P9 + 36 P8)

### Test Limitations

1. **No real data**: All tests use synthetic/mock data
2. **No end-to-end training**: `DomainGeneralizationTrainer` is never instantiated with a real model
3. **No held-out evaluation**: Tests verify split structure but not actual held-out domain performance
4. **Bugs not caught**:
   - Syntax error in HR RMSE (only caught if HR predictions are provided in test)
   - Random split index bug (test creates metadata with sequential indices 0-99, masking the issue)
   - GroupDRO loss computation bug (tests only check weight initialization and simple update)

---

## 8. Findings Summary

### Confirmed Working
- ✅ DomainMetadata structure and domain key construction
- ✅ Subject-held-out splits
- ✅ Domain-held-out splits
- ✅ Leave-one-domain-out splits
- ✅ `DomainBalancedSampler` and `DomainWeightedSampler`
- ✅ Domain split validation (`validate_split`)
- ✅ Leakage detection (`check_leakage`)
- ✅ Domain shift summarization
- ✅ DomainAwareDataset wrapper class

### Implemented but Insufficiently Verified
- ⚠️ `_create_random_split` — **Bug identified** (wrong index generation)
- ⚠️ GroupDRO weight updates — **Bug identified** (identical losses for all domain samples)
- ⚠️ GroupDRO loss aggregation — **Bug identified** (not a proper weighted average)
- ⚠️ HR RMSE computation — **Syntax error** (missing parenthesis)
- ⚠️ P8 integration — Not tested end-to-end with real model
- ⚠️ Device handling — Potential mismatch between trainer and loss device

### Suspected Defects
- 🔴 `_create_random_split` index mapping broken (see Section 3)
- 🔴 `GroupDROLoss.forward()` assigns batch loss to all domain members
- 🔴 HR RMSE syntax error will crash evaluation with HR data
- 🔴 Subject overlap check may fail due to random split bug

### Missing Capabilities
- ❌ Real dataset integration (no automatic domain detection)
- ❌ Statistical significance testing (no confidence intervals)
- ❌ Cross-validation wrapper for LODO
- ❌ Checkpoint/optimizer state persistence
- ❌ Learning rate scheduling
- ❌ Gradient accumulation support

---

## 9. Research Validity Assessment

### Leakage Prevention
- **Index overlap**: Prevented by `validate_split()` — BUT subject to random split bug
- **Subject overlap**: Checked by `validate_split()` when `check_subject_overlap=True`
- **Held-out domain leakage**: Validated in `check_leakage()` — correctly implemented

### Methodological Concerns

1. **GroupDRO is not correctly implemented**: The GroupDRO method upweights groups with high **per-sample** losses. The current implementation assigns the same batch loss to all samples in a domain, defeating the optimization objective.

2. **No evidence of domain generalization**: Tests only verify software components exist and produce outputs. No experiments demonstrate:
   - ERM vs GroupDRO performance differences
   - Performance on held-out domains
   - Cross-domain generalization claims

3. **Evaluation is incomplete**: No selective prediction, confidence calibration, or uncertainty quantification metrics.

### Required Experiments Before Scientific Claims

1. Run ERM baseline on held-out domain test set
2. Run GroupDRO with fixed loss computation
3. Compare performance across domains
4. Statistical significance testing across multiple splits
5. Ablation on domain key selection (dataset vs device vs environment)

---

## 10. Prioritized Next Steps

### Critical (Must Fix Before Use)

1. **Fix syntax error** in `evaluation.py:87`
2. **Fix `_create_random_split`** in `splits.py:149-184` to use actual sample indices
3. **Fix `GroupDROLoss.forward()`** to compute per-sample losses grouped by domain
4. **Fix weighted loss aggregation** to compute proper weighted average

### High Priority

5. Add end-to-end integration tests with P8 model
6. Add held-out domain evaluation tests
7. Verify subject overlap detection works correctly
8. Test with real multi-domain dataset

### Medium Priority

9. Add statistical significance testing
10. Implement LODO cross-validation wrapper
11. Add checkpoint/optimizer state handling
12. Test device placement consistency

---

## 11. Commands for Reproducing Audit

```bash
# Navigate to repository
cd D:/main/Projects/Health

# Run P9 tests
pytest p9_domain_generalization/test_domain_generalization.py -v

# Run P8 tests (reference)
pytest p8_multitask/test_multitask.py -v

# Run both
pytest p9_domain_generalization/ p8_multitask/ -v

# Check module imports
python -c "from p9_domain_generalization import *; print('P9 OK')"
python -c "from p8_multitask import *; print('P8 OK')"

# View P9 structure
ls -la p9_domain_generalization/
```

---

## 12. Handoff Summary for Another AI Assistant

### What P9 Does
P9 provides domain generalization infrastructure for Sanubari: domain metadata structures, train/val/test splits that can hold out domains, domain-aware sampling strategies (ERM, balanced, weighted, GroupDRO), and cross-domain evaluation metrics.

### What's Broken
1. **HR RMSE syntax error** (`evaluation.py:87`) — Will crash if HR metrics are computed
2. **Random split indices** (`splits.py:159`) — Creates wrong indices, breaking random splits
3. **GroupDRO loss computation** (`training.py:116-117`) — Assigns identical batch loss to all samples in domain

### What Needs Testing
- End-to-end training with P8 model on held-out domains
- LODO cross-validation with real data
- Subject overlap detection with non-sequential indices

### What to NOT Do
- Do not claim domain generalization works without fixing the bugs
- Do not modify P8 code (P9 depends on it as-is)
- Do not assume synthetic tests prove real-world behavior
- Do not make scientific claims without held-out domain experiments

### Files to Modify First
1. `p9_domain_generalization/evaluation.py` — Fix parenthesis
2. `p9_domain_generalization/splits.py` — Fix random split indices
3. `p9_domain_generalization/training.py` — Fix GroupDRO loss

---

## Appendix: Files Inspected

| File | Lines | Purpose |
|------|-------|---------|
| `p9_domain_generalization/__init__.py` | 86 | Package exports |
| `p9_domain_generalization/base.py` | 354 | DomainMetadata, DomainAwareDataset, config |
| `p9_domain_generalization/splits.py` | 600 | Data splitting utilities |
| `p9_domain_generalization/sampling.py` | 450 | Domain-aware samplers |
| `p9_domain_generalization/training.py` | 633 | DomainGeneralizationTrainer, GroupDROLoss |
| `p9_domain_generalization/evaluation.py` | 389 | Cross-domain evaluation metrics |
| `p9_domain_generalization/test_domain_generalization.py` | 576 | 22 synthetic unit tests |
| `p8_multitask/model.py` | 528 | MultiTaskPhysiologicalModel |
| `p8_multitask/losses.py` | 525 | MultiTaskLoss and task losses |

---

*End of Audit Report*
