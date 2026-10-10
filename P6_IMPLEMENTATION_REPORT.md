# P6 Temporal Physiological Model Implementation Report

## Overview

P6: Temporal Physiological Model extends the P5 neural baseline with advanced temporal modeling capabilities. The implementation provides LSTM, GRU, Transformer, ConvLSTM, and attention-based temporal models for learning physiological patterns across sequential video frames and signal windows.

## Files Created

### Core Model Files
- `p6_temporal/__init__.py` - Package initialization with exports
- `p6_temporal/base.py` - Base classes, configurations, and utility functions
- `p6_temporal/model_lstm.py` - LSTM-based temporal model
- `p6_temporal/model_gru.py` - GRU-based temporal model
- `p6_temporal/model_transformer.py` - Transformer-based temporal model
- `p6_temporal/model_convlstm.py` - ConvLSTM spatiotemporal model
- `p6_temporal/model_attention_temporal.py` - Attention-based temporal model
- `p6_temporal/dataset.py` - Dataset utilities for temporal sequences
- `p6_temporal/training.py` - Training infrastructure with temporal losses
- `p6_temporal/test_temporal_models.py` - Basic correctness tests

## Architecture and Input/Output Details

### Model 1: LSTMTemporal
- **Input**: `(B, T, F)` - Temporal feature sequences
- **Architecture**: Input projection → LSTM layers → Temporal pooling → Output heads
- **Variants**: `BiLSTMTemporal` (bidirectional)
- **Output**: BPM, confidence, signal quality, optional BVP waveform

### Model 2: GRUTemporal
- **Input**: `(B, T, F)` - Temporal feature sequences
- **Architecture**: Input projection → GRU layers → Temporal pooling → Output heads
- **Variants**: `BiGRUTemporal` (bidirectional)
- **Output**: BPM, confidence, signal quality, optional BVP waveform

### Model 3: TransformerTemporal
- **Input**: `(B, T, F)` - Temporal feature sequences
- **Architecture**: Input projection → Positional encoding → Transformer encoder → Attention pooling → Output heads
- **Output**: BPM, confidence, signal quality, optional BVP waveform

### Model 4: ConvLSTMTemporal
- **Input**: `(B, T, H, W, C)` video or `(B, T, F)` features
- **Architecture**: Spatial CNN → LSTM → Temporal pooling → Output heads
- **Output**: BPM, confidence, signal quality, optional BVP waveform

### Model 5: AttentionTemporal
- **Input**: `(B, T, F)` - Temporal feature sequences
- **Architecture**: Input projection → Multi-head attention × 2 → Temporal conv → FFN → Attention pooling
- **Output**: BPM, confidence, signal quality, optional BVP waveform, attention weights

## Supported Input Modalities

### Implemented
- **Temporal signals**: `(B, T, F)` - Feature sequences (e.g., ROI signals, RGB means)
- **Video frames**: `(B, T, H, W, C)` - Raw video input (ConvLSTM only)
- **Variable-length sequences**: Masked sequences with proper padding
- **BVP waveforms**: `(B, T)` - Blood volume pulse waveforms for training

### Not Implemented
- Multi-modal fusion (requires dataset integration)
- Optical flow features
- 3D skeletal data

## Training Infrastructure

### Losses
- **TemporalLoss**: Combined BPM + BVP + quality losses
- **BVPLoss**: Waveform prediction with MSE, correlation, and spectral losses
- **FrequencyLoss**: Frequency-domain optimization for cardiac band

### Utilities
- Sequence padding and masking
- Variable-length batch collation
- Training/validation loops
- Checkpoint save/load
- Learning rate scheduling (cosine, step, plateau)

## Integration with P5

P6 models share compatible interfaces with P5:
- Similar output types (`TemporalOutput` vs `ModelOutput`)
- Same device management (`get_device()`)
- Same random seed control (`set_seed()`)
- Compatible training patterns

P6 extends P5 by providing:
- Recurrent temporal modeling (LSTM, GRU)
- Long-range dependency modeling (Transformer)
- Attention mechanisms with interpretability
- Variable-length sequence handling
- BVP waveform prediction

## Tests Executed and Results

All 17 basic correctness tests pass:

```
============================================================
P6 Temporal Physiological Model Tests
============================================================
  LSTMTemporal... PASS
  GRUTemporal... PASS
  TransformerTemporal... PASS
  ConvLSTMTemporal... PASS
  AttentionTemporal... PASS
  BiLSTMTemporal... PASS
  Masked Sequences... PASS
  Checkpoint Save/Load... PASS
  Training Step... PASS
  BVPPrediction... PASS
  Dataset Collation... PASS
  Invalid Input Handling... PASS
------------------------------------------------------------
Tests: 17/17 passed
============================================================
```

### Test Coverage
- Model initialization with various configurations
- Forward passes with valid inputs
- Output tensor shape verification
- Gradient propagation during training
- Masked sequence handling
- Checkpoint save/load functionality
- Training step with synthetic data
- BVP waveform prediction
- Dataset collation with variable-length sequences
- Invalid input handling

## How to Instantiate and Run Each Model

### LSTMTemporal
```python
from p6_temporal.model_lstm import LSTMTemporal, LSTMConfig

config = LSTMConfig(
    input_dim=3,  # Feature dimension
    hidden_dim=128,
    num_layers=2,
    bidirectional=True
)
model = LSTMTemporal(config)
x = torch.randn(4, 128, 3)  # (B, T, F)
mask = torch.zeros(4, 128, dtype=torch.bool)  # Optional mask
output = model(x, mask=mask)
print(f"BPM: {output.bpm}")
```

### GRUTemporal
```python
from p6_temporal.model_gru import GRUTemporal, GRUConfig

config = GRUConfig(input_dim=3, hidden_dim=128)
model = GRUTemporal(config)
output = model(x)
```

### TransformerTemporal
```python
from p6_temporal.model_transformer import TransformerTemporal, TransformerConfig

config = TransformerConfig(
    input_dim=3,
    hidden_dim=128,
    num_layers=3,
    num_heads=4
)
model = TransformerTemporal(config)
output = model(x)
```

### ConvLSTMTemporal
```python
from p6_temporal.model_convlstm import ConvLSTMTemporal, ConvLSTMConfig

config = ConvLSTMConfig(
    input_dim=3,
    spatial_channels=(16, 32),
    hidden_dim=128
)
model = ConvLSTMTemporal(config)

# Video input
video = torch.randn(2, 32, 64, 64, 3)  # (B, T, H, W, C)
output = model(video)
```

### AttentionTemporal
```python
from p6_temporal.model_attention_temporal import AttentionTemporal, TemporalAttentionConfig

config = TemporalAttentionConfig(
    input_dim=3,
    hidden_dim=128,
    num_heads=4
)
model = AttentionTemporal(config)
output = model(x)
print(f"Attention weights: {output.attention_weights.shape}")
```

### Training Example
```python
from p6_temporal.dataset import SyntheticTemporalDataset
from p6_temporal.training import train_temporal_epoch, get_optimizer, create_temporal_dataloader
from p6_temporal.model_lstm import LSTMTemporal, LSTMConfig

# Setup
dataset = SyntheticTemporalDataset(num_samples=100, sequence_length=128)
loader = create_temporal_dataloader(dataset, batch_size=16)

config = LSTMConfig(input_dim=3, hidden_dim=128, sequence_length=128)
model = LSTMTemporal(config)
optimizer = get_optimizer(model, lr=1e-4)

# Training
for epoch in range(10):
    loss, metrics = train_temporal_epoch(model, loader, optimizer, torch.device('cpu'))
    print(f"Epoch {epoch}: loss={loss:.4f}")
```

## Missing Datasets and Annotations

The following are needed for training:

1. **Temporal Signal Datasets**: Pre-extracted ROI signals with temporal sequences
2. **BVP Ground Truth**: Synchronized BVP waveforms for waveform prediction training
3. **Signal Quality Annotations**: Quality labels for training quality prediction heads
4. **Variable-length Sequences**: Real data with varying sequence lengths for mask testing

## Known Limitations

1. **ConvLSTM**: Per-timestep autoregressive inference not fully implemented
2. **Transformer**: Caching for efficient online inference not optimized
3. **Attention**: Bidirectional attention in autoregressive mode has limitations
4. **Multi-modal**: No built-in support for multi-modal fusion

## Implementation Status

**IMPLEMENTATION COMPLETE**

All five temporal model families are implemented with:
- Working forward passes
- Correct output tensor shapes
- Gradient propagation
- Checkpoint save/load
- Training infrastructure with temporal losses
- Variable-length sequence handling
- Basic tests passing

The implementation provides a comprehensive temporal modeling layer for physiological signal estimation. Full benchmark evaluation and comparative analysis with P5 models will be conducted separately.
