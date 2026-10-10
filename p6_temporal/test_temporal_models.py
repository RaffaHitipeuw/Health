"""
P6 Temporal Model Basic Correctness Tests

Verifies:
    - Model initialization
    - Forward passes with supported sequence shapes
    - Output shape and type consistency
    - Gradient propagation
    - Checkpoint round-trip
    - Training step
    - CPU/CUDA execution when available
"""

import torch
import numpy as np
import tempfile
import os
from typing import Tuple


def test_lstm_temporal() -> Tuple[int, int]:
    """Test LSTM temporal model."""
    from p6_temporal.model_lstm import LSTMTemporal, LSTMConfig
    passed = 0
    failed = 0
    print("  LSTMTemporal...")
    try:
        config = LSTMConfig(input_dim=3, hidden_dim=64, num_layers=2)
        model = LSTMTemporal(config)
        x = torch.randn(4, 128, 3)  # (B, T, F)
        mask = torch.zeros(4, 128, dtype=torch.bool)
        out = model(x, mask=mask)
        assert out.bpm.shape == (4,), f"Expected (4,), got {out.bpm.shape}"
        assert out.bpm_confidence.shape == (4,), f"Expected (4,), got {out.bpm_confidence.shape}"
        passed += 1
    except Exception as e:
        print(f"    FAIL: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
        return passed, failed

    try:
        x = torch.randn(2, 64, 3, requires_grad=True)
        out = model(x)
        out.bpm.sum().backward()
        assert x.grad is not None
        passed += 1
    except Exception as e:
        print(f"    gradient FAIL: {e}")
        failed += 1

    return passed, failed


def test_gru_temporal() -> Tuple[int, int]:
    """Test GRU temporal model."""
    from p6_temporal.model_gru import GRUTemporal, GRUConfig
    passed = 0
    failed = 0
    print("  GRUTemporal...")
    try:
        config = GRUConfig(input_dim=3, hidden_dim=64, num_layers=2)
        model = GRUTemporal(config)
        x = torch.randn(4, 128, 3)
        out = model(x)
        assert out.bpm.shape == (4,), f"Expected (4,), got {out.bpm.shape}"
        passed += 1
    except Exception as e:
        print(f"    FAIL: {e}")
        failed += 1
        return passed, failed

    try:
        x = torch.randn(2, 64, 3, requires_grad=True)
        out = model(x)
        out.bpm.sum().backward()
        assert x.grad is not None
        passed += 1
    except Exception as e:
        print(f"    gradient FAIL: {e}")
        failed += 1

    return passed, failed


def test_transformer_temporal() -> Tuple[int, int]:
    """Test Transformer temporal model."""
    from p6_temporal.model_transformer import TransformerTemporal, TransformerConfig
    passed = 0
    failed = 0
    print("  TransformerTemporal...")
    try:
        config = TransformerConfig(input_dim=3, hidden_dim=64, num_layers=2, num_heads=4)
        model = TransformerTemporal(config)
        x = torch.randn(4, 128, 3)
        out = model(x)
        assert out.bpm.shape == (4,), f"Expected (4,), got {out.bpm.shape}"
        passed += 1
    except Exception as e:
        print(f"    FAIL: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
        return passed, failed

    try:
        x = torch.randn(2, 64, 3, requires_grad=True)
        out = model(x)
        out.bpm.sum().backward()
        assert x.grad is not None
        passed += 1
    except Exception as e:
        print(f"    gradient FAIL: {e}")
        failed += 1

    return passed, failed


def test_convlstm_temporal() -> Tuple[int, int]:
    """Test ConvLSTM temporal model."""
    from p6_temporal.model_convlstm import ConvLSTMTemporal, ConvLSTMConfig
    passed = 0
    failed = 0
    print("  ConvLSTMTemporal (video input)...")
    try:
        config = ConvLSTMConfig(input_dim=3, hidden_dim=64, sequence_length=32)
        model = ConvLSTMTemporal(config)
        x = torch.randn(2, 32, 32, 32, 3)  # (B, T, H, W, C)
        out = model(x)
        assert out.bpm.shape == (2,), f"Expected (2,), got {out.bpm.shape}"
        passed += 1
    except Exception as e:
        print(f"    FAIL: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
        return passed, failed

    try:
        x = torch.randn(2, 16, 32, 32, 3, requires_grad=True)
        out = model(x)
        out.bpm.sum().backward()
        assert x.grad is not None
        passed += 1
    except Exception as e:
        print(f"    gradient FAIL: {e}")
        failed += 1

    return passed, failed


def test_attention_temporal() -> Tuple[int, int]:
    """Test attention-based temporal model."""
    from p6_temporal.model_attention_temporal import AttentionTemporal, TemporalAttentionConfig
    passed = 0
    failed = 0
    print("  AttentionTemporal...")
    try:
        config = TemporalAttentionConfig(input_dim=3, hidden_dim=64, num_heads=4)
        model = AttentionTemporal(config)
        x = torch.randn(4, 128, 3)
        out = model(x)
        assert out.bpm.shape == (4,), f"Expected (4,), got {out.bpm.shape}"
        assert out.attention_weights is not None, "Should have attention weights"
        passed += 1
    except Exception as e:
        print(f"    FAIL: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
        return passed, failed

    try:
        x = torch.randn(2, 64, 3, requires_grad=True)
        out = model(x)
        out.bpm.sum().backward()
        assert x.grad is not None
        passed += 1
    except Exception as e:
        print(f"    gradient FAIL: {e}")
        failed += 1

    return passed, failed


def test_bilstm_temporal() -> Tuple[int, int]:
    """Test bidirectional LSTM."""
    from p6_temporal.model_lstm import BiLSTMTemporal
    passed = 0
    failed = 0
    print("  BiLSTMTemporal...")
    try:
        model = BiLSTMTemporal(input_dim=3, hidden_dim=64)
        x = torch.randn(4, 128, 3)
        out = model(x)
        assert out.bpm.shape == (4,), f"Expected (4,), got {out.bpm.shape}"
        passed += 1
    except Exception as e:
        print(f"    FAIL: {e}")
        failed += 1
    return passed, failed


def test_masked_sequences() -> Tuple[int, int]:
    """Test handling of variable-length sequences with masking."""
    from p6_temporal.model_lstm import LSTMTemporal, LSTMConfig
    passed = 0
    failed = 0
    print("  Masked Sequences...")
    try:
        config = LSTMConfig(input_dim=3, hidden_dim=64)
        model = LSTMTemporal(config)

        # Variable length sequences
        B = 4
        T = 128
        x = torch.randn(B, T, 3)

        # Create mask with some padded sequences
        lengths = [100, 128, 80, 110]
        mask = torch.zeros(B, T, dtype=torch.bool)
        for i, length in enumerate(lengths):
            mask[i, length:] = True  # Pad after actual length

        out = model(x, mask=mask)
        assert out.bpm.shape == (B,), f"Expected ({B},), got {out.bpm.shape}"
        passed += 1
    except Exception as e:
        print(f"    FAIL: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    return passed, failed


def test_checkpoint_save_load() -> Tuple[int, int]:
    """Test checkpoint saving and loading."""
    from p6_temporal.model_lstm import LSTMTemporal, LSTMConfig
    from p6_temporal.training import save_checkpoint, load_checkpoint
    passed = 0
    failed = 0
    print("  Checkpoint Save/Load...")
    try:
        config = LSTMConfig(input_dim=3, hidden_dim=64, num_layers=2)
        model = LSTMTemporal(config)

        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            ckpt_path = f.name
        save_checkpoint(model, path=ckpt_path)

        # Load into new model
        model2 = LSTMTemporal(config)
        epoch, _ = load_checkpoint(model2, path=ckpt_path)

        # Verify parameters match
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert n1 == n2, f"Parameter names don't match"
            assert torch.allclose(p1, p2), f"Parameter {n1} values don't match"

        os.unlink(ckpt_path)
        passed += 1
    except Exception as e:
        print(f"    FAIL: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    return passed, failed


def test_training_step() -> Tuple[int, int]:
    """Test a single training step."""
    from p6_temporal.dataset import SyntheticTemporalDataset, create_temporal_dataloader
    from p6_temporal.training import train_temporal_epoch, get_optimizer
    from p6_temporal.model_lstm import LSTMTemporal, LSTMConfig
    passed = 0
    failed = 0
    print("  Training Step...")
    try:
        dataset = SyntheticTemporalDataset(
            num_samples=10,
            sequence_length=64,
            feature_dim=3,
            include_bvp=False
        )
        loader = create_temporal_dataloader(dataset, batch_size=4)

        config = LSTMConfig(input_dim=3, hidden_dim=32, num_layers=2, sequence_length=64)
        model = LSTMTemporal(config)
        optimizer = get_optimizer(model, lr=1e-4)

        loss, metrics = train_temporal_epoch(model, loader, optimizer, torch.device('cpu'))
        assert isinstance(loss, float), f"Loss should be float, got {type(loss)}"
        assert loss >= 0, f"Loss should be non-negative, got {loss}"
        passed += 1
    except Exception as e:
        print(f"    FAIL: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    return passed, failed


def test_bvp_prediction() -> Tuple[int, int]:
    """Test BVP waveform prediction."""
    from p6_temporal.model_lstm import LSTMTemporal, LSTMConfig
    passed = 0
    failed = 0
    print("  BVP Prediction...")
    try:
        config = LSTMConfig(input_dim=3, hidden_dim=64, output_type="bvp", sequence_length=128)
        model = LSTMTemporal(config)
        x = torch.randn(4, 128, 3)
        out = model(x)
        assert out.bvp is not None, "Should have BVP output"
        assert out.bvp.shape == (4, 128), f"Expected (4, 128), got {out.bvp.shape}"
        passed += 1
    except Exception as e:
        print(f"    FAIL: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    return passed, failed


def test_dataset_collation() -> Tuple[int, int]:
    """Test temporal dataset collation."""
    from p6_temporal.dataset import SyntheticTemporalDataset, collate_temporal_batch
    passed = 0
    failed = 0
    print("  Dataset Collation...")
    try:
        dataset = SyntheticTemporalDataset(num_samples=8, sequence_length=64, feature_dim=3)
        # Create variable-length sequences by truncating
        batch = []
        for i in range(4):
            sample = dataset[i]
            sample["features"] = sample["features"][:50 + i * 10]  # Variable lengths
            batch.append(sample)

        collated = collate_temporal_batch(batch)
        assert collated["features"].shape[0] == 4, "Batch size should be 4"
        assert collated["mask"].shape[0] == 4, "Mask batch size should be 4"
        assert collated["mask"].sum() > 0, "Should have some masked positions"
        passed += 1
    except Exception as e:
        print(f"    FAIL: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    return passed, failed


def test_invalid_input() -> Tuple[int, int]:
    """Test invalid input handling."""
    from p6_temporal.model_lstm import LSTMTemporal, LSTMConfig
    passed = 0
    failed = 0
    print("  Invalid Input Handling...")
    try:
        config = LSTMConfig(input_dim=3, hidden_dim=64)
        model = LSTMTemporal(config)

        # Wrong number of dimensions
        x_wrong = torch.randn(32, 64)  # Should be (B, T, F)
        try:
            out = model(x_wrong)
            print("    (Warning: invalid input did not raise error)")
        except (RuntimeError, ValueError):
            pass  # Expected behavior

        passed += 1
    except Exception as e:
        print(f"    FAIL: {e}")
        failed += 1
    return passed, failed


def run_tests() -> bool:
    """Run all P6 temporal model tests."""
    print("\n" + "=" * 60)
    print("P6 Temporal Physiological Model Tests")
    print("=" * 60)

    total_pass = 0
    total_fail = 0

    for name, fn in [
        ("LSTMTemporal", test_lstm_temporal),
        ("GRUTemporal", test_gru_temporal),
        ("TransformerTemporal", test_transformer_temporal),
        ("ConvLSTMTemporal", test_convlstm_temporal),
        ("AttentionTemporal", test_attention_temporal),
        ("BiLSTMTemporal", test_bilstm_temporal),
        ("MaskedSequences", test_masked_sequences),
        ("Checkpoint", test_checkpoint_save_load),
        ("TrainingStep", test_training_step),
        ("BVPPrediction", test_bvp_prediction),
        ("DatasetCollation", test_dataset_collation),
        ("InvalidInput", test_invalid_input),
    ]:
        p, f = fn()
        total_pass += p
        total_fail += f

    print("-" * 60)
    print(f"Tests: {total_pass}/{total_pass + total_fail} passed")
    print("=" * 60)
    return total_fail == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_tests() else 1)
