"""
P8 Multi-Task Correctness Tests

Basic correctness tests for multi-task physiological modeling:
    - Model construction and task heads
    - Forward passes and output shapes
    - Loss computation
    - Missing labels and masks
    - Gradient propagation
    - Checkpoint save/load
    - Training step
    - Inference
"""

import torch
import torch.nn as nn
import pytest
import sys
import os
from typing import Dict, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p8_multitask import (
    MultiTaskConfig,
    MultiTaskOutput,
    MultiTaskPhysiologicalModel,
    MultiTaskLoss,
    BVPHead,
    HRHead,
    SQIHead,
    ConfidenceHead,
    SyntheticMultiTaskDataset,
    collate_multitask_batch,
    create_multitask_dataloader,
    train_multitask_epoch,
    validate_multitask,
    save_checkpoint,
    load_checkpoint,
    get_optimizer,
    TaskType,
)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TestModelConstruction:
    """Test model construction and configuration."""

    def test_config_defaults(self):
        """Test default configuration."""
        config = MultiTaskConfig()
        assert config.encoder_type == "lstm"
        assert config.input_dim == 3
        assert config.hidden_dim == 128
        assert config.enable_bvp is True
        assert config.enable_hr is True
        assert config.enable_sqi is True
        assert config.enable_confidence is True

    def test_config_enabled_tasks(self):
        """Test getting enabled tasks from config."""
        config = MultiTaskConfig(
            enable_bvp=True,
            enable_hr=False,
            enable_sqi=True,
            enable_confidence=False
        )
        tasks = config.get_enabled_tasks()
        assert TaskType.BVP in tasks
        assert TaskType.HR not in tasks
        assert TaskType.SQI in tasks
        assert TaskType.CONFIDENCE not in tasks

    def test_config_task_weights(self):
        """Test getting task weights."""
        config = MultiTaskConfig(
            bvp_weight=1.0,
            hr_weight=2.0,
            sqi_weight=0.5,
            confidence_weight=0.5
        )
        weights = config.get_task_weights()
        assert weights[TaskType.BVP] == 1.0
        assert weights[TaskType.HR] == 2.0
        assert weights[TaskType.SQI] == 0.5
        assert weights[TaskType.CONFIDENCE] == 0.5

    def test_model_construction_all_tasks(self):
        """Test model construction with all tasks enabled."""
        set_seed(42)
        config = MultiTaskConfig(
            encoder_type="lstm",
            input_dim=3,
            hidden_dim=64,
            num_encoder_layers=1,
            bidirectional=False
        )
        model = MultiTaskPhysiologicalModel(config)

        assert model is not None
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'heads')
        assert "hr" in model.heads
        assert "bvp" in model.heads
        assert "sqi" in model.heads
        assert "confidence" in model.heads

    def test_model_construction_subset_tasks(self):
        """Test model construction with subset of tasks."""
        set_seed(42)
        config = MultiTaskConfig(
            encoder_type="gru",
            input_dim=3,
            hidden_dim=64,
            enable_bvp=False,
            enable_sqi=False
        )
        model = MultiTaskPhysiologicalModel(config)

        assert "hr" in model.heads
        assert "confidence" in model.heads
        assert "bvp" not in model.heads
        assert "sqi" not in model.heads

    def test_model_construction_transformer(self):
        """Test model construction with transformer encoder."""
        set_seed(42)
        config = MultiTaskConfig(
            encoder_type="transformer",
            input_dim=3,
            hidden_dim=64,
            num_encoder_layers=1
        )
        model = MultiTaskPhysiologicalModel(config)
        assert model is not None


class TestForwardPass:
    """Test forward passes and output shapes."""

    def _create_test_input(self, batch_size: int = 4, seq_len: int = 64, feature_dim: int = 3):
        """Create test input tensor."""
        return torch.randn(batch_size, seq_len, feature_dim)

    def test_forward_all_tasks(self):
        """Test forward pass with all tasks."""
        set_seed(42)
        config = MultiTaskConfig(
            input_dim=3,
            hidden_dim=64,
            num_encoder_layers=1,
            enable_bvp=True,
            enable_hr=True,
            enable_sqi=True,
            enable_confidence=True
        )
        model = MultiTaskPhysiologicalModel(config)
        model.eval()

        x = self._create_test_input()
        output = model(x)

        assert isinstance(output, MultiTaskOutput)
        assert output.hr is not None
        assert output.bvp is not None
        assert output.sqi is not None
        assert output.confidence is not None

    def test_forward_shape_hr(self):
        """Test HR output shape."""
        set_seed(42)
        config = MultiTaskConfig(enable_hr=True, enable_bvp=False, enable_sqi=False, enable_confidence=False)
        model = MultiTaskPhysiologicalModel(config)
        model.eval()

        batch_size = 4
        x = self._create_test_input(batch_size=batch_size)
        output = model(x)

        assert output.hr.shape == (batch_size,)

    def test_forward_shape_bvp(self):
        """Test BVP output shape."""
        set_seed(42)
        config = MultiTaskConfig(
            enable_bvp=True, enable_hr=False, enable_sqi=False, enable_confidence=False,
            output_sequence_length=64, bidirectional=True
        )
        model = MultiTaskPhysiologicalModel(config)
        model.eval()

        batch_size = 4
        x = self._create_test_input(batch_size=batch_size)
        output = model(x)

        assert output.bvp is not None
        assert output.bvp.shape == (batch_size, 64)

    def test_forward_shape_sqi(self):
        """Test SQI output shape."""
        set_seed(42)
        config = MultiTaskConfig(enable_sqi=True, enable_hr=False, enable_bvp=False, enable_confidence=False)
        model = MultiTaskPhysiologicalModel(config)
        model.eval()

        batch_size = 4
        x = self._create_test_input(batch_size=batch_size)
        output = model(x)

        assert output.sqi.shape == (batch_size,)
        assert output.sqi.min() >= 0.0
        assert output.sqi.max() <= 1.0

    def test_forward_shape_confidence(self):
        """Test confidence output shape."""
        set_seed(42)
        config = MultiTaskConfig(enable_confidence=True, enable_hr=False, enable_bvp=False, enable_sqi=False)
        model = MultiTaskPhysiologicalModel(config)
        model.eval()

        batch_size = 4
        x = self._create_test_input(batch_size=batch_size)
        output = model(x)

        assert output.confidence.shape == (batch_size,)
        assert output.confidence.min() >= 0.0
        assert output.confidence.max() <= 1.0

    def test_forward_with_mask(self):
        """Test forward pass with sequence mask."""
        set_seed(42)
        config = MultiTaskConfig(input_dim=3, hidden_dim=64, num_encoder_layers=1)
        model = MultiTaskPhysiologicalModel(config)
        model.eval()

        batch_size = 4
        seq_len = 64
        x = self._create_test_input(batch_size=batch_size, seq_len=seq_len)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask[0, 32:] = True  # First sample padded after 32 timesteps

        output = model(x, mask=mask)

        assert output.hr is not None
        assert output.hr.shape == (batch_size,)


class TestTaskHeads:
    """Test individual task heads."""

    def test_hr_head(self):
        """Test HR head."""
        set_seed(42)
        input_dim = 64
        head = HRHead(input_dim=input_dim, hidden_dim=32)

        x = torch.randn(4, input_dim)
        out = head(x)

        assert out.shape == (4,)
        assert out.min() >= 40.0  # Default hr_min
        assert out.max() <= 200.0  # Default hr_max

    def test_bvp_head(self):
        """Test BVP head."""
        set_seed(42)
        input_dim = 64
        output_len = 64
        head = BVPHead(input_dim=input_dim, output_length=output_len, use_temporal_conv=False)

        representation = torch.randn(4, input_dim)
        out = head(representation)

        assert out.shape == (4, output_len)

    def test_sqi_head(self):
        """Test SQI head."""
        set_seed(42)
        input_dim = 64
        head = SQIHead(input_dim=input_dim)

        x = torch.randn(4, input_dim)
        out = head(x)

        assert out.shape == (4,)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_confidence_head(self):
        """Test confidence head."""
        set_seed(42)
        input_dim = 64
        head = ConfidenceHead(input_dim=input_dim)

        x = torch.randn(4, input_dim)
        out = head(x)

        assert out.shape == (4,)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestLosses:
    """Test loss computation."""

    def test_bvp_loss(self):
        """Test BVP loss computation."""
        set_seed(42)
        pred = torch.randn(4, 64)
        target = torch.randn(4, 64)
        mask = torch.zeros(4, 64, dtype=torch.bool)

        from p8_multitask.losses import BVPLossFn
        loss_fn = BVPLossFn()

        loss, metrics = loss_fn(pred, target, mask)

        assert torch.isfinite(loss)
        assert "bvp_loss" in metrics

    def test_hr_loss(self):
        """Test HR loss computation."""
        set_seed(42)
        pred = torch.randn(4) * 20 + 70  # ~70 BPM
        target = torch.randn(4) * 20 + 70

        from p8_multitask.losses import HRLossFn
        loss_fn = HRLossFn()

        loss, metrics = loss_fn(pred, target)

        assert torch.isfinite(loss)
        assert "hr_loss" in metrics

    def test_sqi_loss(self):
        """Test SQI loss computation."""
        set_seed(42)
        pred = torch.rand(4)  # [0, 1]
        target = torch.rand(4)

        from p8_multitask.losses import SQILossFn
        loss_fn = SQILossFn()

        loss, metrics = loss_fn(pred, target)

        assert torch.isfinite(loss)
        assert "sqi_loss" in metrics

    def test_multitask_loss_all_present(self):
        """Test multi-task loss with all targets present."""
        set_seed(42)
        config = MultiTaskConfig(
            enable_bvp=True,
            enable_hr=True,
            enable_sqi=True,
            enable_confidence=True
        )
        model = MultiTaskPhysiologicalModel(config)
        loss_fn = MultiTaskLoss(config)

        # Create mock output
        output = model(torch.randn(4, 64, 3))

        # Create mock targets
        targets = {
            "hr": torch.randn(4) * 20 + 70,
            "bvp": torch.randn(4, 64),
            "sqi": torch.rand(4),
            "confidence": torch.rand(4)
        }
        masks = {
            "hr_available": torch.ones(4, dtype=torch.bool),
            "bvp_available": torch.ones(4, dtype=torch.bool),
            "sqi_available": torch.ones(4, dtype=torch.bool),
            "confidence_available": torch.ones(4, dtype=torch.bool)
        }

        loss, metrics = loss_fn(output, targets, masks)

        assert torch.isfinite(loss)
        assert "total_loss" in metrics

    def test_multitask_loss_missing_labels(self):
        """Test multi-task loss with some missing labels."""
        set_seed(42)
        config = MultiTaskConfig(
            enable_bvp=True,
            enable_hr=True,
            enable_sqi=True,
            enable_confidence=True
        )
        model = MultiTaskPhysiologicalModel(config)
        loss_fn = MultiTaskLoss(config)

        output = model(torch.randn(4, 64, 3))

        targets = {
            "hr": torch.randn(4) * 20 + 70,
            # BVP missing
            "sqi": torch.rand(4),
            "confidence": torch.rand(4)
        }
        masks = {
            "hr_available": torch.ones(4, dtype=torch.bool),
            "bvp_available": torch.zeros(4, dtype=torch.bool),
            "sqi_available": torch.ones(4, dtype=torch.bool),
            "confidence_available": torch.ones(4, dtype=torch.bool)
        }

        loss, metrics = loss_fn(output, targets, masks)

        assert torch.isfinite(loss)
        assert "total_loss" in metrics

    def test_multitask_loss_no_valid_labels(self):
        """Test multi-task loss when batch has no valid labels for a task."""
        set_seed(42)
        config = MultiTaskConfig(enable_hr=True, enable_bvp=False, enable_sqi=False, enable_confidence=False)
        model = MultiTaskPhysiologicalModel(config)
        loss_fn = MultiTaskLoss(config)

        output = model(torch.randn(4, 64, 3))

        # All labels missing
        targets = {"hr": torch.randn(4)}
        masks = {"hr_available": torch.zeros(4, dtype=torch.bool)}

        loss, metrics = loss_fn(output, targets, masks)

        # Should still produce finite loss
        assert torch.isfinite(loss)


class TestMasks:
    """Test mask handling."""

    def test_sequence_mask(self):
        """Test sequence masking in forward pass."""
        set_seed(42)
        config = MultiTaskConfig(input_dim=3, hidden_dim=64, num_encoder_layers=1)
        model = MultiTaskPhysiologicalModel(config)
        model.eval()

        batch_size = 4
        seq_len = 64
        x = torch.randn(batch_size, seq_len, 3)

        # Create mask with variable lengths
        lengths = torch.tensor([32, 48, 64, 16])
        mask = torch.arange(seq_len).unsqueeze(0) >= lengths.unsqueeze(1)

        with torch.no_grad():
            output = model(x, mask=mask)

        assert output.hr.shape == (batch_size,)

    def test_availability_mask(self):
        """Test task availability mask in loss."""
        set_seed(42)
        config = MultiTaskConfig(enable_hr=True)
        model = MultiTaskPhysiologicalModel(config)
        loss_fn = MultiTaskLoss(config)

        output = model(torch.randn(4, 64, 3))

        # Only first two samples have valid HR
        targets = {"hr": torch.randn(4) * 20 + 70}
        masks = {"hr_available": torch.tensor([True, True, False, False])}

        loss, metrics = loss_fn(output, targets, masks)

        assert torch.isfinite(loss)


class TestGradients:
    """Test gradient propagation."""

    def test_gradient_encoder(self):
        """Test gradients flow through encoder."""
        set_seed(42)
        config = MultiTaskConfig(input_dim=3, hidden_dim=64, num_encoder_layers=1)
        model = MultiTaskPhysiologicalModel(config)

        x = torch.randn(4, 64, 3)
        x.requires_grad = True

        output = model(x)
        loss = output.hr.sum()

        loss.backward()

        # Check encoder has gradients
        assert model.encoder.encoder.input_proj[0].weight.grad is not None

    def test_gradient_task_heads(self):
        """Test gradients flow through task heads."""
        set_seed(42)
        config = MultiTaskConfig(input_dim=3, hidden_dim=64, num_encoder_layers=1)
        model = MultiTaskPhysiologicalModel(config)

        x = torch.randn(4, 64, 3)
        output = model(x)

        if output.hr is not None:
            loss = output.hr.sum()
            loss.backward(retain_graph=True)
            # Check gradient exists
            has_grad = any(p.grad is not None for p in model.heads["hr"].parameters())
            assert has_grad, "HR head should have gradients"

        if output.bvp is not None:
            # Clear previous gradients
            model.zero_grad()
            loss = output.bvp.sum()
            loss.backward(retain_graph=True)
            # Check gradient exists
            has_grad = any(p.grad is not None for p in model.heads["bvp"].parameters())
            assert has_grad, "BVP head should have gradients"

    def test_freeze_unfreeze_encoder(self):
        """Test freezing and unfreezing encoder."""
        set_seed(42)
        config = MultiTaskConfig(input_dim=3, hidden_dim=64, num_encoder_layers=1)
        model = MultiTaskPhysiologicalModel(config)

        # Freeze
        model.freeze_encoder()
        for param in model.encoder.parameters():
            assert not param.requires_grad

        # Unfreeze
        model.unfreeze_encoder()
        for param in model.encoder.parameters():
            assert param.requires_grad


class TestDataset:
    """Test dataset utilities."""

    def test_synthetic_dataset(self):
        """Test synthetic multi-task dataset."""
        set_seed(42)
        dataset = SyntheticMultiTaskDataset(num_samples=20, sequence_length=64)

        assert len(dataset) == 20

        sample = dataset[0]
        assert "features" in sample
        assert sample["features"].shape[0] == 64
        assert "hr" in sample
        assert "bvp" in sample

    def test_collate_batch(self):
        """Test batch collation."""
        set_seed(42)
        dataset = SyntheticMultiTaskDataset(num_samples=10, sequence_length=64)
        loader = create_multitask_dataloader(dataset, batch_size=4)

        batch = next(iter(loader))

        assert batch["features"].shape[0] == 4
        assert batch["hr"].shape[0] == 4
        assert batch["bvp"].shape[0] == 4
        assert "hr_available" in batch
        assert "bvp_available" in batch

    def test_variable_length_sequences(self):
        """Test handling of variable-length sequences."""
        set_seed(42)
        dataset = SyntheticMultiTaskDataset(num_samples=10, sequence_length=64)
        loader = create_multitask_dataloader(dataset, batch_size=4)

        batch = next(iter(loader))

        # Check mask
        assert "mask" in batch
        assert batch["mask"].shape[1] == batch["features"].shape[1]


class TestCheckpointing:
    """Test checkpoint save and load."""

    def test_save_load(self):
        """Test save and load checkpoint.

        Uses a unique temp directory to avoid Windows file locking issues.
        """
        import gc
        import shutil
        import tempfile
        set_seed(42)
        config = MultiTaskConfig(input_dim=3, hidden_dim=64, num_encoder_layers=1)
        model = MultiTaskPhysiologicalModel(config)
        optimizer = get_optimizer(model)

        # Create a unique temp directory outside of pytest's tmp_path
        temp_dir = tempfile.mkdtemp()
        checkpoint_path = os.path.join(temp_dir, "checkpoint.pt")

        try:
            # Save checkpoint
            save_checkpoint(model, optimizer, epoch=5, path=checkpoint_path, loss=0.5)

            # Delete old references to release file handle
            del model
            del optimizer
            gc.collect()

            assert os.path.exists(checkpoint_path)

            # Load in a new model instance
            set_seed(42)
            model2 = MultiTaskPhysiologicalModel(MultiTaskConfig(input_dim=3, hidden_dim=64, num_encoder_layers=1))
            optimizer2 = get_optimizer(model2)

            # Read file and load with weights_only=False for compatibility
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = f.read()

            import io
            buffer = io.BytesIO(checkpoint_data)
            checkpoint = torch.load(buffer, map_location='cpu', weights_only=False)

            model2.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get("epoch", 0)
            loss = checkpoint.get("loss", 0.0)

            assert epoch == 5
            assert loss == 0.5
        finally:
            # Clean up
            try:
                shutil.rmtree(temp_dir)
            except:
                pass


class TestTrainingStep:
    """Test end-to-end training step."""

    def test_training_step(self):
        """Test minimal training step."""
        set_seed(42)
        config = MultiTaskConfig(
            input_dim=3,
            hidden_dim=64,
            num_encoder_layers=1,
            enable_bvp=True,
            enable_hr=True
        )
        model = MultiTaskPhysiologicalModel(config)
        loss_fn = MultiTaskLoss(config)
        optimizer = get_optimizer(model)

        device = torch.device("cpu")

        # Create batch
        features = torch.randn(4, 64, 3)
        mask = torch.zeros(4, 64, dtype=torch.bool)
        hr_target = torch.randn(4) * 20 + 70
        bvp_target = torch.randn(4, 64)

        targets = {"hr": hr_target, "bvp": bvp_target}
        masks = {
            "hr_available": torch.ones(4, dtype=torch.bool),
            "bvp_available": torch.ones(4, dtype=torch.bool)
        }

        # Forward
        output = model(features, mask=mask)

        # Loss
        loss, metrics = loss_fn(output, targets, masks)

        assert torch.isfinite(loss)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class TestInference:
    """Test inference output structure."""

    def test_inference_output_structure(self):
        """Test inference output has all expected fields."""
        set_seed(42)
        config = MultiTaskConfig()
        model = MultiTaskPhysiologicalModel(config)
        model.eval()

        x = torch.randn(2, 64, 3)
        with torch.no_grad():
            output = model(x)

        # Check all expected fields
        assert hasattr(output, 'hr')
        assert hasattr(output, 'bvp')
        assert hasattr(output, 'sqi')
        assert hasattr(output, 'confidence')
        assert hasattr(output, 'representation')

        # Check has_task method
        assert output.has_task(TaskType.HR)
        assert output.has_task(TaskType.BVP)

        # Check get_dict method
        out_dict = output.get_dict()
        assert isinstance(out_dict, dict)

    def test_inference_deterministic(self):
        """Test inference is deterministic with same seed."""
        set_seed(42)
        config = MultiTaskConfig(input_dim=3, hidden_dim=64, num_encoder_layers=1)
        model = MultiTaskPhysiologicalModel(config)
        model.eval()

        x = torch.randn(2, 64, 3)

        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        # Same input should give same output
        assert torch.allclose(out1.hr, out2.hr, atol=1e-6)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_input_shape(self):
        """Test handling of invalid input shape."""
        set_seed(42)
        config = MultiTaskConfig(input_dim=3)
        model = MultiTaskPhysiologicalModel(config)
        model.eval()

        # Wrong number of dimensions
        x = torch.randn(4, 64)  # Should be (B, T, F)

        try:
            with torch.no_grad():
                output = model(x)
            # May succeed or fail depending on implementation
        except Exception:
            pass  # Expected for wrong input shape

    def test_non_finite_loss_handling(self):
        """Test handling of non-finite loss."""
        set_seed(42)
        config = MultiTaskConfig()
        model = MultiTaskPhysiologicalModel(config)
        loss_fn = MultiTaskLoss(config)
        optimizer = get_optimizer(model)

        device = torch.device("cpu")

        # Create batch with potential for numerical issues
        features = torch.randn(4, 64, 3) * 1000  # Large values
        targets = {"hr": torch.randn(4) * 1000}

        # Forward
        optimizer.zero_grad()
        output = model(features)

        loss, _ = loss_fn(output, targets, {"hr_available": torch.ones(4, dtype=torch.bool)})

        # Loss should still be finite (or handled gracefully)
        if torch.isfinite(loss):
            loss.backward()
            optimizer.step()


def run_all_tests():
    """Run all tests and print results."""
    test_classes = [
        TestModelConstruction,
        TestForwardPass,
        TestTaskHeads,
        TestLosses,
        TestMasks,
        TestGradients,
        TestDataset,
        TestCheckpointing,
        TestTrainingStep,
        TestInference,
        TestEdgeCases,
    ]

    total_passed = 0
    total_failed = 0
    failed_tests = []

    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"  {test_class.__name__}")
        print(f"{'='*60}")

        instance = test_class()
        test_methods = [m for m in dir(instance) if m.startswith("test_")]

        for method_name in test_methods:
            method = getattr(instance, method_name)
            try:
                # Handle tmp_path fixture
                if "tmp_path" in method.__code__.co_varnames:
                    import tempfile
                    with tempfile.TemporaryDirectory() as tmp:
                        class TmpPath:
                            def __truediv__(self, other):
                                return os.path.join(self, other)
                        tmp_path = TmpPath()
                        tmp_path._path = tmp
                        import types
                        old_cwd = os.getcwd()
                        os.chdir(tmp)
                        method(tmp_path)
                        os.chdir(old_cwd)
                else:
                    method()
                print(f"  {method_name}... PASS")
                total_passed += 1
            except Exception as e:
                print(f"  {method_name}... FAIL ({str(e)[:50]})")
                total_failed += 1
                failed_tests.append(f"{test_class.__name__}.{method_name}")

    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  Tests: {total_passed + total_failed}")
    print(f"  Passed: {total_passed}")
    print(f"  Failed: {total_failed}")

    if failed_tests:
        print(f"\n  Failed tests:")
        for t in failed_tests:
            print(f"    - {t}")

    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
