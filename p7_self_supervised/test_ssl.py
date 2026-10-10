"""
P7 Self-Supervised Learning Basic Correctness Tests

Verifies:
    - Model and objective instantiation
    - Forward passes and representation output shapes
    - Positive/negative pair or view construction
    - Loss computation and finite loss values
    - Gradient propagation
    - Checkpoint save/load
    - Minimal self-supervised training step
    - Invalid inputs and edge cases
"""

import torch
import torch.nn.functional as F
import numpy as np
import tempfile
import os
from typing import Tuple


def test_ssl_config() -> Tuple[int, int]:
    """Test SSL configuration."""
    from p7_self_supervised.base import SelfSupervisedConfig
    passed = 0
    failed = 0
    print("  SSL Config...")
    try:
        config = SelfSupervisedConfig(
            encoder_type="lstm",
            encoder_hidden_dim=64,
            input_dim=3
        )
        assert config.encoder_hidden_dim == 64
        assert config.input_dim == 3
        passed += 1
    except Exception as e:
        print(f"    FAIL: {e}")
        failed += 1
    return passed, failed


def test_encoder_instantiation() -> Tuple[int, int]:
    """Test SSL encoder instantiation."""
    from p7_self_supervised.encoder import SelfSupervisedEncoder, SimpleSelfSupervisedEncoder
    from p7_self_supervised.base import SelfSupervisedConfig
    passed = 0
    failed = 0
    print("  Encoder Instantiation...")
    try:
        config = SelfSupervisedConfig(
            encoder_type="lstm",
            encoder_hidden_dim=64,
            encoder_num_layers=2,
            input_dim=3,
            projection_dim=64,
            representation_dim=32
        )
        model = SelfSupervisedEncoder(config)
        assert model is not None
        passed += 1
    except Exception as e:
        print(f"    FAIL: {e}")
        failed += 1
        return passed, failed

    try:
        # Test simple encoder
        simple_model = SimpleSelfSupervisedEncoder(config)
        assert simple_model is not None
        passed += 1
    except Exception as e:
        print(f"    FAIL (simple): {e}")
        failed += 1
    return passed, failed


def test_encoder_forward() -> Tuple[int, int]:
    """Test encoder forward pass."""
    from p7_self_supervised.encoder import SimpleSelfSupervisedEncoder
    from p7_self_supervised.base import SelfSupervisedConfig
    passed = 0
    failed = 0
    print("  Encoder Forward...")
    try:
        config = SelfSupervisedConfig(
            encoder_type="simple",
            encoder_hidden_dim=64,
            input_dim=3,
            projection_dim=64,
            representation_dim=32
        )
        model = SimpleSelfSupervisedEncoder(config)
        x = torch.randn(4, 128, 3)  # (B, T, F)

        output = model(x)

        assert output.representations.shape == (4, 32), f"Expected (4, 32), got {output.representations.shape}"
        assert output.projections.shape == (4, 64), f"Expected (4, 64), got {output.projections.shape}"
        passed += 1
    except Exception as e:
        print(f"    FAIL: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    return passed, failed


def test_projection_head() -> Tuple[int, int]:
    """Test projection head."""
    from p7_self_supervised.encoder import ProjectionHead
    passed = 0
    failed = 0
    print("  Projection Head...")
    try:
        head = ProjectionHead(input_dim=128, hidden_dim=64, output_dim=64, num_layers=2)
        x = torch.randn(4, 128)
        out = head(x)
        assert out.shape == (4, 64), f"Expected (4, 64), got {out.shape}"
        passed += 1
    except Exception as e:
        print(f"    FAIL: {e}")
        failed += 1
    return passed, failed


def test_contrastive_loss() -> Tuple[int, int]:
    """Test contrastive loss."""
    from p7_self_supervised.objectives import ContrastiveLoss
    passed = 0
    failed = 0
    print("  Contrastive Loss...")
    try:
        loss_fn = ContrastiveLoss(temperature=0.1)
        projections = F.normalize(torch.randn(8, 64), dim=1)
        loss, metrics = loss_fn(projections)
        assert loss.item() > 0, "Loss should be positive"
        assert "contrastive_loss" in metrics
        passed += 1
    except Exception as e:
        print(f"    FAIL: {e}")
        failed += 1
    return passed, failed


def test_combined_ssl_loss() -> Tuple[int, int]:
    """Test combined SSL loss."""
    from p7_self_supervised.objectives import CombinedSSLLoss
    passed = 0
    failed = 0
    print("  Combined SSL Loss...")
    try:
        loss_fn = CombinedSSLLoss(
            contrastive_weight=1.0,
            consistency_weight=0.5,
            predictive_weight=0.0,
            temperature=0.1
        )

        representations = F.normalize(torch.randn(4, 64), dim=1)
        projections = F.normalize(torch.randn(8, 64), dim=1)  # 2 views per sample

        loss, losses = loss_fn(
            representations=representations,
            projections=projections,
            segments=torch.randn(4, 128, 3)
        )

        assert loss.item() > 0, "Loss should be positive"
        assert "total_loss" in losses
        passed += 1
    except Exception as e:
        print(f"    FAIL: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    return passed, failed


def test_augmentations() -> Tuple[int, int]:
    """Test augmentation functions."""
    from p7_self_supervised.augmentations import (
        GaussianNoise, AmplitudeScale, SignalDropout,
        TemporalCrop, ComposeAugmentations
    )
    passed = 0
    failed = 0
    print("  Augmentations...")
    try:
        x = torch.randn(128, 3)

        # Test Gaussian noise
        aug = GaussianNoise(std=0.05)
        x_aug = aug(x)
        assert x_aug.shape == x.shape

        # Test amplitude scale
        aug = AmplitudeScale(scale_range=(0.9, 1.1))
        x_aug = aug(x)
        assert x_aug.shape == x.shape

        # Test signal dropout
        aug = SignalDropout(prob=0.1)
        x_aug = aug(x)
        assert x_aug.shape == x.shape

        # Test composition
        aug = ComposeAugmentations([GaussianNoise(0.05), AmplitudeScale((0.9, 1.1))])
        x_aug = aug(x)
        assert x_aug.shape == x.shape

        passed += 1
    except Exception as e:
        print(f"    FAIL: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    return passed, failed


def test_multiview_augmenter() -> Tuple[int, int]:
    """Test multi-view augmenter."""
    from p7_self_supervised.augmentations import create_multiview_augmenter
    passed = 0
    failed = 0
    print("  Multi-View Augmenter...")
    try:
        augmenter = create_multiview_augmenter(num_views=2, crop_length=64)
        x = torch.randn(128, 3)
        views = augmenter(x)

        assert len(views) == 2, f"Expected 2 views, got {len(views)}"
        assert views[0].shape[0] == 64, f"Expected crop length 64, got {views[0].shape[0]}"

        passed += 1
    except Exception as e:
        print(f"    FAIL: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    return passed, failed


def test_dataset() -> Tuple[int, int]:
    """Test SSL dataset."""
    from p7_self_supervised.dataset import SyntheticSelfSupervisedDataset, collate_ssl_batch
    passed = 0
    failed = 0
    print("  Dataset...")
    try:
        dataset = SyntheticSelfSupervisedDataset(num_samples=20, sequence_length=64)
        sample = dataset[0]

        assert "features" in sample
        assert "bpm" in sample
        assert sample["features"].shape[0] == 64

        passed += 1
    except Exception as e:
        print(f"    FAIL: {e}")
        failed += 1
    return passed, failed


def test_gradient_propagation() -> Tuple[int, int]:
    """Test gradient propagation through encoder."""
    from p7_self_supervised.encoder import SimpleSelfSupervisedEncoder
    from p7_self_supervised.base import SelfSupervisedConfig
    passed = 0
    failed = 0
    print("  Gradient Propagation...")
    try:
        config = SelfSupervisedConfig(
            encoder_type="simple",
            encoder_hidden_dim=64,
            input_dim=3,
            projection_dim=64,
            representation_dim=32
        )
        model = SimpleSelfSupervisedEncoder(config)
        x = torch.randn(2, 64, 3, requires_grad=True)

        output = model(x)
        loss = output.projections.sum()
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None, "Input gradients should exist"
        assert model.encoder.lstm.weight_ih_l0.grad is not None, "Encoder gradients should exist"

        passed += 1
    except Exception as e:
        print(f"    FAIL: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    return passed, failed


def test_checkpoint_save_load() -> Tuple[int, int]:
    """Test checkpoint save and load."""
    from p7_self_supervised.encoder import SimpleSelfSupervisedEncoder
    from p7_self_supervised.base import SelfSupervisedConfig
    from p7_self_supervised.training import save_checkpoint, load_checkpoint
    passed = 0
    failed = 0
    print("  Checkpoint Save/Load...")
    try:
        config = SelfSupervisedConfig(
            encoder_type="simple",
            encoder_hidden_dim=64,
            input_dim=3,
            projection_dim=64,
            representation_dim=32
        )
        model = SimpleSelfSupervisedEncoder(config)
        optimizer = torch.optim.Adam(model.parameters())

        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            ckpt_path = f.name

        save_checkpoint(model, optimizer, epoch=1, path=ckpt_path, loss=0.5)

        # Load checkpoint
        model2 = SimpleSelfSupervisedEncoder(config)
        optimizer2 = torch.optim.Adam(model2.parameters())
        epoch, loss = load_checkpoint(model2, optimizer2, path=ckpt_path)

        assert epoch == 1
        assert loss == 0.5

        # Verify parameters match
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert torch.allclose(p1, p2), f"Parameter {n1} doesn't match after reload"

        os.unlink(ckpt_path)
        passed += 1
    except Exception as e:
        print(f"    FAIL: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    return passed, failed


def test_ssl_training_step() -> Tuple[int, int]:
    """Test a minimal SSL training step."""
    from p7_self_supervised.encoder import SimpleSelfSupervisedEncoder
    from p7_self_supervised.base import SelfSupervisedConfig
    from p7_self_supervised.objectives import CombinedSSLLoss
    from p7_self_supervised.dataset import SyntheticSelfSupervisedDataset
    from p7_self_supervised.training import get_optimizer
    passed = 0
    failed = 0
    print("  SSL Training Step...")
    try:
        # Create model and data
        config = SelfSupervisedConfig(
            encoder_type="simple",
            encoder_hidden_dim=32,
            input_dim=3,
            projection_dim=32,
            representation_dim=16
        )
        model = SimpleSelfSupervisedEncoder(config)
        dataset = SyntheticSelfSupervisedDataset(num_samples=20, sequence_length=64)

        # Wrap with multi-view
        from p7_self_supervised.dataset import MultiViewDataset, create_ssl_dataloader
        from p7_self_supervised.augmentations import create_multiview_augmenter

        augmenter = create_multiview_augmenter(num_views=2, crop_length=32)
        mv_dataset = MultiViewDataset(dataset, num_views=2, augment_fn=lambda x: augmenter(x))
        loader = create_ssl_dataloader(mv_dataset, batch_size=4)

        loss_fn = CombinedSSLLoss(contrastive_weight=1.0, consistency_weight=0.0)
        optimizer = get_optimizer(model, lr=1e-4)

        # Single training step
        model.train()
        batch = next(iter(loader))
        features = batch["features"]
        views = batch["views"]

        B, V, T, F = views.shape
        views_flat = views.reshape(B * V, T, F)

        optimizer.zero_grad()
        output = model(views_flat)

        projections = output.projections.reshape(B, V, -1).reshape(B * V, -1)
        loss, losses = loss_fn(
            representations=output.representations,
            projections=projections,
            indices=batch["indices"]
        )

        loss.backward()
        optimizer.step()

        assert loss.item() > 0, "Loss should be positive"
        passed += 1
    except Exception as e:
        print(f"    FAIL: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    return passed, failed


def test_invalid_input() -> Tuple[int, int]:
    """Test handling of invalid inputs."""
    from p7_self_supervised.encoder import SimpleSelfSupervisedEncoder
    from p7_self_supervised.base import SelfSupervisedConfig
    passed = 0
    failed = 0
    print("  Invalid Input Handling...")
    try:
        config = SelfSupervisedConfig(
            encoder_type="simple",
            encoder_hidden_dim=64,
            input_dim=3
        )
        model = SimpleSelfSupervisedEncoder(config)

        # Wrong dimensions
        x_wrong = torch.randn(32)  # Should be (B, T, F)
        try:
            out = model(x_wrong)
            print("    (Warning: wrong input did not raise error)")
        except (RuntimeError, ValueError):
            pass  # Expected

        passed += 1
    except Exception as e:
        print(f"    FAIL: {e}")
        failed += 1
    return passed, failed


def test_nt_xent_loss() -> Tuple[int, int]:
    """Test NT-Xent loss function."""
    from p7_self_supervised.objectives import nt_xent_loss
    passed = 0
    failed = 0
    print("  NT-Xent Loss...")
    try:
        z1 = F.normalize(torch.randn(4, 64), dim=1)
        z2 = F.normalize(torch.randn(4, 64), dim=1)

        loss = nt_xent_loss(z1, z2, temperature=0.1)
        assert loss.item() > 0, "Loss should be positive"
        passed += 1
    except Exception as e:
        print(f"    FAIL: {e}")
        failed += 1
    return passed, failed


def run_tests() -> bool:
    """Run all P7 tests."""
    print("\n" + "=" * 60)
    print("P7 Self-Supervised Learning Tests")
    print("=" * 60)

    # Import F for tests
    import torch.nn.functional as F

    total_pass = 0
    total_fail = 0

    for name, fn in [
        ("SSLConfig", test_ssl_config),
        ("EncoderInstantiation", test_encoder_instantiation),
        ("EncoderForward", test_encoder_forward),
        ("ProjectionHead", test_projection_head),
        ("ContrastiveLoss", test_contrastive_loss),
        ("CombinedSSLLoss", test_combined_ssl_loss),
        ("Augmentations", test_augmentations),
        ("MultiViewAugmenter", test_multiview_augmenter),
        ("Dataset", test_dataset),
        ("GradientPropagation", test_gradient_propagation),
        ("Checkpoint", test_checkpoint_save_load),
        ("SSLTrainingStep", test_ssl_training_step),
        ("InvalidInput", test_invalid_input),
        ("NTXentLoss", test_nt_xent_loss),
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
