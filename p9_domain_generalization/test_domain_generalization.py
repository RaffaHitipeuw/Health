"""
P9 Domain Generalization Tests

Basic correctness tests for domain generalization functionality.
"""

import torch
import pytest
import numpy as np
import sys
import os
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p9_domain_generalization import (
    DomainMetadata,
    DomainSplit,
    DomainAwareConfig,
    DomainAwareDataset,
    SplitType,
    EvaluationProtocol,
    create_domain_split,
    create_leave_one_domain_out_splits,
    validate_split,
    DomainSplitResult,
    DomainBalancedSampler,
    DomainWeightedSampler,
    GroupDROSampler,
    create_domain_sampler,
    DomainGeneralizationTrainer,
    GroupDROLoss,
    DomainMetrics,
    compute_domain_metrics,
    aggregate_across_domains,
    check_leakage,
    summarize_domain_shift,
    set_seed,
)


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


class TestDomainMetadata:
    """Test domain metadata structures."""

    def test_metadata_creation(self):
        """Test creating domain metadata."""
        meta = DomainMetadata(
            sample_id="sample_1",
            subject_id=1,
            dataset="dataset_a",
            device="camera_1",
            environment="indoor"
        )

        assert meta.sample_id == "sample_1"
        assert meta.subject_id == 1
        assert meta.dataset == "dataset_a"

    def test_metadata_unknown(self):
        """Test creating metadata with unknown fields."""
        meta = DomainMetadata.create_unknown("sample_1", subject_id=1)

        assert meta.sample_id == "sample_1"
        assert meta.subject_id == 1
        assert meta.dataset is None
        assert meta.device is None

    def test_domain_key(self):
        """Test getting domain key."""
        meta = DomainMetadata(
            sample_id="sample_1",
            dataset="dataset_a"
        )

        assert meta.get_domain_key("dataset") == "dataset_a"
        assert meta.get_domain_key("unknown") == "unknown"

    def test_metadata_to_dict(self):
        """Test converting metadata to dict."""
        meta = DomainMetadata(
            sample_id="sample_1",
            subject_id=1,
            dataset="dataset_a"
        )
        d = meta.to_dict()

        assert d["sample_id"] == "sample_1"
        assert d["subject_id"] == 1
        assert d["dataset"] == "dataset_a"


class TestDomainSplits:
    """Test domain-aware data splitting."""

    def _create_test_metadata(self, n_samples: int = 100) -> List[DomainMetadata]:
        """Create test metadata with multiple domains."""
        metadata = []
        for i in range(n_samples):
            domain = f"domain_{i % 3}"  # 3 domains
            subject = i % 10  # 10 subjects
            meta = DomainMetadata(
                sample_id=f"sample_{i}",
                subject_id=subject,
                dataset=domain,
                device=f"device_{i % 2}",
                environment=f"env_{i % 2}"
            )
            metadata.append(meta)
        return metadata

    def test_random_split(self):
        """Test random split."""
        set_random_seed(42)
        metadata = self._create_test_metadata(100)

        result = create_domain_split(
            metadata=metadata,
            domain_key="dataset",
            split_type=SplitType.RANDOM,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42
        )

        assert result.split.train_count > 0
        assert result.split.val_count > 0
        assert result.split.test_count > 0
        assert result.split.train_count + result.split.val_count + result.split.test_count == 100

    def test_domain_held_out_split(self):
        """Test domain-held-out split."""
        set_random_seed(42)
        metadata = self._create_test_metadata(100)

        result = create_domain_split(
            metadata=metadata,
            domain_key="dataset",
            split_type=SplitType.DOMAIN_HELD_OUT,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42
        )

        assert result.split.split_type == SplitType.DOMAIN_HELD_OUT
        assert result.split.held_out_domains is not None

    def test_leave_one_domain_out(self):
        """Test leave-one-domain-out splits."""
        set_random_seed(42)
        metadata = self._create_test_metadata(90)  # 3 domains, 30 each

        splits = create_leave_one_domain_out_splits(
            metadata=metadata,
            domain_key="dataset",
            train_ratio=0.7,
            val_ratio=0.15,
            seed=42
        )

        assert len(splits) == 3  # One for each domain

        # Each split should have one domain held out
        for split_result in splits:
            assert split_result.split.split_type == SplitType.LEAVE_ONE_DOMAIN_OUT
            assert len(split_result.split.held_out_domains) == 1

    def test_split_deterministic(self):
        """Test that splits are deterministic with same seed."""
        set_random_seed(42)
        metadata = self._create_test_metadata(100)

        result1 = create_domain_split(
            metadata=metadata,
            seed=42
        )

        set_random_seed(42)
        result2 = create_domain_split(
            metadata=metadata,
            seed=42
        )

        assert np.array_equal(result1.split.train_indices, result2.split.train_indices)
        assert np.array_equal(result1.split.val_indices, result2.split.val_indices)
        assert np.array_equal(result1.split.test_indices, result2.split.test_indices)

    def test_validate_split_no_overlap(self):
        """Test split validation for no index overlap."""
        set_random_seed(42)
        metadata = self._create_test_metadata(100)

        result = create_domain_split(
            metadata=metadata,
            seed=42
        )

        is_valid, issues = validate_split(
            result.split,
            metadata,
            check_leakage=True,
            check_subject_overlap=False  # Random split allows subject overlap
        )

        # Should have no index overlap issues
        overlap_issues = [i for i in issues if "overlap" in i.lower() and "sample" in i.lower()]
        assert len(overlap_issues) == 0


class TestDomainAwareDataset:
    """Test DomainAwareDataset wrapper."""

    def test_dataset_wrapper(self):
        """Test wrapping a dataset."""
        # Create mock base dataset
        class MockDataset:
            def __len__(self):
                return 100

            def __getitem__(self, idx):
                return {"features": torch.randn(10, 3)}

        base = MockDataset()
        metadata = [
            DomainMetadata(sample_id=f"s_{i}", dataset=f"d_{i % 3}")
            for i in range(100)
        ]

        dataset = DomainAwareDataset(base, metadata, domain_key="dataset")

        assert len(dataset) == 100
        assert len(dataset.get_all_domains()) == 3

    def test_get_domain_of_sample(self):
        """Test getting domain of sample."""
        class MockDataset:
            def __len__(self):
                return 50

            def __getitem__(self, idx):
                return {}

        base = MockDataset()
        metadata = [
            DomainMetadata(sample_id=f"s_{i}", dataset=f"domain_{i % 2}")
            for i in range(50)
        ]

        dataset = DomainAwareDataset(base, metadata, domain_key="dataset")

        assert dataset.get_domain_of_sample(0) == "domain_0"
        assert dataset.get_domain_of_sample(1) == "domain_1"

    def test_get_indices_by_domain(self):
        """Test getting indices by domain."""
        class MockDataset:
            def __len__(self):
                return 60

            def __getitem__(self, idx):
                return {}

        base = MockDataset()
        metadata = [
            DomainMetadata(sample_id=f"s_{i}", dataset=f"domain_{i % 3}")
            for i in range(60)
        ]

        dataset = DomainAwareDataset(base, metadata, domain_key="dataset")

        indices_0 = dataset.get_indices_by_domain("domain_0")
        assert len(indices_0) == 20


class TestSamplers:
    """Test domain-aware samplers."""

    def test_domain_balanced_sampler(self):
        """Test domain-balanced sampler."""
        set_random_seed(42)

        class MockDataset:
            def __len__(self):
                return 60

            def __getitem__(self, idx):
                return {"features": torch.randn(10, 3)}

        base = MockDataset()
        metadata = [
            DomainMetadata(sample_id=f"s_{i}", dataset=f"domain_{i % 3}")
            for i in range(60)
        ]

        dataset = DomainAwareDataset(base, metadata, domain_key="dataset")
        indices = list(range(60))

        sampler = DomainBalancedSampler(
            dataset=dataset,
            indices=indices,
            batch_size=6,
            shuffle=True,
            drop_last=True,
            seed=42
        )

        batches = list(sampler)
        assert len(batches) > 0

    def test_domain_weighted_sampler(self):
        """Test domain-weighted sampler."""
        set_random_seed(42)

        class MockDataset:
            def __len__(self):
                return 60

            def __getitem__(self, idx):
                return {"features": torch.randn(10, 3)}

        base = MockDataset()
        metadata = [
            DomainMetadata(sample_id=f"s_{i}", dataset=f"domain_{i % 3}")
            for i in range(60)
        ]

        dataset = DomainAwareDataset(base, metadata, domain_key="dataset")
        indices = list(range(60))

        sampler = DomainWeightedSampler(
            dataset=dataset,
            indices=indices,
            weight_mode="inverse",
            batch_size=6,
            seed=42
        )

        assert sampler.num_samples == 60


class TestGroupDRO:
    """Test GroupDRO loss."""

    def test_groupdro_initialization(self):
        """Test GroupDRO loss initialization."""
        base_loss = torch.nn.MSELoss()

        domain_indices = {
            "domain_0": list(range(20)),
            "domain_1": list(range(20, 40)),
            "domain_2": list(range(40, 60)),
        }

        groupdro = GroupDROLoss(
            base_loss_fn=base_loss,
            num_groups=3,
            domain_indices=domain_indices,
            eta=0.01
        )

        weights = groupdro.get_weights()
        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_groupdro_update(self):
        """Test GroupDRO weight update."""
        base_loss = torch.nn.MSELoss()

        domain_indices = {
            "domain_0": list(range(20)),
            "domain_1": list(range(20, 40)),
            "domain_2": list(range(40, 60)),
        }

        groupdro = GroupDROLoss(
            base_loss_fn=base_loss,
            num_groups=3,
            domain_indices=domain_indices,
            eta=0.1
        )

        # Update with losses
        losses = {
            "domain_0": 1.0,
            "domain_1": 2.0,
            "domain_2": 3.0,
        }
        groupdro.update(losses)

        weights = groupdro.get_weights()
        # Higher weight for higher loss
        assert weights["domain_2"] > weights["domain_0"]


class TestEvaluation:
    """Test evaluation utilities."""

    def test_compute_domain_metrics(self):
        """Test computing domain metrics."""
        predictions = {
            "hr": torch.tensor([70.0, 72.0, 75.0, 80.0]),
            "sqi": torch.tensor([0.9, 0.8, 0.7, 0.6]),
        }
        targets = {
            "hr": torch.tensor([72.0, 72.0, 75.0, 78.0]),
            "sqi": torch.tensor([0.85, 0.8, 0.75, 0.65]),
        }

        metrics = compute_domain_metrics(predictions, targets, "test_domain")

        assert metrics.domain == "test_domain"
        assert metrics.hr_mae is not None
        assert metrics.hr_mae > 0  # Should have some error

    def test_aggregate_across_domains(self):
        """Test aggregating metrics across domains."""
        per_domain = {
            "domain_0": DomainMetrics(
                domain="domain_0",
                sample_count=10,
                hr_mae=2.0
            ),
            "domain_1": DomainMetrics(
                domain="domain_1",
                sample_count=10,
                hr_mae=4.0
            ),
            "domain_2": DomainMetrics(
                domain="domain_2",
                sample_count=10,
                hr_mae=6.0
            ),
        }

        agg = aggregate_across_domains(per_domain)

        assert agg.mean_hr_mae == 4.0
        assert agg.std_hr_mae > 0
        assert agg.num_domains == 3

    def test_check_leakage(self):
        """Test leakage checking."""
        # Create a split with overlap
        split = DomainSplit(
            train_indices=np.array([0, 1, 2, 3, 4]),
            val_indices=np.array([3, 4, 5, 6, 7]),  # Overlaps with train
            test_indices=np.array([8, 9]),
            seed=42
        )

        metadata = [
            DomainMetadata(sample_id=f"s_{i}", subject_id=i // 3)
            for i in range(10)
        ]

        has_leakage, issues = check_leakage(split, metadata)

        assert has_leakage
        assert len(issues) > 0

    def test_summarize_domain_shift(self):
        """Test domain shift summary."""
        train_domains = {
            "domain_0": 30,
            "domain_1": 30,
        }
        test_domains = {
            "domain_0": 15,
            "domain_1": 15,
            "domain_2": 20,  # Unseen domain
        }

        summary = summarize_domain_shift(train_domains, test_domains)

        assert len(summary["shared_domains"]) == 2
        assert "domain_2" in summary["unseen_domains"]
        assert summary["domain_shift_detected"] is True


class TestIntegration:
    """Integration tests with P8 model."""

    def test_domain_aware_config(self):
        """Test domain-aware configuration."""
        config = DomainAwareConfig(
            split_type=SplitType.DOMAIN_HELD_OUT,
            domain_key="dataset",
            method="erm",
            batch_size=16,
            epochs=10
        )

        assert config.split_type == SplitType.DOMAIN_HELD_OUT
        assert config.method == "erm"

    def test_split_result_summary(self):
        """Test split result summary."""
        set_random_seed(42)
        metadata = [
            DomainMetadata(sample_id=f"s_{i}", dataset=f"d_{i % 2}")
            for i in range(50)
        ]

        result = create_domain_split(
            metadata=metadata,
            seed=42
        )

        summary = result.get_summary()
        assert "train_count" in summary
        assert "domain_counts" in summary


def run_all_tests():
    """Run all tests and print results."""
    test_classes = [
        TestDomainMetadata,
        TestDomainSplits,
        TestDomainAwareDataset,
        TestSamplers,
        TestGroupDRO,
        TestEvaluation,
        TestIntegration,
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
                method()
                print(f"  {method_name}... PASS")
                total_passed += 1
            except Exception as e:
                print(f"  {method_name}... FAIL ({str(e)[:80]})")
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
