"""
P9 Domain-Aware Sampling

Sampling strategies for domain generalization training.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable
from torch.utils.data import Sampler, Dataset
import torch.utils.data
from collections import defaultdict

from .base import DomainAwareDataset, set_seed


class DomainBalancedSampler(Sampler):
    """
    Sampler that balances samples across domains.

    Ensures each batch contains approximately equal representation
    from all domains in the training set.
    """

    def __init__(
        self,
        dataset: DomainAwareDataset,
        indices: List[int],
        batch_size: int = 16,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42
    ):
        """
        Args:
            dataset: DomainAwareDataset
            indices: Indices of samples to sample from
            batch_size: Batch size
            shuffle: Whether to shuffle
            drop_last: Whether to drop incomplete last batch
            seed: Random seed
        """
        self.dataset = dataset
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

        # Group indices by domain
        self.domain_indices: Dict[str, List[int]] = defaultdict(list)
        for idx in indices:
            domain = dataset.get_domain_of_sample(idx)
            self.domain_indices[domain].append(idx)

        self.num_domains = len(self.domain_indices)
        self.num_samples = len(indices)

        if self.num_domains == 0:
            raise ValueError("No samples to sample from")

    def __iter__(self) -> Iterator[int]:
        """Generate batches of indices."""
        set_seed(self.seed)

        # Shuffle within each domain
        domain_iterators = {}
        for domain, domain_idx in self.domain_indices.items():
            shuffled = domain_idx.copy()
            if self.shuffle:
                np.random.shuffle(shuffled)
            domain_iterators[domain] = iter(shuffled)

        # Generate batches
        batch = []
        while True:
            for domain in self.domain_indices.keys():
                try:
                    idx = next(domain_iterators[domain])
                    batch.append(idx)

                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []
                except StopIteration:
                    # Domain exhausted, continue with other domains
                    pass

            # Check if all domains exhausted
            if len(batch) == 0:
                break

            # Yield remaining samples if not dropping
            if not self.drop_last and len(batch) > 0:
                yield batch
                break

    def __len__(self) -> int:
        """Number of batches."""
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size


class DomainWeightedSampler(Sampler):
    """
    Sampler that weights samples inversely by domain frequency.

    Less frequent domains get higher weights to balance representation.
    """

    def __init__(
        self,
        dataset: DomainAwareDataset,
        indices: List[int],
        weights: Optional[Dict[str, float]] = None,
        weight_mode: str = "inverse",  # "inverse", "sqrt_inverse", "uniform"
        batch_size: int = 16,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42
    ):
        """
        Args:
            dataset: DomainAwareDataset
            indices: Indices of samples to sample from
            weights: Optional explicit domain weights
            weight_mode: Mode for computing weights ("inverse", "sqrt_inverse", "uniform")
            batch_size: Batch size
            shuffle: Whether to shuffle
            drop_last: Whether to drop incomplete last batch
            seed: Random seed
        """
        self.dataset = dataset
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.weight_mode = weight_mode

        # Group indices by domain and count
        domain_counts = defaultdict(int)
        for idx in indices:
            domain = dataset.get_domain_of_sample(idx)
            domain_counts[domain] += 1

        self.domain_counts = dict(domain_counts)
        self.num_samples = len(indices)

        # Compute weights
        if weights is not None:
            self.domain_weights = weights
        else:
            self.domain_weights = self._compute_weights(domain_counts)

        # Create index weights
        self.index_weights = torch.zeros(len(indices))
        self.index_to_original = {}

        for i, idx in enumerate(indices):
            domain = dataset.get_domain_of_sample(idx)
            self.index_weights[i] = self.domain_weights.get(domain, 1.0)
            self.index_to_original[i] = idx

    def _compute_weights(self, domain_counts: Dict[str, int]) -> Dict[str, float]:
        """Compute domain weights based on frequency."""
        if self.weight_mode == "uniform":
            return {d: 1.0 for d in domain_counts}

        total = sum(domain_counts.values())
        weights = {}

        for domain, count in domain_counts.items():
            if self.weight_mode == "inverse":
                # Weight = total / (n_domains * count)
                weights[domain] = total / (len(domain_counts) * count)
            elif self.weight_mode == "sqrt_inverse":
                # Weight = sqrt(total / count)
                weights[domain] = (total / count) ** 0.5
            else:
                weights[domain] = 1.0

        return weights

    def __iter__(self) -> Iterator[int]:
        """Generate batches of indices."""
        set_seed(self.seed)

        # Create weighted random sampling
        num_batches = len(self) if self.drop_last else (len(self) + 1)
        indices = torch.multinomial(
            self.index_weights,
            num_samples=self.batch_size * num_batches,
            replacement=True
        )

        for i in range(num_batches):
            batch_indices = indices[i * self.batch_size:(i + 1) * self.batch_size]
            if len(batch_indices) < self.batch_size and self.drop_last:
                break

            original_indices = [self.index_to_original[int(idx)] for idx in batch_indices]
            if self.shuffle:
                np.random.shuffle(original_indices)

            yield original_indices

    def __len__(self) -> int:
        """Number of batches."""
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size


class GroupDROSampler(Sampler):
    """
    Sampler for Group Distributionally Robust Optimization (GroupDRO).

    Samples proportionally to group weights, which are updated during training
    to upweight groups with high loss.
    """

    def __init__(
        self,
        dataset: DomainAwareDataset,
        indices: List[int],
        batch_size: int = 16,
        initial_weights: Optional[Dict[str, float]] = None,
        eta: float = 0.01,  # Step size for weight updates
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42
    ):
        """
        Args:
            dataset: DomainAwareDataset
            indices: Indices of samples to sample from
            batch_size: Batch size
            initial_weights: Initial domain weights
            eta: Step size for weight updates
            shuffle: Whether to shuffle
            drop_last: Whether to drop incomplete last batch
            seed: Random seed
        """
        self.dataset = dataset
        self.indices = indices
        self.batch_size = batch_size
        self.eta = eta
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

        # Group indices by domain
        self.domain_indices: Dict[str, List[int]] = defaultdict(list)
        for idx in indices:
            domain = dataset.get_domain_of_sample(idx)
            self.domain_indices[domain].append(idx)

        self.num_domains = len(self.domain_indices)

        # Initialize uniform weights
        if initial_weights is None:
            self.domain_weights = {d: 1.0 / self.num_domains for d in self.domain_indices}
        else:
            self.domain_weights = initial_weights

        self.num_samples = len(indices)
        self._current_iter = 0

    def update_weights(self, domain_losses: Dict[str, float]):
        """
        Update domain weights based on recent losses.

        Args:
            domain_losses: Dict mapping domain to average loss
        """
        # Normalize losses
        total_loss = sum(domain_losses.values())
        if total_loss == 0:
            return

        # Update weights using softmax over losses
        new_weights = {}
        for domain in self.domain_weights:
            loss = domain_losses.get(domain, 0.0)
            new_weights[domain] = self.domain_weights[domain] * np.exp(self.eta * loss)

        # Renormalize
        total = sum(new_weights.values())
        if total > 0:
            self.domain_weights = {k: v / total for k, v in new_weights.items()}

    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches of indices."""
        set_seed(self.seed + self._current_iter)
        self._current_iter += 1

        # Sample domains according to weights
        domains = list(self.domain_indices.keys())
        weights = [self.domain_weights.get(d, 1.0 / len(domains)) for d in domains]
        weights = np.array(weights) / sum(weights)

        batch = []
        while len(batch) < self.batch_size or (not self.drop_last and batch):
            # Sample a domain
            domain_idx = np.random.choice(len(domains), p=weights)
            domain = domains[domain_idx]

            # Sample from domain
            domain_sample = np.random.choice(self.domain_indices[domain])
            batch.append(domain_sample)

            if len(batch) == self.batch_size:
                yield batch
                batch = []

    def __len__(self) -> int:
        """Number of batches."""
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size


def create_domain_sampler(
    dataset: DomainAwareDataset,
    indices: List[int],
    method: str = "erm",  # "erm", "domain_balanced", "domain_weighted", "groupdro"
    batch_size: int = 16,
    shuffle: bool = True,
    drop_last: bool = False,
    seed: int = 42,
    **kwargs
) -> Sampler:
    """
    Factory function to create domain-aware samplers.

    Args:
        dataset: DomainAwareDataset
        indices: Indices of samples to sample from
        method: Sampling method
        batch_size: Batch size
        shuffle: Whether to shuffle
        drop_last: Whether to drop incomplete last batch
        seed: Random seed
        **kwargs: Additional arguments for specific samplers

    Returns:
        Appropriate Sampler instance

    Raises:
        ValueError: If method is unknown
    """
    if method == "erm":
        # Standard sequential or shuffled sampler
        if shuffle:
            return torch.utils.data.SubsetRandomSampler(indices)
        else:
            return torch.utils.data.SequentialSampler(indices)

    elif method == "domain_balanced":
        return DomainBalancedSampler(
            dataset=dataset,
            indices=indices,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed
        )

    elif method == "domain_weighted":
        return DomainWeightedSampler(
            dataset=dataset,
            indices=indices,
            weight_mode=kwargs.get("weight_mode", "inverse"),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed
        )

    elif method == "groupdro":
        return GroupDROSampler(
            dataset=dataset,
            indices=indices,
            batch_size=batch_size,
            initial_weights=kwargs.get("initial_weights"),
            eta=kwargs.get("eta", 0.01),
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed
        )

    else:
        raise ValueError(f"Unknown sampling method: {method}")


def create_domain_aware_dataloader(
    dataset: DomainAwareDataset,
    indices: List[int],
    batch_size: int = 16,
    method: str = "erm",
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
    seed: int = 42,
    collate_fn: Optional[Callable] = None,
    **sampler_kwargs
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader with domain-aware sampling.

    Args:
        dataset: DomainAwareDataset
        indices: Indices of samples
        batch_size: Batch size
        method: Sampling method
        shuffle: Whether to shuffle (only for ERM)
        num_workers: Number of workers
        drop_last: Whether to drop incomplete last batch
        seed: Random seed
        collate_fn: Optional collate function
        **sampler_kwargs: Additional sampler arguments

    Returns:
        DataLoader instance
    """
    sampler = create_domain_sampler(
        dataset=dataset,
        indices=indices,
        method=method,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
        **sampler_kwargs
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last
    )
