"""
P9 Domain Generalization Training

Training utilities for domain-aware and domain generalization training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import torch.optim as optim
from collections import defaultdict

from .base import (
    DomainAwareDataset,
    DomainSplit,
    DomainAwareConfig,
    set_seed,
)
from .splits import create_domain_split, validate_split
from .sampling import create_domain_sampler, GroupDROSampler
from .evaluation import compute_domain_metrics, aggregate_across_domains, DomainMetrics


@dataclass
class GroupDROLoss:
    """
    Group Distributionally Robust Optimization Loss.

    Updates group weights during training to minimize worst-case loss
    across groups (domains).
    """

    def __init__(
        self,
        base_loss_fn: nn.Module,
        num_groups: int,
        domain_indices: Dict[str, List[int]],
        eta: float = 0.01,
        device: str = "cpu"
    ):
        """
        Args:
            base_loss_fn: Base loss function (e.g., MultiTaskLoss from P8)
            num_groups: Number of domain groups
            domain_indices: Mapping from domain to sample indices
            eta: Step size for weight updates
            device: Device for computation
        """
        self.base_loss_fn = base_loss_fn
        self.num_groups = num_groups
        self.domain_indices = domain_indices
        self.eta = eta

        # Initialize uniform weights
        self.group_weights = {d: 1.0 / num_groups for d in domain_indices}
        self.group_losses = {d: 0.0 for d in domain_indices}

        self.device = device

    def update(self, losses: Dict[str, float]):
        """
        Update group weights based on losses.

        Args:
            losses: Dict mapping domain to average loss
        """
        self.group_losses = losses

        # Compute new weights using softmax over losses
        new_weights = {}
        for domain in self.group_weights:
            loss = losses.get(domain, 0.0)
            new_weights[domain] = self.group_weights[domain] * np.exp(self.eta * loss)

        # Renormalize
        total = sum(new_weights.values())
        if total > 0:
            self.group_weights = {k: v / total for k, v in new_weights.items()}

    def get_weights(self) -> Dict[str, float]:
        """Get current group weights."""
        return self.group_weights.copy()

    def forward(
        self,
        output: Any,
        targets: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]],
        batch_indices: List[int],
        batch_domains: List[str],
        per_sample_loss: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted loss.

        Args:
            output: Model output
            targets: Target tensors
            masks: Availability masks
            batch_indices: Original indices in batch
            batch_domains: Domains for each batch element
            per_sample_loss: Optional per-sample losses (B,) tensor. If provided,
                           each sample's individual loss is used for domain loss
                           aggregation. If None, uses batch loss for all samples.

        Returns:
            Tuple of (weighted_loss, loss_dict)
        """
        # Compute base loss
        base_loss, base_metrics = self.base_loss_fn(output, targets, masks)

        # Group losses by domain
        domain_losses = defaultdict(list)
        domain_counts = defaultdict(int)

        # Track which samples belong to which domain
        for i, domain in enumerate(batch_domains):
            domain_counts[domain] += 1
            # Use per_sample_loss if available, otherwise use batch loss
            if per_sample_loss is not None:
                sample_loss = float(per_sample_loss[i].detach())
            else:
                # Fallback: use batch loss (note: this is approximate for weight updates)
                sample_loss = float(base_loss.detach())
            domain_losses[domain].append(sample_loss)

        # Compute average loss per domain
        avg_domain_losses = {}
        for domain, losses in domain_losses.items():
            avg_domain_losses[domain] = np.mean(losses)

        # Update weights based on average domain losses
        self.update(avg_domain_losses)

        # Compute weighted loss as proper weighted average
        # weighted_loss = sum(w_d * avg_loss_d) where d ranges over unique domains in batch
        weighted_loss = torch.tensor(0.0, device=self.device)
        batch_size = len(batch_domains)

        # Count samples per domain in this batch
        for domain in set(batch_domains):
            count = batch_domains.count(domain)
            weight = self.group_weights.get(domain, 1.0 / self.num_groups)
            # Weighted contribution = weight * (count/batch_size) * avg_loss
            # But since avg_loss = sum(samples in domain)/count, this simplifies
            # to: weight * sum(samples in domain * loss) / batch_size
            if per_sample_loss is not None:
                # Get losses for samples in this domain
                domain_mask = [d == domain for d in batch_domains]
                domain_losses_list = [domain_losses[domain]]
                # We need per-sample losses for averaging
                pass  # Handled above

        # Simplified weighted average: weight * loss for each sample, normalized
        weighted_loss = torch.tensor(0.0, device=self.device)
        for i, domain in enumerate(batch_domains):
            weight = self.group_weights.get(domain, 1.0 / self.num_groups)
            if per_sample_loss is not None:
                sample_loss = per_sample_loss[i]
            else:
                sample_loss = base_loss
            # Add weighted contribution, normalized by batch size
            weighted_loss = weighted_loss + (weight * sample_loss) / len(batch_domains)

        metrics = {
            **base_metrics,
            "groupdro_weighted_loss": float(weighted_loss.detach()),
        }

        # Add domain weights to metrics
        for domain, weight in self.group_weights.items():
            metrics[f"weight_{domain}"] = weight

        # Add average domain losses to metrics
        for domain, loss in avg_domain_losses.items():
            metrics[f"loss_{domain}"] = loss

        return weighted_loss, metrics


class DomainGeneralizationTrainer:
    """
    Trainer for domain generalization.

    Handles training with domain-aware splits, multiple domains,
    and various generalization methods.
    """

    def __init__(
        self,
        model: nn.Module,
        config: DomainAwareConfig,
        domain_dataset: DomainAwareDataset,
        collate_fn: Optional[Callable] = None
    ):
        """
        Args:
            model: P8 MultiTaskPhysiologicalModel
            config: DomainAwareConfig
            domain_dataset: Dataset with domain metadata
            collate_fn: Collate function for batching
        """
        self.model = model
        self.config = config
        self.domain_dataset = domain_dataset
        self.collate_fn = collate_fn

        self.device = torch.device(config.device)

        # Create split
        self.split_result = create_domain_split(
            metadata=domain_dataset.metadata,
            domain_key=config.domain_key,
            split_type=config.split_type,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            seed=config.seed
        )
        self.split = self.split_result.split

        # Validate split
        is_valid, issues = validate_split(
            self.split,
            domain_dataset.metadata,
            check_leakage=True,
            check_subject_overlap=True
        )
        if not is_valid:
            print(f"Warning: Split validation issues: {issues}")

        # Initialize training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')

        # Domain tracking for GroupDRO
        self.groupdro_loss: Optional[GroupDROLoss] = None
        if config.method == "groupdro":
            self._init_groupdro()

    def _init_groupdro(self):
        """Initialize GroupDRO loss."""
        from p8_multitask.losses import MultiTaskLoss

        domain_indices = {}
        for idx in self.split.train_indices:
            domain = self.domain_dataset.get_domain_of_sample(idx)
            if domain not in domain_indices:
                domain_indices[domain] = []
            domain_indices[domain].append(idx)

        base_loss = MultiTaskLoss(self.model.config)
        self.groupdro_loss = GroupDROLoss(
            base_loss_fn=base_loss,
            num_groups=len(domain_indices),
            domain_indices=domain_indices,
            eta=self.config.groupdro_eta,
            device=str(self.device)
        )

    def create_dataloaders(self) -> Tuple:
        """Create train, val, and test dataloaders."""
        from torch.utils.data import DataLoader

        # Create samplers based on method
        if self.config.method in ["groupdro", "domain_balanced", "domain_weighted"]:
            train_sampler = create_domain_sampler(
                dataset=self.domain_dataset,
                indices=self.split.train_indices.tolist(),
                method=self.config.method,
                batch_size=self.config.batch_size,
                shuffle=True,
                drop_last=True,
                seed=self.config.seed,
                eta=self.config.groupdro_eta,
                weight_mode=self.config.domain_weighting
            )
            train_shuffle = False  # Sampler handles shuffling
        else:
            train_sampler = None
            train_shuffle = True

        train_loader = DataLoader(
            self.domain_dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            shuffle=train_shuffle,
            num_workers=0,
            collate_fn=self.collate_fn,
            drop_last=True
        )

        val_loader = DataLoader(
            self.domain_dataset,
            batch_size=self.config.batch_size,
            sampler=self.split.val_indices.tolist(),
            shuffle=False,
            num_workers=0,
            collate_fn=self.collate_fn,
            drop_last=False
        )

        test_loader = DataLoader(
            self.domain_dataset,
            batch_size=self.config.batch_size,
            sampler=self.split.test_indices.tolist(),
            shuffle=False,
            num_workers=0,
            collate_fn=self.collate_fn,
            drop_last=False
        )

        return train_loader, val_loader, test_loader

    def train_epoch(
        self,
        train_loader,
        optimizer,
        epoch: int
    ) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        all_metrics = defaultdict(float)

        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            features = batch["features"].to(self.device)
            mask = batch.get("mask")
            if mask is not None:
                mask = mask.to(self.device)

            # Prepare targets
            targets = {}
            masks = {}

            for key in ["hr", "bvp", "sqi", "confidence"]:
                if key in batch:
                    targets[key] = batch[key].to(self.device)
                avail_key = f"{key}_available"
                if avail_key in batch:
                    masks[avail_key] = batch[avail_key].to(self.device)

            # Forward pass
            optimizer.zero_grad()
            output = self.model(features, mask=mask)

            # Compute loss
            if self.groupdro_loss is not None:
                # Get batch domains
                batch_indices = batch.get("indices", torch.arange(len(features))).tolist()
                batch_domains = [self.domain_dataset.get_domain_of_sample(idx) for idx in batch_indices]

                loss, metrics = self.groupdro_loss(
                    output, targets, masks, batch_indices, batch_domains
                )
            else:
                from p8_multitask.losses import MultiTaskLoss
                loss_fn = MultiTaskLoss(self.model.config)
                loss, metrics = loss_fn(output, targets, masks)

            # Backward
            loss.backward()
            optimizer.step()

            # Accumulate
            total_loss += float(loss.detach())
            for k, v in metrics.items():
                all_metrics[k] += v
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_metrics = {k: v / max(num_batches, 1) for k, v in all_metrics.items()}
        avg_metrics["loss"] = avg_loss

        return avg_loss, avg_metrics

    def validate(self, val_loader) -> Tuple[float, Dict[str, float]]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_metrics = defaultdict(float)

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(self.device)
                mask = batch.get("mask")
                if mask is not None:
                    mask = mask.to(self.device)

                targets = {}
                masks = {}

                for key in ["hr", "bvp", "sqi", "confidence"]:
                    if key in batch:
                        targets[key] = batch[key].to(self.device)
                    avail_key = f"{key}_available"
                    if avail_key in batch:
                        masks[avail_key] = batch[avail_key].to(self.device)

                output = self.model(features, mask=mask)

                from p8_multitask.losses import MultiTaskLoss
                loss_fn = MultiTaskLoss(self.model.config)
                loss, metrics = loss_fn(output, targets, masks)

                total_loss += float(loss)
                for k, v in metrics.items():
                    all_metrics[k] += v
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_metrics = {k: v / max(num_batches, 1) for k, v in all_metrics.items()}
        avg_metrics["loss"] = avg_loss

        return avg_loss, avg_metrics

    def train(
        self,
        epochs: Optional[int] = None,
        save_best: bool = True,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, List]:
        """
        Train the model.

        Args:
            epochs: Number of epochs (default: from config)
            save_best: Whether to save best model
            checkpoint_path: Path for checkpoint saving

        Returns:
            Training history
        """
        if epochs is None:
            epochs = self.config.epochs

        train_loader, val_loader, test_loader = self.create_dataloaders()

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
            "group_weights": [] if self.groupdro_loss else None
        }

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Train
            train_loss, train_metrics = self.train_epoch(train_loader, optimizer, epoch)

            # Validate
            val_loss, val_metrics = self.validate(val_loader)

            # Record
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_metrics"].append(train_metrics)
            history["val_metrics"].append(val_metrics)

            if self.groupdro_loss:
                history["group_weights"].append(self.groupdro_loss.get_weights())

            # Log
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            # Save best
            if save_best and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if checkpoint_path:
                    self.save_checkpoint(checkpoint_path, optimizer, epoch, val_loss)

        return history

    def evaluate_on_test(self, test_loader) -> Dict[str, Any]:
        """Evaluate on held-out test set."""
        self.model.eval()

        # Collect predictions by domain
        domain_predictions: Dict[str, Dict] = defaultdict(lambda: defaultdict(list))
        domain_targets: Dict[str, Dict] = defaultdict(lambda: defaultdict(list))

        with torch.no_grad():
            for batch in test_loader:
                features = batch["features"].to(self.device)
                mask = batch.get("mask")
                if mask is not None:
                    mask = mask.to(self.device)

                output = self.model(features, mask=mask)

                # Get domains for this batch
                indices = batch.get("indices", torch.arange(len(features))).tolist()
                for i, idx in enumerate(indices):
                    domain = self.domain_dataset.get_domain_of_sample(idx)

                    for key in ["hr", "bvp", "sqi", "confidence"]:
                        if getattr(output, key, None) is not None:
                            domain_predictions[domain][key].append(
                                output.__dict__[key][i].cpu()
                            )
                        if key in batch:
                            domain_targets[domain][key].append(batch[key][i])

        # Compute metrics per domain
        results = {
            "per_domain": {},
            "aggregate": {}
        }

        for domain in domain_predictions:
            domain_preds = {}
            domain_tgts = {}

            for key in ["hr", "bvp", "sqi", "confidence"]:
                if domain_predictions[domain][key]:
                    domain_preds[key] = torch.stack(domain_predictions[domain][key])
                    domain_tgts[key] = torch.stack(domain_targets[domain][key])

            if domain_preds:
                metrics = compute_domain_metrics(domain_preds, domain_tgts)
                results["per_domain"][domain] = metrics

        # Aggregate across domains
        results["aggregate"] = aggregate_across_domains(results["per_domain"])

        return results

    def save_checkpoint(
        self,
        path: str,
        optimizer,
        epoch: int,
        loss: float
    ):
        """Save checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
            "config": self.config,
            "split_summary": self.split.get_summary(),
            "group_weights": self.groupdro_loss.get_weights() if self.groupdro_loss else None
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=str(self.device))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint


def train_domain_generalization(
    model: nn.Module,
    domain_dataset: DomainAwareDataset,
    config: DomainAwareConfig,
    base_loss_fn: Optional[nn.Module] = None,
    collate_fn: Optional[Callable] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train a model for domain generalization.

    Args:
        model: P8 MultiTaskPhysiologicalModel
        domain_dataset: Dataset with domain metadata
        config: DomainAwareConfig
        base_loss_fn: Optional base loss function
        collate_fn: Optional collate function

    Returns:
        Tuple of (trained_model, training_history)
    """
    trainer = DomainGeneralizationTrainer(
        model=model,
        config=config,
        domain_dataset=domain_dataset,
        collate_fn=collate_fn
    )

    history = trainer.train()

    return trainer.model, history


def evaluate_per_domain(
    model: nn.Module,
    domain_dataset: DomainAwareDataset,
    split: DomainSplit,
    collate_fn: Optional[Callable] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Evaluate model per domain.

    Args:
        model: Trained model
        domain_dataset: Dataset with domain metadata
        split: Domain split with test indices
        collate_fn: Optional collate function
        device: Device for computation

    Returns:
        Dictionary with per-domain metrics
    """
    from torch.utils.data import DataLoader

    model.eval()
    device = torch.device(device)

    # Create dataloader for test set
    test_loader = DataLoader(
        domain_dataset,
        batch_size=16,
        sampler=split.test_indices.tolist(),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # Collect predictions by domain
    domain_results: Dict[str, Dict] = defaultdict(lambda: {
        "predictions": defaultdict(list),
        "targets": defaultdict(list)
    })

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            mask = batch.get("mask")
            if mask is not None:
                mask = mask.to(device)

            output = model(features, mask=mask)

            # Get domains for batch
            indices = batch.get("indices", torch.arange(len(features))).tolist()
            for i, idx in enumerate(indices):
                domain = domain_dataset.get_domain_of_sample(idx)

                for key in ["hr", "bvp", "sqi", "confidence"]:
                    if getattr(output, key, None) is not None:
                        domain_results[domain]["predictions"][key].append(
                            output.__dict__[key][i].cpu()
                        )
                    if key in batch:
                        domain_results[domain]["targets"][key].append(batch[key][i])

    # Compute metrics per domain
    metrics = {}
    for domain, results in domain_results.items():
        if results["predictions"]:
            preds = {k: torch.stack(v) for k, v in results["predictions"].items()}
            tgts = {k: torch.stack(v) for k, v in results["targets"].items()}
            metrics[domain] = compute_domain_metrics(preds, tgts)

    return metrics
