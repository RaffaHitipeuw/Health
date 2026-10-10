"""
P7 Self-Supervised Training Infrastructure

Training utilities for self-supervised physiological representation learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass

from .base import SelfSupervisedConfig, SSLOutput
from .encoder import SelfSupervisedEncoder
from .objectives import CombinedSSLLoss
from .augmentations import create_multiview_augmenter, PhysiologicalAugmenter


def train_ssl_epoch(
    model: SelfSupervisedEncoder,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: CombinedSSLLoss,
    device: torch.device,
    augmenter: Optional[PhysiologicalAugmenter] = None,
    epoch: int = 0,
    log_interval: int = 10
) -> Tuple[float, Dict[str, float]]:
    """
    Train for one epoch using self-supervised learning.

    Args:
        model: Self-supervised encoder
        loader: Training data loader
        optimizer: Optimizer
        loss_fn: Combined SSL loss function
        device: Device to train on
        augmenter: Optional augmenter for creating views
        epoch: Current epoch number
        log_interval: Logging interval

    Returns:
        Tuple of (avg_loss, metrics_dict)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    all_metrics = {}

    for batch_idx, batch in enumerate(loader):
        # Move to device
        features = batch["features"].to(device)
        indices = batch["indices"].to(device)
        subject_ids = batch["subject_ids"].to(device)

        # Get views from batch or create augmented views
        if "views" in batch:
            views = batch["views"].to(device)  # (B, num_views, T, F)
            B, V, T, F = views.shape
            # Flatten views: (B, V, T, F) -> (B*V, T, F)
            views_flat = views.reshape(B * V, T, F)
        else:
            # Create augmented views on the fly
            if augmenter is None:
                augmenter = create_multiview_augmenter(num_views=2)
            views_list = []
            for i in range(features.shape[0]):
                views = augmenter(features[i].cpu().numpy())
                views_list.extend([torch.from_numpy(v) for v in views])
            views_flat = torch.stack(views_list).to(device)

        # Create mask if needed
        mask = None

        # Forward pass
        optimizer.zero_grad()

        # Encode each view
        outputs = model(views_flat, mask=mask)

        # For contrastive loss, we need to handle multiple views per sample
        if "views" in batch:
            # Reshape projections: (B*V, D) -> (B, V, D)
            projections = outputs.projections.reshape(B, V, -1)
            # Flatten for contrastive: (B*V, D)
            projections = projections.reshape(B * V, -1)

            # Positive pairs are within same original sample
            # Loss will handle this
        else:
            projections = outputs.projections

        # Compute loss
        loss, losses = loss_fn(
            representations=outputs.representations,
            projections=projections,
            indices=indices
        )

        # Backward
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss += float(loss.detach())
        for k, v in losses.items():
            if k not in all_metrics:
                all_metrics[k] = 0.0
            all_metrics[k] += v
        num_batches += 1

        # Logging
        if (batch_idx + 1) % log_interval == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | "
                  f"Loss: {float(loss.detach()):.4f}")

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    metrics = {k: v / num_batches for k, v in all_metrics.items()}
    metrics["loss"] = avg_loss

    return avg_loss, metrics


def validate_ssl(
    model: SelfSupervisedEncoder,
    loader: DataLoader,
    loss_fn: CombinedSSLLoss,
    device: torch.device
) -> Dict[str, float]:
    """
    Validate self-supervised model.

    Args:
        model: Model to validate
        loader: Validation data loader
        loss_fn: Loss function
        device: Device

    Returns:
        Dictionary of metrics
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_metrics = {}

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            views = batch["views"].to(device)

            B, V, T, F = views.shape
            views_flat = views.reshape(B * V, T, F)

            outputs = model(views_flat)

            projections = outputs.projections.reshape(B, V, -1).reshape(B * V, -1)

            loss, losses = loss_fn(
                representations=outputs.representations,
                projections=projections,
                indices=batch["indices"].to(device)
            )

            total_loss += float(loss.detach())
            for k, v in losses.items():
                if k not in all_metrics:
                    all_metrics[k] = 0.0
                all_metrics[k] += v
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    metrics = {k: v / num_batches for k, v in all_metrics.items()}
    metrics["loss"] = avg_loss

    return metrics


def save_checkpoint(
    model: SelfSupervisedEncoder,
    optimizer: optim.Optimizer,
    epoch: int,
    path: str,
    loss: float = 0.0
) -> None:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        path: Save path
        loss: Current loss value
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    model: SelfSupervisedEncoder,
    optimizer: Optional[optim.Optimizer],
    path: str,
    device: str = "cpu"
) -> Tuple[int, float]:
    """
    Load model checkpoint.

    Args:
        model: Model to load into
        optimizer: Optional optimizer to load state into
        path: Path to checkpoint
        device: Device to load to

    Returns:
        Tuple of (epoch, loss)
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", 0.0)

    return epoch, loss


def get_optimizer(model: nn.Module, lr: float = 1e-4, wd: float = 1e-5, kind: str = "adam") -> optim.Optimizer:
    """Get optimizer."""
    if kind.lower() == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif kind.lower() == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif kind.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


def get_scheduler(optimizer: optim.Optimizer, steps_per_epoch: int = 100, kind: str = "cosine") -> optim.lr_scheduler._LRScheduler:
    """Get learning rate scheduler."""
    if kind == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps_per_epoch)
    elif kind == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif kind == "warmup_cosine":
        # Warmup + cosine decay
        warmup_epochs = 5
        def lr_lambda(step):
            if step < warmup_epochs * steps_per_epoch:
                return float(step) / (warmup_epochs * steps_per_epoch)
            else:
                progress = float(step - warmup_epochs * steps_per_epoch) / ((100 - warmup_epochs) * steps_per_epoch)
                return 0.5 * (1.0 + np.cos(np.pi * progress))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        return optim.lr_scheduler.StepLR(optimizer, step_size=1)


def extract_representations(
    model: SelfSupervisedEncoder,
    loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract representations from trained encoder.

    Args:
        model: Trained encoder
        loader: Data loader
        device: Device

    Returns:
        Tuple of (representations, indices)
    """
    model.eval()
    all_representations = []
    all_indices = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)

            # Extract representation
            rep = model.extract_representation(features)
            all_representations.append(rep.cpu().numpy())
            all_indices.append(batch["indices"].numpy())

    representations = np.concatenate(all_representations, axis=0)
    indices = np.concatenate(all_indices, axis=0)

    return representations, indices


@dataclass
class SSLTrainingConfig:
    """Configuration for SSL training."""
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-5
    log_interval: int = 10
    save_interval: int = 10
    device: str = "cpu"
    seed: int = 42
