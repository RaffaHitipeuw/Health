"""
P6 Temporal Training Infrastructure

Training utilities for temporal physiological models including:
    - Temporal-aware losses
    - BVP waveform losses
    - Frequency domain losses
    - Training and validation loops
    - Checkpoint management
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Any
import numpy as np


class TemporalLoss(nn.Module):
    """Combined loss for temporal physiological models.

    Combines BPM regression with optional BVP waveform and quality losses.
    """

    def __init__(
        self,
        bpm_weight: float = 1.0,
        bvp_weight: float = 0.5,
        quality_weight: float = 0.1,
        confidence_weight: float = 0.1
    ):
        super().__init__()
        self.bpm_weight = bpm_weight
        self.bvp_weight = bvp_weight
        self.quality_weight = quality_weight
        self.confidence_weight = confidence_weight

        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            pred: Dictionary of predictions
            target: Dictionary of targets
            mask: Optional sequence mask (B, T)

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        losses = {}
        total = 0.0

        # BPM loss (primary)
        if "bpm" in pred and "bpm" in target:
            bpm_loss = self.mse(pred["bpm"], target["bpm"])
            losses["bpm"] = float(bpm_loss.detach())
            total += self.bpm_weight * bpm_loss

        # BVP waveform loss
        if "bvp" in pred and "bvp" in target:
            pred_bvp = pred["bvp"]
            target_bvp = target["bvp"]

            if mask is not None:
                # Apply mask to BVP loss
                valid_mask = ~mask
                valid_count = valid_mask.sum()
                if valid_count > 0:
                    bvp_diff = (pred_bvp - target_bvp) ** 2
                    bvp_loss = (bvp_diff * valid_mask.float()).sum() / valid_count
                else:
                    bvp_loss = torch.tensor(0.0, device=pred_bvp.device)
            else:
                bvp_loss = self.mse(pred_bvp, target_bvp)

            losses["bvp"] = float(bvp_loss.detach() if hasattr(bvp_loss, 'detach') else bvp_loss)
            total += self.bvp_weight * bvp_loss

        # Quality loss
        if "quality" in pred and "quality" in target:
            quality_loss = self.mse(pred["quality"], target["quality"])
            losses["quality"] = float(quality_loss.detach())
            total += self.quality_weight * quality_loss

        # Confidence loss
        if "confidence" in pred and "confidence" in target:
            confidence_loss = self.bce(pred["confidence"], target["confidence"])
            losses["confidence"] = float(confidence_loss.detach())
            total += self.confidence_weight * confidence_loss

        return total, losses


class BVPLoss(nn.Module):
    """Loss focused on BVP waveform estimation.

    Uses multiple loss components:
        - MSE for overall waveform shape
        - Correlation loss for temporal patterns
        - Spectral loss for frequency domain
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        corr_weight: float = 0.5,
        spectral_weight: float = 0.3
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.corr_weight = corr_weight
        self.spectral_weight = spectral_weight

    def pearson_correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Pearson correlation coefficient."""
        x_centered = x - x.mean(dim=-1, keepdim=True)
        y_centered = y - y.mean(dim=-1, keepdim=True)
        corr = (x_centered * y_centered).sum(dim=-1) / (
            x_centered.norm(dim=-1) * y_centered.norm(dim=-1) + 1e-9
        )
        return 1 - corr.mean()  # Convert to loss

    def spectral_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """L1 loss in frequency domain."""
        # FFT along temporal dimension
        pred_fft = torch.abs(torch.fft.rfft(pred, dim=-1))
        target_fft = torch.abs(torch.fft.rfft(target, dim=-1))

        # Normalize
        pred_fft = pred_fft / (pred_fft.max(dim=-1, keepdim=True)[0] + 1e-9)
        target_fft = target_fft / (target_fft.max(dim=-1, keepdim=True)[0] + 1e-9)

        return F.l1_loss(pred_fft, target_fft)

    def forward(self, pred_bvp: torch.Tensor, target_bvp: torch.Tensor) -> torch.Tensor:
        """Compute BVP loss."""
        mse = F.mse_loss(pred_bvp, target_bvp)
        corr = self.pearson_correlation(pred_bvp, target_bvp)
        spectral = self.spectral_loss(pred_bvp, target_bvp)

        return self.mse_weight * mse + self.corr_weight * corr + self.spectral_weight * spectral


class FrequencyLoss(nn.Module):
    """Frequency domain loss for physiological signals.

    Optimizes for correct frequency components in cardiac band.
    """

    def __init__(
        self,
        cardiac_band: Tuple[float, float] = (0.833, 3.0),
        weight: float = 1.0
    ):
        super().__init__()
        self.cardiac_band = cardiac_band
        self.weight = weight

    def get_cardiac_mask(self, freqs: torch.Tensor) -> torch.Tensor:
        """Create mask for cardiac frequency band."""
        min_hz, max_hz = self.cardiac_band
        return (freqs >= min_hz) & (freqs <= max_hz)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        fps: float = 30.0
    ) -> torch.Tensor:
        """Compute frequency domain loss.

        Args:
            pred: Predicted signal (B, T)
            target: Target signal (B, T)
            fps: Frames per second

        Returns:
            Frequency loss
        """
        B, T = pred.shape

        # Compute FFT
        pred_fft = torch.abs(torch.fft.rfft(pred, dim=-1))
        target_fft = torch.abs(torch.fft.rfft(target, dim=-1))

        # Frequency bins
        freqs = torch.fft.rfftfreq(T, d=1.0 / fps, device=pred.device)

        # Get cardiac band mask
        cardiac_mask = self.get_cardiac_mask(freqs)

        # Cardiac band loss
        cardiac_loss = F.l1_loss(pred_fft[:, cardiac_mask], target_fft[:, cardiac_mask])

        # Non-cardiac (should be minimized)
        non_cardiac_mask = ~cardiac_mask
        non_cardiac_loss = F.l1_loss(pred_fft[:, non_cardiac_mask], target_fft[:, non_cardiac_mask])

        return self.weight * (cardiac_loss + 0.1 * non_cardiac_loss)


def train_temporal_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    loss_fn: Optional[TemporalLoss] = None,
    use_mask: bool = True
) -> Tuple[float, Dict[str, float]]:
    """
    Train for one epoch.

    Args:
        model: Model to train
        loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        loss_fn: Loss function (uses default if None)
        use_mask: Whether to use sequence masking

    Returns:
        Tuple of (avg_loss, metrics_dict)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    all_losses = {"bpm": 0.0, "bvp": 0.0, "quality": 0.0}

    if loss_fn is None:
        loss_fn = TemporalLoss()

    for batch in loader:
        # Move to device
        features = batch["features"].to(device)
        bpm_target = batch["bpm"].to(device)
        mask = batch["mask"].to(device) if use_mask and "mask" in batch else None

        # Prepare targets
        target_dict = {"bpm": bpm_target}
        if "bvp" in batch:
            target_dict["bvp"] = batch["bvp"].to(device)
        if "quality" in batch:
            target_dict["quality"] = batch["quality"].to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(features, mask=mask)

        # Prepare predictions
        pred_dict = {"bpm": output.bpm}
        if output.bvp is not None:
            pred_dict["bvp"] = output.bvp
        if output.signal_quality is not None:
            pred_dict["quality"] = output.signal_quality
        if output.bpm_confidence is not None:
            pred_dict["confidence"] = output.bpm_confidence

        # Compute loss
        loss, losses = loss_fn(pred_dict, target_dict, mask)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach())
        for k, v in losses.items():
            if k in all_losses:
                all_losses[k] += v
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    metrics = {k: v / num_batches for k, v in all_losses.items()}
    metrics["loss"] = avg_loss

    return avg_loss, metrics


def validate_temporal(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_mask: bool = True
) -> Dict[str, float]:
    """
    Validate model.

    Args:
        model: Model to validate
        loader: Validation data loader
        device: Device
        use_mask: Whether to use sequence masking

    Returns:
        Dictionary of metrics
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_bvp_preds = []
    all_bvp_targets = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            bpm_target = batch["bpm"].to(device)
            mask = batch["mask"].to(device) if use_mask and "mask" in batch else None

            output = model(features, mask=mask)

            all_preds.append(output.bpm.cpu())
            all_targets.append(bpm_target.cpu())

            if output.bvp is not None and "bvp" in batch:
                all_bvp_preds.append(output.bvp.cpu())
                all_bvp_targets.append(batch["bvp"].to(device).cpu())

    # Concatenate
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    # Compute metrics
    mae = float(torch.mean(torch.abs(preds - targets)))
    rmse = float(torch.sqrt(torch.mean((preds - targets) ** 2)))
    bias = float(torch.mean(preds - targets))

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "bias": bias
    }

    # BVP metrics if available
    if len(all_bvp_preds) > 0:
        bvp_preds = torch.cat(all_bvp_preds)
        bvp_targets = torch.cat(all_bvp_targets)
        bvp_mae = float(torch.mean(torch.abs(bvp_preds - bvp_targets)))
        metrics["bvp_mae"] = bvp_mae

    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    epoch: int = 0,
    path: str = "checkpoint.pt"
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    path: str = "checkpoint.pt",
    device: str = "cpu"
) -> Tuple[int, Optional[optim.Optimizer]]:
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint.get("epoch", 0)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return epoch, optimizer

    return epoch, optimizer


def get_optimizer(model: nn.Module, lr: float = 1e-4, wd: float = 1e-5, kind: str = "adam") -> optim.Optimizer:
    """Get optimizer for model."""
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
    elif kind == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    else:
        return optim.lr_scheduler.StepLR(optimizer, step_size=1)
