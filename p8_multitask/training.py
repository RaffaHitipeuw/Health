"""
P8 Multi-Task Training Infrastructure

Training utilities for multi-task physiological modeling:
    - Training and validation loops
    - Per-task loss logging
    - Checkpoint management
    - Optimizer and scheduler configuration
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List, Any, Union
import numpy as np
from dataclasses import dataclass

from .base import MultiTaskConfig, MultiTaskOutput, TaskType
from .model import MultiTaskPhysiologicalModel
from .losses import MultiTaskLoss, compute_per_task_loss


def train_multitask_epoch(
    model: MultiTaskPhysiologicalModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: MultiTaskLoss,
    device: torch.device,
    epoch: int = 0,
    log_interval: int = 10,
    use_mask: bool = True,
    gradient_clip: Optional[float] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Train for one epoch.

    Args:
        model: Multi-task model
        loader: Training data loader
        optimizer: Optimizer
        loss_fn: Multi-task loss function
        device: Device to train on
        epoch: Current epoch number
        log_interval: Logging interval
        use_mask: Whether to use sequence masking
        gradient_clip: Optional gradient clipping value

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
        mask = batch["mask"].to(device) if use_mask and "mask" in batch else None

        # Prepare targets and masks
        targets = {}
        masks = {}

        if "hr" in batch:
            targets["hr"] = batch["hr"].to(device)
            masks["hr_available"] = batch["hr_available"].to(device)

        if "bvp" in batch:
            targets["bvp"] = batch["bvp"].to(device)
            masks["bvp_available"] = batch["bvp_available"].to(device)

        if "sqi" in batch:
            targets["sqi"] = batch["sqi"].to(device)
            masks["sqi_available"] = batch["sqi_available"].to(device)

        if "confidence" in batch:
            targets["confidence"] = batch["confidence"].to(device)
            masks["confidence_available"] = batch["confidence_available"].to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(features, mask=mask)

        # Compute loss
        loss, losses = loss_fn(output, targets, masks)

        # Check for non-finite loss
        if not torch.isfinite(loss):
            print(f"Warning: Non-finite loss at batch {batch_idx}: {loss.item()}")
            continue

        # Backward
        loss.backward()

        # Gradient clipping
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

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

    # Compute averages
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    metrics = {k: v / num_batches for k, v in all_metrics.items()}
    metrics["loss"] = avg_loss

    return avg_loss, metrics


def validate_multitask(
    model: MultiTaskPhysiologicalModel,
    loader: DataLoader,
    loss_fn: Optional[MultiTaskLoss],
    device: torch.device,
    use_mask: bool = True
) -> Dict[str, float]:
    """
    Validate multi-task model.

    Args:
        model: Model to validate
        loader: Validation data loader
        loss_fn: Optional loss function
        device: Device
        use_mask: Whether to use sequence masking

    Returns:
        Dictionary of metrics
    """
    model.eval()
    all_preds = {}
    all_targets = {}
    all_losses = []
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device) if use_mask and "mask" in batch else None

            # Forward pass
            output = model(features, mask=mask)

            # Collect predictions
            if output.hr is not None:
                if "hr" not in all_preds:
                    all_preds["hr"] = []
                    all_targets["hr"] = []
                all_preds["hr"].append(output.hr.cpu())
                if "hr" in batch:
                    all_targets["hr"].append(batch["hr"])

            if output.bvp is not None:
                if "bvp" not in all_preds:
                    all_preds["bvp"] = []
                    all_targets["bvp"] = []
                all_preds["bvp"].append(output.bvp.cpu())

            if "bvp" in batch:
                all_targets["bvp"].append(batch["bvp"])

            if output.sqi is not None:
                if "sqi" not in all_preds:
                    all_preds["sqi"] = []
                    all_targets["sqi"] = []
                all_preds["sqi"].append(output.sqi.cpu())
                if "sqi" in batch:
                    all_targets["sqi"].append(batch["sqi"])

            # Compute loss if available
            if loss_fn is not None:
                targets = {}
                masks = {}

                if "hr" in batch:
                    targets["hr"] = batch["hr"].to(device)
                    masks["hr_available"] = batch["hr_available"].to(device)
                if "bvp" in batch:
                    targets["bvp"] = batch["bvp"].to(device)
                    masks["bvp_available"] = batch["bvp_available"].to(device)
                if "sqi" in batch:
                    targets["sqi"] = batch["sqi"].to(device)
                    masks["sqi_available"] = batch["sqi_available"].to(device)
                if "confidence" in batch:
                    targets["confidence"] = batch["confidence"].to(device)
                    masks["confidence_available"] = batch["confidence_available"].to(device)

                loss, _ = loss_fn(output, targets, masks)
                all_losses.append(float(loss))
                num_batches += 1

    # Compute metrics
    metrics = {}

    # Loss
    if all_losses:
        metrics["val_loss"] = sum(all_losses) / len(all_losses)

    # HR metrics
    if "hr" in all_preds and "hr" in all_targets:
        preds = torch.cat(all_preds["hr"])
        targets = torch.cat(all_targets["hr"])
        metrics["hr_mae"] = float(torch.mean(torch.abs(preds - targets)))
        metrics["hr_rmse"] = float(torch.sqrt(torch.mean((preds - targets) ** 2)))
        metrics["hr_bias"] = float(torch.mean(preds - targets))

    # BVP metrics
    if "bvp" in all_preds and all_targets.get("bvp"):
        preds = torch.cat(all_preds["bvp"])
        targets = torch.cat(all_targets["bvp"])
        metrics["bvp_mae"] = float(torch.mean(torch.abs(preds - targets)))

    # SQI metrics
    if "sqi" in all_preds and "sqi" in all_targets:
        preds = torch.cat(all_preds["sqi"])
        targets = torch.cat(all_targets["sqi"])
        metrics["sqi_mae"] = float(torch.mean(torch.abs(preds - targets)))

    return metrics


def train_multitask_with_finetuning(
    model: MultiTaskPhysiologicalModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: MultiTaskLoss,
    device: torch.device,
    epochs: int = 10,
    finetune_epochs: int = 5,
    finetune_task: Optional[TaskType] = None,
    log_interval: int = 10,
    save_best: bool = True,
    checkpoint_dir: str = "checkpoints"
) -> Dict[str, List[float]]:
    """
    Train with optional encoder finetuning phase.

    Args:
        model: Multi-task model
        train_loader: Training loader
        val_loader: Validation loader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device
        epochs: Total training epochs
        finetune_epochs: Epochs for encoder finetuning (0 = no finetuning)
        finetune_task: Task to finetune encoder for
        log_interval: Logging interval
        save_best: Whether to save best model
        checkpoint_dir: Directory for checkpoints

    Returns:
        Dictionary of training histories
    """
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_metrics": []
    }

    best_val_loss = float('inf')

    # Phase 1: Freeze encoder, train heads
    print("Phase 1: Training task heads...")
    model.freeze_encoder()
    for epoch in range(epochs - finetune_epochs):
        train_loss, train_metrics = train_multitask_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch, log_interval
        )
        val_metrics = validate_multitask(model, val_loader, loss_fn, device)
        val_loss = val_metrics.get("val_loss", 0.0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_metrics"].append(val_metrics)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if save_best and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, f"{checkpoint_dir}/best_model.pt")

    # Phase 2: Unfreeze encoder, finetune
    if finetune_epochs > 0:
        print("Phase 2: Finetuning encoder...")
        model.unfreeze_encoder()

        # Use lower learning rate for encoder
        encoder_params = model.encoder.parameters()
        head_params = [p for n, p in model.named_parameters() if "encoder" not in n]

        optimizer = optim.Adam([
            {"params": head_params, "lr": optimizer.param_groups[0]["lr"]},
            {"params": encoder_params, "lr": optimizer.param_groups[0]["lr"] * 0.1}
        ])

        for epoch in range(epochs - finetune_epochs, epochs):
            train_loss, train_metrics = train_multitask_epoch(
                model, train_loader, optimizer, loss_fn, device, epoch, log_interval
            )
            val_metrics = validate_multitask(model, val_loader, loss_fn, device)
            val_loss = val_metrics.get("val_loss", 0.0)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_metrics"].append(val_metrics)

            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch, f"{checkpoint_dir}/best_model.pt")

    return history


def save_checkpoint(
    model: MultiTaskPhysiologicalModel,
    optimizer: Optional[optim.Optimizer] = None,
    epoch: int = 0,
    path: str = "checkpoint.pt",
    loss: float = 0.0,
    metrics: Optional[Dict] = None
) -> None:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optional optimizer state
        epoch: Current epoch
        path: Save path
        loss: Current loss value
        metrics: Optional metrics dict
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model.config,
        "epoch": epoch,
        "loss": loss
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if metrics is not None:
        checkpoint["metrics"] = metrics

    # Save to a buffer first, then write to file
    import io
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)

    with open(path, 'wb') as f:
        f.write(buffer.getvalue())

    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    model: MultiTaskPhysiologicalModel,
    optimizer: Optional[optim.Optimizer] = None,
    path: str = "checkpoint.pt",
    device: str = "cpu",
    strict: bool = False
) -> Tuple[int, float, Optional[Dict]]:
    """
    Load model checkpoint.

    Args:
        model: Model to load into
        optimizer: Optional optimizer to load state into
        path: Path to checkpoint
        device: Device to load to
        strict: Whether to strictly load state dict

    Returns:
        Tuple of (epoch, loss, metrics)
    """
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", 0.0)
    metrics = checkpoint.get("metrics")

    print(f"Loaded checkpoint from {path} (epoch={epoch}, loss={loss:.4f})")

    return epoch, loss, metrics


def get_optimizer(
    model: MultiTaskPhysiologicalModel,
    lr: float = 1e-4,
    wd: float = 1e-5,
    kind: str = "adam",
    encoder_lr_scale: float = 1.0
) -> optim.Optimizer:
    """
    Get optimizer for multi-task model.

    Args:
        model: Model to optimize
        lr: Learning rate for heads
        wd: Weight decay
        kind: Optimizer type ("adam", "adamw", "sgd")
        encoder_lr_scale: Scale for encoder learning rate

    Returns:
        Optimizer instance
    """
    encoder_params = list(model.encoder.parameters())
    head_params = list(model.heads.parameters())

    if encoder_lr_scale != 1.0:
        param_groups = [
            {"params": head_params, "lr": lr, "weight_decay": wd},
            {"params": encoder_params, "lr": lr * encoder_lr_scale, "weight_decay": wd}
        ]
    else:
        param_groups = list(model.parameters())

    if kind.lower() == "adam":
        return optim.Adam(param_groups, lr=lr, weight_decay=wd)
    elif kind.lower() == "adamw":
        return optim.AdamW(param_groups, lr=lr, weight_decay=wd)
    elif kind.lower() == "sgd":
        return optim.SGD(param_groups, lr=lr, momentum=0.9, weight_decay=wd)
    else:
        return optim.Adam(param_groups, lr=lr, weight_decay=wd)


def get_scheduler(
    optimizer: optim.Optimizer,
    steps_per_epoch: int = 100,
    kind: str = "cosine",
    warmup_epochs: int = 5,
    min_lr: float = 1e-6
) -> optim.lr_scheduler._LRScheduler:
    """
    Get learning rate scheduler.

    Args:
        optimizer: Optimizer
        steps_per_epoch: Steps per epoch
        kind: Scheduler type
        warmup_epochs: Warmup epochs
        min_lr: Minimum learning rate

    Returns:
        Scheduler instance
    """
    if kind == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps_per_epoch, eta_min=min_lr
        )
    elif kind == "step":
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=10 * steps_per_epoch, gamma=0.1
        )
    elif kind == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    elif kind == "warmup_cosine":
        total_steps = 100 * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / warmup_steps
            else:
                progress = float(step - warmup_steps) / (total_steps - warmup_steps)
                return max(0.5 * (1.0 + np.cos(np.pi * progress)), min_lr / (optimizer.param_groups[0]["lr"]))

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        return optim.lr_scheduler.StepLR(optimizer, step_size=1)


@dataclass
class MultiTaskTrainingConfig:
    """Configuration for multi-task training."""
    epochs: int = 50
    batch_size: int = 16
    lr: float = 1e-4
    encoder_lr_scale: float = 0.1
    weight_decay: float = 1e-5
    log_interval: int = 10
    save_interval: int = 10
    gradient_clip: Optional[float] = 1.0
    finetune_epochs: int = 10
    checkpoint_dir: str = "checkpoints"
    device: str = "cpu"
    seed: int = 42


class MultiTaskTrainer:
    """Training manager for multi-task models."""

    def __init__(
        self,
        model: MultiTaskPhysiologicalModel,
        config: MultiTaskTrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Set device
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Loss function
        self.loss_fn = MultiTaskLoss(model.config)

        # Optimizer
        self.optimizer = get_optimizer(
            model,
            lr=config.lr,
            wd=config.weight_decay,
            encoder_lr_scale=config.encoder_lr_scale
        )

        # Scheduler
        self.scheduler = get_scheduler(self.optimizer, len(train_loader))

    def train(self) -> Dict[str, List]:
        """Run full training."""
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_metrics": [],
            "lr": []
        }

        best_val_loss = float('inf')

        for epoch in range(self.config.epochs):
            # Train
            train_loss, train_metrics = train_multitask_epoch(
                self.model, self.train_loader, self.optimizer, self.loss_fn,
                self.device, epoch, self.config.log_interval,
                gradient_clip=self.config.gradient_clip
            )

            # Validate
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = validate_multitask(
                    self.model, self.val_loader, self.loss_fn, self.device
                )

            # Update scheduler
            if isinstance(self.scheduler, optim.lr_scheduler._LRScheduler):
                if not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()

            # Record history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics.get("val_loss", 0.0))
            history["val_metrics"].append(val_metrics)
            history["lr"].append(self.optimizer.param_groups[0]["lr"])

            # Print
            print(f"Epoch {epoch}: loss={train_loss:.4f}, val_loss={val_metrics.get('val_loss', 0.0):.4f}")

            # Save checkpoint
            val_loss = val_metrics.get("val_loss", 0.0)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    self.model, self.optimizer, epoch,
                    f"{self.config.checkpoint_dir}/best.pt",
                    val_loss, val_metrics
                )

            if (epoch + 1) % self.config.save_interval == 0:
                save_checkpoint(
                    self.model, self.optimizer, epoch,
                    f"{self.config.checkpoint_dir}/epoch_{epoch}.pt"
                )

        return history

    def resume(self, checkpoint_path: str):
        """Resume from checkpoint."""
        epoch, _, _ = load_checkpoint(self.model, self.optimizer, checkpoint_path, str(self.device))
        return epoch
