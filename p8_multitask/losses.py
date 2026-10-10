"""
P8 Multi-Task Loss Functions

Loss functions for each task and configurable multi-task loss aggregation.
Handles missing labels gracefully.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Union

from .base import MultiTaskConfig, MultiTaskOutput, TaskType


class BVPLossFn(nn.Module):
    """Loss function for BVP waveform prediction.

    Combines MSE with frequency-domain loss for better physiological signal capture.
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        corr_weight: float = 0.5,
        spectral_weight: float = 0.3,
        use_mask: bool = True
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.corr_weight = corr_weight
        self.spectral_weight = spectral_weight
        self.use_mask = use_mask

    def pearson_correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Pearson correlation (1 - correlation as loss)."""
        x_centered = x - x.mean(dim=-1, keepdim=True)
        y_centered = y - y.mean(dim=-1, keepdim=True)
        corr = (x_centered * y_centered).sum(dim=-1) / (
            x_centered.norm(dim=-1) * y_centered.norm(dim=-1) + 1e-9
        )
        return 1 - corr.mean()

    def spectral_loss(self, pred: torch.Tensor, target: torch.Tensor, fps: float = 30.0) -> torch.Tensor:
        """L1 loss in frequency domain."""
        pred_fft = torch.abs(torch.fft.rfft(pred, dim=-1))
        target_fft = torch.abs(torch.fft.rfft(target, dim=-1))

        # Normalize
        pred_fft = pred_fft / (pred_fft.max(dim=-1, keepdim=True)[0] + 1e-9)
        target_fft = target_fft / (target_fft.max(dim=-1, keepdim=True)[0] + 1e-9)

        return F.l1_loss(pred_fft, target_fft)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        fps: float = 30.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute BVP loss.

        Args:
            pred: (B, T) predicted BVP waveform
            target: (B, T) target BVP waveform
            mask: (B, T) optional mask for valid positions
            fps: Frames per second

        Returns:
            Tuple of (loss, metrics_dict)
        """
        if mask is not None and self.use_mask:
            valid_mask = ~mask
            valid_count = valid_mask.sum()

            if valid_count == 0:
                return torch.tensor(0.0, device=pred.device, requires_grad=pred.requires_grad), {"bvp_loss": 0.0}

            # MSE with mask
            diff = (pred - target) ** 2
            mse_loss = (diff * valid_mask.float()).sum() / valid_count

            # Correlation (use all valid positions)
            valid_pred = pred[valid_mask].reshape(-1, pred.shape[-1])
            valid_target = target[valid_mask].reshape(-1, target.shape[-1])

            if valid_pred.shape[0] > 1:
                corr_loss = self.pearson_correlation(valid_pred, valid_target)
            else:
                corr_loss = torch.tensor(0.0, device=pred.device)

            # Spectral loss (per sample then average)
            valid_pred_per_sample = pred[valid_mask.any(dim=-1)]
            valid_target_per_sample = target[valid_mask.any(dim=-1)]
            if len(valid_pred_per_sample) > 0:
                spectral_loss = self.spectral_loss(valid_pred_per_sample, valid_target_per_sample, fps)
            else:
                spectral_loss = torch.tensor(0.0, device=pred.device)

        else:
            # No mask - use all positions
            mse_loss = F.mse_loss(pred, target)
            corr_loss = self.pearson_correlation(pred, target)
            spectral_loss = self.spectral_loss(pred, target, fps)

        total_loss = (
            self.mse_weight * mse_loss +
            self.corr_weight * corr_loss +
            self.spectral_weight * spectral_loss
        )

        metrics = {
            "bvp_loss": float(total_loss.detach()),
            "bvp_mse": float(mse_loss.detach()),
            "bvp_corr": float(corr_loss.detach()),
            "bvp_spectral": float(spectral_loss.detach())
        }

        return total_loss, metrics


class HRLossFn(nn.Module):
    """Loss function for heart rate estimation.

    Uses MSE loss with optional focal weighting for extreme HR values.
    """

    def __init__(
        self,
        use_mse: bool = True,
        use_l1: bool = False,
        focal_weight: float = 0.0,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        self.use_mse = use_mse
        self.use_l1 = use_l1
        self.focal_weight = focal_weight
        self.focal_gamma = focal_gamma

        if use_mse:
            self.mse = nn.MSELoss()
        if use_l1:
            self.l1 = nn.L1Loss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        available_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute HR loss.

        Args:
            pred: (B,) predicted HR values
            target: (B,) target HR values
            available_mask: (B,) boolean mask for valid targets

        Returns:
            Tuple of (loss, metrics_dict)
        """
        if available_mask is not None:
            # Filter to valid targets
            valid_indices = available_mask.nonzero(as_tuple=True)[0]
            if len(valid_indices) == 0:
                return torch.tensor(0.0, device=pred.device, requires_grad=pred.requires_grad), {"hr_loss": 0.0}

            pred_valid = pred[valid_indices]
            target_valid = target[valid_indices]
        else:
            pred_valid = pred
            target_valid = target

        # Compute losses
        total_loss = torch.tensor(0.0, device=pred.device, requires_grad=pred.requires_grad)
        metrics = {}

        if self.use_mse:
            mse = self.mse(pred_valid, target_valid)
            total_loss = total_loss + mse
            metrics["hr_mse"] = float(mse.detach())

        if self.use_l1:
            l1 = self.l1(pred_valid, target_valid)
            total_loss = total_loss + l1
            metrics["hr_l1"] = float(l1.detach())

        # Focal weighting for extreme values
        if self.focal_weight > 0:
            # Weight samples with extreme HR more heavily
            hr_range = target_valid.max() - target_valid.min()
            if hr_range > 0:
                normalized = (target_valid - target_valid.min()) / hr_range
                focal_weights = (1 - normalized) ** self.focal_gamma
                focal_loss = (focal_weights * (pred_valid - target_valid) ** 2).mean()
                total_loss = total_loss + self.focal_weight * focal_loss
                metrics["hr_focal"] = float(focal_loss.detach())

        metrics["hr_loss"] = float(total_loss.detach())

        return total_loss, metrics


class SQILossFn(nn.Module):
    """Loss function for signal quality index prediction.

    Uses BCE loss for binary-like quality scores.
    """

    def __init__(
        self,
        use_bce: bool = True,
        use_mse: bool = False,
        bce_pos_weight: float = 1.0
    ):
        super().__init__()
        self.use_bce = use_bce
        self.use_mse = use_mse

        if use_bce:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(bce_pos_weight))
        if use_mse:
            self.mse = nn.MSELoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        available_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute SQI loss.

        Args:
            pred: (B,) predicted SQI values [0, 1]
            target: (B,) target SQI values [0, 1]
            available_mask: (B,) boolean mask for valid targets

        Returns:
            Tuple of (loss, metrics_dict)
        """
        if available_mask is not None:
            valid_indices = available_mask.nonzero(as_tuple=True)[0]
            if len(valid_indices) == 0:
                return torch.tensor(0.0, device=pred.device, requires_grad=pred.requires_grad), {"sqi_loss": 0.0}

            pred_valid = pred[valid_indices]
            target_valid = target[valid_indices]
        else:
            pred_valid = pred
            target_valid = target

        # Clamp targets to valid range
        target_valid = target_valid.clamp(0.0, 1.0)

        total_loss = torch.tensor(0.0, device=pred.device, requires_grad=pred.requires_grad)
        metrics = {}

        if self.use_bce:
            bce = F.binary_cross_entropy(pred_valid, target_valid)
            total_loss = total_loss + bce
            metrics["sqi_bce"] = float(bce.detach())

        if self.use_mse:
            mse = F.mse_loss(pred_valid, target_valid)
            total_loss = total_loss + mse
            metrics["sqi_mse"] = float(mse.detach())

        metrics["sqi_loss"] = float(total_loss.detach())

        return total_loss, metrics


class ConfidenceLossFn(nn.Module):
    """Loss function for prediction confidence estimation.

    Encourages high confidence when predictions are accurate,
    low confidence when predictions are wrong.
    """

    def __init__(
        self,
        accuracy_threshold: float = 0.1,
        uncertainty_weight: float = 0.1
    ):
        super().__init__()
        self.accuracy_threshold = accuracy_threshold
        self.uncertainty_weight = uncertainty_weight

    def forward(
        self,
        pred_confidence: torch.Tensor,
        hr_pred: Optional[torch.Tensor] = None,
        hr_target: Optional[torch.Tensor] = None,
        available_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute confidence loss.

        Args:
            pred_confidence: (B,) predicted confidence [0, 1]
            hr_pred: (B,) HR predictions (for accuracy computation)
            hr_target: (B,) HR targets (for accuracy computation)
            available_mask: (B,) mask for valid samples

        Returns:
            Tuple of (loss, metrics_dict)
        """
        if hr_pred is None or hr_target is None:
            # Without ground truth, use uniform targets
            return torch.tensor(0.0, device=pred_confidence.device, requires_grad=pred_confidence.requires_grad), {"confidence_loss": 0.0}

        if available_mask is not None:
            valid_indices = available_mask.nonzero(as_tuple=True)[0]
            if len(valid_indices) == 0:
                return torch.tensor(0.0, device=pred_confidence.device, requires_grad=pred_confidence.requires_grad), {"confidence_loss": 0.0}

            pred_conf = pred_confidence[valid_indices]
            pred_hr = hr_pred[valid_indices]
            target_hr = hr_target[valid_indices]
        else:
            pred_conf = pred_confidence
            pred_hr = hr_pred
            target_hr = hr_target

        # Compute prediction error
        error = torch.abs(pred_hr - target_hr)

        # Accuracy indicator (1 if correct, 0 if wrong)
        accuracy = (error < self.accuracy_threshold).float()

        # Confidence loss: high confidence when accurate, low when inaccurate
        # This is similar to calibration loss
        confidence_loss = F.binary_cross_entropy(pred_conf, accuracy)

        # Uncertainty regularization: don't predict all 0s or all 1s
        uncertainty_loss = pred_conf.var()

        total_loss = confidence_loss + self.uncertainty_weight * uncertainty_loss

        metrics = {
            "confidence_loss": float(total_loss.detach()),
            "confidence_main": float(confidence_loss.detach()),
            "confidence_uncertainty": float(uncertainty_loss.detach()),
            "mean_confidence": float(pred_conf.mean().detach()),
            "accuracy": float(accuracy.mean().detach())
        }

        return total_loss, metrics


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss with configurable task weights.

    Handles missing labels by checking availability masks.
    """

    def __init__(self, config: MultiTaskConfig):
        super().__init__()
        self.config = config
        self.tasks = config.get_enabled_tasks()
        self.weights = config.get_task_weights()
        self.aggregation = config.loss_aggregation

        # Initialize task-specific losses
        self.task_losses = nn.ModuleDict()

        if config.enable_bvp:
            self.task_losses["bvp"] = BVPLossFn()

        if config.enable_hr:
            self.task_losses["hr"] = HRLossFn()

        if config.enable_sqi:
            self.task_losses["sqi"] = SQILossFn()

        if config.enable_confidence:
            self.task_losses["confidence"] = ConfidenceLossFn()

        # Set device for loss computation
        self._device = torch.device(config.device)

    def forward(
        self,
        output: MultiTaskOutput,
        target: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task loss.

        Args:
            output: MultiTaskOutput with predictions
            target: Dict with target tensors {"hr": ..., "bvp": ..., "sqi": ..., "confidence": ...}
            masks: Optional dict with availability masks

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        if masks is None:
            masks = {}

        # Get device from first tensor or use stored device
        device = self._device
        for key in ["hr", "bvp", "sqi", "confidence"]:
            if key in target and target[key] is not None:
                device = target[key].device
                break
        else:
            # Only executed if loop completed without break
            if output.hr is not None:
                device = output.hr.device
            elif output.bvp is not None:
                device = output.bvp.device

        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

        all_metrics = {}

        # BVP loss
        if TaskType.BVP in self.tasks and output.bvp is not None:
            if "bvp" in target:
                bvp_target = target["bvp"]
                bvp_mask = masks.get("bvp_available")
                loss, metrics = self.task_losses["bvp"](output.bvp, bvp_target, bvp_mask)
                total_loss = total_loss + self.weights[TaskType.BVP] * loss
                all_metrics.update(metrics)

        # HR loss
        if TaskType.HR in self.tasks and output.hr is not None:
            if "hr" in target:
                hr_target = target["hr"]
                hr_mask = masks.get("hr_available")
                loss, metrics = self.task_losses["hr"](output.hr, hr_target, hr_mask)
                total_loss = total_loss + self.weights[TaskType.HR] * loss
                all_metrics.update(metrics)

        # SQI loss
        if TaskType.SQI in self.tasks and output.sqi is not None:
            if "sqi" in target:
                sqi_target = target["sqi"]
                sqi_mask = masks.get("sqi_available")
                loss, metrics = self.task_losses["sqi"](output.sqi, sqi_target, sqi_mask)
                total_loss = total_loss + self.weights[TaskType.SQI] * loss
                all_metrics.update(metrics)

        # Confidence loss
        if TaskType.CONFIDENCE in self.tasks and output.confidence is not None:
            if "confidence" in target:
                conf_target = target["confidence"]
                conf_mask = masks.get("confidence_available")
                loss, metrics = self.task_losses["confidence"](
                    output.confidence,
                    output.hr if output.hr is not None else None,
                    target.get("hr"),
                    conf_mask
                )
                total_loss = total_loss + self.weights[TaskType.CONFIDENCE] * loss
                all_metrics.update(metrics)

        all_metrics["total_loss"] = float(total_loss.detach())

        return total_loss, all_metrics

    def get_task_losses(self) -> Dict[str, nn.Module]:
        """Get individual task loss modules."""
        return dict(self.task_losses)


def compute_per_task_loss(
    output: MultiTaskOutput,
    target: Dict[str, torch.Tensor],
    masks: Optional[Dict[str, torch.Tensor]] = None,
    config: Optional[MultiTaskConfig] = None
) -> Dict[str, float]:
    """
    Compute loss for each task individually (without aggregation).

    Useful for analysis and monitoring.
    """
    if masks is None:
        masks = {}

    losses = {}

    # HR MSE
    if output.hr is not None and "hr" in target:
        mask = masks.get("hr_available")
        if mask is not None:
            valid = mask.nonzero(as_tuple=True)[0]
            if len(valid) > 0:
                losses["hr_mse"] = float(F.mse_loss(output.hr[valid], target["hr"][valid]))
        else:
            losses["hr_mse"] = float(F.mse_loss(output.hr, target["hr"]))

    # BVP MSE
    if output.bvp is not None and "bvp" in target:
        mask = masks.get("bvp_available")
        if mask is not None:
            valid = (~mask).any(dim=-1).nonzero(as_tuple=True)[0]
            if len(valid) > 0:
                losses["bvp_mse"] = float(F.mse_loss(output.bvp[valid], target["bvp"][valid]))
        else:
            losses["bvp_mse"] = float(F.mse_loss(output.bvp, target["bvp"]))

    # SQI BCE
    if output.sqi is not None and "sqi" in target:
        mask = masks.get("sqi_available")
        if mask is not None:
            valid = mask.nonzero(as_tuple=True)[0]
            if len(valid) > 0:
                losses["sqi_bce"] = float(F.binary_cross_entropy(output.sqi[valid], target["sqi"][valid].clamp(0, 1)))
        else:
            losses["sqi_bce"] = float(F.binary_cross_entropy(output.sqi, target["sqi"].clamp(0, 1)))

    # Confidence
    if output.confidence is not None:
        losses["mean_confidence"] = float(output.confidence.mean())

    return losses
