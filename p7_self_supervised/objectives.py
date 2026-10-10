"""
P7 Self-Supervised Learning Objectives

Implementation of self-supervised learning objectives for physiological signals:
    - Contrastive Loss (SimCLR-style)
    - Temporal Consistency Loss
    - Predictive Loss (masked reconstruction)
    - BYOL-style loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class ContrastiveLoss(nn.Module):
    """
    SimCLR-style contrastive loss for temporal sequences.

    Maximizes agreement between differently augmented views of the same sequence
    while minimizing agreement with other sequences in the batch.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        projections: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
        subject_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute contrastive loss.

        Args:
            projections: (B*V, D) - L2-normalized projection vectors
            indices: (B,) - Original sample indices
            subject_ids: (B,) - Subject IDs for same-subject handling

        Returns:
            Tuple of (loss, metrics_dict)
        """
        B = projections.shape[0]

        # Compute pairwise similarities (already L2-normalized)
        # similarities[i,j] = cosine_similarity(projections[i], projections[j])
        similarities = projections @ projections.T / self.temperature

        # Create positive mask (same original sample)
        if indices is not None:
            # Match original samples before augmentation
            # We assume alternating order: [view1_s1, view2_s1, view1_s2, view2_s2, ...]
            # Or all views of sample i are similar
            batch_size = B // 2 if B % 2 == 0 else B
            # Simple case: first half is positives for second half
            mask = torch.eye(batch_size, device=projections.device)
            # Expand for views
            if B > batch_size:
                mask = torch.block_diag(*[mask] * (B // batch_size))
            else:
                mask = torch.zeros(B, B, device=projections.device)
                for i in range(batch_size):
                    mask[i, i] = 1.0
        else:
            # Diagonal is positive pairs
            mask = torch.eye(B, device=projections.device)

        # Numerical stability
        logits = similarities
        # Mask out diagonal for negative calculation
        logits = logits - mask * 1e9

        # Compute loss using InfoNCE
        exp_logits = torch.exp(logits)
        log_probs = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))

        # Mean of log-likelihood over positives
        mask_flat = mask.sum(dim=1) > 0
        loss = -(log_probs * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        loss = loss.mean()

        # Metrics
        with torch.no_grad():
            # Accuracy of predicting positive pairs
            pred = similarities.argmax(dim=1)
            acc = (pred == torch.arange(B, device=projections.device)).float().mean()

        metrics = {
            "contrastive_loss": float(loss.detach()),
            "contrastive_acc": float(acc)
        }

        return loss, metrics


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency loss for physiological signals.

    Encourages the encoder to produce similar representations for
    temporally close segments of the same sequence.
    """

    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma

    def forward(
        self,
        representations: torch.Tensor,
        segments: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute temporal consistency loss.

        Args:
            representations: (B, D) - Learned representations
            segments: (B, T, F) - Original segment features
            mask: (B, T) - Optional mask

        Returns:
            Tuple of (loss, metrics_dict)
        """
        B, T, F = segments.shape

        # Compute temporal distance in feature space
        # Similar segments should have similar representations
        if mask is not None:
            # Weight by valid positions
            valid_mask = ~mask

        # Simple approach: encourage smooth temporal dynamics
        # Compute difference between consecutive timesteps in representation
        # This is a simplified version - full implementation would
        # compare representations from different temporal crops

        # For now, return a regularization-style loss
        # that encourages the representation to vary smoothly
        rep_std = representations.std(dim=1).mean()

        # Loss encourages reasonable variance in representations
        # (not too collapsed, not too noisy)
        loss = -rep_std * 0.1  # Negative because we want to maximize std

        metrics = {
            "consistency_loss": float(loss.detach()),
            "rep_std": float(rep_std.detach())
        }

        return loss, metrics


class PredictiveLoss(nn.Module):
    """
    Predictive / masked reconstruction loss.

    Trains the model to predict masked portions of the input signal,
    learning rich temporal representations.
    """

    def __init__(self, mask_ratio: float = 0.3):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute predictive loss.

        Args:
            predictions: (B, T, F) or (B*T, F) - Predicted values for masked positions
            targets: (B, T, F) or (B*T, F) - Original values at masked positions
            mask: (B, T) or (B*T,) - Mask indicating masked positions

        Returns:
            Tuple of (loss, metrics_dict)
        """
        if mask is not None:
            # Apply mask to loss
            valid_mask = mask.reshape(-1)
            pred_flat = predictions.reshape(-1, predictions.shape[-1])
            target_flat = targets.reshape(-1, targets.shape[-1])

            mse = ((pred_flat - target_flat) ** 2).mean(dim=1)
            loss = (mse * valid_mask.float()).sum() / (valid_mask.sum() + 1e-9)
        else:
            loss = F.mse_loss(predictions, targets)

        metrics = {
            "predictive_loss": float(loss.detach())
        }

        return loss, metrics


class BYOLLoss(nn.Module):
    """
    BYOL-style loss for self-supervised learning.

    Uses online and target networks with exponential moving average.
    Predictor network learns to match online predictions to target representations.
    """

    def __init__(self, tau: float = 0.1):
        super().__init__()
        self.tau = tau

    def forward(
        self,
        online_preds: torch.Tensor,
        target_reprs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute BYOL loss.

        Args:
            online_preds: (B, D) - Predictions from online network
            target_reprs: (B, D) - Target representations

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Normalize
        online_preds = F.normalize(online_preds, dim=1)
        target_reprs = F.normalize(target_reprs, dim=1)

        # Compute loss
        loss = 2 - 2 * (online_preds * target_reprs).sum(dim=1).mean()

        with torch.no_grad():
            sim = (online_preds * target_reprs).sum(dim=1).mean()

        metrics = {
            "byol_loss": float(loss.detach()),
            "byol_similarity": float(sim.detach())
        }

        return loss, metrics


class CombinedSSLLoss(nn.Module):
    """
    Combined self-supervised loss.

    Combines multiple SSL objectives with configurable weights.
    """

    def __init__(
        self,
        contrastive_weight: float = 1.0,
        consistency_weight: float = 0.5,
        predictive_weight: float = 0.3,
        byol_weight: float = 0.0,
        temperature: float = 0.1
    ):
        super().__init__()

        self.contrastive_weight = contrastive_weight
        self.consistency_weight = consistency_weight
        self.predictive_weight = predictive_weight
        self.byol_weight = byol_weight

        self.contrastive_loss = ContrastiveLoss(temperature)
        self.consistency_loss = TemporalConsistencyLoss()
        self.predictive_loss = PredictiveLoss()
        self.byol_loss = BYOLLoss()

    def forward(
        self,
        representations: torch.Tensor,
        projections: torch.Tensor,
        segments: Optional[torch.Tensor] = None,
        predictions: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        target_reprs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined SSL loss.

        Args:
            representations: (B, D) - Learned representations
            projections: (B*V, D) - Projection for contrastive
            segments: (B, T, F) - Original segments for consistency
            predictions: Predicted values for masked positions
            targets: Original values for masked positions
            mask: Mask for masked positions
            indices: Sample indices
            target_reprs: Target representations for BYOL

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        total_loss = torch.tensor(0.0, device=representations.device)
        losses = {}

        # Contrastive loss
        if self.contrastive_weight > 0 and projections is not None:
            c_loss, c_metrics = self.contrastive_loss(projections, indices)
            total_loss = total_loss + self.contrastive_weight * c_loss
            losses.update(c_metrics)

        # Temporal consistency loss
        if self.consistency_weight > 0 and segments is not None:
            cons_loss, cons_metrics = self.consistency_loss(representations, segments, mask)
            total_loss = total_loss + self.consistency_weight * cons_loss
            losses.update(cons_metrics)

        # Predictive loss
        if self.predictive_weight > 0 and predictions is not None and targets is not None:
            pred_loss, pred_metrics = self.predictive_loss(predictions, targets, mask)
            total_loss = total_loss + self.predictive_weight * pred_loss
            losses.update(pred_metrics)

        # BYOL loss
        if self.byol_weight > 0 and target_reprs is not None:
            byol_loss, byol_metrics = self.byol_loss(representations, target_reprs)
            total_loss = total_loss + self.byol_weight * byol_loss
            losses.update(byol_metrics)

        losses["total_loss"] = float(total_loss.detach())

        return total_loss, losses


def nt_xent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.1
) -> torch.Tensor:
    """
    Simplified NT-Xent loss for two views.

    Args:
        z1: (B, D) - First view projections
        z2: (B, D) - Second view projections
        temperature: Temperature parameter

    Returns:
        Scalar loss
    """
    B = z1.shape[0]
    D = z1.shape[1]

    # Normalize
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Concatenate all representations
    z = torch.cat([z1, z2], dim=0)  # (2B, D)

    # Compute similarities
    sim = z @ z.T / temperature  # (2B, 2B)

    # Mask out self-similarity
    mask = torch.eye(2 * B, device=z.device).bool()
    sim.masked_fill_(mask, float('-inf'))

    # Positive pairs are (i, i+B) and (i+B, i)
    pos_sim = torch.cat([
        (z1 * z2).sum(dim=1) / temperature,
        (z2 * z1).sum(dim=1) / temperature
    ])  # (2B,)

    # Loss
    loss = -pos_sim + torch.log(torch.exp(sim).sum(dim=1))
    return loss.mean()
