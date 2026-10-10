"""
P5 Training Infrastructure
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple
import numpy as np


class RMSELoss(nn.Module):
    def forward(self, pred, target):
        return torch.sqrt(torch.mean((pred - target) ** 2))


class MAPELoss(nn.Module):
    def forward(self, pred, target):
        return torch.mean(torch.abs((target - pred) / (target.abs() + 1e-8)) * 100)


class CombinedLoss(nn.Module):
    def __init__(self, bpm_weight=1.0, conf_weight=0.1, quality_weight=0.1):
        super().__init__()
        self.bpm_weight = bpm_weight
        self.conf_weight = conf_weight
        self.quality_weight = quality_weight
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def forward(self, pred_bpm, target_bpm, pred_conf=None, target_conf=None,
                 pred_quality=None, target_quality=None):
        losses = {}
        total = self.bpm_weight * self.mse(pred_bpm, target_bpm)
        losses["bpm"] = float(total)
        if pred_conf is not None and target_conf is not None:
            c = self.conf_weight * self.bce(pred_conf, target_conf)
            losses["conf"] = float(c)
            total = total + c
        if pred_quality is not None and target_quality is not None:
            q = self.quality_weight * self.bce(pred_quality, target_quality)
            losses["quality"] = float(q)
            total = total + q
        return total, losses


def get_optimizer(model, lr=1e-4, wd=1e-5, kind="adam"):
    if kind.lower() == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if kind.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


def compute_metrics(pred, target):
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    mae = float(np.mean(np.abs(pred - target)))
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    bias = float(np.mean(pred - target))
    return {"mae": mae, "rmse": rmse, "bias": bias}


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        video = batch["video"].to(device)
        bpm_target = batch["bpm"].to(device).squeeze(-1)  # Ensure shape (B,)
        optimizer.zero_grad()
        out = model(video)
        loss = torch.nn.functional.mse_loss(out.bpm, bpm_target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    metrics = {"mae": 0.0, "rmse": 0.0, "bias": 0.0}
    with torch.no_grad():
        for batch in loader:
            video = batch["video"].to(device)
            bpm_target = batch["bpm"].to(device)
            out = model(video)
            m = compute_metrics(out.bpm, bpm_target)
            for k, v in m.items():
                metrics[k] += v
    n = len(loader.dataset)
    return {k: v / n for k, v in metrics.items()}


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)


def get_scheduler(optimizer, steps_per_epoch=100):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps_per_epoch)


def load_checkpoint(model, path, device="cpu"):
    model.load_state_dict(torch.load(path, map_location=device))
    return model
