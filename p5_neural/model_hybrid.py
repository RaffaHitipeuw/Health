"""
P5 Model 5: Sanubari Hybrid

Combines neural video features with classical Sanubari signals.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .base import BaseModel, ModelConfig, ModelOutput


class LightweightHybrid(torch.nn.Module):
    """Lightweight hybrid model."""

    def __init__(self, video_channels: int = 3, classical_features: int = 8, hidden: int = 32):
        super().__init__()
        self.video_enc = torch.nn.Sequential(
            torch.nn.Conv2d(video_channels, hidden, 3, padding=1),
            torch.nn.BatchNorm2d(hidden),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((4, 4)),
            torch.nn.Flatten()
        )
        self.classical_enc = torch.nn.Sequential(
            torch.nn.Linear(classical_features, hidden),
            torch.nn.ReLU()
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(hidden + hidden * 16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 3)
        )

    def forward(self, video, classical):
        v = self.video_enc(video.mean(1).permute(0, 3, 1, 2))
        c = self.classical_enc(classical)
        x = torch.cat([v, c], dim=1)
        out = self.out(x)
        return out[:, 0], torch.sigmoid(out[:, 1:2]), torch.sigmoid(out[:, 2:])


class HybridModel(BaseModel):
    """Full hybrid model with BaseModel interface."""

    def __init__(self, config: ModelConfig, video_hidden: int = 64, classical_hidden: int = 32, fusion_dim: int = 128):
        super().__init__(config)

        self.video_enc = torch.nn.Sequential(
            torch.nn.Conv2d(config.input_channels, video_hidden, 3, padding=1),
            torch.nn.BatchNorm2d(video_hidden),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((4, 4)),
            torch.nn.Flatten()
        )

        self.classical_enc = torch.nn.Sequential(
            torch.nn.Linear(8, classical_hidden),
            torch.nn.LayerNorm(classical_hidden),
            torch.nn.ReLU()
        )

        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(video_hidden * 16 + classical_hidden, fusion_dim),
            torch.nn.LayerNorm(fusion_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(fusion_dim, fusion_dim // 2),
            torch.nn.ReLU()
        )

        self.head_bpm = torch.nn.Linear(fusion_dim // 2, 1)
        self.head_conf = torch.nn.Sequential(
            torch.nn.Linear(fusion_dim // 2, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )
        self.head_quality = torch.nn.Sequential(
            torch.nn.Linear(fusion_dim // 2, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, video, classical=None):
        # video: (B, T, H, W, C) or (B*T, H, W, C)
        if video.dim() == 5:
            spatial = video.mean(1)  # mean over time
        else:
            spatial = video

        v_feat = self.video_enc(spatial.permute(0, 3, 1, 2))

        if classical is not None:
            c_feat = self.classical_enc(classical)
            fused = torch.cat([v_feat, c_feat], dim=1)
        else:
            fused = v_feat

        x = self.fusion(fused)
        return ModelOutput(
            bpm=self.head_bpm(x).squeeze(-1),
            bpm_confidence=self.head_conf(x).squeeze(-1),
            signal_quality=self.head_quality(x).squeeze(-1)
        )
