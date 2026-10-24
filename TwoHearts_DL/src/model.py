"""
TwoHearts Model Architecture: End-to-End Multi-Task 1D-CNN.
Shared multi-scale residual backbone with Squeeze-and-Excitation (SE) channel attention,
split into two task heads:
  1. Fetal Arrhythmia Classification (Binary)
  2. Maternal Cardiovascular Stress / Autonomic Index (Continuous regression [0, 1])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation1D(nn.Module):
    """Squeeze-and-Excitation attention for 1D temporal signals."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced_ch = max(4, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced_ch, bias=False)
        self.fc2 = nn.Linear(reduced_ch, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        b, c, _ = x.size()
        w = x.mean(dim=2)  # Global average pooling -> (B, C)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))  # (B, C)
        return x * w.unsqueeze(2)


class ResBlock1D(nn.Module):
    """Residual block with 1D convolutions, batch norm, dropout, and SE attention."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, kernel_size: int = 7, dropout: float = 0.2):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)
        self.drop = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SqueezeExcitation1D(out_channels)
        self.act2 = nn.LeakyReLU(0.1, inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.act2(out + residual)
        return out


class MultiScaleStem(nn.Module):
    """Multi-scale front-end to capture both sharp QRS spikes and broad cardiac waves."""

    def __init__(self, in_channels: int = 4, out_channels: int = 32):
        super().__init__()
        ch_per_branch = out_channels // 3
        rem = out_channels - 2 * ch_per_branch

        self.branch_short = nn.Conv1d(in_channels, ch_per_branch, kernel_size=7, padding=3, bias=False)
        self.branch_med = nn.Conv1d(in_channels, ch_per_branch, kernel_size=15, padding=7, bias=False)
        self.branch_long = nn.Conv1d(in_channels, rem, kernel_size=31, padding=15, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch_short(x)
        b2 = self.branch_med(x)
        b3 = self.branch_long(x)
        out = torch.cat([b1, b2, b3], dim=1)
        return self.act(self.bn(out))


class TwoHeartsNet(nn.Module):
    """
    Multi-Task 1D Deep CNN for Maternal-Fetal Cardiac Screening.
    Takes 5 abdominal leads (B, 5, L) and predicts:
      - fetal_logits: (B, 1) logits for arrhythmia classification
      - stress_pred: (B, 1) continuous predicted maternal autonomic stress index [0, 1]
    """

    def __init__(self, in_channels: int = 4, base_channels: int = 32, dropout: float = 0.2):
        super().__init__()
        self.stem = MultiScaleStem(in_channels, base_channels)

        # 4 Residual Stages with downsampling (stride 2)
        self.stage1 = ResBlock1D(base_channels, base_channels, stride=2, kernel_size=7, dropout=dropout)
        self.stage2 = ResBlock1D(base_channels, base_channels * 2, stride=2, kernel_size=7, dropout=dropout)
        self.stage3 = ResBlock1D(base_channels * 2, base_channels * 4, stride=2, kernel_size=7, dropout=dropout)
        self.stage4 = ResBlock1D(base_channels * 4, base_channels * 8, stride=2, kernel_size=7, dropout=dropout)

        feature_dim = base_channels * 8  # e.g., 256

        self.gap = nn.AdaptiveAvgPool1d(1)

        # Head 1: Fetal Arrhythmia Classifier
        self.fetal_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        # Head 2: Maternal Autonomic Stress Score (Regression in [0, 1])
        self.maternal_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial-temporal feature map before pooling (used for Grad-CAM)."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x

    def forward(self, x: torch.Tensor):
        feat_map = self.extract_features(x)
        pooled = self.gap(feat_map).squeeze(-1)  # (B, feature_dim)

        fetal_logits = self.fetal_head(pooled)
        stress_pred = self.maternal_head(pooled)

        return {
            "fetal_logits": fetal_logits,
            "fetal_prob": torch.sigmoid(fetal_logits),
            "stress_pred": stress_pred,
            "feature_map": feat_map
        }

    def get_cam_target_layer(self) -> nn.Module:
        """Returns the final convolutional block of the shared backbone for Grad-CAM."""
        return self.stage4.conv2
