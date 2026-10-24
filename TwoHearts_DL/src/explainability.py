"""
TwoHearts Explainable AI (XAI) Module: 1D Grad-CAM for Biomedical Time Series.
Generates localized temporal saliency maps for both the Fetal Arrhythmia Head
and the Maternal Autonomic Stress Head, overlaid on raw abdominal ECG traces.
"""

from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM1D:
    """1D Grad-CAM engine for Multi-Task 1D-CNN."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer if target_layer is not None else model.get_cam_target_layer()

        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        # grad_output[0] is (B, C, L)
        self.gradients = grad_output[0].detach()

    def generate_cam(self, x: torch.Tensor, target_head: str = "fetal") -> np.ndarray:
        """
        Generate 1D Grad-CAM saliency map for the given input tensor.
        x: (1, channels, length)
        target_head: "fetal" or "maternal"
        Returns: (length,) float32 array normalized in [0, 1]
        """
        self.model.zero_grad()
        x.requires_grad_(True)

        out = self.model(x)

        if target_head == "fetal":
            score = out["fetal_logits"][0, 0]
        elif target_head == "maternal":
            score = out["stress_pred"][0, 0]
        else:
            raise ValueError(f"Unknown target_head: {target_head}. Choose 'fetal' or 'maternal'.")

        score.backward(retain_graph=True)

        # Gradients: (1, C, L_feat), Activations: (1, C, L_feat)
        grads = self.gradients[0]  # (C, L_feat)
        acts = self.activations[0]  # (C, L_feat)

        # Global average pooling over temporal dimension to get channel weights
        weights = grads.mean(dim=1, keepdim=True)  # (C, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * acts, dim=0)  # (L_feat,)
        cam = F.relu(cam)  # keep only positive influences

        cam_np = cam.cpu().numpy()

        # Upsample to original sequence length
        input_len = x.shape[2]
        cam_upsampled = np.interp(
            np.linspace(0, len(cam_np) - 1, input_len),
            np.arange(len(cam_np)),
            cam_np
        )

        # Normalize to [0, 1]
        denom = cam_upsampled.max() - cam_upsampled.min()
        if denom > 1e-8:
            cam_norm = (cam_upsampled - cam_upsampled.min()) / denom
        else:
            cam_norm = np.zeros_like(cam_upsampled)

        return cam_norm


def plot_gradcam_overlay(x_np: np.ndarray, cam: np.ndarray, fs: float, title: str,
                         save_path: Optional[str] = None, lead_names: Optional[list] = None) -> plt.Figure:
    """
    Plots the 5 abdominal ECG channels with the Grad-CAM heatmap overlaid.
    x_np: (channels, length)
    cam: (length,) normalized in [0, 1]
    """
    n_channels, length = x_np.shape
    time = np.arange(length) / fs
    if lead_names is None:
        lead_names = [f"Abdomen {i+1}" for i in range(n_channels)]

    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 2.2 * n_channels), sharex=True)
    if n_channels == 1:
        axes = [axes]

    cmap = plt.get_cmap("hot")

    for i, ax in enumerate(axes):
        sig = x_np[i]
        ax.plot(time, sig, color="#1f77b4", linewidth=1.2, label=lead_names[i], zorder=2)

        # Draw colored background spans representing Grad-CAM saliency
        # Downsample background for fast rendering
        step = max(1, length // 1000)
        t_sub = time[::step]
        cam_sub = cam[::step]

        for j in range(len(t_sub) - 1):
            saliency = cam_sub[j]
            if saliency > 0.2:  # Highlight significant attention regions
                ax.axvspan(t_sub[j], t_sub[j+1], color=cmap(saliency), alpha=0.35, zorder=1)

        ax.set_ylabel(lead_names[i], fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
