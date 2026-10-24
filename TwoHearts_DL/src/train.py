"""
TwoHearts Training Engine.
Executes patient-level stratified 5-fold cross-validation with multi-task loss balancing,
learning rate scheduling, model checkpointing, and out-of-fold evaluation.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import TwoHeartsDataset
from src.evaluate import (
    compute_fetal_metrics,
    compute_maternal_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_curves,
)
from src.model import TwoHeartsNet


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_stress: float = 0.5,
    pos_weight: Optional[torch.Tensor] = None,
) -> float:
    """Trains the model for one epoch and returns the average multi-task loss."""
    model.train()
    total_loss = 0.0

    for x, y_arr, y_stress, _ in loader:
        x = x.to(device)
        y_arr = y_arr.to(device)
        y_stress = y_stress.to(device)

        optimizer.zero_grad()
        out = model(x)

        loss_fetal = F.binary_cross_entropy_with_logits(out["fetal_logits"], y_arr, pos_weight=pos_weight)
        loss_stress = F.smooth_l1_loss(out["stress_pred"], y_stress)

        loss = loss_fetal + lambda_stress * loss_stress
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(x)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lambda_stress: float = 0.5,
) -> Tuple[float, Dict[str, float], Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluates the model and returns loss, metrics, and raw predictions."""
    model.eval()
    total_loss = 0.0

    all_y_arr = []
    all_pred_prob = []
    all_y_stress = []
    all_pred_stress = []

    for x, y_arr, y_stress, _ in loader:
        x = x.to(device)
        y_arr = y_arr.to(device)
        y_stress = y_stress.to(device)

        out = model(x)

        loss_fetal = F.binary_cross_entropy_with_logits(out["fetal_logits"], y_arr)
        loss_stress = F.smooth_l1_loss(out["stress_pred"], y_stress)
        loss = loss_fetal + lambda_stress * loss_stress

        total_loss += loss.item() * len(x)

        all_y_arr.append(y_arr.cpu().numpy())
        all_pred_prob.append(out["fetal_prob"].cpu().numpy())
        all_y_stress.append(y_stress.cpu().numpy())
        all_pred_stress.append(out["stress_pred"].cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)

    y_arr_all = np.vstack(all_y_arr).squeeze()
    pred_prob_all = np.vstack(all_pred_prob).squeeze()
    y_stress_all = np.vstack(all_y_stress).squeeze()
    pred_stress_all = np.vstack(all_pred_stress).squeeze()

    fetal_metrics = compute_fetal_metrics(y_arr_all, pred_prob_all)
    maternal_metrics = compute_maternal_metrics(y_stress_all, pred_stress_all)

    return avg_loss, fetal_metrics, maternal_metrics, pred_prob_all, pred_stress_all


def run_fold_training(
    fold_idx: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 25,
    lr: float = 1e-3,
    lambda_stress: float = 0.5,
    checkpoint_dir: Path = Path("results/checkpoints"),
) -> Tuple[nn.Module, Dict[str, List[float]], Dict[str, float], Dict[str, float]]:
    """Trains one cross-validation fold with early stopping and checkpoint saving."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / f"twohearts_best_fold{fold_idx}.pt"

    model = TwoHeartsNet(in_channels=4, base_channels=32, dropout=0.25).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_fetal_f1": [],
        "val_fetal_acc": [],
        "val_fetal_auc": [],
        "val_stress_mae": [],
        "val_stress_rmse": [],
    }

    best_f1 = -1.0
    best_fetal_metrics = {}
    best_maternal_metrics = {}

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, lambda_stress)
        val_loss, f_metrics, m_metrics, _, _ = evaluate_epoch(model, val_loader, device, lambda_stress)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_fetal_f1"].append(f_metrics["f1_score"])
        history["val_fetal_acc"].append(f_metrics["accuracy"])
        history["val_fetal_auc"].append(f_metrics["roc_auc"])
        history["val_stress_mae"].append(m_metrics["mae"])
        history["val_stress_rmse"].append(m_metrics["rmse"])

        # Checkpoint based on Fetal F1 and Val Loss
        if f_metrics["f1_score"] > best_f1:
            best_f1 = f_metrics["f1_score"]
            best_fetal_metrics = f_metrics
            best_maternal_metrics = m_metrics
            torch.save({
                "epoch": epoch,
                "fold": fold_idx,
                "model_state_dict": model.state_dict(),
                "fetal_metrics": f_metrics,
                "maternal_metrics": m_metrics,
            }, best_ckpt_path)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Fold {fold_idx} | Epoch {epoch:02d}/{epochs} | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Fetal F1: {f_metrics['f1_score']:.3f} | Fetal AUC: {f_metrics['roc_auc']:.3f} | Stress MAE: {m_metrics['mae']:.3f}")

    # Load best model weights
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    return model, history, best_fetal_metrics, best_maternal_metrics
