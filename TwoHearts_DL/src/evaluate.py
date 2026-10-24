"""
TwoHearts Evaluation & Metrics Module.
Calculates comprehensive performance metrics for both tasks and generates
publication-grade figures (ROC curve, Confusion Matrix, Loss Curves).
"""

from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_fetal_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Computes clinical diagnostic metrics for Fetal Arrhythmia classification."""
    y_pred = (y_prob >= threshold).astype(int)
    y_true_int = y_true.astype(int)

    acc = accuracy_score(y_true_int, y_pred)
    prec = precision_score(y_true_int, y_pred, zero_division=0)
    rec = recall_score(y_true_int, y_pred, zero_division=0)
    f1 = f1_score(y_true_int, y_pred, zero_division=0)

    # Specificity = TN / (TN + FP)
    tn, fp, fn, tp = confusion_matrix(y_true_int, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        auc = roc_auc_score(y_true_int, y_prob)
    except Exception:
        auc = float("nan")

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall_sensitivity": float(rec),
        "specificity": float(spec),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def compute_maternal_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes regression metrics for Maternal Autonomic Stress scoring."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Pearson r
    if np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8:
        corr = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        corr = 0.0

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2_score": float(r2),
        "pearson_r": float(corr),
    }


def plot_training_curves(history: Dict[str, List[float]], save_path: str):
    """Plots training and validation loss curves over epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Total Multi-Task Loss
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", linewidth=1.5, markersize=4)
    axes[0].plot(epochs, history["val_loss"], "r--s", label="Val Loss", linewidth=1.5, markersize=4)
    axes[0].set_title("Total Multi-Task Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Fetal Arrhythmia F1
    axes[1].plot(epochs, history["val_fetal_f1"], "g-^", label="Val Fetal F1", linewidth=1.5, markersize=4)
    axes[1].set_title("Fetal Arrhythmia Val F1-Score", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 Score")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    # Maternal Stress MAE
    axes[2].plot(epochs, history["val_stress_mae"], "m-d", label="Val Stress MAE", linewidth=1.5, markersize=4)
    axes[2].set_title("Maternal Stress Val MAE", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Mean Absolute Error")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, auc_score: float, save_path: str):
    """Generates ROC-AUC curve with chance diagonal."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2ca02c", linewidth=2.5, label=f"TwoHearts 1D-CNN (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Chance Level")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    plt.title("ROC Curve: Fetal Arrhythmia Detection", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_prob: np.ndarray, save_path: str, threshold: float = 0.5):
    """Plots normalized and raw confusion matrix."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im)

    classes = ["Normal Rhythm", "Arrhythmia"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)

    # Text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12, fontweight="bold")

    ax.set_ylabel("True Ground Truth Label", fontsize=11, fontweight="bold")
    ax.set_xlabel("TwoHearts Predicted Label", fontsize=11, fontweight="bold")
    ax.set_title("Confusion Matrix: Fetal Arrhythmia", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
