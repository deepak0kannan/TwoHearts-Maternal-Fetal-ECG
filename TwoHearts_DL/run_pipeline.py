"""
========================================================================================
TWOHEARTS MASTER PIPELINE RUNNER (run_pipeline.py)
========================================================================================

Purpose & System Architecture:
------------------------------
This script serves as the centralized orchestration engine for the TwoHearts deep
learning framework. It handles the complete lifecycle from raw non-invasive abdominal
ECG signal ingestion to clinical cross-validation, model training, performance
benchmarking, and Explainable AI (1D Grad-CAM) saliency visualization.

The TwoHearts network is a Multi-Task 1D-CNN designed for dual clinical outputs:
  1. Head 1 (Classification): Detects fetal arrhythmias from multi-lead abdominal traces.
  2. Head 2 (Regression): Estimates maternal autonomic wellness (Baevsky Stress Index).

Pipeline Stages:
  [1/4] Dataset Ingestion & Caching: Preprocesses or loads standardized 500 Hz 10s windows.
  [2/4] Patient-Level 5-Fold Cross-Validation: Trains models with zero inter-patient data leakage.
  [3/4] Out-of-Fold Performance Metrics: Calculates clinical ROC-AUC, F1, Specificity, MAE.
  [4/4] Explainable AI (1D Grad-CAM): Visualizes convolutional feature saliency maps.
========================================================================================
"""

import argparse
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# --------------------------------------------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------------------------------------------
# Resolve the directory of this script to allow relative imports of local modules
# (src.dataset, src.model, src.train, src.evaluate, src.explainability)
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Import modular components from the TwoHearts deep learning package
from src.dataset import (
    TwoHeartsDataset,
    get_patient_level_folds,
    prepare_twohearts_data,
)
from src.evaluate import (
    compute_fetal_metrics,
    compute_maternal_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_curves,
)
from src.explainability import GradCAM1D, plot_gradcam_overlay
from src.model import TwoHeartsNet
from src.train import run_fold_training


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments to configure pipeline execution parameters.

    Purpose:
    --------
    Enables flexible parameter tuning via terminal commands without hardcoding
    hyperparameters directly into the source code.

    Returns:
    --------
    argparse.Namespace: Object containing all verified runtime flags and parameters.
    """
    parser = argparse.ArgumentParser(
        description="TwoHearts End-to-End Deep Learning Training & Validation Pipeline"
    )

    # Path to the source PhysioNet Non-Invasive Fetal ECG Arrhythmia Database (NIFEA)
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../non-invasive-fetal-ecg-arrhythmia-database-1.0.0",
        help="Path to folder containing raw PhysioNet NIFEA .hea and .dat records"
    )

    # Destination directory where all artifacts (checkpoints, metrics CSV, plots) are saved
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Root directory for saving checkpoints, publication plots, and evaluation metrics"
    )

    # Number of training epochs per cross-validation fold
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of complete training iterations through the dataset per fold"
    )

    # Mini-batch size passed into the 1D-CNN per optimization step
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of 10-second multi-channel ECG windows processed per batch"
    )

    # Initial learning rate for the AdamW optimizer
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate for backpropagation weight updates"
    )

    # Number of patient-level splits for cross-validation (5-fold is standard for clinical trials)
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of patient-isolated cross-validation folds"
    )

    # Pseudorandom seed for reproducible data splits and weight initializations
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed ensuring deterministic data splitting and model reproducibility"
    )

    # Flag to force re-filtering and re-segmentation of raw signals even if cache exists
    parser.add_argument(
        "--force_reprocess",
        action="store_true",
        help="Force full signal preprocessing pipeline even if cached .npz dataset is present"
    )

    return parser.parse_args()


def main():
    """
    Master pipeline controller function.
    Orchestrates data caching, 5-fold training, performance aggregation, and XAI generation.
    """
    # ----------------------------------------------------------------------------------
    # SNIPPET: Argument Parsing & Deterministic Seed Setup
    # ----------------------------------------------------------------------------------
    # Parse command line flags or defaults
    args = parse_arguments()

    # Seed all random number generators to guarantee reproducible scientific experiments
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Select GPU acceleration if available, otherwise fallback cleanly to multi-core CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 75)
    print("TWOHEARTS: Multi-Task Deep Learning Framework for Maternal-Fetal Screening")
    print(f"Device: {device} | Cross-Validation Folds: {args.n_splits} | Epochs/Fold: {args.epochs}")
    print("=" * 75)

    # ----------------------------------------------------------------------------------
    # SNIPPET: Output Directories Initialization
    # ----------------------------------------------------------------------------------
    # Create structured folders to organize generated model artifacts
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"              # ROC curves, confusion matrices, training losses
    xai_dir = output_dir / "xai"                  # 1D Grad-CAM saliency overlays
    ckpt_dir = output_dir / "checkpoints"          # Saved PyTorch model weights (.pt files)

    for d in [output_dir, plots_dir, xai_dir, ckpt_dir]:
        d.mkdir(parents=True, exist_ok=True)

    cache_file = output_dir / "twohearts_dataset.npz"
    summary_file = output_dir / "patient_summary.csv"
    data_dir = Path(args.data_dir).resolve()

    # ----------------------------------------------------------------------------------
    # STAGE 1: Dataset Preparation & High-Speed Binary Caching
    # ----------------------------------------------------------------------------------
    # Purpose:
    # Processing 24 high-resolution 16-bit WFDB recordings (bandpass filtering 0.5-100 Hz,
    # notch filtering at 50 Hz, resampling from 1000 Hz to 500 Hz, R-peak extraction, and
    # 10s sliding window segmentation) takes computational time.
    # To optimize iteration speed, we serialize the preprocessed arrays into an .npz cache.
    # Subsequent runs load the entire 3,304 windows in < 1 second.
    if cache_file.exists() and summary_file.exists() and not args.force_reprocess:
        print(f"\n[1/4] Loading cached preprocessed dataset from {cache_file.name}...")
        cached = np.load(cache_file, allow_pickle=True)
        segments = cached["segments"]           # Multi-lead abdominal ECG tensor (N, 4, 5000)
        arr_labels = cached["arr_labels"]       # Fetal arrhythmia binary labels (0=Normal, 1=Arrhythmia)
        stress_labels = cached["stress_labels"] # Continuous maternal Baevsky Stress Index values
        record_ids = cached["record_ids"].tolist() # Patient ID corresponding to each window
        summary_df = pd.read_csv(summary_file)  # Per-patient demographic and vitals summary
    else:
        print(f"\n[1/4] Processing raw recordings from {data_dir}...")
        # Execute signal filtering, HRV extraction, and windowing on raw WFDB files
        segments, arr_labels, stress_labels, record_ids, summary_df = prepare_twohearts_data(data_dir)

        # Compress and save to disk for immediate reuse in subsequent runs
        np.savez_compressed(
            cache_file,
            segments=segments,
            arr_labels=arr_labels,
            stress_labels=stress_labels,
            record_ids=np.array(record_ids)
        )
        summary_df.to_csv(summary_file, index=False)
        print(f"  Successfully cached dataset to {cache_file} ({segments.shape[0]} windows)")

    print(f"  Total 10s Windows: {len(segments)} | Segment Shape: {segments.shape[1:]} (4 leads, 5000 samples @ 500 Hz)")
    print(f"  Fetal Class Balance: {int(np.sum(arr_labels == 1))} Arrhythmic vs {int(np.sum(arr_labels == 0))} Normal windows")
    print(f"  Patient Cohort Summary:\n{summary_df[['record_id', 'arrhythmia_label', 'maternal_hr', 'maternal_rmssd_ms', 'segment_count']].to_string(index=False)}")

    # ----------------------------------------------------------------------------------
    # STAGE 2: Patient-Level 5-Fold Stratified Cross-Validation
    # ----------------------------------------------------------------------------------
    # Purpose & Clinical Rationale:
    # A common flaw in published ECG machine learning papers is "window-level splitting",
    # where windows from the same patient appear in both train and validation splits.
    # Because ECG morphologies are highly patient-specific, window-level splits result in
    # artificial overfitting ("memorizing the patient") rather than learning true pathology.
    #
    # TwoHearts strictly enforces Patient-Level Isolation: All windows from any given patient
    # belong EXCLUSIVELY to either the training set or the validation set. This guarantees
    # zero data leakage and reflects true real-world hospital deployment.
    print(f"\n[2/4] Running {args.n_splits}-Fold Patient-Level Stratified Cross-Validation...")
    folds = get_patient_level_folds(summary_df, n_splits=args.n_splits, seed=args.seed)

    # Out-Of-Fold (OOF) collectors to record unbiased predictions across the entire dataset
    oof_fetal_true = []
    oof_fetal_pred = []
    oof_stress_true = []
    oof_stress_pred = []
    fold_metrics = []

    best_overall_model = None
    best_overall_f1 = -1.0

    record_ids_arr = np.array(record_ids)

    # Iterate through each cross-validation fold
    for fold_idx, (train_patients, val_patients) in enumerate(folds, 1):
        print(f"\n--- Fold {fold_idx}/{args.n_splits} ---")
        print(f"  Train cohort ({len(train_patients)} patients): {train_patients}")
        print(f"  Val cohort   ({len(val_patients)} patients): {val_patients}")

        # Boolean masks matching patient IDs to isolate train vs validation windows
        train_mask = np.isin(record_ids_arr, train_patients)
        val_mask = np.isin(record_ids_arr, val_patients)

        # Instantiate PyTorch Dataset objects with isolated segment arrays
        train_ds = TwoHeartsDataset(
            segments[train_mask],
            arr_labels[train_mask],
            stress_labels[train_mask],
            record_ids_arr[train_mask].tolist()
        )
        val_ds = TwoHeartsDataset(
            segments[val_mask],
            arr_labels[val_mask],
            stress_labels[val_mask],
            record_ids_arr[val_mask].tolist()
        )

        # Create DataLoaders for mini-batch stochastic gradient descent
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

        # ------------------------------------------------------------------------------
        # SNIPPET: Multi-Task Model Training per Fold
        # ------------------------------------------------------------------------------
        # Train TwoHeartsNet on this fold using multi-task loss:
        #   Total Loss = L_focal(fetal_pred, fetal_true) + 0.3 * L_smooth_l1(stress_pred, stress_true)
        model, history, f_metrics, m_metrics = run_fold_training(
            fold_idx=fold_idx,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            checkpoint_dir=ckpt_dir
        )

        # Save training and validation loss/accuracy curves for this fold
        plot_training_curves(history, str(plots_dir / f"training_curves_fold_{fold_idx}.png"))

        # ------------------------------------------------------------------------------
        # SNIPPET: Validation Inference & Metric Computation
        # ------------------------------------------------------------------------------
        # Set model to evaluation mode (disables dropout and batch norm running updates)
        model.eval()
        with torch.no_grad():
            fold_val_preds_arr = []
            fold_val_preds_stress = []
            for vx, _, _, _ in val_loader:
                out = model(vx.to(device))
                fold_val_preds_arr.append(out["fetal_prob"].cpu().numpy())
                fold_val_preds_stress.append(out["stress_pred"].cpu().numpy())

            val_preds_arr = np.vstack(fold_val_preds_arr).squeeze()
            val_preds_stress = np.vstack(fold_val_preds_stress).squeeze()

        val_true_arr = arr_labels[val_mask]
        val_true_stress = stress_labels[val_mask]

        # Accumulate out-of-fold predictions for holistic evaluation
        oof_fetal_true.extend(val_true_arr)
        oof_fetal_pred.extend(val_preds_arr)
        oof_stress_true.extend(val_true_stress)
        oof_stress_pred.extend(val_preds_stress)

        # Calculate fold-level classification and regression metrics
        f_eval = compute_fetal_metrics(val_true_arr, val_preds_arr)
        m_eval = compute_maternal_metrics(val_true_stress, val_preds_stress)

        fold_metrics.append({
            "fold": fold_idx,
            "fetal_acc": f_eval["accuracy"],
            "fetal_precision": f_eval["precision"],
            "fetal_recall": f_eval["recall_sensitivity"],
            "fetal_specificity": f_eval["specificity"],
            "fetal_f1": f_eval["f1_score"],
            "fetal_auc": f_eval["roc_auc"],
            "maternal_mae": m_eval["mae"],
            "maternal_rmse": m_eval["rmse"],
            "maternal_pearson_r": m_eval["pearson_r"],
        })

        # Track the best performing checkpoint to use for downstream Explainable AI
        if f_eval["f1_score"] > best_overall_f1:
            best_overall_f1 = f_eval["f1_score"]
            best_overall_model = model

    # ----------------------------------------------------------------------------------
    # STAGE 3: Overall Out-Of-Fold Cross-Validation Performance
    # ----------------------------------------------------------------------------------
    # Purpose:
    # Out-Of-Fold (OOF) evaluation compiles predictions from each patient when they were
    # held out as the unseen test cohort. This produces an unbiased assessment across
    # all 3,304 windows without any data leakage.
    print("\n" + "=" * 75)
    print("[3/4] Overall Out-Of-Fold Cross-Validation Performance:")
    print("=" * 75)

    oof_fetal_true_arr = np.array(oof_fetal_true)
    oof_fetal_pred_arr = np.array(oof_fetal_pred)
    oof_stress_true_arr = np.array(oof_stress_true)
    oof_stress_pred_arr = np.array(oof_stress_pred)

    overall_fetal = compute_fetal_metrics(oof_fetal_true_arr, oof_fetal_pred_arr)
    overall_maternal = compute_maternal_metrics(oof_stress_true_arr, oof_stress_pred_arr)

    # Create summary metrics table across all 5 folds + aggregate Mean (SD)
    metrics_df = pd.DataFrame(fold_metrics)
    summary_row = {
        "fold": "Mean (SD)",
        "fetal_acc": f"{metrics_df['fetal_acc'].mean():.3f} ({metrics_df['fetal_acc'].std():.3f})",
        "fetal_precision": f"{metrics_df['fetal_precision'].mean():.3f} ({metrics_df['fetal_precision'].std():.3f})",
        "fetal_recall": f"{metrics_df['fetal_recall'].mean():.3f} ({metrics_df['fetal_recall'].std():.3f})",
        "fetal_specificity": f"{metrics_df['fetal_specificity'].mean():.3f} ({metrics_df['fetal_specificity'].std():.3f})",
        "fetal_f1": f"{metrics_df['fetal_f1'].mean():.3f} ({metrics_df['fetal_f1'].std():.3f})",
        "fetal_auc": f"{metrics_df['fetal_auc'].mean():.3f} ({metrics_df['fetal_auc'].std():.3f})",
        "maternal_mae": f"{metrics_df['maternal_mae'].mean():.3f} ({metrics_df['maternal_mae'].std():.3f})",
        "maternal_rmse": f"{metrics_df['maternal_rmse'].mean():.3f} ({metrics_df['maternal_rmse'].std():.3f})",
        "maternal_pearson_r": f"{metrics_df['maternal_pearson_r'].mean():.3f} ({metrics_df['maternal_pearson_r'].std():.3f})",
    }
    metrics_df_full = pd.concat([metrics_df, pd.DataFrame([summary_row])], ignore_index=True)
    metrics_df_full.to_csv(output_dir / "cv_performance_metrics.csv", index=False)

    print("\nFold-by-Fold Performance Table:")
    print(metrics_df_full.to_string(index=False))

    print(f"\nOverall Out-of-Fold Fetal Arrhythmia ROC-AUC: {overall_fetal['roc_auc']:.4f}")
    print(f"Overall Out-of-Fold Fetal Arrhythmia F1-Score: {overall_fetal['f1_score']:.4f}")
    print(f"Overall Out-of-Fold Fetal Arrhythmia Accuracy: {overall_fetal['accuracy'] * 100:.2f}%")
    print(f"Overall Out-of-Fold Maternal Stress MAE:       {overall_maternal['mae']:.4f}")
    print(f"Overall Out-of-Fold Maternal Stress Pearson r: {overall_maternal['pearson_r']:.4f}")

    # ----------------------------------------------------------------------------------
    # SNIPPET: Publication-Grade Figures Generation
    # ----------------------------------------------------------------------------------
    # Generate high-resolution plots for research review presentations and papers:
    # 1. Receiver Operating Characteristic (ROC) Curve with AUC score
    plot_roc_curve(
        oof_fetal_true_arr,
        oof_fetal_pred_arr,
        overall_fetal["roc_auc"],
        str(plots_dir / "overall_roc_curve.png")
    )
    # 2. Confusion Matrix across all 3,304 windows
    plot_confusion_matrix(
        oof_fetal_true_arr,
        oof_fetal_pred_arr,
        str(plots_dir / "overall_confusion_matrix.png")
    )

    print(f"\n  Saved Publication ROC Curve to:        {plots_dir / 'overall_roc_curve.png'}")
    print(f"  Saved Publication Confusion Matrix to: {plots_dir / 'overall_confusion_matrix.png'}")
    print(f"  Saved Formal Metrics Table to:        {output_dir / 'cv_performance_metrics.csv'}")

    # ----------------------------------------------------------------------------------
    # STAGE 4: Explainable AI via 1D Grad-CAM (Gradient-Weighted Class Activation Mapping)
    # ----------------------------------------------------------------------------------
    # Purpose & Clinical Rationale:
    # Deep neural networks are frequently criticized as "black boxes" in medical settings.
    # 1D Grad-CAM addresses this by tracking gradient flow from target output heads back
    # into the final convolutional layer (Layer 4).
    #
    # This highlights exact time intervals that drove the decision:
    #  - Fetal Head: Highlights rapid ectopic spikes and irregular RR intervals.
    #  - Maternal Head: Locks onto prominent maternal R-wave complexes for HRV estimation.
    # This provides obstetricians with physiological evidence for automated triage.
    print("\n[4/4] Generating 1D Grad-CAM Explainable AI Visualizations...")
    xai_engine = GradCAM1D(best_overall_model)

    # Locate representative sample segments (one confirmed Arrhythmia, one confirmed Normal)
    arr_idx = np.where(arr_labels == 1)[0][10]
    nr_idx = np.where(arr_labels == 0)[0][10]

    for sample_idx, prefix in [(arr_idx, "Arrhythmia_Case"), (nr_idx, "Normal_Rhythm_Case")]:
        sample_x = torch.tensor(segments[sample_idx:sample_idx+1], dtype=torch.float32).to(device)
        sample_np = segments[sample_idx]
        rec_id = record_ids[sample_idx]

        # Generate Grad-CAM saliency for the Fetal Arrhythmia classification head
        cam_fetal = xai_engine.generate_cam(sample_x, target_head="fetal")
        plot_gradcam_overlay(
            sample_np,
            cam_fetal,
            fs=500,
            title=f"TwoHearts 1D Grad-CAM Saliency: Fetal Arrhythmia Head ({prefix} - {rec_id})",
            save_path=str(xai_dir / f"gradcam_fetal_{prefix.lower()}.png")
        )

        # Generate Grad-CAM saliency for the Maternal Wellness regression head
        cam_maternal = xai_engine.generate_cam(sample_x, target_head="maternal")
        plot_gradcam_overlay(
            sample_np,
            cam_maternal,
            fs=500,
            title=f"TwoHearts 1D Grad-CAM Saliency: Maternal Stress Head ({prefix} - {rec_id})",
            save_path=str(xai_dir / f"gradcam_maternal_{prefix.lower()}.png")
        )

    print(f"  Saved Saliency Heatmap Figures to:     {xai_dir}")
    print("\n" + "=" * 75)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 75)


# --------------------------------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
