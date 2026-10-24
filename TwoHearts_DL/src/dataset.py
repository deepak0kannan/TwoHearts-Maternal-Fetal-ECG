"""
TwoHearts Dataset and Segmentation Module.
Handles loading, resampling to 500 Hz, bandpass filtering, window segmentation,
maternal HRV ground-truth derivation from chest lead, and patient-level splitting.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.signal as signal
import torch
from torch.utils.data import DataLoader, Dataset
import wfdb


EXCLUDED_RECORDS = {"NR_13", "NR_14"}  # Exact bit-for-bit duplicates in NIFEA DB
TARGET_FS = 500  # Standardized sampling rate (Hz)
WINDOW_SEC = 10  # Window duration in seconds
OVERLAP_SEC = 5  # Overlap between consecutive windows (seconds)


def filter_abdominal_signals(signals: np.ndarray, fs: float) -> np.ndarray:
    """
    Apply zero-phase Butterworth bandpass filter (0.5 - 45 Hz)
    and per-channel Z-score normalization.
    Input: (channels, samples)
    """
    nyq = 0.5 * fs
    low = 0.5 / nyq
    high = min(45.0 / nyq, 0.99)
    b, a = signal.butter(4, [low, high], btype="bandpass")
    
    filtered = np.zeros_like(signals, dtype=np.float32)
    for ch in range(signals.shape[0]):
        f_sig = signal.filtfilt(b, a, signals[ch])
        std = np.std(f_sig)
        if std > 1e-8:
            filtered[ch] = (f_sig - np.mean(f_sig)) / std
        else:
            filtered[ch] = f_sig - np.mean(f_sig)
    return filtered


def extract_maternal_chest_hrv(chest_signal: np.ndarray, fs: float) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Detect maternal R-peaks from the chest lead and compute clinical HRV metrics.
    Returns:
      peaks: sample indices of detected maternal R-peaks
      metrics: dict with HR, RMSSD, SDNN, Baevsky Stress Index (SI), and Normalized Stress Score
    """
    nyq = 0.5 * fs
    b, a = signal.butter(3, [5.0 / nyq, 30.0 / nyq], btype="bandpass")
    filt = signal.filtfilt(b, a, chest_signal)
    diff = np.diff(filt)
    squared = diff ** 2
    w = max(1, int(0.12 * fs))
    integrated = np.convolve(squared, np.ones(w) / w, mode="same")
    
    min_dist = int(0.35 * fs)  # up to ~170 bpm
    threshold = np.mean(integrated) + 0.3 * np.std(integrated)
    peaks, _ = signal.find_peaks(integrated, distance=min_dist, height=threshold)
    
    if len(peaks) < 3:
        return peaks, {"HR": 75.0, "RMSSD": 35.0, "SDNN": 40.0, "stress_index": 100.0, "normalized_stress": 0.5}
    
    rr_sec = np.diff(peaks) / fs
    rr_ms = rr_sec * 1000.0
    valid = (rr_ms >= 400.0) & (rr_ms <= 1500.0)
    valid_rr = rr_ms[valid]
    
    if len(valid_rr) < 3:
        return peaks, {"HR": 75.0, "RMSSD": 35.0, "SDNN": 40.0, "stress_index": 100.0, "normalized_stress": 0.5}
    
    hr = 60000.0 / np.mean(valid_rr)
    sdnn = np.std(valid_rr, ddof=1)
    diff_rr = np.diff(valid_rr)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))
    
    # Baevsky Stress Index (SI) = AMo / (2 * Mo * MxDMn)
    # Mo: mode of RR in seconds; AMo: % of RR in modal bin; MxDMn: RR_max - RR_min in seconds
    rr_in_s = valid_rr / 1000.0
    bins = np.arange(np.min(rr_in_s), np.max(rr_in_s) + 0.05, 0.05)
    if len(bins) > 1:
        hist, edges = np.histogram(rr_in_s, bins=bins)
        modal_bin_idx = np.argmax(hist)
        mo = 0.5 * (edges[modal_bin_idx] + edges[modal_bin_idx + 1])
        amo = (hist[modal_bin_idx] / len(rr_in_s)) * 100.0
        mxdmn = max(0.05, np.max(rr_in_s) - np.min(rr_in_s))
        si = amo / (2.0 * mo * mxdmn)
    else:
        si = 100.0
    
    # Normalized stress score [0, 1] using clinical sigmoid scaling
    # Normal SI is 50-150. Moderate stress >150, severe >300.
    normalized_stress = float(1.0 / (1.0 + np.exp(-(si - 120.0) / 60.0)))
    
    metrics = {
        "HR": float(hr),
        "RMSSD": float(rmssd),
        "SDNN": float(sdnn),
        "stress_index": float(si),
        "normalized_stress": normalized_stress,
    }
    return peaks, metrics


def resample_signal(sig: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    """Resample signal along axis 1 to target_fs."""
    if orig_fs == target_fs:
        return sig
    num_target = int(round(sig.shape[1] * (target_fs / orig_fs)))
    return signal.resample(sig, num_target, axis=1)


class TwoHeartsDataset(Dataset):
    """PyTorch Dataset for multi-lead abdominal ECG segments."""

    def __init__(self, segments: np.ndarray, labels_arr: np.ndarray, labels_stress: np.ndarray, record_ids: List[str]):
        """
        segments: (N, channels, length) float32
        labels_arr: (N,) float32 (0.0 = Normal, 1.0 = Arrhythmia)
        labels_stress: (N,) float32 (continuous stress score 0.0 - 1.0)
        record_ids: list of str of length N
        """
        self.segments = torch.tensor(segments, dtype=torch.float32)
        self.labels_arr = torch.tensor(labels_arr, dtype=torch.float32).unsqueeze(1)
        self.labels_stress = torch.tensor(labels_stress, dtype=torch.float32).unsqueeze(1)
        self.record_ids = record_ids

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int):
        return self.segments[idx], self.labels_arr[idx], self.labels_stress[idx], self.record_ids[idx]


def prepare_twohearts_data(data_dir: Path, target_fs: int = TARGET_FS,
                           window_sec: int = WINDOW_SEC,
                           overlap_sec: int = OVERLAP_SEC) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """
    Processes all unique recordings in data_dir, extracts standardized 10s windows,
    and returns dataset arrays and patient summary table.
    """
    hea_files = sorted(list(data_dir.glob("*.hea")))
    record_names = [p.stem for p in hea_files if not p.name.startswith(".")]
    record_names = [r for r in record_names if r not in EXCLUDED_RECORDS]

    all_segments = []
    all_arr_labels = []
    all_stress_labels = []
    all_record_ids = []
    summary_rows = []

    window_len = int(window_sec * target_fs)
    step_len = int((window_sec - overlap_sec) * target_fs)

    for r_name in record_names:
        rec_path = str(data_dir / r_name)
        rec = wfdb.rdrecord(rec_path)
        fs = float(rec.fs)
        sig_names = rec.sig_name

        chest_idx = sig_names.index("ECG")
        abdomen_indices = [i for i, name in enumerate(sig_names) if name.startswith("Abdomen")]

        chest_sig = rec.p_signal[:, chest_idx]
        abdomen_sigs = rec.p_signal[:, abdomen_indices[:4]].T  # Standard 4 abdominal channels across all records

        # Resample to target_fs
        abdomen_resamp = resample_signal(abdomen_sigs, fs, target_fs)
        chest_resamp = signal.resample(chest_sig, abdomen_resamp.shape[1])

        # Filter abdominal signals
        abdomen_clean = filter_abdominal_signals(abdomen_resamp, target_fs)

        # Derive maternal ground truth metrics
        _, maternal_metrics = extract_maternal_chest_hrv(chest_resamp, target_fs)

        # Fetal Arrhythmia Label (1 for ARR, 0 for NR)
        arr_label = 1.0 if r_name.startswith("ARR") else 0.0
        stress_label = maternal_metrics["normalized_stress"]

        n_samples = abdomen_clean.shape[1]
        rec_segment_count = 0

        for start in range(0, n_samples - window_len + 1, step_len):
            end = start + window_len
            seg = abdomen_clean[:, start:end]
            all_segments.append(seg)
            all_arr_labels.append(arr_label)
            all_stress_labels.append(stress_label)
            all_record_ids.append(r_name)
            rec_segment_count += 1

        summary_rows.append({
            "record_id": r_name,
            "orig_fs": fs,
            "duration_sec": n_samples / target_fs,
            "arrhythmia_label": int(arr_label),
            "maternal_hr": maternal_metrics["HR"],
            "maternal_rmssd_ms": maternal_metrics["RMSSD"],
            "stress_index": maternal_metrics["stress_index"],
            "normalized_stress": maternal_metrics["normalized_stress"],
            "segment_count": rec_segment_count,
        })

    segments_arr = np.array(all_segments, dtype=np.float32)
    arr_labels_arr = np.array(all_arr_labels, dtype=np.float32)
    stress_labels_arr = np.array(all_stress_labels, dtype=np.float32)
    summary_df = pd.DataFrame(summary_rows)

    return segments_arr, arr_labels_arr, stress_labels_arr, all_record_ids, summary_df


def get_patient_level_folds(summary_df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> List[Tuple[List[str], List[str]]]:
    """
    Splits unique recording IDs into n_splits stratified folds (balanced ARR vs NR).
    Returns list of (train_record_ids, val_record_ids) for each fold.
    """
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    records = summary_df["record_id"].values
    labels = summary_df["arrhythmia_label"].values

    folds = []
    for train_idx, val_idx in skf.split(records, labels):
        train_records = records[train_idx].tolist()
        val_records = records[val_idx].tolist()
        folds.append((train_records, val_records))
    return folds
