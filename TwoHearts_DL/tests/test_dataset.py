import unittest
import numpy as np
import torch
from pathlib import Path
import sys

# Ensure TwoHearts_DL root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "TwoHearts_DL"))

from src.dataset import filter_abdominal_signals, extract_maternal_chest_hrv, TwoHeartsDataset

class TestDataset(unittest.TestCase):
    def test_filter_abdominal_signals(self):
        fs = 500
        t = np.linspace(0, 10, fs * 10)
        # 5 channels with artificial sine + noise
        sig = np.array([np.sin(2 * np.pi * 2.0 * t) + 0.1 * np.random.randn(len(t)) for _ in range(5)])
        filtered = filter_abdominal_signals(sig, fs)
        self.assertEqual(filtered.shape, sig.shape)
        self.assertTrue(np.all(np.isfinite(filtered)))
        # Check near-zero mean
        for ch in range(5):
            self.assertAlmostEqual(filtered[ch].mean(), 0.0, places=2)

    def test_hrv_extraction(self):
        fs = 500
        t = np.linspace(0, 60, fs * 60)
        # Create pulses at 75 bpm (every 0.8s)
        chest = np.zeros_like(t)
        for beat_t in np.arange(0.5, 59.5, 0.8):
            idx = int(beat_t * fs)
            chest[idx-5:idx+5] = 1.0
        peaks, metrics = extract_maternal_chest_hrv(chest, fs)
        self.assertGreater(len(peaks), 50)
        self.assertAlmostEqual(metrics["HR"], 75.0, delta=5.0)
        self.assertIn("normalized_stress", metrics)

    def test_twohearts_dataset(self):
        segs = np.random.randn(20, 5, 5000).astype(np.float32)
        y_arr = np.random.randint(0, 2, 20).astype(np.float32)
        y_stress = np.random.rand(20).astype(np.float32)
        rec_ids = [f"REC_{i}" for i in range(20)]
        ds = TwoHeartsDataset(segs, y_arr, y_stress, rec_ids)
        self.assertEqual(len(ds), 20)
        x, ya, ys, rid = ds[0]
        self.assertEqual(x.shape, (5, 5000))
        self.assertEqual(ya.shape, (1,))
        self.assertEqual(ys.shape, (1,))
        self.assertEqual(rid, "REC_0")

if __name__ == "__main__":
    unittest.main()
