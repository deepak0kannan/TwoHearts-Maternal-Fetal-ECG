import unittest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "TwoHearts_DL"))

from src.model import TwoHeartsNet

class TestModel(unittest.TestCase):
    def test_forward_pass(self):
        batch_size = 4
        channels = 4
        length = 5000
        x = torch.randn(batch_size, channels, length)

        model = TwoHeartsNet(in_channels=4, base_channels=32)
        model.eval()

        with torch.no_grad():
            out = model(x)

        self.assertIn("fetal_logits", out)
        self.assertIn("fetal_prob", out)
        self.assertIn("stress_pred", out)
        self.assertIn("feature_map", out)

        self.assertEqual(out["fetal_logits"].shape, (batch_size, 1))
        self.assertEqual(out["fetal_prob"].shape, (batch_size, 1))
        self.assertEqual(out["stress_pred"].shape, (batch_size, 1))

        # Probabilities should be strictly in [0, 1]
        self.assertTrue(torch.all(out["fetal_prob"] >= 0.0))
        self.assertTrue(torch.all(out["fetal_prob"] <= 1.0))
        self.assertTrue(torch.all(out["stress_pred"] >= 0.0))
        self.assertTrue(torch.all(out["stress_pred"] <= 1.0))

    def test_backprop(self):
        x = torch.randn(2, 4, 5000)
        y_arr = torch.tensor([[1.0], [0.0]])
        y_stress = torch.tensor([[0.7], [0.3]])

        model = TwoHeartsNet(in_channels=4, base_channels=16)
        model.train()

        out = model(x)
        loss_fetal = torch.nn.functional.binary_cross_entropy_with_logits(out["fetal_logits"], y_arr)
        loss_stress = torch.nn.functional.smooth_l1_loss(out["stress_pred"], y_stress)
        total_loss = loss_fetal + 0.5 * loss_stress

        total_loss.backward()
        # Verify gradients exist
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)

if __name__ == "__main__":
    unittest.main()
