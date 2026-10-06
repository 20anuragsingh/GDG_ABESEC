"""Unit tests for Simple Linear Regression Student Performance Model."""

import unittest
from pathlib import Path
import pandas as pd

import sys
_root_dir = Path(__file__).resolve().parent.parent
_pkg_dir = _root_dir / "packages"
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))
if _pkg_dir.exists() and str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

from src.model import (
    CLASSES,
    predict_batch,
    predict_student,
    score_to_class,
    train_linear_model,
)


class TestSimpleLinearModel(unittest.TestCase):

    def test_01_training(self):
        """Test training of the Linear Regression model."""
        res = train_linear_model()
        self.assertIn("accuracy", res)
        self.assertGreater(res["accuracy"], 0.70)
        self.assertIn("confusion_matrix", res)
        self.assertEqual(res["confusion_matrix"].shape, (3, 3))

    def test_02_single_prediction(self):
        """Test prediction for single student."""
        student = {"studytime": 3, "failures": 0, "absences": 2, "G1": 16, "G2": 17}
        res = predict_student(student)

        self.assertIn("predicted_score", res)
        self.assertIn("predicted_class", res)
        self.assertEqual(res["predicted_class"], "High")
        self.assertGreaterEqual(res["confidence"], 50.0)
        self.assertIn("risk_info", res)
        self.assertIn("top_factors", res)
        self.assertIn("suggestions", res)

    def test_03_at_risk_flagging(self):
        """Test that struggling student is flagged at-risk."""
        struggling = {"studytime": 1, "failures": 2, "absences": 15, "G1": 6, "G2": 7}
        res = predict_student(struggling)
        self.assertEqual(res["predicted_class"], "Low")
        self.assertTrue(res["risk_info"]["is_at_risk"])
        self.assertIn("⚠️ AT RISK", res["risk_info"]["status"])

    def test_04_batch_prediction(self):
        """Test batch prediction with sample students CSV."""
        sample_path = Path("data/sample_students.csv")
        self.assertTrue(sample_path.exists())
        df = pd.read_csv(sample_path)
        out_df = predict_batch(df)

        self.assertEqual(len(out_df), len(df))
        self.assertIn("Predicted_Class", out_df.columns)
        self.assertIn("Predicted_Score", out_df.columns)
        self.assertIn("Confidence_%", out_df.columns)
        self.assertIn("Risk_Status", out_df.columns)


if __name__ == "__main__":
    unittest.main()
