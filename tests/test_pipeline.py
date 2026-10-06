"""Unit and integration test suite for Student Performance ML pipeline."""

import json
import os
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

# Add local packages directory to sys.path if available
_pkg_dir = Path(__file__).resolve().parent.parent / "packages"
if _pkg_dir.exists() and str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

from src.data_loader import (
    ALL_FEATURE_COLUMNS,
    PERFORMANCE_LABELS,
    get_default_student,
    load_dataset,
    prepare_features_and_target,
)
from src.explainer import (
    assess_student_risk,
    compute_feature_contributions,
    generate_improvement_suggestions,
)
from src.predict import StudentPerformancePredictor


class TestStudentPerformancePipeline(unittest.TestCase):
    """Test suite covering data loading, training artifacts, inference, and explainability."""

    @classmethod
    def setUpClass(cls):
        """Initialize predictor for tests."""
        cls.predictor = StudentPerformancePredictor()

    def test_01_data_loading_and_target_preparation(self):
        """Test dataset loading and 3-tier target discretization."""
        df = load_dataset(data_dir="data", source="combined")
        self.assertGreater(len(df), 1000)
        self.assertIn("G3", df.columns)

        X, y = prepare_features_and_target(df)
        self.assertEqual(len(X), len(y))
        self.assertNotIn("G3", X.columns)
        self.assertNotIn("performance", X.columns)

        unique_labels = set(y.unique())
        self.assertTrue(unique_labels.issubset(set(PERFORMANCE_LABELS)))
        self.assertEqual(len(unique_labels), 3)

    def test_02_single_student_prediction(self):
        """Test single student inference, confidence scoring, and output structure."""
        student = get_default_student()
        result = self.predictor.predict_single(student)

        self.assertIn("predicted_class", result)
        self.assertIn(result["predicted_class"], PERFORMANCE_LABELS)

        self.assertIn("confidence", result)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

        self.assertIn("probabilities", result)
        self.assertEqual(len(result["probabilities"]), 3)
        prob_sum = sum(result["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=2)

    def test_03_high_risk_flagging(self):
        """Test that struggling student attributes trigger High Risk status."""
        struggling_student = get_default_student()
        struggling_student.update({
            "failures": 3,
            "absences": 18,
            "studytime": 1,
            "G1": 6,
            "G2": 7,
        })

        result = self.predictor.predict_single(struggling_student)
        risk = result["risk_assessment"]

        self.assertEqual(risk["risk_level"], "High Risk")
        self.assertGreaterEqual(risk["risk_score"], 50)
        self.assertGreater(len(risk["triggers"]), 0)
        self.assertEqual(result["predicted_class"], "Low")

    def test_04_low_risk_high_achiever(self):
        """Test that top-performing student attributes yield High performance and Low risk."""
        top_student = get_default_student()
        top_student.update({
            "failures": 0,
            "absences": 1,
            "studytime": 4,
            "G1": 17,
            "G2": 18,
            "higher": "yes",
        })

        result = self.predictor.predict_single(top_student)
        risk = result["risk_assessment"]

        self.assertEqual(result["predicted_class"], "High")
        self.assertEqual(risk["risk_level"], "Low Risk")
        self.assertLess(risk["risk_score"], 25)

    def test_05_feature_contributions_explainability(self):
        """Test feature contribution calculation for explainability."""
        student = get_default_student()
        student["G2"] = 18  # Strong positive indicator
        student_df = self.predictor._prepare_student_df(student)

        explanation = compute_feature_contributions(self.predictor.pipeline, student_df)
        self.assertIn("all_contributions", explanation)
        self.assertGreater(len(explanation["all_contributions"]), 0)

        # Check structure of contribution items
        first_item = explanation["all_contributions"][0]
        self.assertIn("feature", first_item)
        self.assertIn("delta_probability", first_item)
        self.assertIn("direction", first_item)

    def test_06_prescriptive_suggestions_generation(self):
        """Test that prescriptive recommendations address specific student weaknesses."""
        weak_student = get_default_student()
        weak_student.update({
            "studytime": 1,
            "absences": 12,
            "failures": 1,
        })

        suggestions = generate_improvement_suggestions(weak_student, "Low", 0.85)
        self.assertGreater(len(suggestions), 0)

        categories = [s["category"] for s in suggestions]
        self.assertIn("Study Habits", categories)
        self.assertIn("Attendance", categories)
        self.assertIn("Academic Support", categories)

    def test_07_batch_csv_prediction(self):
        """Test batch processing from sample students CSV."""
        sample_csv = Path("data/sample_students.csv")
        self.assertTrue(sample_csv.exists())

        input_df = pd.read_csv(sample_csv)
        output_df = self.predictor.predict_batch(input_df)

        self.assertEqual(len(output_df), len(input_df))
        self.assertIn("Predicted_Performance", output_df.columns)
        self.assertIn("Confidence_Score", output_df.columns)
        self.assertIn("Risk_Level", output_df.columns)
        self.assertIn("Key_Improvement_Suggestion", output_df.columns)


if __name__ == "__main__":
    unittest.main()
