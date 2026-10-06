"""Inference and batch prediction engine for student performance."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import pandas as pd

# Add local packages directory to sys.path if available
_pkg_dir = Path(__file__).resolve().parent.parent / "packages"
if _pkg_dir.exists() and str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

from src.data_loader import (
    ALL_FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    FEATURE_METADATA,
    NUMERICAL_FEATURES,
    get_default_student,
)
from src.explainer import (
    assess_student_risk,
    compute_feature_contributions,
    generate_improvement_suggestions,
)


class StudentPerformancePredictor:
    """Predictor class encapsulating model inference, explainability, and risk flags."""

    def __init__(
        self,
        model_path: str = "models/best_model.joblib",
        metadata_path: str = "models/model_metadata.json",
    ):
        model_file = Path(model_path)
        metadata_file = Path(metadata_path)

        if not model_file.exists():
            raise FileNotFoundError(f"Model artifact not found at '{model_path}'. Run src/train.py first.")

        self.pipeline = joblib.load(model_file)
        self.classes = list(self.pipeline.classes_)

        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

    def _prepare_student_df(self, student_input: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
        """Validate, cast, and ensure all required features are present."""
        defaults = get_default_student()

        if isinstance(student_input, dict):
            row = defaults.copy()
            row.update(student_input)
            df = pd.DataFrame([row])
        elif isinstance(student_input, pd.DataFrame):
            df = student_input.copy()
            for col, val in defaults.items():
                if col not in df.columns:
                    df[col] = val
        else:
            raise TypeError("student_input must be a dict or pd.DataFrame")

        # Cast column types
        for col in NUMERICAL_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(defaults[col])
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.strip('"\'')

        return df[ALL_FEATURE_COLUMNS]

    def predict_single(self, student_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict performance tier for an individual student.

        Returns comprehensive diagnostic dictionary including:
        - predicted_class: 'High', 'Average', or 'Low'
        - confidence: confidence percentage (float)
        - probabilities: full probability distribution
        - risk_assessment: risk level, score, and flags
        - top_factors: positive and negative contributing factors
        - suggestions: prioritized improvement plan
        """
        student_df = self._prepare_student_df(student_dict)

        # Explainability & probabilities
        explanation = compute_feature_contributions(self.pipeline, student_df)
        predicted_class = explanation["predicted_class"]
        confidence = explanation["confidence"]
        probabilities = explanation["probabilities"]

        # Risk assessment
        risk = assess_student_risk(student_dict, predicted_class, probabilities)

        # Prescriptive suggestions
        suggestions = generate_improvement_suggestions(student_dict, predicted_class, confidence)

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "confidence_percentage": round(confidence * 100.0, 2),
            "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
            "risk_assessment": risk,
            "top_contributing_factors": {
                "positive": explanation["top_positive"],
                "negative": explanation["top_negative"],
                "all": explanation["all_contributions"][:10],
            },
            "suggestions": suggestions,
            "input_features": student_dict,
        }

    def predict_batch(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform batch predictions for multiple students from a DataFrame.

        Appends prediction, confidence, risk score, and primary recommendation.
        """
        prepared_df = self._prepare_student_df(input_df)

        predictions = self.pipeline.predict(prepared_df)
        probabilities = self.pipeline.predict_proba(prepared_df)
        classes = list(self.pipeline.classes_)

        output_df = input_df.copy()
        output_df["Predicted_Performance"] = predictions
        output_df["Confidence_Score"] = [f"{max(p) * 100:.1f}%" for p in probabilities]

        # Probabilities per class
        for i, cls in enumerate(classes):
            output_df[f"Prob_{cls}"] = [round(float(p[i]), 3) for p in probabilities]

        risk_levels = []
        risk_scores = []
        primary_risks = []
        key_suggestions = []

        for idx, row in prepared_df.iterrows():
            row_dict = row.to_dict()
            pred_cls = predictions[idx]
            probs_dict = {cls: float(probabilities[idx][i]) for i, cls in enumerate(classes)}

            risk = assess_student_risk(row_dict, pred_cls, probs_dict)
            sugg = generate_improvement_suggestions(row_dict, pred_cls, max(probabilities[idx]))

            risk_levels.append(risk["risk_level"])
            risk_scores.append(risk["risk_score"])
            primary_risks.append(risk["triggers"][0] if risk["triggers"] else "None")
            key_suggestions.append(sugg[0]["title"] if sugg else "Maintain Routine")

        output_df["Risk_Level"] = risk_levels
        output_df["Risk_Score"] = risk_scores
        output_df["Primary_Risk_Trigger"] = primary_risks
        output_df["Key_Improvement_Suggestion"] = key_suggestions

        return output_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run student performance predictions.")
    parser.add_argument("--input_csv", type=str, default=None, help="Path to input CSV for batch prediction.")
    parser.add_argument("--output_csv", type=str, default="batch_predictions.csv", help="Output path for batch results.")
    args = parser.parse_args()

    predictor = StudentPerformancePredictor()

    if args.input_csv:
        print(f"Loading students from '{args.input_csv}'...")
        # Automatically detect delimiter
        with open(args.input_csv, "r") as f:
            first_line = f.readline()
            sep = ";" if ";" in first_line else ","
        df_in = pd.read_csv(args.input_csv, sep=sep)
        df_out = predictor.predict_batch(df_in)
        df_out.to_csv(args.output_csv, index=False)
        print(f"Saved batch predictions for {len(df_out)} students to '{args.output_csv}'.")
    else:
        # Run demo prediction
        sample = get_default_student()
        sample.update({"studytime": 1, "failures": 2, "absences": 12, "G1": 7, "G2": 8})
        res = predictor.predict_single(sample)
        print("\nSingle Prediction Result:")
        print(f"  Class:      {res['predicted_class']}")
        print(f"  Confidence: {res['confidence_percentage']}%")
        print(f"  Risk Level: {res['risk_assessment']['risk_level']} (Score: {res['risk_assessment']['risk_score']})")
        print(f"  Risk Flags: {res['risk_assessment']['triggers']}")
        print("\nTop Contributing Factors:")
        for factor in res["top_contributing_factors"]["all"][:5]:
            print(f"  - {factor['label']}: impact {factor['impact_percentage']:+.1f}% ({factor['direction']})")
        print("\nTop Suggestions:")
        for s in res["suggestions"][:2]:
            print(f"  - [{s['category']}] {s['title']}: {s['description']}")

