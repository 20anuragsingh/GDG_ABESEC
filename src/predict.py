"""Simple CLI prediction script for student performance."""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add local packages directory to sys.path if available
_pkg_dir = Path(__file__).resolve().parent.parent / "packages"
if _pkg_dir.exists() and str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

from src.model import predict_batch, predict_student


def main():
    parser = argparse.ArgumentParser(description="Predict student performance using Linear Regression.")
    parser.add_argument("--input_csv", type=str, default=None, help="Input CSV for batch predictions.")
    parser.add_argument("--output_csv", type=str, default="batch_predictions.csv", help="Output CSV path.")
    args = parser.parse_args()

    if args.input_csv:
        df_in = pd.read_csv(args.input_csv)
        df_out = predict_batch(df_in)
        df_out.to_csv(args.output_csv, index=False)
        print(f"Batch predictions saved to '{args.output_csv}'.")
        print(df_out[["Predicted_Score", "Predicted_Class", "Confidence_%", "Risk_Status"]].head())
    else:
        # Demo single prediction
        demo_student = {
            "studytime": 1,
            "failures": 2,
            "absences": 12,
            "G1": 7,
            "G2": 8,
        }
        res = predict_student(demo_student)
        print("\n=== Single Student Prediction (Linear Regression) ===")
        print(f"Predicted Score : {res['predicted_score']} / 20")
        print(f"Predicted Class : {res['predicted_class']}")
        print(f"Confidence      : {res['confidence']}%")
        print(f"Risk Status     : {res['risk_info']['status']}")
        print("Risk Reasons    :", ", ".join(res['risk_info']['reasons']))
        print("\nFactor Contributions (Coefficients x Values):")
        for f in res["top_factors"]:
            print(f"  {f['feature']:10s} (value={f['value']}) -> Impact: {f['impact']:+5.2f} [{f['effect']}]")
        print("\nSuggestions:")
        for s in res["suggestions"]:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
