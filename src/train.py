"""Simple script to train the Linear Regression student performance model."""

import json
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Add local packages directory to sys.path if available
_pkg_dir = Path(__file__).resolve().parent.parent / "packages"
if _pkg_dir.exists() and str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

from src.model import CLASSES, train_linear_model

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")


def main():
    print("=== Training Simple Linear Regression Model ===")
    results = train_linear_model(
        data_path="data/student-mat.csv",
        model_save_path="models/linear_model.joblib",
    )

    print(f"\nAccuracy: {results['accuracy'] * 100:.2f}%")
    print(f"\nModel Coefficients:")
    for feat, coef in results["coefficients"].items():
        print(f"  {feat:10s} : {coef:+.4f}")
    print(f"  Intercept  : {results['intercept']:+.4f}")

    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])

    # Save visual Confusion Matrix plot
    cm = results["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        ax=ax,
        cbar=False,
    )
    ax.set_title("Confusion Matrix (Linear Regression)")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png", dpi=200)
    plt.close()
    print("Saved confusion matrix image to 'models/confusion_matrix.png'.")

    # Save metadata
    meta = {
        "model_type": "LinearRegression",
        "accuracy": round(float(results["accuracy"]), 4),
        "coefficients": results["coefficients"],
        "intercept": results["intercept"],
        "confusion_matrix": results["confusion_matrix"].tolist(),
        "classes": CLASSES,
    }
    with open("models/model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved metadata to 'models/model_metadata.json'.")


if __name__ == "__main__":
    main()
