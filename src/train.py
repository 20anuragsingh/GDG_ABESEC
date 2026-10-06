"""Model training, benchmarking, evaluation, and serialization pipeline."""

import argparse
import json
import os
import sys
from pathlib import Path

# Add local packages directory to sys.path if available
_pkg_dir = Path(__file__).resolve().parent.parent / "packages"
if _pkg_dir.exists() and str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from src.data_loader import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PERFORMANCE_LABELS,
    load_dataset,
    prepare_features_and_target,
)

# Ensure Matplotlib config uses writable directory
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")


def build_preprocessor(cat_cols: list, num_cols: list) -> ColumnTransformer:
    """Build scikit-learn ColumnTransformer for numerical and categorical features."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ]
    )


def benchmark_models(X: pd.DataFrame, y: pd.Series, preprocessor: ColumnTransformer) -> dict:
    """Run 5-fold cross-validation on candidates and return summary results."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidate_models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=2, class_weight="balanced", random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Support Vector Machine": SVC(
            kernel="rbf", class_weight="balanced", random_state=42
        ),
    }

    results = {}
    print("\n--- 5-Fold Cross-Validation Model Benchmark ---")
    for name, clf in candidate_models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        scores = cross_validate(
            pipe,
            X,
            y,
            cv=cv,
            scoring=["accuracy", "f1_weighted", "f1_macro"],
            n_jobs=1,
        )
        acc_mean = float(scores["test_accuracy"].mean())
        acc_std = float(scores["test_accuracy"].std())
        f1_w = float(scores["test_f1_weighted"].mean())
        f1_m = float(scores["test_f1_macro"].mean())

        results[name] = {
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "f1_weighted": f1_w,
            "f1_macro": f1_m,
        }
        print(f"  {name:25s} | Accuracy: {acc_mean:.4f} (±{acc_std:.4f}) | F1-Weighted: {f1_w:.4f} | F1-Macro: {f1_m:.4f}")

    return results


def train_and_evaluate(
    data_dir: str = "data",
    source: str = "combined",
    output_dir: str = "models",
    random_state: int = 42,
) -> dict:
    """Train the best model, evaluate on holdout test set, and persist artifacts."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/5] Loading {source} student dataset...")
    df = load_dataset(data_dir=data_dir, source=source)
    X, y = prepare_features_and_target(df)
    print(f"Loaded {len(df)} student records with {X.shape[1]} input features.")
    print(f"Target distribution:\n{y.value_counts(sort=False).to_dict()}")

    # Feature column identification
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    num_cols = [c for c in NUMERICAL_FEATURES if c in X.columns]
    preprocessor = build_preprocessor(cat_cols, num_cols)

    # Benchmark candidate models
    print("\n[2/5] Benchmarking candidate architectures...")
    cv_benchmarks = benchmark_models(X, y, preprocessor)

    # Train / Test split (80/20 stratified)
    print("\n[3/5] Splitting data into stratified Train (80%) and Test (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )

    # Best Model: Optimized Random Forest Classifier with balanced weights
    best_classifier = RandomForestClassifier(
        n_estimators=250,
        max_depth=14,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    best_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", best_classifier),
    ])

    print("\n[4/5] Training final production pipeline...")
    best_pipeline.fit(X_train, y_train)

    # Test set evaluation
    y_pred = best_pipeline.predict(X_test)
    y_prob = best_pipeline.predict_proba(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    f1_w = float(f1_score(y_test, y_pred, average="weighted"))
    f1_m = float(f1_score(y_test, y_pred, average="macro"))
    prec_w = float(precision_score(y_test, y_pred, average="weighted"))
    rec_w = float(recall_score(y_test, y_pred, average="weighted"))

    cm = confusion_matrix(y_test, y_pred, labels=PERFORMANCE_LABELS)
    report = classification_report(y_test, y_pred, labels=PERFORMANCE_LABELS, output_dict=True)

    print("\n[5/5] Test Set Evaluation Results:")
    print(f"  Accuracy:           {acc * 100:.2f}%")
    print(f"  F1 Score (Weighted): {f1_w:.4f}")
    print(f"  F1 Score (Macro):    {f1_m:.4f}")
    print(f"  Precision:          {prec_w:.4f}")
    print(f"  Recall:             {rec_w:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, labels=PERFORMANCE_LABELS))

    # Compute Global Feature Importances
    fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
    cat_feature_names = fitted_preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)
    all_feature_names = num_cols + list(cat_feature_names)
    raw_importances = best_pipeline.named_steps["classifier"].feature_importances_

    feature_importances = [
        {"feature": name, "importance": float(imp)}
        for name, imp in sorted(zip(all_feature_names, raw_importances), key=lambda x: x[1], reverse=True)
    ]

    # Save Confusion Matrix Heatmap Image
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=PERFORMANCE_LABELS,
        yticklabels=PERFORMANCE_LABELS,
        ax=ax,
        cbar=True,
        annot_kws={"size": 13, "weight": "bold"},
    )
    ax.set_title("Test Confusion Matrix (Student Performance)", fontsize=13, pad=12, weight="bold")
    ax.set_xlabel("Predicted Class", fontsize=11, labelpad=8)
    ax.set_ylabel("True Class", fontsize=11, labelpad=8)
    plt.tight_layout()
    cm_plot_path = out_path / "confusion_matrix.png"
    fig.savefig(cm_plot_path, dpi=300)
    plt.close(fig)
    print(f"Confusion matrix plot saved to: {cm_plot_path}")

    # Serialize Model Pipeline
    model_file = out_path / "best_model.joblib"
    joblib.dump(best_pipeline, model_file)
    print(f"Serialized model pipeline saved to: {model_file}")

    # Metadata dictionary
    metadata = {
        "model_type": "RandomForestClassifier",
        "dataset_source": source,
        "sample_count": len(df),
        "test_size": 0.20,
        "labels": PERFORMANCE_LABELS,
        "classes": list(best_pipeline.classes_),
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "metrics": {
            "test_accuracy": acc,
            "test_f1_weighted": f1_w,
            "test_f1_macro": f1_m,
            "test_precision_weighted": prec_w,
            "test_recall_weighted": rec_w,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        },
        "cv_benchmarks": cv_benchmarks,
        "top_features": feature_importances[:20],
        "all_feature_importances": feature_importances,
    }

    metadata_file = out_path / "model_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata saved to: {metadata_file}")

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train student performance classifier.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing CSV files.")
    parser.add_argument("--source", type=str, default="combined", choices=["combined", "math", "portuguese"])
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save model artifacts.")
    args = parser.parse_args()

    train_and_evaluate(
        data_dir=args.data_dir,
        source=args.source,
        output_dir=args.output_dir,
    )

