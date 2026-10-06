"""Simple linear regression for predicting student grades."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

FEATURES = ["studytime", "failures", "absences", "G1", "G2"]
TARGET = "G3"
CLASSES = ["Low", "Average", "High"]


def score_to_class(score):
    return "High" if score >= 15 else "Average" if score >= 10 else "Low"


def calculate_confidence(score, performance_class):
    edge = min(score - 10, 15 - score) if performance_class == "Average" else 0
    return round(float(np.clip(60 + max(0, edge) * 14, 55, 95)), 1)


def check_at_risk(student, performance_class):
    reasons = []
    if performance_class == "Low": reasons.append("Predicted grade is below 10/20")
    if int(student.get("failures", 0)) > 0: reasons.append("Past class failures")
    if int(student.get("absences", 0)) > 8: reasons.append("High absences")
    if int(student.get("studytime", 2)) == 1: reasons.append("Low weekly study time")
    risk = bool(reasons)
    return {"is_at_risk": risk, "status": "⚠️ AT RISK" if risk else "✅ SAFE",
            "reasons": reasons or ["Good attendance and study habits."]}


def get_improvement_suggestions(student):
    tips = []
    if int(student.get("studytime", 2)) <= 1: tips.append("Increase weekly study time.")
    if int(student.get("absences", 0)) > 5: tips.append("Try to reduce absences.")
    if int(student.get("failures", 0)) > 0: tips.append("Ask for help with difficult topics.")
    if float(student.get("G2", 10)) < 10: tips.append("Practice with weekly mock tests.")
    return tips or ["Keep up your regular study routine."]


def train_linear_model(data_path="data/student-mat.csv", model_save_path="models/linear_model.joblib"):
    df = pd.read_csv(data_path, sep=";")
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_save_path)
    actual = [score_to_class(x) for x in y_test]
    predicted = [score_to_class(x) for x in model.predict(X_test)]
    return {"model": model, "accuracy": accuracy_score(actual, predicted),
            "confusion_matrix": confusion_matrix(actual, predicted, labels=CLASSES),
            "report": classification_report(actual, predicted, labels=CLASSES, output_dict=True),
            "coefficients": dict(zip(FEATURES, model.coef_.round(4))),
            "intercept": round(float(model.intercept_), 4)}


def predict_student(student, model_path="models/linear_model.joblib"):
    model = joblib.load(model_path)
    values = {f: float(student.get(f, 0)) for f in FEATURES}
    score = float(np.clip(model.predict(pd.DataFrame([values]))[0], 0, 20))
    label = score_to_class(score)
    factors = [{"feature": f, "value": values[f], "coefficient": round(float(c), 4),
                "impact": round(float(c * values[f]), 2),
                "effect": "Positive (+)" if c * values[f] > 0 else "Negative (-)"}
               for f, c in zip(FEATURES, model.coef_)]
    return {"predicted_score": round(score, 2), "predicted_class": label,
            "confidence": calculate_confidence(score, label),
            "risk_info": check_at_risk(student, label),
            "top_factors": sorted(factors, key=lambda x: abs(x["impact"]), reverse=True),
            "suggestions": get_improvement_suggestions(student)}


def predict_batch(df, model_path="models/linear_model.joblib"):
    result = df.copy()
    for feature in FEATURES:
        if feature not in result: result[feature] = 0.0
    scores = np.clip(joblib.load(model_path).predict(result[FEATURES].astype(float)), 0, 20)
    labels = [score_to_class(s) for s in scores]
    result["Predicted_Score"] = np.round(scores, 1)
    result["Predicted_Class"] = labels
    result["Confidence_%"] = [calculate_confidence(s, c) for s, c in zip(scores, labels)]
    result["Risk_Status"] = [check_at_risk(row, c)["status"] for row, c in zip(result.to_dict("records"), labels)]
    result["Key_Suggestion"] = [get_improvement_suggestions(row)[0] for row in result.to_dict("records")]
    return result
