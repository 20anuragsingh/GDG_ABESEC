"""Feature explainability and prescriptive recommendation engine for student performance."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

# Add local packages directory to sys.path if available
_pkg_dir = Path(__file__).resolve().parent.parent / "packages"
if _pkg_dir.exists() and str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

from src.data_loader import (
    CATEGORICAL_FEATURES,
    FEATURE_METADATA,
    NUMERICAL_FEATURES,
    PERFORMANCE_LABELS,
    get_default_student,
)


def compute_feature_contributions(
    pipeline: Any,
    student_df: pd.DataFrame,
    top_k: int = 6,
) -> Dict[str, Any]:
    """
    Compute local feature contributions for an individual student prediction.

    Uses marginal impact perturbation against baseline reference medians/modes:
    Delta_j = P(predicted_class | student) - P(predicted_class | student with feature j set to baseline)

    Returns:
        Dict with predicted_class, confidence, probabilities, and top contributing factors.
    """
    classes = list(pipeline.classes_)
    probs = pipeline.predict_proba(student_df)[0]
    prob_dict = {cls: float(p) for cls, p in zip(classes, probs)}
    predicted_class = pipeline.predict(student_df)[0]
    target_idx = classes.index(predicted_class)
    base_prob = float(probs[target_idx])

    # Establish reference baseline values from defaults
    reference = get_default_student()

    contributions = []
    # Test each feature's contribution
    features_to_test = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    for feature in features_to_test:
        if feature not in student_df.columns:
            continue

        actual_val = student_df.iloc[0][feature]
        baseline_val = reference.get(feature, actual_val)

        # Skip if identical to baseline
        if actual_val == baseline_val:
            continue

        perturbed_df = student_df.copy()
        perturbed_df.at[perturbed_df.index[0], feature] = baseline_val

        perturbed_prob = float(pipeline.predict_proba(perturbed_df)[0][target_idx])
        # Delta: how much having actual_val changed probability vs baseline
        delta = base_prob - perturbed_prob

        meta = FEATURE_METADATA.get(feature, {})
        label = meta.get("label", feature)

        contributions.append({
            "feature": feature,
            "label": label,
            "student_value": actual_val,
            "baseline_value": baseline_val,
            "delta_probability": delta,
            "impact_percentage": delta * 100.0,
            "direction": "positive" if delta > 0 else "negative",
        })

    # Sort contributions by absolute delta magnitude
    contributions.sort(key=lambda x: abs(x["delta_probability"]), reverse=True)

    positive_contributors = [c for c in contributions if c["delta_probability"] > 0][:top_k]
    negative_contributors = [c for c in contributions if c["delta_probability"] < 0][:top_k]

    return {
        "predicted_class": predicted_class,
        "confidence": base_prob,
        "probabilities": prob_dict,
        "all_contributions": contributions,
        "top_positive": positive_contributors,
        "top_negative": negative_contributors,
    }


def assess_student_risk(
    student_dict: Dict[str, Any],
    predicted_class: str,
    class_probs: Dict[str, float],
) -> Dict[str, Any]:
    """
    Evaluate academic risk level and trigger flags.

    Returns:
        Dict with risk_level ('High Risk', 'Moderate Risk', 'Low Risk'),
        risk_score (0-100), and specific risk triggers.
    """
    risk_score = 0
    triggers = []

    prob_low = class_probs.get("Low", 0.0)
    prob_high = class_probs.get("High", 0.0)

    # 1. Model prediction tier
    if predicted_class == "Low":
        risk_score += 45 + int(prob_low * 20)
        triggers.append(f"Model predicts Low performance tier ({prob_low:.1%} probability)")
    elif predicted_class == "Average" and prob_low >= 0.30:
        risk_score += 25
        triggers.append(f"Elevated risk of slipping into Low tier ({prob_low:.1%} probability)")

    # 2. Prior academic failures
    failures = int(student_dict.get("failures", 0))
    if failures >= 2:
        risk_score += 30
        triggers.append(f"Severe prior academic failures ({failures} subjects)")
    elif failures == 1:
        risk_score += 15
        triggers.append("History of 1 prior class failure")

    # 3. Absenteeism
    absences = int(student_dict.get("absences", 0))
    if absences >= 15:
        risk_score += 25
        triggers.append(f"Chronic absenteeism ({absences} missed school days)")
    elif absences >= 8:
        risk_score += 12
        triggers.append(f"Elevated absence rate ({absences} school days missed)")

    # 4. Study time deficiency
    studytime = int(student_dict.get("studytime", 2))
    if studytime == 1:
        risk_score += 15
        triggers.append("Critically low weekly study time (<2 hours/week)")

    # 5. Period scores (G1 & G2)
    g1 = float(student_dict.get("G1", 10))
    g2 = float(student_dict.get("G2", 10))
    if g2 < 10:
        risk_score += 20
        triggers.append(f"Failing second period score (G2 = {g2}/20)")
    elif g1 < 10:
        risk_score += 10
        triggers.append(f"Failing first period score (G1 = {g1}/20)")

    # 6. High alcohol consumption or social distraction
    dalc = int(student_dict.get("Dalc", 1))
    walc = int(student_dict.get("Walc", 1))
    goout = int(student_dict.get("goout", 3))
    if dalc >= 3 or walc >= 4:
        risk_score += 10
        triggers.append("High alcohol intake impacting academic readiness")
    if goout >= 4 and studytime <= 2:
        risk_score += 8
        triggers.append("Imbalanced leisure vs. study ratio")

    # Cap risk score between 0 and 100
    risk_score = min(100, max(0, risk_score))

    if risk_score >= 50 or predicted_class == "Low":
        level = "High Risk"
        badge = "🔴"
        color = "#e53e3e"
        summary = "Immediate academic intervention and targeted mentoring recommended."
    elif risk_score >= 25:
        level = "Moderate Risk"
        badge = "🟡"
        color = "#dd6b20"
        summary = "Student requires focused monitoring in study habits and attendance."
    else:
        level = "Low Risk"
        badge = "🟢"
        color = "#38a169"
        summary = "Student is academically stable and performing consistently."

    return {
        "risk_level": level,
        "risk_score": risk_score,
        "badge": badge,
        "color": color,
        "summary": summary,
        "triggers": triggers,
    }


def generate_improvement_suggestions(
    student_dict: Dict[str, Any],
    predicted_class: str,
    confidence: float,
) -> List[Dict[str, str]]:
    """
    Generate prescriptive, actionable improvement recommendations based on weaker factors.

    Returns:
        List of dicts with category, priority, and concrete action plan.
    """
    suggestions = []

    # Factor: Study Time
    studytime = int(student_dict.get("studytime", 2))
    if studytime == 1:
        suggestions.append({
            "category": "Study Habits",
            "priority": "High",
            "icon": "📚",
            "title": "Increase Weekly Structured Study Hours",
            "description": "Weekly study time is currently <2 hours. Adopt a 5–8 hours/week study timetable using time-blocking and the Pomodoro technique (25m study / 5m break).",
            "impact": "+15-25% Grade Boost"
        })
    elif studytime == 2 and predicted_class != "High":
        suggestions.append({
            "category": "Study Habits",
            "priority": "Medium",
            "icon": "📖",
            "title": "Optimize Study Depth and Consistency",
            "description": "Transition from passive revision (2–5 hours) to active recall and problem-solving practice (5–10 hours/week) before midterm examinations.",
            "impact": "+10% Grade Boost"
        })

    # Factor: Absences
    absences = int(student_dict.get("absences", 0))
    if absences >= 10:
        suggestions.append({
            "category": "Attendance",
            "priority": "High",
            "icon": "⏰",
            "title": "Attendance Recovery & Absence Cap",
            "description": f"Student has missed {absences} classes. Coordinate an attendance recovery agreement. Consistent class attendance strongly correlates with high test scores.",
            "impact": "Crucial for Passing"
        })
    elif absences >= 5:
        suggestions.append({
            "category": "Attendance",
            "priority": "Medium",
            "icon": "📅",
            "title": "Minimize Class Absences",
            "description": f"{absences} missed school days recorded. Ensure missed lecture notes and homework assignments are reviewed promptly within 48 hours.",
            "impact": "+8% Retention"
        })

    # Factor: Class Failures
    failures = int(student_dict.get("failures", 0))
    if failures > 0:
        suggestions.append({
            "category": "Academic Support",
            "priority": "High",
            "icon": "🎯",
            "title": "Targeted Remedial Tutoring",
            "description": f"Address prerequisite concept gaps from {failures} past course failure(s). Enroll in instructor office hours or departmental peer tutoring.",
            "impact": "Core Prerequisite Fix"
        })

    # Factor: Educational Support
    schoolsup = student_dict.get("schoolsup", "no")
    if (predicted_class == "Low" or failures > 0) and schoolsup == "no":
        suggestions.append({
            "category": "Academic Support",
            "priority": "High",
            "icon": "🤝",
            "title": "Enroll in School Academic Support",
            "description": "Register for after-school instructional support and structured group study to receive guided feedback on problem sets.",
            "impact": "+12% Passing Odds"
        })

    # Factor: Term Scores (G1, G2)
    g1 = float(student_dict.get("G1", 10))
    g2 = float(student_dict.get("G2", 10))
    if g2 < 10 or g1 < 10:
        suggestions.append({
            "category": "Exam Preparation",
            "priority": "High",
            "icon": "📝",
            "title": "Exam Simulation & Mock Practice",
            "description": "Recent period scores indicate testing struggles. Complete timed past-paper mock tests under exam conditions weekly to build test confidence.",
            "impact": "+20% Exam Readiness"
        })

    # Factor: Leisure & Lifestyle Balance
    dalc = int(student_dict.get("Dalc", 1))
    walc = int(student_dict.get("Walc", 1))
    goout = int(student_dict.get("goout", 3))
    if dalc >= 3 or walc >= 4 or goout >= 4:
        suggestions.append({
            "category": "Lifestyle Balance",
            "priority": "Medium",
            "icon": "⚖️",
            "title": "Establish Healthy Social & Sleep Routine",
            "description": "High weekday/weekend social exhaustion impairs morning cognitive focus. Reserve weekdays strictly for academic prep and sleep hygiene (7-8 hours).",
            "impact": "Sharper Daily Focus"
        })

    # Factor: Health
    health = int(student_dict.get("health", 4))
    if health <= 2:
        suggestions.append({
            "category": "Health & Well-being",
            "priority": "Medium",
            "icon": "🌱",
            "title": "Wellness & Health Support",
            "description": "Low health score reported. Connect with campus wellness/counseling services to address physical stamina or health concerns.",
            "impact": "Sustained Stamina"
        })

    # Factor: Higher Education Motivation
    higher = student_dict.get("higher", "yes")
    if higher == "no":
        suggestions.append({
            "category": "Motivation",
            "priority": "Low",
            "icon": "🚀",
            "title": "Career Mentoring & Goal Exploration",
            "description": "Engage with career counselors to explore long-term career pathways, internships, and the practical value of academic qualification.",
            "impact": "Higher Intrinsic Motivation"
        })

    # If already high performer and few weaknesses:
    if len(suggestions) == 0 or (predicted_class == "High" and len(suggestions) <= 1):
        suggestions.append({
            "category": "Advanced Enrichment",
            "priority": "Low",
            "icon": "⭐",
            "title": "Advanced Problem Solving & Peer Mentoring",
            "description": "Strong foundational performance. Consider participating in academic competitions, honors research, or mentoring peers in study groups.",
            "impact": "Excellence & Leadership"
        })

    return suggestions

