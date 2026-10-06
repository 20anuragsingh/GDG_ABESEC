"""Data loader and preprocessing utilities for the Student Performance Dataset."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# Performance classification thresholds (standard 0-20 Portuguese academic scale)
PERFORMANCE_BINS = [-1, 9, 14, 20]
PERFORMANCE_LABELS = ["Low", "Average", "High"]

# Column type definitions
CATEGORICAL_FEATURES = [
    "school", "sex", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian",
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic"
]

NUMERICAL_FEATURES = [
    "age", "Medu", "Fedu", "traveltime", "studytime",
    "failures", "famrel", "freetime", "goout",
    "Dalc", "Walc", "health", "absences", "G1", "G2"
]

ALL_FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

FEATURE_METADATA = {
    "G1": {"label": "First Period Grade", "min": 0, "max": 20, "default": 11, "help": "Score from 0 to 20"},
    "G2": {"label": "Second Period Grade", "min": 0, "max": 20, "default": 11, "help": "Score from 0 to 20"},
    "studytime": {"label": "Weekly Study Time", "min": 1, "max": 4, "default": 2, "help": "1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h"},
    "failures": {"label": "Past Class Failures", "min": 0, "max": 4, "default": 0, "help": "Number of previous failures (0-4)"},
    "absences": {"label": "School Absences", "min": 0, "max": 93, "default": 4, "help": "Number of days absent"},
    "age": {"label": "Age", "min": 15, "max": 22, "default": 16, "help": "Student age in years"},
    "Medu": {"label": "Mother's Education", "min": 0, "max": 4, "default": 2, "help": "0: None, 1: 4th grade, 2: 5-9th grade, 3: Secondary, 4: Higher"},
    "Fedu": {"label": "Father's Education", "min": 0, "max": 4, "default": 2, "help": "0: None, 1: 4th grade, 2: 5-9th grade, 3: Secondary, 4: Higher"},
    "traveltime": {"label": "Home to School Travel Time", "min": 1, "max": 4, "default": 1, "help": "1: <15m, 2: 15-30m, 3: 30m-1h, 4: >1h"},
    "famrel": {"label": "Family Relationship Quality", "min": 1, "max": 5, "default": 4, "help": "1: Very poor to 5: Excellent"},
    "freetime": {"label": "Free Time After School", "min": 1, "max": 5, "default": 3, "help": "1: Very low to 5: Very high"},
    "goout": {"label": "Going Out with Friends", "min": 1, "max": 5, "default": 3, "help": "1: Very low to 5: Very high"},
    "Dalc": {"label": "Workday Alcohol Consumption", "min": 1, "max": 5, "default": 1, "help": "1: Very low to 5: Very high"},
    "Walc": {"label": "Weekend Alcohol Consumption", "min": 1, "max": 5, "default": 1, "help": "1: Very low to 5: Very high"},
    "health": {"label": "Current Health Status", "min": 1, "max": 5, "default": 4, "help": "1: Very poor to 5: Very good"},
    "school": {"label": "School", "options": ["GP", "MS"], "default": "GP"},
    "sex": {"label": "Sex", "options": ["F", "M"], "default": "F"},
    "address": {"label": "Home Address Type", "options": ["U", "R"], "default": "U"},
    "famsize": {"label": "Family Size", "options": ["GT3", "LE3"], "default": "GT3"},
    "Pstatus": {"label": "Parents Cohabitation", "options": ["T", "A"], "default": "T"},
    "Mjob": {"label": "Mother's Job", "options": ["other", "services", "at_home", "teacher", "health"], "default": "other"},
    "Fjob": {"label": "Father's Job", "options": ["other", "services", "teacher", "at_home", "health"], "default": "other"},
    "reason": {"label": "Reason to Choose School", "options": ["course", "home", "reputation", "other"], "default": "course"},
    "guardian": {"label": "Guardian", "options": ["mother", "father", "other"], "default": "mother"},
    "schoolsup": {"label": "Extra Educational Support", "options": ["no", "yes"], "default": "no"},
    "famsup": {"label": "Family Educational Support", "options": ["yes", "no"], "default": "yes"},
    "paid": {"label": "Extra Paid Classes", "options": ["no", "yes"], "default": "no"},
    "activities": {"label": "Extracurricular Activities", "options": ["yes", "no"], "default": "yes"},
    "nursery": {"label": "Attended Nursery School", "options": ["yes", "no"], "default": "yes"},
    "higher": {"label": "Desire Higher Education", "options": ["yes", "no"], "default": "yes"},
    "internet": {"label": "Home Internet Access", "options": ["yes", "no"], "default": "yes"},
    "romantic": {"label": "In Romantic Relationship", "options": ["no", "yes"], "default": "no"}
}


def get_default_student() -> Dict:
    """Return a baseline student dictionary with realistic default values."""
    defaults = {}
    for col, meta in FEATURE_METADATA.items():
        defaults[col] = meta["default"]
    return defaults


def load_dataset(data_dir: str = "data", source: str = "combined") -> pd.DataFrame:
    """
    Load student dataset(s).

    Parameters:
        data_dir: Directory containing CSV files.
        source: 'combined', 'math', or 'portuguese'.

    Returns:
        pd.DataFrame with raw student attributes.
    """
    mat_path = Path(data_dir) / "student-mat.csv"
    por_path = Path(data_dir) / "student-por.csv"

    if source == "math":
        df = pd.read_csv(mat_path, sep=";")
    elif source == "portuguese":
        df = pd.read_csv(por_path, sep=";")
    elif source == "combined":
        df_mat = pd.read_csv(mat_path, sep=";")
        df_por = pd.read_csv(por_path, sep=";")
        df = pd.concat([df_mat, df_por], ignore_index=True)
    else:
        raise ValueError(f"Invalid dataset source '{source}'. Choose 'combined', 'math', or 'portuguese'.")

    return df


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Discretize G3 into High, Average, Low classes and extract X, y.

    Parameters:
        df: Input DataFrame containing student features and G3.

    Returns:
        Tuple of (X, y)
    """
    data = df.copy()
    if "G3" not in data.columns:
        raise ValueError("Target column 'G3' not found in dataset.")

    # Target categorization:
    # Low: 0-9 (Fail)
    # Average: 10-14 (Pass/Satisfactory)
    # High: 15-20 (Very Good/Excellent)
    data["performance"] = pd.cut(
        data["G3"],
        bins=PERFORMANCE_BINS,
        labels=PERFORMANCE_LABELS,
        ordered=True
    )

    drop_cols = ["G3", "performance"]
    if "subject" in data.columns:
        drop_cols.append("subject")

    X = data.drop(columns=[c for c in drop_cols if c in data.columns])
    y = data["performance"]

    return X, y

