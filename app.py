"""Simple Streamlit Web App for Student Performance Prediction using Linear Regression.
GDG ABESEC ML Project.
"""

import io
import json
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Set writable cache for matplotlib
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

# Automatically add local packages directory to sys.path if available
_pkg_dir = Path(__file__).resolve().parent / "packages"
if _pkg_dir.exists() and str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

from src.model import (
    CLASSES,
    predict_batch,
    predict_student,
)

# App Configuration
st.set_page_config(
    page_title="EduPredict | Simple Linear Regression Model",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 EduPredict: Student Performance Classifier")
st.markdown("**Simple Linear Regression Model** that predicts student performance (**High**, **Average**, **Low**) based on academic factors.")

tab1, tab2, tab3 = st.tabs([
    "🎯 Predict Single Student",
    "📂 Batch CSV Upload",
    "📊 Model & Confusion Matrix",
])

# ==============================================================================
# TAB 1: PREDICT SINGLE STUDENT
# ==============================================================================
with tab1:
    st.subheader("Enter Student Academic Attributes")

    c1, c2 = st.columns(2)
    with c1:
        g1 = st.slider("First Period Grade (G1)", min_value=0, max_value=20, value=11, help="Score from 0 to 20")
        g2 = st.slider("Second Period Grade (G2)", min_value=0, max_value=20, value=11, help="Score from 0 to 20")
        studytime = st.selectbox(
            "Weekly Study Time",
            options=[1, 2, 3, 4],
            format_func=lambda x: {1: "1: < 2 hours", 2: "2: 2 - 5 hours", 3: "3: 5 - 10 hours", 4: "4: > 10 hours"}[x],
            index=1,
        )

    with c2:
        failures = st.selectbox("Past Class Failures", options=[0, 1, 2, 3, 4], index=0)
        absences = st.slider("School Absences (days)", min_value=0, max_value=50, value=4)

    if st.button("🔮 Predict Performance", type="primary", use_container_width=True):
        student_data = {
            "studytime": studytime,
            "failures": failures,
            "absences": absences,
            "G1": g1,
            "G2": g2,
        }

        res = predict_student(student_data)
        score = res["predicted_score"]
        pred_class = res["predicted_class"]
        conf = res["confidence"]
        risk = res["risk_info"]
        factors = res["top_factors"]
        suggestions = res["suggestions"]

        st.markdown("---")
        st.subheader("Diagnostic Results")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Predicted Final Score", f"{score:.1f} / 20")

        badge_color = "🟢" if pred_class == "High" else ("🟡" if pred_class == "Average" else "🔴")
        r2.metric("Performance Class", f"{badge_color} {pred_class}")
        r3.metric("Confidence Score", f"{conf}%")
        r4.metric("Risk Status", risk["status"])

        # Risk details
        if risk["is_at_risk"]:
            st.error(f"**Academic Risk Flags:** {', '.join(risk['reasons'])}")
        else:
            st.success("✅ **Academic Status:** Steady performance with no critical risk triggers.")

        # Contributing Factors
        st.markdown("##### 🔍 Factors that Contributed Most to Prediction")
        st.caption("Contribution calculated as: `Model Coefficient × Student Attribute Value`")
        f_df = pd.DataFrame(factors)[["feature", "value", "coefficient", "impact", "effect"]]
        f_df.columns = ["Factor", "Student Value", "Coefficient", "Score Impact", "Effect Direction"]
        st.dataframe(f_df, use_container_width=True, hide_index=True)

        # Improvement Suggestions
        st.markdown("##### 💡 Improvement Suggestions (Prescriptive Feedback)")
        for s in suggestions:
            st.info(s)

# ==============================================================================
# TAB 2: BATCH CSV UPLOAD
# ==============================================================================
with tab2:
    st.subheader("📂 Batch Student Prediction via CSV Upload")
    st.caption("Upload a CSV file containing `studytime, failures, absences, G1, G2` to predict for multiple students.")

    sample_csv_path = Path("data/sample_students.csv")
    if sample_csv_path.exists():
        with open(sample_csv_path, "rb") as f:
            st.download_button(
                label="📥 Download Sample CSV Template",
                data=f.read(),
                file_name="sample_students.csv",
                mime="text/csv",
            )

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        try:
            content = uploaded_file.getvalue().decode("utf-8")
            delimiter = ";" if ";" in content.splitlines()[0] else ","
            df_in = pd.read_csv(io.StringIO(content), sep=delimiter)

            st.success(f"Loaded {len(df_in)} student records!")
            df_out = predict_batch(df_in)

            # High-level Metrics
            m1, m2, m3, m4 = st.columns(4)
            total = len(df_out)
            high_n = (df_out["Predicted_Class"] == "High").sum()
            avg_n = (df_out["Predicted_Class"] == "Average").sum()
            low_n = (df_out["Predicted_Class"] == "Low").sum()
            risk_n = (df_out["Risk_Status"] == "⚠️ AT RISK").sum()

            m1.metric("Total Students", f"{total}")
            m2.metric("High Performers", f"{high_n}")
            m3.metric("Average Performers", f"{avg_n}")
            m4.metric("At-Risk Students", f"{risk_n}")

            st.dataframe(df_out, use_container_width=True)

            csv_data = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv_data,
                file_name="student_predictions.csv",
                mime="text/csv",
                type="primary",
            )
        except Exception as e:
            st.error(f"Error processing CSV: {e}")

# ==============================================================================
# TAB 3: MODEL PERFORMANCE & CONFUSION MATRIX
# ==============================================================================
with tab3:
    st.subheader("📊 Model Performance & Confusion Matrix")

    meta_file = Path("models/model_metadata.json")
    if meta_file.exists():
        with open(meta_file, "r") as f:
            meta = json.load(f)

        st.metric("Model Classification Accuracy", f"{meta.get('accuracy', 0.785) * 100:.2f}%")

        c_m1, c_m2 = st.columns(2)
        with c_m1:
            st.markdown("##### 🔲 Confusion Matrix")
            cm_arr = np.array(meta.get("confusion_matrix", []))
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm_arr,
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
            st.pyplot(fig)
            plt.close()

        with c_m2:
            st.markdown("##### 📐 Linear Regression Equation")
            st.latex(r"G3 = w_1 \cdot \text{studytime} + w_2 \cdot \text{failures} + w_3 \cdot \text{absences} + w_4 \cdot G1 + w_5 \cdot G2 + b")

            coef_df = pd.DataFrame(
                list(meta.get("coefficients", {}).items()),
                columns=["Feature", "Learned Weight (w)"],
            )
            st.dataframe(coef_df, use_container_width=True, hide_index=True)
            st.markdown(f"**Intercept (b):** `{meta.get('intercept', -1.6213)}`")
            st.markdown(r"""
            **Threshold Classification Rule:**
            - **High**: Predicted Score $\ge 15$
            - **Average**: Predicted Score between $10$ and $14.9$
            - **Low**: Predicted Score $< 10$
            """)
