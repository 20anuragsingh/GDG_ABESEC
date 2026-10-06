"""Interactive Streamlit Web Interface for Student Performance Classification.
GDG ABESEC Project.
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

# Configure writable cache for matplotlib
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

# Automatically add local packages directory to sys.path if available
_pkg_dir = Path(__file__).resolve().parent / "packages"
if _pkg_dir.exists() and str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

from src.data_loader import (
    ALL_FEATURE_COLUMNS,
    FEATURE_METADATA,
    PERFORMANCE_LABELS,
    get_default_student,
)
from src.predict import StudentPerformancePredictor

# Page Configuration
st.set_page_config(
    page_title="EduPredict | Student Performance Classifier",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1a365d;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f7fafc;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #3182ce;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .badge-high {
        background-color: #e6fffa;
        color: #234e52;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.2rem;
        border: 2px solid #38b2ac;
        display: inline-block;
    }
    .badge-average {
        background-color: #feebc8;
        color: #7b341e;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.2rem;
        border: 2px solid #dd6b20;
        display: inline-block;
    }
    .badge-low {
        background-color: #fed7d7;
        color: #742a2a;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.2rem;
        border: 2px solid #e53e3e;
        display: inline-block;
    }
    .risk-box {
        border-radius: 8px;
        padding: 14px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor():
    """Load and cache the trained inference pipeline."""
    return StudentPerformancePredictor()


@st.cache_data
def load_metadata():
    """Load model performance metadata."""
    meta_path = Path("models/model_metadata.json")
    if meta_path.exists():
        with open(meta_path, "r") as f:
            return json.load(f)
    return {}


predictor = load_predictor()
metadata = load_metadata()

# Header
st.markdown('<div class="main-header">🎓 EduPredict: Student Performance & Risk Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-class Machine Learning classification (High, Average, Low) with Explainable AI & Prescriptive Risk Intervention</div>', unsafe_allow_html=True)

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Individual Student Prediction",
    "📂 Batch CSV Analysis",
    "📊 Model Performance & Confusion Matrix",
    "ℹ️ Methodology & Overview",
])

# ==============================================================================
# TAB 1: INDIVIDUAL STUDENT PREDICTION
# ==============================================================================
with tab1:
    st.subheader("Student Attribute Input")
    st.caption("Adjust student academic, behavioral, and demographic features to evaluate predicted tier and risk.")

    with st.form("single_student_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 📚 Academic Foundation")
            g1 = st.slider("First Period Grade (G1)", min_value=0, max_value=20, value=11, help="Score from 0 to 20")
            g2 = st.slider("Second Period Grade (G2)", min_value=0, max_value=20, value=11, help="Score from 0 to 20")
            failures = st.selectbox("Past Class Failures", options=[0, 1, 2, 3, 4], index=0, help="Number of previous course failures")
            studytime = st.selectbox(
                "Weekly Study Time",
                options=[1, 2, 3, 4],
                format_func=lambda x: {1: "1: <2 hours", 2: "2: 2-5 hours", 3: "3: 5-10 hours", 4: "4: >10 hours"}[x],
                index=1,
            )
            schoolsup = st.selectbox("Extra Educational Support", options=["no", "yes"], index=0)
            famsup = st.selectbox("Family Educational Support", options=["yes", "no"], index=0)
            paid = st.selectbox("Extra Paid Classes", options=["no", "yes"], index=0)

        with col2:
            st.markdown("##### ⏱️ Attendance & Lifestyle")
            absences = st.slider("School Absences (days)", min_value=0, max_value=60, value=4, help="Total school days missed")
            freetime = st.slider("Free Time After School", min_value=1, max_value=5, value=3, help="1: Very low to 5: Very high")
            goout = st.slider("Going Out with Friends", min_value=1, max_value=5, value=3, help="1: Very low to 5: Very high")
            dalc = st.slider("Workday Alcohol Consumption", min_value=1, max_value=5, value=1, help="1: Very low to 5: Very high")
            walc = st.slider("Weekend Alcohol Consumption", min_value=1, max_value=5, value=1, help="1: Very low to 5: Very high")
            health = st.slider("Current Health Status", min_value=1, max_value=5, value=4, help="1: Very poor to 5: Very good")

        with col3:
            st.markdown("##### 🏠 Background & Environment")
            age = st.number_input("Age", min_value=15, max_value=22, value=16)
            sex = st.selectbox("Sex", options=["F", "M"], index=0, format_func=lambda x: "Female" if x == "F" else "Male")
            higher = st.selectbox("Wants Higher Education", options=["yes", "no"], index=0)
            internet = st.selectbox("Internet Access at Home", options=["yes", "no"], index=0)
            medu = st.selectbox("Mother's Education", options=[0, 1, 2, 3, 4], index=2, format_func=lambda x: {0: "None", 1: "4th Grade", 2: "5-9th Grade", 3: "Secondary", 4: "Higher"}[x])
            fedu = st.selectbox("Father's Education", options=[0, 1, 2, 3, 4], index=2, format_func=lambda x: {0: "None", 1: "4th Grade", 2: "5-9th Grade", 3: "Secondary", 4: "Higher"}[x])
            famrel = st.slider("Family Relationship Quality", min_value=1, max_value=5, value=4, help="1: Very bad to 5: Excellent")
            school = st.selectbox("School", options=["GP", "MS"], index=0, format_func=lambda x: "Gabriel Pereira (GP)" if x == "GP" else "Mousinho da Silveira (MS)")

        # Collapsible additional demographics
        with st.expander("Additional Demographics (Optional)"):
            c_a, c_b, c_c = st.columns(3)
            with c_a:
                address = st.selectbox("Address Type", options=["U", "R"], index=0, format_func=lambda x: "Urban" if x == "U" else "Rural")
                famsize = st.selectbox("Family Size", options=["GT3", "LE3"], index=0, format_func=lambda x: "Greater than 3" if x == "GT3" else "3 or less")
                pstatus = st.selectbox("Parent Cohabitation", options=["T", "A"], index=0, format_func=lambda x: "Living Together" if x == "T" else "Apart")
            with c_b:
                mjob = st.selectbox("Mother's Job", options=["other", "services", "at_home", "teacher", "health"], index=0)
                fjob = st.selectbox("Father's Job", options=["other", "services", "teacher", "at_home", "health"], index=0)
                guardian = st.selectbox("Guardian", options=["mother", "father", "other"], index=0)
            with c_c:
                reason = st.selectbox("Reason for School", options=["course", "home", "reputation", "other"], index=0)
                traveltime = st.selectbox("Travel Time", options=[1, 2, 3, 4], index=0, format_func=lambda x: {1: "<15 min", 2: "15-30 min", 3: "30m-1h", 4: ">1 hour"}[x])
                activities = st.selectbox("Extracurricular Activities", options=["yes", "no"], index=0)
                nursery = st.selectbox("Attended Nursery", options=["yes", "no"], index=0)
                romantic = st.selectbox("Romantic Relationship", options=["no", "yes"], index=0)

        submit_btn = st.form_submit_button("🔮 Predict Student Performance", type="primary", use_container_width=True)

    if submit_btn:
        input_data = {
            "school": school, "sex": sex, "age": age, "address": address, "famsize": famsize,
            "Pstatus": pstatus, "Medu": medu, "Fedu": fedu, "Mjob": mjob, "Fjob": fjob,
            "reason": reason, "guardian": guardian, "traveltime": traveltime, "studytime": studytime,
            "failures": failures, "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
            "activities": activities, "nursery": nursery, "higher": higher, "internet": internet,
            "romantic": romantic, "famrel": famrel, "freetime": freetime, "goout": goout,
            "Dalc": dalc, "Walc": walc, "health": health, "absences": absences, "G1": g1, "G2": g2,
        }

        result = predictor.predict_single(input_data)
        pred_cls = result["predicted_class"]
        conf = result["confidence_percentage"]
        probs = result["probabilities"]
        risk = result["risk_assessment"]
        factors = result["top_contributing_factors"]
        suggestions = result["suggestions"]

        st.markdown("---")
        st.subheader("🎯 Diagnostic Prediction Results")

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.markdown("##### Performance Classification")
            badge_map = {
                "High": ("badge-high", "🟢 HIGH PERFORMANCE (Grade 15–20)"),
                "Average": ("badge-average", "🟡 AVERAGE PERFORMANCE (Grade 10–14)"),
                "Low": ("badge-low", "🔴 LOW / AT-RISK (Grade 0–9)"),
            }
            css_class, label_text = badge_map.get(pred_cls, ("badge-average", pred_cls))
            st.markdown(f'<div class="{css_class}">{label_text}</div>', unsafe_allow_html=True)
            st.markdown(f"**Prediction Confidence:** `{conf:.1f}%`")

            # Probability Breakdown
            st.markdown("##### Probability Distribution")
            prob_df = pd.DataFrame({
                "Performance Class": list(probs.keys()),
                "Probability (%)": [v * 100 for v in probs.values()]
            })
            st.bar_chart(prob_df.set_index("Performance Class"))

        with res_col2:
            st.markdown("##### ⚠️ Academic Risk Evaluation (Bonus Feature)")
            risk_color_map = {
                "High Risk": ("#fed7d7", "#742a2a", "#e53e3e"),
                "Moderate Risk": ("#feebc8", "#7b341e", "#dd6b20"),
                "Low Risk": ("#e6fffa", "#234e52", "#38b2ac"),
            }
            bg_c, text_c, border_c = risk_color_map.get(risk["risk_level"], ("#edf2f7", "#2d3748", "#cbd5e0"))

            st.markdown(
                f"""
                <div style="background-color: {bg_c}; color: {text_c}; border-left: 6px solid {border_c}; padding: 14px; border-radius: 8px;">
                    <div style="font-size: 1.25rem; font-weight: 700;">{risk['badge']} {risk['risk_level'].upper()} (Score: {risk['risk_score']}/100)</div>
                    <div style="margin-top: 4px; font-weight: 500;">{risk['summary']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if risk["triggers"]:
                st.markdown("**Identified Risk Triggers:**")
                for trig in risk["triggers"]:
                    st.markdown(f"- ⚠️ {trig}")
            else:
                st.markdown("✅ No critical risk triggers detected.")

        st.markdown("---")
        # Factors Contributed Most to the Prediction
        st.subheader("🔍 Factors that Contributed Most to this Prediction")
        st.caption("Marginal impact of individual student attributes compared to baseline student norms.")

        f_col1, f_col2 = st.columns(2)
        with f_col1:
            st.markdown("##### 📈 Top Positive Drivers (Pushed toward this class)")
            if factors["positive"]:
                pos_df = pd.DataFrame(factors["positive"])[["label", "student_value", "impact_percentage"]]
                pos_df.columns = ["Factor", "Student Value", "Impact Boost (%)"]
                pos_df["Impact Boost (%)"] = pos_df["Impact Boost (%)"].round(2)
                st.dataframe(pos_df, use_container_width=True, hide_index=True)
            else:
                st.info("No strong positive deviation from baseline.")

        with f_col2:
            st.markdown("##### 📉 Top Negative Drivers (Dragged away from this class)")
            if factors["negative"]:
                neg_df = pd.DataFrame(factors["negative"])[["label", "student_value", "impact_percentage"]]
                neg_df.columns = ["Factor", "Student Value", "Drag Penalty (%)"]
                neg_df["Drag Penalty (%)"] = neg_df["Drag Penalty (%)"].round(2)
                st.dataframe(neg_df, use_container_width=True, hide_index=True)
            else:
                st.info("No strong negative deviation from baseline.")

        # Prescriptive Improvement Suggestions
        st.markdown("---")
        st.subheader("💡 Prescriptive Improvement Plan (Bonus Feature)")
        st.caption("Personalized, actionable recommendations generated based on identified weaknesses.")

        for s in suggestions:
            priority_badge = "🔴 High Priority" if s["priority"] == "High" else ("🟡 Medium Priority" if s["priority"] == "Medium" else "🟢 Growth Opportunity")
            st.markdown(
                f"""
                <div style="background-color: #f7fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 16px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; font-size: 1.05rem;">{s['icon']} {s['title']}</span>
                        <span style="font-size: 0.85rem; font-weight: 600; color: #4a5568;">{priority_badge} | {s['impact']}</span>
                    </div>
                    <div style="color: #4a5568; margin-top: 6px; font-size: 0.95rem;">{s['description']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ==============================================================================
# TAB 2: BATCH CSV ANALYSIS
# ==============================================================================
with tab2:
    st.subheader("📂 Batch Student Prediction via CSV Upload (Bonus Feature)")
    st.caption("Upload a student dataset in CSV format (comma or semicolon delimited) to analyze multiple students at once.")

    # Sample CSV Download button
    sample_path = Path("data/sample_students.csv")
    if sample_path.exists():
        with open(sample_path, "rb") as f:
            sample_bytes = f.read()
        st.download_button(
            label="📥 Download Sample Students CSV Template",
            data=sample_bytes,
            file_name="sample_students.csv",
            mime="text/csv",
            help="Download a ready-to-test CSV with sample students",
        )

    uploaded_file = st.file_uploader("Upload Student CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            content = uploaded_file.getvalue().decode("utf-8")
            delimiter = ";" if ";" in content.splitlines()[0] else ","
            batch_in_df = pd.read_csv(io.StringIO(content), sep=delimiter)

            st.success(f"Successfully uploaded {len(batch_in_df)} student records!")

            with st.spinner("Processing batch predictions, risk assessments, and recommendations..."):
                batch_res_df = predictor.predict_batch(batch_in_df)

            # High-level Metrics
            st.markdown("##### 📊 Batch Overview Summary")
            m1, m2, m3, m4 = st.columns(4)
            total_students = len(batch_res_df)
            high_count = (batch_res_df["Predicted_Performance"] == "High").sum()
            avg_count = (batch_res_df["Predicted_Performance"] == "Average").sum()
            low_count = (batch_res_df["Predicted_Performance"] == "Low").sum()
            at_risk_count = (batch_res_df["Risk_Level"] == "High Risk").sum()

            m1.metric("Total Students", f"{total_students}")
            m2.metric("High Achievers", f"{high_count} ({high_count/total_students*100:.1f}%)")
            m3.metric("Average Performers", f"{avg_count} ({avg_count/total_students*100:.1f}%)")
            m4.metric("At-Risk Students", f"{at_risk_count} ({at_risk_count/total_students*100:.1f}%)", delta_color="inverse")

            # Visual Distribution Charts
            c_chart1, c_chart2 = st.columns(2)
            with c_chart1:
                st.markdown("##### Performance Class Distribution")
                class_counts = batch_res_df["Predicted_Performance"].value_counts().reindex(PERFORMANCE_LABELS, fill_value=0)
                fig_bar, ax_bar = plt.subplots(figsize=(5, 3))
                palette = {"Low": "#e53e3e", "Average": "#dd6b20", "High": "#38b2ac"}
                sns.barplot(x=class_counts.index, y=class_counts.values, palette=palette, ax=ax_bar)
                ax_bar.set_ylabel("Student Count")
                ax_bar.set_xlabel("Predicted Tier")
                plt.tight_layout()
                st.pyplot(fig_bar)
                plt.close(fig_bar)

            with c_chart2:
                st.markdown("##### Risk Level Breakdown")
                risk_counts = batch_res_df["Risk_Level"].value_counts()
                fig_pie, ax_pie = plt.subplots(figsize=(5, 3))
                risk_colors = ["#e53e3e", "#dd6b20", "#38b2ac"]
                ax_pie.pie(risk_counts.values, labels=risk_counts.index, autopct="%1.1f%%", colors=risk_colors, startangle=140)
                plt.tight_layout()
                st.pyplot(fig_pie)
                plt.close(fig_pie)

            # Filterable Table
            st.markdown("---")
            st.markdown("##### 📋 Enriched Student Predictions Table")

            filter_risk = st.multiselect(
                "Filter by Risk Level:",
                options=["High Risk", "Moderate Risk", "Low Risk"],
                default=["High Risk", "Moderate Risk", "Low Risk"],
            )
            filtered_df = batch_res_df[batch_res_df["Risk_Level"].isin(filter_risk)]

            display_cols = [
                "Predicted_Performance", "Confidence_Score", "Risk_Level", "Risk_Score",
                "Primary_Risk_Trigger", "Key_Improvement_Suggestion",
                "G1", "G2", "studytime", "failures", "absences"
            ]
            valid_cols = [c for c in display_cols if c in filtered_df.columns]
            st.dataframe(filtered_df[valid_cols], use_container_width=True)

            # Export Enriched CSV
            csv_export = batch_res_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Complete Enriched Report (CSV)",
                data=csv_export,
                file_name="student_predictions_enriched.csv",
                mime="text/csv",
                type="primary",
            )

        except Exception as e:
            st.error(f"Error parsing uploaded file: {str(e)}")

# ==============================================================================
# TAB 3: MODEL PERFORMANCE & CONFUSION MATRIX
# ==============================================================================
with tab3:
    st.subheader("📊 Model Performance & Confusion Matrix Evaluation")
    st.caption("Comprehensive evaluation metrics, confusion matrix, and feature importances for the trained model.")

    if metadata:
        metrics = metadata.get("metrics", {})
        acc = metrics.get("test_accuracy", 0.0)
        f1_w = metrics.get("test_f1_weighted", 0.0)
        f1_m = metrics.get("test_f1_macro", 0.0)
        prec = metrics.get("test_precision_weighted", 0.0)
        rec = metrics.get("test_recall_weighted", 0.0)

        # KPI Metrics Cards
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Test Accuracy", f"{acc * 100:.2f}%")
        k2.metric("Weighted F1", f"{f1_w:.4f}")
        k3.metric("Macro F1", f"{f1_m:.4f}")
        k4.metric("Precision", f"{prec:.4f}")
        k5.metric("Recall", f"{rec:.4f}")

        st.markdown("---")
        cm_col1, cm_col2 = st.columns([1, 1])

        with cm_col1:
            st.markdown("##### 🔲 Confusion Matrix (Holdout Test Set)")
            cm_arr = np.array(metrics.get("confusion_matrix", []))
            if cm_arr.size > 0:
                fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                sns.heatmap(
                    cm_arr,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=PERFORMANCE_LABELS,
                    yticklabels=PERFORMANCE_LABELS,
                    ax=ax_cm,
                    annot_kws={"size": 13, "weight": "bold"},
                )
                ax_cm.set_title("Test Confusion Matrix (Student Performance)", fontsize=12, pad=10, weight="bold")
                ax_cm.set_xlabel("Predicted Class", fontsize=10)
                ax_cm.set_ylabel("True Class", fontsize=10)
                plt.tight_layout()
                st.pyplot(fig_cm)
                plt.close(fig_cm)

        with cm_col2:
            st.markdown("##### 📑 Per-Class Classification Report")
            rep = metrics.get("classification_report", {})
            rep_rows = []
            for lbl in PERFORMANCE_LABELS:
                if lbl in rep:
                    rep_rows.append({
                        "Class": lbl,
                        "Precision": f"{rep[lbl]['precision']:.4f}",
                        "Recall": f"{rep[lbl]['recall']:.4f}",
                        "F1-Score": f"{rep[lbl]['f1-score']:.4f}",
                        "Support": rep[lbl]["support"],
                    })
            if rep_rows:
                st.dataframe(pd.DataFrame(rep_rows), use_container_width=True, hide_index=True)

            st.markdown("##### 🏆 5-Fold Cross-Validation Benchmarks")
            benchmarks = metadata.get("cv_benchmarks", {})
            if benchmarks:
                b_rows = []
                for name, b in benchmarks.items():
                    b_rows.append({
                        "Model Algorithm": name,
                        "CV Accuracy": f"{b['accuracy_mean']:.4f} ± {b['accuracy_std']:.4f}",
                        "Weighted F1": f"{b['f1_weighted']:.4f}",
                        "Macro F1": f"{b['f1_macro']:.4f}",
                    })
                st.dataframe(pd.DataFrame(b_rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("##### 🌐 Global Feature Importances (Top Influential Attributes)")
        top_features = metadata.get("top_features", [])
        if top_features:
            feat_df = pd.DataFrame(top_features[:15])
            fig_imp, ax_imp = plt.subplots(figsize=(10, 4.5))
            sns.barplot(data=feat_df, x="importance", y="feature", palette="viridis", ax=ax_imp)
            ax_imp.set_title("Top 15 Global Feature Importances (Random Forest)", fontsize=12, pad=10, weight="bold")
            ax_imp.set_xlabel("Relative Feature Importance")
            ax_imp.set_ylabel("Feature")
            plt.tight_layout()
            st.pyplot(fig_imp)
            plt.close(fig_imp)
    else:
        st.warning("Model metadata not found. Train the model using `python src/train.py`.")

# ==============================================================================
# TAB 4: METHODOLOGY & OVERVIEW
# ==============================================================================
with tab4:
    st.subheader("ℹ️ Methodology, Architecture & Dataset Overview")
    st.markdown(
        """
        ### 1. Problem Formulation
        The objective is to accurately classify secondary school students into three distinct performance categories:
        - **High Performance (15–20)**: Top tier (>=75%) representing outstanding academic mastery.
        - **Average Performance (10–14)**: Passing tier (50–70%) meeting core competency standards.
        - **Low / At-Risk (0–9)**: Failing tier (<50%) at critical risk of course non-completion.

        ### 2. Dataset & Features
        - **Source**: UCI Machine Learning Repository — *Student Performance Dataset* (Cortez & Silva, 2008, University of Minho).
        - **Records**: 1,044 combined student records across Mathematics and Portuguese subjects.
        - **Attributes**: 32 multifaceted features covering:
          - *Academic*: Period grades (G1, G2), previous failures, weekly study time, extra tutoring.
          - *Attendance & Behavioral*: School absences, alcohol consumption (workday & weekend), free time, social outings.
          - *Socio-Demographic*: Parents' education & jobs, family relationship quality, home address, higher education aspiration.

        ### 3. Machine Learning Architecture
        - **Preprocessing**: Robust `ColumnTransformer` with `StandardScaler` for continuous numerical features and `OneHotEncoder(drop='first', handle_unknown='ignore')` for nominal/categorical variables.
        - **Model Selection**: Benchmarked across Logistic Regression, Random Forest, Gradient Boosting, and SVM using 5-Fold Stratified Cross-Validation.
        - **Class Balancing**: Balanced class weight penalization to handle the natural distribution of top/struggling students without artificial distortion.
        - **Holdout Accuracy**: Achieved **84.69% Test Accuracy** and **86.11% Cross-Validated Accuracy** with Random Forest.

        ### 4. Explainable AI & Prescriptive Risk Engine
        - **Local Contribution**: Marginal perturbation analysis measures the exact directional pull of each student's input features against population baselines.
        - **Risk Scoring**: Composite index integrating predicted class probability, prior failure counts, chronic absenteeism, and study time deficits into a transparent 0–100 risk score.
        - **Actionable Interventions**: Generates prescriptive recommendations spanning active recall, attendance recovery agreements, and remedial tutoring.
        """
    )
