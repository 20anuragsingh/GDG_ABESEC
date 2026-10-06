# 🎓 Student Performance Classification (Linear Regression)

A clean, beginner-friendly Machine Learning system built with **Linear Regression** that classifies student performance into **High**, **Average**, or **Low** based on academic factors.

---

## 📌 Project Overview & Features
- **Classification**: Predicts final grade ($0-20$) and classifies students into:
  - **High**: Score $\ge 15$
  - **Average**: Score between $10$ and $14.9$
  - **Low**: Score $< 10$ (Failing / At-Risk)
- **Confidence Score**: Distance-based certainty score ($55\% - 98.5\%$).
- **Explainable Factors**: Calculates exact factor contributions using learned Linear Regression weights ($w_i \times x_i$).
- **Confusion Matrix**: Evaluates model predictions across all 3 classes.
- **Bonus Features**:
  - ⚠️ **At-Risk Flagging**: Alerts on failing predictions, repeated failures, or high absenteeism.
  - 💡 **Improvement Suggestions**: Actionable recommendations for weaker factors.
  - 📂 **Batch CSV Upload**: Analyze cohorts of students at once.
  - 🖥️ **Interactive Web App**: Simple Streamlit UI.

---

## 🛠️ Technology Stack
- **Language**: Python
- **Libraries**: Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Streamlit

---

## 🚀 Setup & How to Run

```bash
# 1. Clone repo
git clone https://github.com/20anuragsingh/GDG_ABESEC.git
cd GDG_ABESEC

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Web App
streamlit run app.py
```

### CLI Commands
- **Train Model**: `python src/train.py`
- **Predict Demo Student**: `python src/predict.py`
- **Batch CSV Prediction**: `python src/predict.py --input_csv data/sample_students.csv`
- **Run Tests**: `python -m unittest tests/test_pipeline.py`

---

## 📊 Dataset & Source
- **Dataset**: [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance) (`student-mat.csv`).
- **Core 5 Features**:
  1. `G1`: First period exam grade ($0-20$)
  2. `G2`: Second period exam grade ($0-20$)
  3. `studytime`: Weekly study hours ($1: <2\text{h}, 2: 2-5\text{h}, 3: 5-10\text{h}, 4: >10\text{h}$)
  4. `failures`: Number of past failed classes ($0-4$)
  5. `absences`: School days missed ($0-93$)

---

## 🧠 Model Approach & Math

The model fits an Ordinary Least Squares (OLS) **Linear Regression**:

$$\hat{y} = w_1 \cdot \text{studytime} + w_2 \cdot \text{failures} + w_3 \cdot \text{absences} + w_4 \cdot G1 + w_5 \cdot G2 + b$$

- **Learned Equation**:
  - $G2$ weight: `+0.9796` (strongest positive driver)
  - $G1$ weight: `+0.1445` (positive driver)
  - `failures` weight: `-0.4558` (penalty per failure)
  - `absences` weight: `+0.0392`
  - `studytime` weight: `-0.0712`
  - Intercept ($b$): `-1.6213`

### Evaluation Results (Test Set)
- **Classification Accuracy**: **78.48%**
- **Confusion Matrix**:

| True \ Pred | Low | Average | High |
| :--- | :---: | :---: | :---: |
| **Low** | 27 | 0 | 0 |
| **Average** | 11 | 21 | 0 |
| **High** | 0 | 6 | 14 |

![Confusion Matrix](models/confusion_matrix.png)

---

## 💡 Challenges & Simple Solutions
1. **Converting Continuous Regression to 3-Class Labels**:
   - Used domain thresholding based on the standard European/Portuguese academic grading scale ($<10$ Fail, $10-14$ Pass, $\ge 15$ Distinction).
2. **Explainability for Students**:
   - Used linear coefficients directly to show clear positive/negative contributions ($w_i \times x_i$) so non-technical users immediately understand why a prediction was made.

---

## 📁 Repository Structure
```
GDG_ABESEC/
├── app.py                      # Simple Streamlit web interface
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
├── data/
│   ├── student-mat.csv         # UCI Math dataset
│   └── sample_students.csv     # 5-column CSV template for batch test
├── models/
│   ├── linear_model.joblib     # Trained Linear Regression model
│   ├── model_metadata.json     # Accuracy and learned weights
│   └── confusion_matrix.png    # Confusion matrix visual
├── src/
│   ├── __init__.py
│   ├── model.py                # Core training, inference & explainability
│   ├── train.py                # Training runner script
│   └── predict.py              # CLI inference script
└── tests/
    ├── __init__.py
    └── test_pipeline.py        # Automated test suite (all passing)
```
