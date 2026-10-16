from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from sklearn.model_selection import train_test_split

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="SHAP Explainability | Heart Disease AI",
    page_icon="🔎",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "heart_cleaned.csv"
MODEL_DIR = ROOT / "models"

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
    "XGBoost": MODEL_DIR / "xgboost.joblib",
}

FEATURE_DESCRIPTIONS = {
    "age": "Age (years)",
    "sex": "Sex (dataset coding)",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl",
    "restecg": "Resting ECG result",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (oldpeak)",
    "slope": "Peak exercise ST slope",
    "ca": "Major vessels (ca)",
    "thal": "Thal test result",
}

st.title("🔎 SHAP Explainability Dashboard")
st.caption(
    "Explore which input features influence the predictions of your trained models."
)
st.info(
    "Educational machine-learning explanation only. SHAP describes model behavior; "
    "it is not a medical explanation, diagnosis, or estimate of clinical risk."
)

# --------------------------------------------------
# CHECK PROJECT FILES
# --------------------------------------------------
if not DATA_PATH.exists():
    st.error("Dataset not found at data/heart_cleaned.csv. Run the preprocessing script first.")
    st.stop()

for model_path in MODEL_FILES.values():
    if not model_path.exists():
        st.error(f"Trained model not found: {model_path.name}. Run `python src/train.py` first.")
        st.stop()

df = pd.read_csv(DATA_PATH)
if "target" not in df.columns:
    st.error("The dataset must contain a 'target' column.")
    st.stop()

X = df.drop(columns="target")
y = df["target"].astype(int)

# Use the same split settings as the project's training workflow.
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

model_name = st.selectbox("Choose a trained model", list(MODEL_FILES.keys()))
model = load_model(str(MODEL_FILES[model_name]))

st.divider()
st.header("1. Global Feature Importance")
st.write(
    "This chart ranks features by the average absolute SHAP value. Larger values "
    "mean the model's output is more sensitive to that feature in this dataset."
)

# Keep computation responsive while preserving a reproducible sample.
max_rows = min(100, len(X_test))
sample_size = st.slider(
    "Number of test records to explain",
    min_value=min(10, max_rows),
    max_value=max_rows,
    value=max_rows,
    step=1,
)
X_sample = X_test.sample(n=sample_size, random_state=42)

try:
    with st.spinner("Calculating SHAP values..."):
        # The fitted model is a sklearn pipeline; SHAP's model-agnostic
        # permutation explainer handles preprocessing inside that pipeline.
        background = X.sample(n=min(40, len(X)), random_state=42)
        predict_fn = lambda data: model.predict_proba(
            pd.DataFrame(data, columns=X.columns)
        )[:, 1]

        explainer = shap.Explainer(
            predict_fn,
            background,
            algorithm="permutation",
            feature_names=list(X.columns),
        )
        shap_values = explainer(
            X_sample,
            max_evals=2 * len(X.columns) + 1,
        )

    mean_abs = pd.Series(
        abs(shap_values.values).mean(axis=0),
        index=X.columns,
        name="Mean absolute SHAP value",
    ).sort_values()

    importance = mean_abs.rename(index=FEATURE_DESCRIPTIONS).to_frame()
    fig, ax = plt.subplots(figsize=(9, 6))
    importance.plot(kind="barh", legend=False, ax=ax)
    ax.set_title(f"{model_name}: Mean Absolute SHAP Values")
    ax.set_xlabel("Mean absolute SHAP value (model output scale)")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(
        importance.sort_values("Mean absolute SHAP value", ascending=False).style.format(
            {"Mean absolute SHAP value": "{:.5f}"}
        ),
        use_container_width=True,
    )

    st.divider()
    st.header("2. SHAP Summary Plot")
    st.write(
        "Each point represents one test record. The horizontal position indicates "
        "whether a feature pushes the model output higher or lower relative to its "
        "baseline. This is a model-output explanation, not a causal relationship."
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values.values,
        X_sample.rename(columns=FEATURE_DESCRIPTIONS),
        show=False,
        plot_size=None,
    )
    fig = plt.gcf()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.header("3. Explain One Test Record")

    selected_index = st.selectbox(
        "Choose a test record",
        options=list(X_sample.index),
        format_func=lambda idx: f"Record {idx}",
    )
    row_position = list(X_sample.index).index(selected_index)
    one_row = X_sample.loc[[selected_index]]
    probability = float(model.predict_proba(one_row)[0, 1])
    predicted_class = int(model.predict(one_row)[0])

    a, b, c = st.columns(3)
    a.metric("Model output for positive class", f"{probability:.1%}")
    b.metric("Predicted dataset class", "Positive (1)" if predicted_class == 1 else "Negative (0)")
    c.metric("Explained test record", str(selected_index))

    st.caption(
        "The percentage above is the model's predicted probability for the dataset's "
        "positive class, not a clinically validated probability of heart disease."
    )

    local_values = shap_values.values[row_position]
    local_series = pd.Series(local_values, index=X.columns)
    local_series = local_series.reindex(local_series.abs().sort_values(ascending=True).index)
    local_series.index = [FEATURE_DESCRIPTIONS.get(name, name) for name in local_series.index]

    fig, ax = plt.subplots(figsize=(9, 6))
    local_series.plot(kind="barh", ax=ax)
    ax.axvline(0, linewidth=1)
    ax.set_title("Feature contributions for the selected record")
    ax.set_xlabel("SHAP value (positive-class model output)")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("View input values for this record"):
        shown = one_row.T.rename(columns={selected_index: "Input value"})
        shown.index = [FEATURE_DESCRIPTIONS.get(name, name) for name in shown.index]
        st.dataframe(shown, use_container_width=True)

except Exception as exc:
    st.error("SHAP calculation did not complete.")
    st.exception(exc)
    st.info(
        "If the error mentions a SHAP evaluation limit, reduce the number of test records "
        "and try again. Your existing prediction and evaluation pages are not changed."
    )

st.divider()
st.caption(
    "Dataset: UCI Heart Disease dataset (303 records). Results depend on this dataset, "
    "the fitted model, and the selected sample; they should not guide medical care."
)
