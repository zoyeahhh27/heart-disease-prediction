from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report

st.set_page_config(page_title="Model Evaluation | Heart Disease AI", page_icon="📊", layout="wide")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "heart_cleaned.csv"
MODEL_DIR = ROOT / "models"
TEST_RESULTS_PATH = MODEL_DIR / "model_comparison.csv"
CV_RESULTS_PATH = MODEL_DIR / "cross_validation_results.csv"

sns.set_theme(style="whitegrid")

st.title("📊 Model Evaluation Dashboard")
st.caption("Evaluate Logistic Regression, Random Forest, and XGBoost using a held-out test set and stratified cross-validation.")
st.info("Educational project only. These models are not clinically validated diagnostic tools and must not be used for diagnosis.")

for path in [DATA_PATH, TEST_RESULTS_PATH, CV_RESULTS_PATH]:
    if not path.exists():
        st.error(f"Required file not found: {path.relative_to(ROOT)}")
        st.info("Run `python src/preprocess.py` and `python src/train.py` from the project root, then refresh this page.")
        st.stop()

model_files = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
    "XGBoost": MODEL_DIR / "xgboost.joblib",
}
for path in model_files.values():
    if not path.exists():
        st.error(f"Trained model missing: {path.name}")
        st.stop()

df = pd.read_csv(DATA_PATH)
test_results = pd.read_csv(TEST_RESULTS_PATH)
cv_results = pd.read_csv(CV_RESULTS_PATH)

test_metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
cv_metrics = ["CV ACCURACY Mean", "CV PRECISION Mean", "CV RECALL Mean", "CV F1 Mean", "CV ROC_AUC Mean"]

if not set(["Model"] + test_metrics).issubset(test_results.columns):
    st.error("Unexpected model_comparison.csv format. Rerun src/train.py.")
    st.stop()
if not set(["Model"] + cv_metrics).issubset(cv_results.columns):
    st.error("Unexpected cross_validation_results.csv format. Rerun the updated src/train.py.")
    st.stop()

best_test = test_results.loc[test_results["Accuracy"].idxmax()]
best_cv = cv_results.loc[cv_results["CV ACCURACY Mean"].idxmax()]
best_cv_auc = cv_results.loc[cv_results["CV ROC_AUC Mean"].idxmax()]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Models evaluated", len(test_results))
c2.metric("Best test accuracy", f"{best_test['Accuracy']:.1%}", str(best_test["Model"]))
c3.metric("Best CV accuracy", f"{best_cv['CV ACCURACY Mean']:.1%}", str(best_cv["Model"]))
c4.metric("Best mean CV ROC-AUC", f"{best_cv_auc['CV ROC_AUC Mean']:.3f}", str(best_cv_auc["Model"]))
st.divider()

st.header("1. Held-Out Test Performance")
st.caption("Metrics below come from the held-out test set produced by src/train.py.")
fig, ax = plt.subplots(figsize=(10, 5))
test_results.set_index("Model")[test_metrics].plot(kind="bar", ax=ax, width=0.82)
ax.set_title("Test Metrics by Model")
ax.set_xlabel("Model")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.set_yticks([0, .2, .4, .6, .8, 1.0])
ax.tick_params(axis="x", rotation=0)
ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
ax.grid(axis="y", alpha=.25)
ax.set_axisbelow(True)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.dataframe(test_results.style.format({m: "{:.3f}" for m in test_metrics}), use_container_width=True, hide_index=True)
st.divider()

st.header("2. Five-Fold Cross-Validation")
st.caption("Each training fold fits its own preprocessing pipeline. Scores show mean performance across five folds.")
cv_display = cv_results[["Model"] + [col for col in cv_results.columns if col.startswith("CV ")]].copy()

fig, ax = plt.subplots(figsize=(10, 5))
cv_results.set_index("Model")[cv_metrics].plot(kind="bar", ax=ax, width=0.82)
ax.set_title("Mean Cross-Validation Metrics")
ax.set_xlabel("Model")
ax.set_ylabel("Mean score")
ax.set_ylim(0, 1)
ax.set_yticks([0, .2, .4, .6, .8, 1.0])
ax.tick_params(axis="x", rotation=0)
ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
ax.grid(axis="y", alpha=.25)
ax.set_axisbelow(True)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("Cross-Validation Results: Mean ± Standard Deviation")
st.dataframe(cv_display.style.format({col: "{:.4f}" for col in cv_display.columns if col != "Model"}), use_container_width=True, hide_index=True)
st.download_button("Download Cross-Validation Results CSV", data=cv_results.to_csv(index=False).encode("utf-8"), file_name="cross_validation_results.csv", mime="text/csv")
st.caption("Standard deviation describes variation across folds. It is not a confidence interval and does not establish clinical validity.")
st.divider()

st.header("3. Cross-Validation vs Test Accuracy")
comparison = test_results[["Model", "Accuracy"]].merge(
    cv_results[["Model", "CV ACCURACY Mean", "CV ACCURACY Std"]],
    on="Model", how="inner"
)
comparison = comparison.rename(columns={
    "Accuracy": "Test Accuracy",
    "CV ACCURACY Mean": "Cross-Validation Accuracy",
    "CV ACCURACY Std": "CV Standard Deviation",
})
comparison_long = comparison.melt(
    id_vars=["Model"],
    value_vars=["Test Accuracy", "Cross-Validation Accuracy"],
    var_name="Evaluation",
    value_name="Score",
)

fig, ax = plt.subplots(figsize=(9, 5))
comparison_long.pivot(index="Model", columns="Evaluation", values="Score").plot(kind="bar", ax=ax, width=.8)
ax.set_title("Test vs Cross-Validation Accuracy")
ax.set_xlabel("Model")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
ax.tick_params(axis="x", rotation=0)
ax.legend(title="Evaluation")
ax.grid(axis="y", alpha=.25)
ax.set_axisbelow(True)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.dataframe(comparison.style.format({
    "Test Accuracy": "{:.3f}",
    "Cross-Validation Accuracy": "{:.3f}",
    "CV Standard Deviation": "{:.3f}",
}), use_container_width=True, hide_index=True)
st.caption("Test and cross-validation accuracy use different subsets. The standard deviation describes variation across the five folds.")
st.divider()

st.header("4. Confusion Matrix")
X = df.drop(columns="target")
y = df["target"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42, stratify=y)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

selected_model = st.selectbox("Select a model for detailed evaluation", list(model_files.keys()))
model = load_model(str(model_files[selected_model]))
predictions = model.predict(X_test)

matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicted negative", "Predicted positive"],
            yticklabels=["Actual negative", "Actual positive"], ax=ax)
ax.set_xlabel("Predicted class")
ax.set_ylabel("Actual class")
ax.set_title(f"{selected_model} — Confusion Matrix")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

tn, fp, fn, tp = matrix.ravel()
c1, c2, c3, c4 = st.columns(4)
c1.metric("True negatives", int(tn))
c2.metric("False positives", int(fp))
c3.metric("False negatives", int(fn))
c4.metric("True positives", int(tp))

with st.expander("View classification report"):
    report = classification_report(y_test, predictions, labels=[0, 1],
        target_names=["No recorded disease", "Recorded disease"],
        output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

st.divider()

st.header("5. ROC Curve Comparison")
fig, ax = plt.subplots(figsize=(8, 5))
for name, path in model_files.items():
    pipeline = load_model(str(path))
    probs = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_value = roc_auc_score(y_test, probs)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_value:.3f})")

ax.plot([0, 1], [0, 1], "--", label="Random classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Receiver Operating Characteristic")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.legend(loc="lower right")
ax.grid(alpha=.25)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.caption("ROC-AUC measures ranking ability across thresholds; it does not measure calibration or establish clinical safety.")
st.success("Evaluation dashboard loaded successfully.")
