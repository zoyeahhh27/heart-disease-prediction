
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from xgboost import XGBClassifier


# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "heart_cleaned.csv"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}. "
        "Run python src/preprocess.py first."
    )

df = pd.read_csv(DATA_PATH)

if "target" not in df.columns:
    raise ValueError("Dataset must contain a 'target' column.")

X = df.drop(columns="target")
y = df["target"].astype(int)

# This project uses binary target labels: 0 and 1.
if set(y.unique()) != {0, 1}:
    raise ValueError(
        "Expected binary target labels 0 and 1. "
        "Check your preprocessing script."
    )

# Identify categorical columns.
# These features represent categories, not continuous measurements.
categorical_columns = [
    col for col in [
        "sex", "cp", "fbs", "restecg", "exang",
        "slope", "ca", "thal"
    ]
    if col in X.columns
]

numerical_columns = [
    col for col in X.columns
    if col not in categorical_columns
]


# --------------------------------------------------
# PREPROCESSING PIPELINE
# --------------------------------------------------
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numerical_columns),
    ("categorical", categorical_pipeline, categorical_columns),
])


# --------------------------------------------------
# DEFINE MODELS
# --------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
    ),
}


# --------------------------------------------------
# TRAIN/TEST SPLIT
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("=" * 65)
print("HEART DISEASE PREDICTION — MODEL EVALUATION")
print("=" * 65)
print(f"Dataset size: {len(df)}")
print(f"Training records: {len(X_train)}")
print(f"Test records: {len(X_test)}")
print(f"Features: {X.shape[1]}")
print(f"Cross-validation folds: {N_SPLITS}")


# --------------------------------------------------
# STRATIFIED 5-FOLD CROSS-VALIDATION
# --------------------------------------------------
cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

cv_rows = []
test_rows = []

for name, classifier in models.items():
    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    # Each fold fits preprocessing only on that fold's training data.
    cv_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    cv_results = cross_validate(
        cv_pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=1,
        error_score="raise",
    )

    cv_row = {"Model": name}

    for metric in scoring:
        scores = cv_results[f"test_{metric}"]
        cv_row[f"CV {metric.upper()} Mean"] = scores.mean()
        cv_row[f"CV {metric.upper()} Std"] = scores.std(ddof=1)

    cv_rows.append(cv_row)

    print("\n5-fold cross-validation:")
    for metric in scoring:
        scores = cv_results[f"test_{metric}"]
        print(
            f"{metric.upper():10s}: "
            f"{scores.mean():.4f} ± {scores.std(ddof=1):.4f}"
        )

    # Fit a final pipeline on the full training split.
    final_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    final_pipeline.fit(X_train, y_train)

    predictions = final_pipeline.predict(X_test)
    probabilities = final_pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    auc = roc_auc_score(y_test, probabilities)

    test_rows.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": auc,
    })

    print("\nHeld-out test-set report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["No recorded disease", "Recorded disease"],
        zero_division=0,
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print(f"Test ROC-AUC: {auc:.4f}")

    # Save the fitted preprocessing + model pipeline.
    filename = name.lower().replace(" ", "_") + ".joblib"
    model_path = MODEL_DIR / filename
    joblib.dump(final_pipeline, model_path)

    print(f"Saved model: {model_path}")


# --------------------------------------------------
# SAVE COMPARISON RESULTS
# --------------------------------------------------
test_results = pd.DataFrame(test_rows)
cv_results_df = pd.DataFrame(cv_rows)

test_results = test_results.sort_values(
    "ROC-AUC", ascending=False
)

test_results.to_csv(
    MODEL_DIR / "model_comparison.csv",
    index=False,
)

cv_results_df.to_csv(
    MODEL_DIR / "cross_validation_results.csv",
    index=False,
)

print("\n" + "=" * 65)
print("HELD-OUT TEST SET — MODEL COMPARISON")
print("=" * 65)
print(test_results.round(4).to_string(index=False))

print("\n" + "=" * 65)
print("5-FOLD CROSS-VALIDATION — MEAN ± STANDARD DEVIATION")
print("=" * 65)
print(cv_results_df.round(4).to_string(index=False))

print("\nSaved:")
print(f"- {MODEL_DIR / 'model_comparison.csv'}")
print(f"- {MODEL_DIR / 'cross_validation_results.csv'}")
print("- All three fitted model pipelines")
print("\nTraining and evaluation completed successfully.")
