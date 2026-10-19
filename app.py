
import streamlit as st
import pandas as pd
import joblib

from pathlib import Path
from src.database import (
    init_db,
    save_prediction,
    get_prediction_history,
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
)

st.title("❤️ Heart Disease Prediction System")
st.caption("Machine learning demonstration using the UCI Heart Disease dataset.")

st.warning(
    "Educational use only. This application is not a medical diagnostic tool "
    "and must not be used to make medical or treatment decisions."
)

# --------------------------------------------------
# DATABASE INITIALIZATION
# --------------------------------------------------
try:
    init_db()
    database_ready = True
except Exception:
    database_ready = False
    st.warning(
        "The database is currently unavailable. "
        "You can still use the prediction feature, "
        "but prediction history will not be saved."
    )
# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------
MODEL_PATH = Path("models/xgboost.joblib")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("Trained model not found. Run: python src/train.py")
    st.stop()

model = load_model()

# --------------------------------------------------
# PATIENT INPUT FORM
# --------------------------------------------------
st.header("Patient Measurements")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (years)", 18, 100, 50)

        sex = st.selectbox(
            "Sex (dataset coding)",
            [0, 1],
            format_func=lambda x: {
                0: "Female (0)",
                1: "Male (1)",
            }[x],
        )

        cp = st.selectbox(
            "Chest pain type",
            [1, 2, 3, 4],
            format_func=lambda x: {
                1: "Typical angina (1)",
                2: "Atypical angina (2)",
                3: "Non-anginal pain (3)",
                4: "Asymptomatic (4)",
            }[x],
        )

        trestbps = st.number_input(
            "Resting blood pressure (mm Hg)",
            70, 250, 120,
        )

        chol = st.number_input(
            "Serum cholesterol (mg/dl)",
            100, 700, 200,
        )

    with col2:
        fbs = st.selectbox(
            "Fasting blood sugar > 120 mg/dl?",
            [0, 1],
            format_func=lambda x: {
                0: "No (0)",
                1: "Yes (1)",
            }[x],
        )

        restecg = st.selectbox(
            "Resting ECG result",
            [0, 1, 2],
            format_func=lambda x: {
                0: "Normal (0)",
                1: "ST-T abnormality (1)",
                2: "LV hypertrophy (2)",
            }[x],
        )

        thalach = st.number_input(
            "Maximum heart rate achieved",
            60, 230, 150,
        )

        exang = st.selectbox(
            "Exercise-induced angina",
            [0, 1],
            format_func=lambda x: {
                0: "No (0)",
                1: "Yes (1)",
            }[x],
        )

    with col3:
        oldpeak = st.number_input(
            "ST depression (oldpeak)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
        )

        slope = st.selectbox(
            "Peak exercise ST slope",
            [1, 2, 3],
            format_func=lambda x: {
                1: "Upsloping (1)",
                2: "Flat (2)",
                3: "Downsloping (3)",
            }[x],
        )

        ca = st.selectbox("Major vessels (ca)", [0, 1, 2, 3])

        thal = st.selectbox(
            "Thal test result",
            [3, 6, 7],
            format_func=lambda x: {
                3: "Normal (3)",
                6: "Fixed defect (6)",
                7: "Reversible defect (7)",
            }[x],
        )

    submitted = st.form_submit_button(
        "Run Prediction",
        type="primary",
        use_container_width=True,
    )

# --------------------------------------------------
# MAKE PREDICTION AND SAVE TO DATABASE
# --------------------------------------------------
if submitted:
    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }])

    prediction = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]

    positive_index = list(model.classes_).index(1)
    probability = float(probabilities[positive_index])

    st.divider()
    st.header("Prediction Result")

    metric_col1, metric_col2 = st.columns(2)

    metric_col1.metric(
        "Positive-class model probability",
        f"{probability:.1%}",
    )

    metric_col2.metric(
        "Predicted class",
        "Positive (1)" if prediction == 1 else "Negative (0)",
    )

    if prediction == 1:
        st.error(
            "The model classified this input as the positive class "
            "(recorded disease in the dataset)."
        )
    else:
        st.success(
            "The model classified this input as the negative class "
            "(no recorded disease in the dataset)."
        )

    st.caption(
        "This probability is a model output, not a validated estimate "
        "of an individual's actual disease risk."
    )

    if database_ready:
        try:
            save_prediction(
                age=age,
                sex=sex,
                prediction=prediction,
                probability=probability,
                model_name="XGBoost",
            )
            st.success("Prediction saved to PostgreSQL.")
        except Exception as exc:
            st.error(f"Prediction could not be saved: {exc}")

# --------------------------------------------------
# PREDICTION HISTORY DASHBOARD
# --------------------------------------------------
st.divider()
st.header("Prediction History")

if database_ready:
    try:
        history = get_prediction_history()

        if history:
            history_df = pd.DataFrame(history)

            total = len(history_df)
            positive = int((history_df["prediction"] == 1).sum())
            negative = int((history_df["prediction"] == 0).sum())

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Predictions", total)
            col2.metric("Positive Classifications", positive)
            col3.metric("Negative Classifications", negative)

            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True,
            )

            csv = history_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download History as CSV",
                data=csv,
                file_name="prediction_history.csv",
                mime="text/csv",
            )
        else:
            st.info(
                "No prediction history yet. Submit the form above "
                "to save your first record."
            )

    except Exception as exc:
        st.error(f"Could not load prediction history: {exc}")
else:
    st.info("Prediction history is unavailable until the database reconnects.")
