# Heart Disease Prediction System

An educational Streamlit machine-learning application built with the UCI Heart Disease dataset.

## Features

- Prediction form using the trained XGBoost pipeline
- PostgreSQL prediction history
- Model evaluation dashboard comparing Logistic Regression, Random Forest, and XGBoost
- SHAP explainability dashboard
- Export prediction history as CSV

## Project structure

```text
.
├── app.py
├── pages/
│   ├── Model_Evaluation.py
│   └── SHAP_Explainability.py
├── src/
│   ├── database.py
│   ├── download_data.py
│   ├── eda.py
│   ├── preprocess.py
│   └── train.py
├── models/
│   ├── xgboost.joblib
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── model_comparison.csv
│   └── cross_validation_results.csv
├── data/
├── requirements.txt
└── .gitignore
```

## Run locally

Use Python 3.12 or another Python version compatible with all packages and your saved model artifacts.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The application expects `models/xgboost.joblib` to exist. The evaluation and explainability pages may require the other model artifacts and the saved evaluation files.

## PostgreSQL configuration

Set `DATABASE_URL` to a PostgreSQL connection URL. For example:

```bash
export DATABASE_URL='postgresql+psycopg2://USERNAME:PASSWORD@HOST:5432/DATABASE_NAME'
```

Do not commit database credentials to GitHub. Configure `DATABASE_URL` in your deployment platform's secrets/environment settings.

If `DATABASE_URL` is not set, the current application attempts to connect to a local PostgreSQL database named `heart_disease_db`; that local database will not automatically exist in the cloud.

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository.
3. Set the main file path to `app.py`.
4. Configure `DATABASE_URL` using the app's secrets/environment settings supported by the deployment platform.
5. Deploy and test a prediction, prediction history, Model Evaluation, and SHAP Explainability.

## Important notes

- Keep the trained model files required by the app in the repository or provide a reliable model-download process.
- Never commit `.env` files, database passwords, or credentials.
- Test database connectivity and every dashboard after deployment.
- This is an educational demonstration, not a clinically validated diagnostic tool. Predictions are not medical advice and must not be used for diagnosis or treatment decisions.
