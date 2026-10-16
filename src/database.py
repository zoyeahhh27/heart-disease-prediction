
import os
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2:///heart_disease_db"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)


def init_db():
    with engine.begin() as connection:
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id SERIAL PRIMARY KEY,
                age INTEGER NOT NULL,
                sex INTEGER NOT NULL,
                prediction INTEGER NOT NULL,
                probability REAL NOT NULL,
                model_name VARCHAR(50) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))


def save_prediction(age, sex, prediction, probability, model_name):
    with engine.begin() as connection:
        connection.execute(
            text("""
                INSERT INTO prediction_history
                (age, sex, prediction, probability, model_name)
                VALUES (:age, :sex, :prediction, :probability, :model_name)
            """),
            {
                "age": int(age),
                "sex": int(sex),
                "prediction": int(prediction),
                "probability": float(probability),
                "model_name": model_name
            }
        )


def get_prediction_history():
    with engine.connect() as connection:
        result = connection.execute(text("""
            SELECT id, age, sex, prediction, probability,
                   model_name, created_at
            FROM prediction_history
            ORDER BY created_at DESC
        """))
        return result.mappings().all()
