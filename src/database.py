
import os
import streamlit as st
from sqlalchemy import create_engine, text

def get_database_url():
    try:
        secret_url = st.secrets.get("DATABASE_URL")
        if secret_url:
            return secret_url
    except Exception:
        pass

    return os.getenv("DATABASE_URL")


DATABASE_URL = get_database_url()

if DATABASE_URL:
    if DATABASE_URL.startswith("postgresql+psycopg://"):
        DATABASE_URL = DATABASE_URL.replace(
            "postgresql+psycopg://",
            "postgresql+psycopg2://",
            1
        )
    elif DATABASE_URL.startswith("postgresql://"):
        DATABASE_URL = DATABASE_URL.replace(
            "postgresql://",
            "postgresql+psycopg2://",
            1
        )
    elif DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace(
            "postgres://",
            "postgresql+psycopg2://",
            1
        )

engine = (
    create_engine(DATABASE_URL, pool_pre_ping=True)
    if DATABASE_URL
    else None
)


def init_db():
    if engine is None:
        raise RuntimeError("DATABASE_URL is not configured.")

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
    if engine is None:
        raise RuntimeError("Database is not configured.")

    with engine.begin() as connection:
        connection.execute(text("""
            INSERT INTO prediction_history
            (age, sex, prediction, probability, model_name)
            VALUES (:age, :sex, :prediction, :probability, :model_name)
        """), {
            "age": int(age),
            "sex": int(sex),
            "prediction": int(prediction),
            "probability": float(probability),
            "model_name": model_name
        })


def get_prediction_history():
    if engine is None:
        return []

    with engine.connect() as connection:
        result = connection.execute(text("""
            SELECT id, age, sex, prediction, probability,
                   model_name, created_at
            FROM prediction_history
            ORDER BY created_at DESC
        """))
        return result.mappings().all()
