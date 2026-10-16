
from src.database import init_db, engine
from sqlalchemy import text

init_db()

with engine.connect() as connection:
    result = connection.execute(text("SELECT current_database()"))
    print("Connected to database:", result.scalar())

print("Database initialized successfully!")
