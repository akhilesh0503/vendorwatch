FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] sqlalchemy asyncpg psycopg2-binary \
    pydantic pydantic-settings python-dotenv \
    numpy pandas scikit-learn shap joblib \
    prometheus-fastapi-instrumentator structlog
COPY src/ src/
ENV PYTHONPATH=/app
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
