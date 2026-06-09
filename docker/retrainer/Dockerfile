FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir \
    sqlalchemy psycopg2-binary pydantic pydantic-settings python-dotenv \
    numpy pandas scikit-learn shap joblib \
    apscheduler
COPY src/ src/
ENV PYTHONPATH=/app
CMD ["python", "-m", "src.services.retrainer"]
