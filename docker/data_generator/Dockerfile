FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir psycopg2-binary numpy faker
COPY src/data/generate_data.py .
CMD ["python", "generate_data.py"]
