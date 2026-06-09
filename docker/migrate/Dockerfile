FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir alembic sqlalchemy psycopg2-binary pydantic pydantic-settings python-dotenv
COPY alembic/ alembic/
COPY alembic.ini .
COPY src/ src/
CMD ["alembic", "upgrade", "head"]
