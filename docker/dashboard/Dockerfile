FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir streamlit plotly pandas httpx
COPY dashboard/ dashboard/
ENV PYTHONPATH=/app/dashboard
WORKDIR /app/dashboard
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
