FROM python:3.11-slim

WORKDIR /mlflow

# Install mlflow
RUN pip install --no-cache-dir mlflow

# Expose MLflow port
EXPOSE 5050

# Default command
CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5050", "--backend-store-uri", "/mlruns"]
