# ML Model serving Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn

COPY config/ config/
COPY src/ src/

EXPOSE 8000

CMD ["uvicorn", "src.ml.api:app", "--host", "0.0.0.0", "--port", "8000"]
