FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install streamlit plotly openai python-dotenv evidently

# Copy application code and artifacts
COPY . /app

# Expose Streamlit ports
EXPOSE 8501 8502 8503 8504

# Set environment variables
ENV PYTHONPATH=/app

# Default command (can be overridden in docker-compose)
CMD ["streamlit", "run", "src/dashboards/persona_app.py", "--server.port", "8502", "--server.address", "0.0.0.0"]
