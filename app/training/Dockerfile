# Stage 1: Build dependencies
FROM --platform=linux/amd64 python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /app/venv \
    && . /app/venv/bin/activate \
    && pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# Install git for MLflow versioning
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=/app/mlruns/tracking

# Copy application code and data
COPY app app/
COPY data data/

# Create necessary directories
RUN mkdir -p /app/model/artifacts /app/mlruns

# Set default command
CMD ["python", "-m", "app.training.train"]
