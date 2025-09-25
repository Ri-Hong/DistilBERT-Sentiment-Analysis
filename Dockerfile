# Stage 1: Build dependencies
FROM --platform=linux/amd64 python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Stage 2: Runtime
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy model files and service code
COPY model/artifacts/best /app/model/artifacts/best
COPY app/service.py /app/app/service.py

# Set environment variables
ENV MODEL_PATH=/app/model/artifacts/best
ENV TOKENIZER_NAME=distilbert-base-uncased
ENV PYTHONPATH=/app

# Create a non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 3000

# Run the service
CMD ["uvicorn", "app.service:app", "--host", "0.0.0.0", "--port", "3000"]
