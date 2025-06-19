# TEMPO Docker Image
# Multi-stage build for efficient image size

# Stage 1: Python base with dependencies
FROM python:3.10-slim as python-base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-test.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Stage 2: Node.js for frontend build
FROM node:18-slim as frontend-builder

WORKDIR /app/frontend

# Copy frontend files
COPY frontend/package*.json ./
RUN npm ci --only=production

COPY frontend/ ./
RUN npm run build

# Stage 3: Final image
FROM python-base as final

# Copy application code
COPY . /app/

# Copy built frontend from previous stage
COPY --from=frontend-builder /app/frontend/build /app/frontend/build

# Create non-root user
RUN useradd -m -u 1000 tempo && \
    chown -R tempo:tempo /app

USER tempo

# Create necessary directories
RUN mkdir -p /app/output /app/logs /app/.cache

# Expose ports
EXPOSE 8000 5173 5174

# Set environment variables
ENV TEMPO_API_HOST=0.0.0.0 \
    TEMPO_API_PORT=8000 \
    HF_HOME=/app/.cache/huggingface

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]