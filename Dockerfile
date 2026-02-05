# Multi-stage build for AI Learning Accelerator
FROM python:3.11-slim as base

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
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install pytest pytest-asyncio pytest-cov black isort mypy

# Copy source code
COPY . .

# Change ownership to app user
RUN chown -R appuser:appuser /app

# Switch to app user
USER appuser

# Expose port
EXPOSE 8000

# Command for development
CMD ["python", "-m", "uvicorn", "src.ai_learning_accelerator.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Copy source code
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini ./

# Create necessary directories
RUN mkdir -p /app/logs /app/models /app/data

# Change ownership to app user
RUN chown -R appuser:appuser /app

# Switch to app user
USER appuser

# Install the package
RUN pip install -e .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health/ || exit 1

# Command for production
CMD ["python", "-m", "uvicorn", "src.ai_learning_accelerator.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Default to production
FROM production