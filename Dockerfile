# Use the official Python image from Google Cloud
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

# Set working directory
WORKDIR /app

# Install system dependencies for PostgreSQL and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Required for psycopg2 (PostgreSQL adapter)
    libpq-dev \
    # For compiling Python packages
    build-essential \
    # Required for OpenCV
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install production dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8080

# Run the application with a single worker
# Cloud Run manages scaling, so we don't need multiple workers per container
CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1