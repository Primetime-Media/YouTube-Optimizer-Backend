# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy Flask-specific requirements and install Python dependencies
COPY requirements-flask.txt .
RUN pip install --no-cache-dir -r requirements-flask.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_ENV=production

# Expose port (Cloud Run will set PORT env var)
EXPOSE 8080

# Run the Flask app
CMD ["python", "main.py"]