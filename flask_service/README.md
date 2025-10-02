[# YouTube Optimizer - Flask Authentication Service

A lightweight Flask service for handling user authentication and onboarding for the YouTube Optimizer platform.

## Purpose

This service handles:
- User authentication processing from the frontend
- User data storage in PostgreSQL
- Video queueing for optimization (30-day filter)
- Independent scaling from the main optimization service

## Endpoints

- `GET /health` - Health check
- `POST /api/auth/process` - Process user authentication data

## Environment Variables

- `DATABASE_URL` - PostgreSQL connection string
- `DEBUG` - Debug mode (default: false)
- `SECRET_KEY` - Flask secret key
- `FLASK_HOST` - Host to bind to (default: 0.0.0.0)
- `FLASK_PORT` - Port to listen on (default: 5001)

## Cloud Run Deployment

This service is designed to deploy independently to Cloud Run with:
- Automatic scaling (0-5 instances for dev, 1-10 for prod)
- Direct Cloud SQL database access
- Lightweight container (~200MB vs 2GB+ for main service)

## Local Development

```bash
cd flask_service
python -m venv flask-venv
source flask-venv/bin/activate
pip install -r requirements-flask.txt
python main.py
```

## Architecture

- **Independent service** - Can scale and deploy separately from FastAPI service
- **Fault isolation** - Auth failures don't affect video optimization
- **Cost efficient** - Smaller container, lower resource usage](https://claude.ai/chat/fc262d4b-1c34-4406-ac9d-837eb2309b54)
