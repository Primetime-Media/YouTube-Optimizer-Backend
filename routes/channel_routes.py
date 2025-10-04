# .env.example
# Copy this file to .env and fill in your values

# ==================================
# Application Settings
# ==================================
APP_NAME="Channel Optimizer API"
APP_VERSION="1.0.0"
DEBUG=false
ENVIRONMENT=production  # development, staging, production

# ==================================
# API Configuration
# ==================================
API_V1_PREFIX="/api/v1"
ALLOWED_HOSTS=["yourdomain.com"]
CORS_ORIGINS=["https://yourdomain.com","https://app.yourdomain.com"]

# ==================================
# Security
# ==================================
# CRITICAL: Generate a secure random key for production
# python -c "import secrets; print(secrets.token_urlsafe(32))"
SECRET_KEY=your-super-secret-key-change-this-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# ==================================
# Database Configuration
# ==================================
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=channel_optimizer
DATABASE_USER=postgres
DATABASE_PASSWORD=your-database-password
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20
DATABASE_POOL_TIMEOUT=30

# ==================================
# Redis Configuration
# ==================================
REDIS_URL=redis://localhost:6379/0
CACHE_TTL=300

# ==================================
# Rate Limiting
# ==================================
RATE_LIMITING_ENABLED=true
RATE_LIMIT_STORAGE=redis

# ==================================
# Logging
# ==================================
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT=json  # json or text

# ==================================
# Metrics & Monitoring
# ==================================
METRICS_ENABLED=true
METRICS_BACKEND=prometheus  # prometheus, statsd, cloudwatch, datadog, console
METRICS_PORT=9090

# ==================================
# YouTube API
# ==================================
YOUTUBE_API_KEY=your-youtube-api-key
YOUTUBE_QUOTA_PER_DAY=10000

# ==================================
# Background Tasks (Celery)
# ==================================
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2

# ==================================
# Error Tracking (Sentry)
# ==================================
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id
SENTRY_ENVIRONMENT=production
SENTRY_TRACES_SAMPLE_RATE=0.1

# ==================================
# Feature Flags
# ==================================
ENABLE_SWAGGER_UI=true
ENABLE_RATE_LIMITING=true
ENABLE_METRICS=true
ENABLE_CACHING=true

# ==================================
# AWS Configuration (if using CloudWatch)
# ==================================
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key

# ==================================
# Development Settings
# ==================================
# Uncomment for local development
# DEBUG=true
# ENVIRONMENT=development
# LOG_LEVEL=DEBUG
# ENABLE_SWAGGER_UI=true
# CORS_ORIGINS=["http://localhost:3000"]
