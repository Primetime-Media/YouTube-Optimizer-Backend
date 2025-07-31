#!/bin/bash

# YouTube Optimizer Flask Auth Service - Cloud Run Deployment Script
# Usage: ./deploy.sh [environment]
# Environment: dev (default) or prod

ENVIRONMENT=${1:-dev}
PROJECT_ID="youtube-optimizer-454919"
REGION="us-central1"
SERVICE_NAME="youtube-optimizer-auth"

echo "🚀 Deploying Flask Auth Service to Cloud Run"
echo "Environment: $ENVIRONMENT"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Service: $SERVICE_NAME"

# Set environment-specific configurations
if [ "$ENVIRONMENT" = "prod" ]; then
    MIN_INSTANCES=1
    MAX_INSTANCES=10
    MEMORY="512Mi"
    CPU="1"
    TIMEOUT="300"
else
    MIN_INSTANCES=0
    MAX_INSTANCES=5
    MEMORY="256Mi"
    CPU="0.5"
    TIMEOUT="60"
fi

echo "📦 Building and deploying to Cloud Run..."

gcloud run deploy $SERVICE_NAME \
    --source . \
    --platform managed \
    --region $REGION \
    --project $PROJECT_ID \
    --allow-unauthenticated \
    --port 8080 \
    --memory $MEMORY \
    --cpu $CPU \
    --timeout $TIMEOUT \
    --min-instances $MIN_INSTANCES \
    --max-instances $MAX_INSTANCES \
    --set-env-vars "FLASK_ENV=production" \
    --set-env-vars "DEBUG=false" \
    --concurrency 80

if [ $? -eq 0 ]; then
    echo "✅ Deployment successful!"
    echo "🌐 Service URL:"
    gcloud run services describe $SERVICE_NAME --region=$REGION --project=$PROJECT_ID --format="value(status.url)"
    
    echo ""
    echo "🔧 Next steps:"
    echo "1. Set up environment variables (DATABASE_URL, etc.) in Cloud Run console"
    echo "2. Configure service-to-service authentication if needed"
    echo "3. Update your frontend to use the new service URL"
else
    echo "❌ Deployment failed!"
    exit 1
fi