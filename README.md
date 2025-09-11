# YouTube Optimizer Backend

A comprehensive FastAPI-based backend service for optimizing YouTube channels and videos using AI-powered analysis and recommendations.

## 🎯 What It Does

The YouTube Optimizer helps content creators automatically improve their YouTube presence through:

- **AI-Powered Content Optimization**: Smart title generation, description enhancement, and tag optimization using Claude AI
- **Intelligent Thumbnail Analysis**: AI frame selection and quality evaluation using Google Gemini
- **Comprehensive Analytics**: Video and channel performance tracking with optimization impact analysis
- **Automated Workflows**: Background processing and smart scheduling for optimization tasks

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │    │   PostgreSQL    │
│   (React/Vue)   │◄──►│   Backend       │◄──►│   Database      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   AI Services   │
                       │ • Claude AI     │
                       │ • Google Gemini │
                       │ • YouTube API   │
                       └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- PostgreSQL database
- Google Cloud Project with YouTube API access
- Anthropic API key

### Installation
```bash
# Clone and setup
git clone <repository-url>
cd YouTube-Optimizer-Backend

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your configuration

# Initialize database
python -c "from utils.db import init_db; init_db()"

# Run the application
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

**API Documentation**: Visit `http://localhost:8080/docs` for interactive OpenAPI documentation.

## 📖 Usage Examples

### Authenticate and Get User Info
```bash
# Start OAuth flow
curl http://localhost:8080/login

# Get current user
curl http://localhost:8080/api/me
```

### Optimize a Video
```bash
# Generate optimization for a video
curl -X POST http://localhost:8080/video/{video_id}/optimize-all

# Check optimization status
curl http://localhost:8080/video/{video_id}/optimization-status

# Apply optimization to YouTube
curl -X POST http://localhost:8080/video/{video_id}/apply-optimization
```

### Get Analytics
```bash
# Get video analyticsSdDsLgYKqhY
curl "http://localhost:8080/analytics/video/{video_id}?start_date=2024-01-01&end_date=2024-01-31"

# Get channel analytics
curl "http://localhost:8080/analytics/channel/{channel_id}?start_date=2024-01-01&end_date=2024-01-31"
```

## 🔧 Environment Variables

Create a `.env` file with the following configuration:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/youtube_optimizer
POSTGRES_USER=your_db_user
POSTGRES_PASSWORD=your_db_password
POSTGRES_DB=youtube_optimizer

# API Keys
ANTHROPIC_API_KEY=your_anthropic_api_key

# OAuth Configuration
CLIENT_SECRET_FILE=path/to/your/google_client_secret.json
FRONTEND_URL=http://localhost:3000
REDIRECT_URI=http://localhost:8080/auth/callback

# Security
SESSION_SECRET=your_secure_session_secret_32_chars_minimum

# Optional
DEBUG=true
ENVIRONMENT=development
```

See `.env.example` for complete configuration options.

## 🐳 Deployment

### Docker
```bash
# Build and run
docker build -t youtube-optimizer-backend .
docker run -p 8080:8080 \
  -e DATABASE_URL="postgresql://user:pass@host:5432/db" \
  -e ANTHROPIC_API_KEY="your_key" \
  youtube-optimizer-backend
```

### Docker Compose
```yaml
version: '3.8'
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: youtube_optimizer
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"

  backend:
    build: .
    ports:
      - "8080:8080"
    environment:
      DATABASE_URL: postgresql://postgres:password@postgres:5432/youtube_optimizer
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
    depends_on:
      - postgres
```

### Cloud Deployment
- **Google Cloud Run**: Ready for serverless deployment
- **AWS ECS/Fargate**: Container orchestration support
- **Kubernetes**: Helm charts available

## 🔒 Security

- **OAuth2 Authentication**: Secure Google OAuth integration
- **Session Management**: Encrypted session tokens with expiration
- **CORS Protection**: Configured for trusted frontend domains
- **Input Validation**: Pydantic models for request/response validation
- **Rate Limiting**: Built-in protection against abuse

## 🛠️ Development

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

# Format code
black .
isort .
```

### Project Structure
```
YouTube-Optimizer-Backend/
├── main.py                 # FastAPI application
├── routes/                  # API endpoints
├── services/               # Business logic
├── utils/                  # Utilities
├── flask_service/          # Flask auth service
└── tests/                  # Test files
```

## 📚 Documentation

- **API Reference**: `http://localhost:8080/docs` (Swagger UI)
- **Alternative Docs**: `http://localhost:8080/redoc` (ReDoc)
- **OpenAPI Schema**: `http://localhost:8080/openapi.json`

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Quick Contribution Steps
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Submit a pull request


## 📄 License

This project is open source and follows YouTube's Terms of Service and API guidelines.

## 🆘 Support

- **Documentation**: Check this README and API docs first
- **Issues**: Search existing GitHub issues
- **Discussions**: Use GitHub Discussions for questions

### Common Issues

**Database Connection Issues**
```bash
curl http://localhost:8080/db-test
```

**Authentication Issues**
- Verify Google OAuth configuration
- Check client secret file path
- Ensure proper redirect URI

---

**Made with ❤️ for YouTube creators worldwide**