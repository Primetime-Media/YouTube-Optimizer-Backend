# YouTube Optimizer Backend

A comprehensive FastAPI-based backend service for optimizing YouTube channels and videos using AI-powered analysis and recommendations.

## Overview

The YouTube Optimizer is an intelligent system that helps content creators optimize their YouTube presence through AI-driven insights, automated improvements, and comprehensive analytics. The system integrates with YouTube's API to fetch channel data, analyze content performance, and apply optimizations directly to YouTube.

## Key Features

### 🎯 AI-Powered Content Optimization
- **Smart Title Generation**: Uses Anthropic's Claude to create compelling, SEO-optimized video titles
- **Description Enhancement**: Automatically improves video descriptions with relevant keywords and formatting
- **Tag Optimization**: Generates relevant tags based on content analysis and trending topics
- **Multilingual Support**: Detects content language and creates appropriate hashtags
- **Chapter Generation**: Extracts chapter timestamps from video transcripts

### 🖼️ Intelligent Thumbnail Optimization
- **AI Frame Selection**: Identifies optimal moments in videos for thumbnail extraction
- **Quality Evaluation**: Uses Google's Gemini AI to score thumbnail effectiveness
- **Competitor Analysis**: Analyzes competitor thumbnails for insights and best practices
- **Custom Thumbnail Upload**: Automated thumbnail updates via YouTube API

### 📊 Comprehensive Analytics
- **Video Performance Metrics**: Detailed analytics including views, likes, comments, and engagement
- **Channel Analytics**: Channel-level performance tracking and optimization insights
- **Time-Series Data**: Granular view data for trend analysis
- **Optimization Impact**: Before/after comparison of optimization effectiveness

### 🔄 Automated Workflow Management
- **Background Processing**: Asynchronous optimization tasks with APScheduler
- **Batch Operations**: Process multiple videos and channels efficiently
- **Smart Scheduling**: Intelligent timing for optimization application
- **Progress Tracking**: Real-time status updates for all optimization tasks

## Architecture

### Core Components

**Main Application (`main.py`)**
- FastAPI application with OAuth2 authentication
- Session management with secure cookie handling
- CORS configuration for frontend integration
- Database initialization and connection management

**Routes**
- `analytics.py`: Analytics data retrieval and visualization endpoints
- `channel_routes.py`: Channel management and optimization endpoints  
- `scheduler_routes.py`: Background task scheduling and monitoring
- `video_routes.py`: Video optimization and management endpoints

**Services**
- `youtube.py`: YouTube API integration and data management
- `llm_optimization.py`: AI-powered content optimization using Claude
- `thumbnail_optimizer.py`: Thumbnail analysis and optimization
- `optimizer.py`: Main orchestration for applying optimizations
- `scheduler.py`: Background task management
- `competitor_analysis.py`: Competitor research and insights

**Utilities**
- `auth.py`: Authentication and credential management
- `db.py`: Database connection and query utilities

## Technology Stack

- **Framework**: FastAPI for high-performance API development
- **Database**: PostgreSQL with psycopg2 for data persistence
- **AI/ML**: Anthropic Claude for content optimization, Google Gemini for thumbnail analysis
- **YouTube Integration**: Google APIs for comprehensive YouTube interaction
- **Authentication**: OAuth2 with Google for secure user authentication
- **Task Processing**: APScheduler for background job management
- **Image Processing**: FFmpeg for video frame extraction and thumbnail generation

## Installation & Setup

### Prerequisites
- Python 3.10
- PostgreSQL database
- Google Cloud Project with YouTube API access
- Anthropic API key
- FFmpeg installed on system

### Environment Configuration
Create a `.env` file with the required configuration:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost/youtube_optimizer
POSTGRES_USER=your_db_user
POSTGRES_PASSWORD=your_db_password
POSTGRES_DB=youtube_optimizer

# API Keys
ANTHROPIC_API_KEY=your_anthropic_api_key

# OAuth
CLIENT_SECRET_FILE=path/to/your/google_client_secret.json
FRONTEND_URL=http://localhost:3000
REDIRECT_URI=http://localhost:8080/auth/callback

# Security
SESSION_SECRET=your_secure_session_secret
```

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database
python -c "from utils.db import init_db; init_db()"

# Run the application
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

## API Endpoints

### Authentication
- `GET /login` - Initiate OAuth2 flow
- `GET /auth/callback` - Handle OAuth2 callback
- `GET /api/me` - Get current user info
- `POST /logout` - End user session

### YouTube Data Management
- `GET /youtube-data/{user_id}` - Fetch user's YouTube data
- `POST /refresh-youtube-data/{user_id}` - Refresh YouTube data
- `POST /videos/{video_id}/fetch-transcript` - Fetch video transcript

### Optimization
- `POST /videos/{video_id}/optimize-all` - Comprehensive video optimization
- `PUT /video/{video_id}/optimization-status` - Update optimization status
- `GET /optimized-videos/{user_id}` - Get optimized videos list

### Analytics
- Video analytics endpoints
- Channel performance metrics
- Optimization impact tracking

## Security Features

- **OAuth2 Authentication**: Secure Google OAuth integration
- **Session Management**: Encrypted session tokens with expiration
- **CORS Protection**: Configured for frontend security
- **Credential Encryption**: Secure storage of user credentials
- **Permission Levels**: Role-based access control

## Optimization Process

1. **Data Collection**: Fetch YouTube channel and video data
2. **Content Analysis**: AI-powered analysis of titles, descriptions, and transcripts
3. **Competitor Research**: Analyze similar channels for insights
4. **Optimization Generation**: Create improved content using Claude AI
5. **Quality Validation**: Score and validate optimizations
6. **Application**: Apply approved optimizations to YouTube
7. **Performance Tracking**: Monitor impact and effectiveness

## Performance & Scalability

- **Asynchronous Processing**: Non-blocking operations for better performance
- **Background Tasks**: Resource-intensive operations handled asynchronously
- **Database Optimization**: Efficient queries and connection pooling
- **Caching**: Strategic caching for frequently accessed data
- **Rate Limiting**: YouTube API quota management

## Development

### Running in Development
```bash
# Start with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8080

# Database testing
curl http://localhost:8080/db-test
```

### Docker Support
```bash
# Build and run with Docker
docker build -t youtube-optimizer-backend .
docker run -p 8080:8080 youtube-optimizer-backend
```

## Contributing

The system is designed with modularity in mind - each service handles a specific domain and can be extended independently. Key areas for contribution:

- Additional AI model integration
- Enhanced analytics and reporting
- Advanced thumbnail optimization techniques
- Performance optimizations
- New optimization strategies

## License

This project is designed for YouTube content optimization and follows YouTube's Terms of Service and API guidelines.