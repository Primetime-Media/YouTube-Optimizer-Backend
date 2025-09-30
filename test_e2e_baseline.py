"""
End-to-End Test Suite for YouTube Optimizer Backend
Tests the existing code with dummy data to establish baseline
"""

import sqlite3
import json
import sys
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestDatabase:
    """In-memory SQLite database for testing"""
    
    def __init__(self):
        self.conn = sqlite3.connect(':memory:')
        self.conn.row_factory = sqlite3.Row
        self.setup_schema()
        self.insert_test_data()
    
    def setup_schema(self):
        """Create database schema matching PostgreSQL structure"""
        cursor = self.conn.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # YouTube channels table
        cursor.execute("""
            CREATE TABLE youtube_channels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                channel_id TEXT UNIQUE NOT NULL,
                kind TEXT,
                etag TEXT,
                title TEXT,
                description TEXT,
                custom_url TEXT,
                published_at TIMESTAMP,
                view_count INTEGER DEFAULT 0,
                subscriber_count INTEGER DEFAULT 0,
                hidden_subscriber_count BOOLEAN DEFAULT FALSE,
                video_count INTEGER DEFAULT 0,
                thumbnail_url_default TEXT,
                thumbnail_url_medium TEXT,
                thumbnail_url_high TEXT,
                uploads_playlist_id TEXT,
                banner_url TEXT,
                privacy_status TEXT,
                is_linked BOOLEAN DEFAULT FALSE,
                long_uploads_status TEXT,
                is_monetization_enabled BOOLEAN DEFAULT FALSE,
                topic_ids TEXT,
                topic_categories TEXT,
                overall_good_standing BOOLEAN DEFAULT TRUE,
                community_guidelines_good_standing BOOLEAN DEFAULT TRUE,
                copyright_strikes_good_standing BOOLEAN DEFAULT TRUE,
                content_id_claims_good_standing BOOLEAN DEFAULT TRUE,
                branding_settings TEXT,
                audit_details TEXT,
                topic_details TEXT,
                status_details TEXT,
                is_optimized BOOLEAN DEFAULT FALSE,
                last_optimized_at TIMESTAMP,
                last_optimization_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # YouTube videos table
        cursor.execute("""
            CREATE TABLE youtube_videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id INTEGER NOT NULL,
                video_id TEXT UNIQUE NOT NULL,
                kind TEXT,
                etag TEXT,
                playlist_item_id TEXT,
                title TEXT,
                description TEXT,
                published_at TIMESTAMP,
                channel_title TEXT,
                playlist_id TEXT,
                position INTEGER DEFAULT 0,
                tags TEXT,
                thumbnail_url_default TEXT,
                thumbnail_url_medium TEXT,
                thumbnail_url_high TEXT,
                thumbnail_url_standard TEXT,
                thumbnail_url_maxres TEXT,
                view_count INTEGER DEFAULT 0,
                like_count INTEGER DEFAULT 0,
                comment_count INTEGER DEFAULT 0,
                duration TEXT,
                transcript TEXT,
                has_captions BOOLEAN DEFAULT FALSE,
                caption_language TEXT,
                privacy_status TEXT,
                upload_status TEXT,
                license TEXT,
                embeddable BOOLEAN,
                public_stats_viewable BOOLEAN,
                definition TEXT,
                dimension TEXT,
                has_custom_thumbnail BOOLEAN DEFAULT FALSE,
                projection TEXT,
                category_id TEXT,
                category_name TEXT,
                topic_ids TEXT,
                topic_categories TEXT,
                content_details TEXT,
                status_details TEXT,
                topic_details TEXT,
                is_optimized BOOLEAN DEFAULT FALSE,
                last_optimized_at TIMESTAMP,
                last_optimization_id INTEGER,
                last_analytics_refresh TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (channel_id) REFERENCES youtube_channels(id)
            )
        """)
        
        # Channel optimizations table
        cursor.execute("""
            CREATE TABLE channel_optimizations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id INTEGER NOT NULL,
                original_description TEXT,
                optimized_description TEXT,
                original_keywords TEXT,
                optimized_keywords TEXT,
                optimization_notes TEXT,
                is_applied BOOLEAN DEFAULT FALSE,
                applied_at TIMESTAMP,
                status TEXT DEFAULT 'pending',
                progress INTEGER DEFAULT 0,
                created_by TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (channel_id) REFERENCES youtube_channels(id)
            )
        """)
        
        # Video optimizations table
        cursor.execute("""
            CREATE TABLE video_optimizations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                original_title TEXT,
                optimized_title TEXT,
                original_description TEXT,
                optimized_description TEXT,
                original_tags TEXT,
                optimized_tags TEXT,
                optimization_notes TEXT,
                optimization_step INTEGER DEFAULT 1,
                is_applied BOOLEAN DEFAULT FALSE,
                applied_at TIMESTAMP,
                status TEXT DEFAULT 'pending',
                progress INTEGER DEFAULT 0,
                created_by TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES youtube_videos(id)
            )
        """)
        
        # Video timeseries data table
        cursor.execute("""
            CREATE TABLE video_timeseries_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                views INTEGER DEFAULT 0,
                estimated_minutes_watched REAL,
                average_view_percentage REAL,
                raw_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(video_id, timestamp),
                FOREIGN KEY (video_id) REFERENCES youtube_videos(id)
            )
        """)
        
        # Channel optimization schedules table
        cursor.execute("""
            CREATE TABLE channel_optimization_schedules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id INTEGER NOT NULL,
                auto_apply BOOLEAN DEFAULT FALSE,
                is_active BOOLEAN DEFAULT TRUE,
                last_run TIMESTAMP,
                next_run TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (channel_id) REFERENCES youtube_channels(id)
            )
        """)
        
        # Scheduler run history table
        cursor.execute("""
            CREATE TABLE scheduler_run_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                schedule_id INTEGER NOT NULL,
                optimization_id INTEGER,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                status TEXT DEFAULT 'running',
                applied BOOLEAN DEFAULT FALSE,
                error_message TEXT,
                FOREIGN KEY (schedule_id) REFERENCES channel_optimization_schedules(id)
            )
        """)
        
        self.conn.commit()
    
    def insert_test_data(self):
        """Insert test data for E2E tests"""
        cursor = self.conn.cursor()
        
        # Insert test user
        cursor.execute("""
            INSERT INTO users (id, email) VALUES (1, 'test@example.com')
        """)
        
        # Insert test channel
        cursor.execute("""
            INSERT INTO youtube_channels (
                id, user_id, channel_id, title, description, 
                subscriber_count, video_count, uploads_playlist_id,
                branding_settings, published_at
            ) VALUES (
                1, 1, 'UC_test_channel_id', 'Test Channel', 
                'Original channel description',
                10000, 50, 'UU_test_uploads',
                ?, ?
            )
        """, (
            json.dumps({
                "channel": {
                    "description": "Original channel description",
                    "keywords": "test, youtube, channel"
                }
            }),
            datetime.now(timezone.utc).isoformat()
        ))
        
        # Insert test videos
        videos = [
            {
                'id': 1,
                'video_id': 'test_video_1',
                'title': 'How to Make Perfect Coffee',
                'description': 'Learn the secrets to brewing perfect coffee at home',
                'tags': json.dumps(['coffee', 'brewing', 'tutorial']),
                'view_count': 5000,
                'like_count': 250,
                'comment_count': 50,
                'transcript': 'Welcome to this coffee brewing tutorial. Today I will show you...',
                'has_captions': True,
                'category_id': '22',
                'category_name': 'People & Blogs',
                'duration': 'PT10M30S',
                'published_at': (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
            },
            {
                'id': 2,
                'video_id': 'test_video_2',
                'title': 'Best Coffee Beans 2024',
                'description': 'Review of the top coffee beans this year',
                'tags': json.dumps(['coffee', 'review', 'beans']),
                'view_count': 8000,
                'like_count': 400,
                'comment_count': 80,
                'transcript': 'In this video, I review the best coffee beans available...',
                'has_captions': True,
                'category_id': '22',
                'category_name': 'People & Blogs',
                'duration': 'PT15M45S',
                'published_at': (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
            },
            {
                'id': 3,
                'video_id': 'test_video_3',
                'title': 'Coffee Shop Setup Guide',
                'description': 'Everything you need to know about setting up a coffee shop',
                'tags': json.dumps(['coffee', 'business', 'guide']),
                'view_count': 12000,
                'like_count': 600,
                'comment_count': 120,
                'transcript': 'Starting a coffee shop requires careful planning...',
                'has_captions': True,
                'category_id': '22',
                'category_name': 'People & Blogs',
                'duration': 'PT20M15S',
                'published_at': (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
            }
        ]
        
        for video in videos:
            cursor.execute("""
                INSERT INTO youtube_videos (
                    id, channel_id, video_id, title, description, tags,
                    view_count, like_count, comment_count, transcript, has_captions,
                    category_id, category_name, duration, published_at
                ) VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                video['id'], video['video_id'], video['title'], 
                video['description'], video['tags'],
                video['view_count'], video['like_count'], video['comment_count'],
                video['transcript'], video['has_captions'],
                video['category_id'], video['category_name'], 
                video['duration'], video['published_at']
            ))
        
        # Insert timeseries data for video analytics
        for day in range(7):
            date = datetime.now(timezone.utc) - timedelta(days=day)
            cursor.execute("""
                INSERT INTO video_timeseries_data (
                    video_id, timestamp, views, estimated_minutes_watched, average_view_percentage
                ) VALUES (1, ?, ?, ?, ?)
            """, (
                date.isoformat(),
                100 + day * 10,
                50.0 + day * 5.0,
                45.0 + day * 2.0
            ))
        
        self.conn.commit()
    
    def get_connection(self):
        """Return database connection (mimics utils.db.get_connection)"""
        return self.conn


class MockExternalAPIs:
    """Mock all external API calls"""
    
    @staticmethod
    def mock_anthropic_client():
        """Mock Anthropic Claude API"""
        mock_client = Mock()
        
        # Mock message creation
        mock_response = Mock()
        mock_response.content = [Mock(text=json.dumps({
            "optimized_title": "How to Make Perfect Coffee - Complete Guide 2024",
            "optimized_description": "Learn professional coffee brewing techniques in this comprehensive tutorial. Perfect for beginners and coffee enthusiasts!",
            "optimized_tags": ["coffee", "brewing", "tutorial", "howto", "coffeemaking", "barista", "espresso", "guide"],
            "optimization_notes": "Improved title with year and 'Complete Guide' for better SEO. Enhanced description with target audience mention.",
            "optimization_score": 0.85
        }))]
        
        mock_client.messages.create.return_value = mock_response
        return mock_client
    
    @staticmethod
    def mock_youtube_client():
        """Mock YouTube Data API"""
        mock_client = Mock()
        
        # Mock videos().list()
        mock_client.videos().list().execute.return_value = {
            'items': [{
                'id': 'test_video_1',
                'snippet': {
                    'title': 'How to Make Perfect Coffee',
                    'description': 'Learn the secrets...',
                    'tags': ['coffee', 'brewing']
                }
            }]
        }
        
        # Mock videos().update()
        mock_client.videos().update().execute.return_value = {'id': 'test_video_1'}
        
        # Mock channels().list()
        mock_client.channels().list().execute.return_value = {
            'items': [{
                'id': 'UC_test_channel_id',
                'brandingSettings': {
                    'channel': {
                        'description': 'Test channel',
                        'keywords': 'test, youtube'
                    }
                }
            }]
        }
        
        return mock_client
    
    @staticmethod
    def mock_youtube_analytics_client():
        """Mock YouTube Analytics API"""
        mock_client = Mock()
        
        # Mock reports().query()
        mock_client.reports().query().execute.return_value = {
            'columnHeaders': [
                {'name': 'day'},
                {'name': 'views'},
                {'name': 'estimatedMinutesWatched'},
                {'name': 'averageViewPercentage'}
            ],
            'rows': [
                ['2024-01-01', 100, 50.0, 45.0],
                ['2024-01-02', 120, 60.0, 47.0],
                ['2024-01-03', 110, 55.0, 46.0]
            ]
        }
        
        return mock_client
    
    @staticmethod
    def mock_serpapi():
        """Mock SerpAPI for Google Trends"""
        mock_search = Mock()
        mock_search.data = {
            'interest_over_time': {
                'timeline_data': [
                    {
                        'values': [
                            {'query': 'coffee brewing', 'extracted_value': 75},
                            {'query': 'coffee tutorial', 'extracted_value': 65}
                        ]
                    }
                ]
            },
            'related_queries': {
                'coffee': {
                    'top': [
                        {'query': 'best coffee', 'value': 80},
                        {'query': 'coffee maker', 'value': 70}
                    ],
                    'rising': [
                        {'query': 'coffee recipe', 'value': '+150%'}
                    ]
                }
            }
        }
        return mock_search


class E2ETestRunner:
    """Run end-to-end tests on existing code"""
    
    def __init__(self):
        self.db = TestDatabase()
        self.mocks = MockExternalAPIs()
        self.results = []
    
    def setup_mocks(self):
        """Setup all necessary mocks"""
        # Mock database connection
        self.db_patch = patch('utils.db.get_connection', return_value=self.db.get_connection())
        self.db_patch.start()
        
        # Mock Anthropic client
        self.anthropic_patch = patch('services.llm_optimization.client', self.mocks.mock_anthropic_client())
        self.anthropic_patch.start()
        
        # Mock YouTube clients
        self.youtube_patch = patch('services.youtube.build_youtube_client', return_value=self.mocks.mock_youtube_client())
        self.youtube_patch.start()
        
        self.analytics_patch = patch('services.youtube.build_youtube_analytics_client', return_value=self.mocks.mock_youtube_analytics_client())
        self.analytics_patch.start()
        
        # Mock SerpAPI
        self.serpapi_patch = patch('serpapi.search', return_value=self.mocks.mock_serpapi())
        self.serpapi_patch.start()
    
    def cleanup_mocks(self):
        """Stop all mocks"""
        self.db_patch.stop()
        self.anthropic_patch.stop()
        self.youtube_patch.stop()
        self.analytics_patch.stop()
        self.serpapi_patch.stop()
    
    def test_video_optimization(self):
        """Test video optimization flow"""
        print("\n" + "="*80)
        print("TEST 1: Video Optimization Flow")
        print("="*80)
        
        try:
            from services.video import get_video_data, create_optimization, generate_video_optimization
            
            # Step 1: Get video data
            print("\n1. Fetching video data...")
            video_data = get_video_data('test_video_1')
            assert video_data is not None, "Failed to get video data"
            print(f"   ✓ Retrieved video: {video_data['title']}")
            
            # Step 2: Create optimization record
            print("\n2. Creating optimization record...")
            optimization_id = create_optimization(video_data['id'])
            assert optimization_id > 0, "Failed to create optimization"
            print(f"   ✓ Created optimization ID: {optimization_id}")
            
            # Step 3: Generate optimization (mocked LLM call)
            print("\n3. Generating optimization...")
            result = generate_video_optimization(
                video=video_data,
                user_id=1,
                optimization_id=optimization_id,
                apply_optimization=False
            )
            
            assert 'error' not in result, f"Optimization failed: {result.get('error')}"
            assert result['optimized_title'], "No optimized title generated"
            print(f"   ✓ Original title: {result['original_title']}")
            print(f"   ✓ Optimized title: {result['optimized_title']}")
            print(f"   ✓ Original tags: {result['original_tags']}")
            print(f"   ✓ Optimized tags: {result['optimized_tags']}")
            
            self.results.append({
                'test': 'video_optimization',
                'status': 'PASSED',
                'input': {'video_id': 'test_video_1'},
                'output': result
            })
            print("\n✅ Video Optimization Test PASSED")
            
        except Exception as e:
            print(f"\n❌ Video Optimization Test FAILED: {e}")
            self.results.append({
                'test': 'video_optimization',
                'status': 'FAILED',
                'error': str(e)
            })
            raise
    
    def test_channel_optimization(self):
        """Test channel optimization flow"""
        print("\n" + "="*80)
        print("TEST 2: Channel Optimization Flow")
        print("="*80)
        
        try:
            from services.channel import get_channel_data, create_optimization, generate_channel_optimization
            
            # Step 1: Get channel data
            print("\n1. Fetching channel data...")
            channel_data = get_channel_data(1)
            assert channel_data is not None, "Failed to get channel data"
            print(f"   ✓ Retrieved channel: {channel_data['title']}")
            
            # Step 2: Create optimization record
            print("\n2. Creating optimization record...")
            optimization_id = create_optimization(1)
            assert optimization_id > 0, "Failed to create optimization"
            print(f"   ✓ Created optimization ID: {optimization_id}")
            
            # Step 3: Generate optimization
            print("\n3. Generating channel optimization...")
            result = generate_channel_optimization(channel_data, optimization_id)
            
            assert 'error' not in result, f"Optimization failed: {result.get('error')}"
            assert result.get('optimized_description'), "No optimized description"
            print(f"   ✓ Original description: {result['original_description'][:100]}...")
            print(f"   ✓ Optimized description: {result['optimized_description'][:100]}...")
            
            self.results.append({
                'test': 'channel_optimization',
                'status': 'PASSED',
                'input': {'channel_id': 1},
                'output': result
            })
            print("\n✅ Channel Optimization Test PASSED")
            
        except Exception as e:
            print(f"\n❌ Channel Optimization Test FAILED: {e}")
            self.results.append({
                'test': 'channel_optimization',
                'status': 'FAILED',
                'error': str(e)
            })
            raise
    
    def test_analytics_fetch(self):
        """Test analytics data fetching"""
        print("\n" + "="*80)
        print("TEST 3: Analytics Fetching Flow")
        print("="*80)
        
        try:
            from services.youtube import fetch_video_timeseries_data
            
            # Fetch timeseries data
            print("\n1. Fetching video analytics...")
            result = fetch_video_timeseries_data('test_video_1', interval='day')
            
            assert 'error' not in result, f"Analytics fetch failed: {result.get('error')}"
            assert 'timeseries_data' in result, "No timeseries data"
            print(f"   ✓ Retrieved {len(result['timeseries_data'])} data points")
            print(f"   ✓ Total views: {result['summary']['total_views']}")
            
            self.results.append({
                'test': 'analytics_fetch',
                'status': 'PASSED',
                'input': {'video_id': 'test_video_1'},
                'output': result['summary']
            })
            print("\n✅ Analytics Fetching Test PASSED")
            
        except Exception as e:
            print(f"\n❌ Analytics Fetching Test FAILED: {e}")
            self.results.append({
                'test': 'analytics_fetch',
                'status': 'FAILED',
                'error': str(e)
            })
            raise
    
    def run_all_tests(self):
        """Run all E2E tests"""
        print("\n" + "="*80)
        print("STARTING E2E BASELINE TESTS")
        print("="*80)
        
        self.setup_mocks()
        
        try:
            self.test_video_optimization()
            self.test_channel_optimization()
            self.test_analytics_fetch()
            
            print("\n" + "="*80)
            print("TEST SUMMARY")
            print("="*80)
            
            passed = sum(1 for r in self.results if r['status'] == 'PASSED')
            failed = sum(1 for r in self.results if r['status'] == 'FAILED')
            
            print(f"\nTotal Tests: {len(self.results)}")
            print(f"✅ Passed: {passed}")
            print(f"❌ Failed: {failed}")
            
            # Save results to file
            with open('baseline_test_results.json', 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            print("\n📄 Results saved to: baseline_test_results.json")
            
            return failed == 0
            
        finally:
            self.cleanup_mocks()


if __name__ == '__main__':
    runner = E2ETestRunner()
    success = runner.run_all_tests()
    sys.exit(0 if success else 1)
