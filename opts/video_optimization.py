"""
Video Optimization Testing Module

This module provides testing utilities and performance validation for the
YouTube Optimizer video optimization endpoints. It includes local testing
functions for API endpoints and performance monitoring.

Key functionalities:
- Local API endpoint testing using FastAPI TestClient
- Performance validation for video optimization endpoints
- Endpoint response validation and error handling
- Testing utilities for development and debugging

This module is designed for development and testing purposes, allowing
developers to validate API functionality without external dependencies.

Author: YouTube Optimizer Team
Version: 1.0.0
"""

import sys
import os
# =============================================================================
# PATH MANIPULATION FOR DIRECT SCRIPT EXECUTION
# =============================================================================

# Ensures 'backend' can be imported when running the script directly
# This allows the module to be executed as a standalone script for testing
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
backend_dir = os.path.dirname(current_dir)

# Add necessary paths to sys.path for imports
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# =============================================================================
# FASTAPI TEST CLIENT SETUP
# =============================================================================

from fastapi.testclient import TestClient

# Import FastAPI app and target functions for testing
# This allows testing of API endpoints without running the full server
try:
    from main import app
    # Import the target functions to be tested
    from routes.video_routes import get_channel_videos_performance, get_all_videos_performance
except ImportError as e:
    print(f"Error importing FastAPI app or function: {e}")
    print("Make sure your FastAPI app instance and function are correctly defined.")
    sys.exit(1)

# Create a TestClient instance pointing to the FastAPI app
# This provides a test interface for making HTTP requests to the API
client = TestClient(app)

# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def test_performance_all_endpoint_locally():
    """
    Test the /video/performance/all endpoint using FastAPI TestClient.
    
    This function tests the video performance endpoint locally without requiring
    a running server. It validates the endpoint's response format, status codes,
    and basic functionality.
    
    Test Parameters:
    - interval: "30m" - 30-minute data interval
    - limit: 1 - Limit results for testing (keeps response small)
    - offset: 0 - Start from beginning of results
    - refresh: True - Force data refresh (slower but tests full functionality)
    """
    print("--- Testing /video/performance/all Endpoint ---")
    endpoint_url = "/video/performance/all"
    
    # Test parameters - configured for testing with minimal data load
    params = {
        "interval": "30m",    # 30-minute data interval
        "limit": 1,           # Keep limit low for testing
        "offset": 0,          # Start from beginning
        "refresh": True       # Set to True to test refresh logic (will be slower)
    }

    try:
        # Make a GET request to the endpoint using TestClient
        response = client.get(endpoint_url, params=params)

        # Print status code and response data for validation
        print(f"Status Code: {response.status_code}")

        # Try to print JSON response, handle potential errors
        try:
            response_json = response.json()
            print("Response JSON:")
            # Pretty print if possible (optional)
            import json
            print(json.dumps(response_json, indent=2))
        except json.JSONDecodeError:
            print("Response Content (not valid JSON):")
            print(response.text)

        # Basic assertion (optional, but good practice)
        if response.status_code == 200:
            print("\nTest basic check: Status code 200 OK")
            # assert "videos" in response_json # Example assertion
        else:
            print(f"\nTest basic check: Failed with status code {response.status_code}")

    except Exception as e:
        print(f"\nAn error occurred during the test client request: {e}")

    print("--- Test Finished ---")

# --- New Async Test Function ---
async def test_get_channel_videos_performance_directly():
    """
    Calls the get_channel_videos_performance function directly.
    """
    print("\n--- Testing get_channel_videos_performance Directly ---")
    # !!! IMPORTANT: Replace 1 with a valid channel_id from your local database !!!
    test_channel_id = 1

    print(f"Calling function for channel_id: {test_channel_id}")
    try:
        # Call the async function directly
        # Using refresh=False to avoid needing YouTube credentials

        optimization_data = await get_all_videos_performance()

        performance_data = await get_channel_videos_performance(
            channel_id=test_channel_id,
            refresh=True,
            include_optimizations=True # Or False, depending on what you want to test
        )

        # Print the results
        print("Function Result:")
        import json
        print(json.dumps(performance_data, indent=2, default=str)) # Use default=str for datetime objects

        # Add assertions based on expected results
        assert isinstance(performance_data, dict)
        assert "channel_id" in performance_data
        assert performance_data["channel_id"] == test_channel_id
        assert "videos" in performance_data
        assert isinstance(performance_data["videos"], list)
        print(f"\nDirect function call test: Basic checks passed for channel {test_channel_id}")

    except Exception as e:
        print(f"\nAn error occurred during the direct function call: {e}")
        # Optionally re-raise or handle the error
        # raise

    print("--- Direct Function Test Finished ---")
# --- ---

# Run the test functions directly
if __name__ == "__main__":
    # You can run one or both tests
    # test_performance_all_endpoint_locally()
    import asyncio
    # Run the new async test using asyncio.run()
    asyncio.run(test_get_channel_videos_performance_directly())