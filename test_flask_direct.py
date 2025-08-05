#!/usr/bin/env python3
"""
Direct Flask Test - Run the app and test the API endpoint
"""

import requests
import time
import threading
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def start_flask_server():
    """Start the Flask server in a separate thread"""
    try:
        from web.app import app
        print("Starting Flask server...")
        app.run(debug=False, host='127.0.0.1', port=5000, use_reloader=False)
    except Exception as e:
        print(f"Error starting Flask server: {e}")

def test_api():
    """Test the API endpoint"""
    print("Waiting for server to start...")
    time.sleep(3)
    
    try:
        # Test health endpoint first
        print("Testing health endpoint...")
        response = requests.get('http://127.0.0.1:5000/api/health', timeout=10)
        print(f"Health endpoint response: {response.status_code}")
        if response.status_code == 200:
            print(f"Health response: {response.json()}")
        
        # Test main endpoint
        print("\nTesting analyze endpoint...")
        test_data = {
            'symbol': 'AAPL',
            'timeframe': '1d'
        }
        
        response = requests.post(
            'http://127.0.0.1:5000/api/analyze',
            json=test_data,
            timeout=30
        )
        
        print(f"API Response Status: {response.status_code}")
        print(f"API Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success! Validation Score: {result.get('validation_score', 'N/A')}")
            print(f"Analysis Status: {result.get('status', 'N/A')}")
            print(f"Wave Structure: {result.get('wave_structure', 'N/A')}")
            print(f"Number of waves detected: {len(result.get('waves', []))}")
            print(f"Number of validation results: {len(result.get('validation_results', []))}")
            print(f"Number of fibonacci levels: {len(result.get('fibonacci_levels', []))}")
            print(f"Chart data present: {'Yes' if result.get('chart') else 'No'}")
            
            # Show first few waves
            if result.get('waves'):
                print(f"\nFirst wave: {result['waves'][0]}")
        else:
            print(f"Error Response: {response.text}")
            
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error: {e}")
    except Exception as e:
        print(f"Test error: {e}")

if __name__ == '__main__':
    # Start Flask server in background thread
    flask_thread = threading.Thread(target=start_flask_server, daemon=True)
    flask_thread.start()
    
    # Test the API
    test_api()
    
    print("\nTest completed. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")
