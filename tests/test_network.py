#!/usr/bin/env python3
"""
Quick network and server test for Elliott Bot
"""

import requests
import json
import sys
import time

def test_server_connectivity():
    """Test if the Elliott Bot server is responding correctly."""
    
    print("🔧 Elliott Bot Network Test")
    print("=" * 50)
    
    # Test 1: Basic connectivity
    try:
        print("1. Testing basic connectivity...")
        response = requests.get("http://localhost:5000", timeout=10)
        print(f"   ✅ Server responding: HTTP {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("   ❌ Connection failed: Server not running")
        return False
    except requests.exceptions.Timeout:
        print("   ❌ Connection timeout: Server too slow")
        return False
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
        return False
    
    # Test 2: Health check
    try:
        print("2. Testing health check endpoint...")
        response = requests.get("http://localhost:5000/api/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Health check OK: {health_data.get('status', 'unknown')}")
        else:
            print(f"   ⚠️ Health check returned: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
    
    # Test 3: API endpoint test
    try:
        print("3. Testing analysis API...")
        test_payload = {
            "symbol": "AAPL",
            "timeframe": "1d"
        }
        
        response = requests.post(
            "http://localhost:5000/api/analyze",
            json=test_payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                wave_count = len(data.get('waves', []))
                print(f"   ✅ API test successful: {wave_count} waves detected")
            else:
                print(f"   ⚠️ API responded but analysis failed: {data.get('error', 'Unknown error')}")
        else:
            print(f"   ❌ API test failed: HTTP {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            
    except requests.exceptions.Timeout:
        print("   ❌ API test timeout: Analysis taking too long")
    except Exception as e:
        print(f"   ❌ API test error: {e}")
    
    # Test 4: Network speed test
    try:
        print("4. Testing network speed...")
        start_time = time.time()
        response = requests.get("http://localhost:5000", timeout=5)
        end_time = time.time()
        response_time = (end_time - start_time) * 1000
        
        if response_time < 1000:
            print(f"   ✅ Fast response: {response_time:.0f}ms")
        elif response_time < 3000:
            print(f"   ⚠️ Slow response: {response_time:.0f}ms")
        else:
            print(f"   ❌ Very slow response: {response_time:.0f}ms")
            
    except Exception as e:
        print(f"   ❌ Speed test failed: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Test completed. If you see errors above, these might be causing the 'Network error' message.")
    print("\n💡 Recommendations:")
    print("   - Make sure the server is running: python launch_web.py")
    print("   - Check Windows Firewall settings")
    print("   - Try accessing http://localhost:5000 in your browser")
    print("   - Clear browser cache and try again")
    
    return True

if __name__ == "__main__":
    test_server_connectivity()
