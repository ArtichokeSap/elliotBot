#!/usr/bin/env python3
"""
Test the integrated web app with technical confluence analysis
"""

import requests
import json
import time

def test_web_app_integration():
    """Test the web app with the new technical analysis integration."""
    
    print("🚀 Testing Elliott Wave Web App with Technical Confluence Integration")
    print("=" * 70)
    
    # Test data
    test_symbol = "AAPL"
    test_timeframe = "1d"
    
    # Test payload
    payload = {
        "symbol": test_symbol,
        "timeframe": test_timeframe
    }
    
    print(f"📊 Testing symbol: {test_symbol}")
    print(f"⏰ Timeframe: {test_timeframe}")
    print(f"📡 Payload: {json.dumps(payload, indent=2)}")
    print()
    
    try:
        # Start the web app in background (if not already running)
        print("🌐 Testing connection to web app...")
        
        # Test main analysis endpoint
        response = requests.post(
            "http://localhost:5000/api/analyze",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print("✅ Web App Analysis Successful!")
            print(f"📈 Success: {data.get('success', False)}")
            print(f"🎯 Validation Score: {data.get('validation_score', 0):.1%}")
            print(f"🌊 Wave Structure: {data.get('wave_structure', 'unknown').upper()}")
            print(f"📊 Direction: {data.get('direction', 'neutral').upper()}")
            print(f"🔢 Waves Detected: {data.get('wave_count', 0)}")
            
            # Check for new technical confluence features
            if 'target_zones' in data:
                target_zones = data['target_zones']
                print(f"🎯 Target Zones: {len(target_zones)}")
                
                if target_zones:
                    best_target = target_zones[0]
                    print(f"🏆 Best Target: ${best_target['price_level']:.4f}")
                    print(f"📈 Expected Move: {best_target['price_change_pct']:+.2f}%")
                    print(f"🔥 Confidence: {best_target['confidence_level']}")
                    print(f"🧩 Confluences: {len(best_target['confluences'])}")
            
            if 'confluence_summary' in data:
                summary = data['confluence_summary']
                print(f"📊 High Confidence Targets: {summary['high_confidence']}")
                print(f"⚠️ Medium Confidence Targets: {summary['medium_confidence']}")
                print(f"🔽 Low Confidence Targets: {summary['low_confidence']}")
            
            print(f"🕐 Analysis Mode: {data.get('analysis_mode', 'standard')}")
            print(f"📅 Timestamp: {data.get('analysis_timestamp', 'unknown')}")
            
            print("\n✅ Web App Integration Test PASSED!")
            
        else:
            print(f"❌ Web App Test Failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Web app not running on localhost:5000")
        print("💡 Start the web app with: python web/app.py")
        
    except Exception as e:
        print(f"❌ Test Error: {e}")

def test_new_api_endpoints():
    """Test the new technical analysis API endpoints."""
    
    print("\n🔬 Testing New Technical Analysis API Endpoints")
    print("=" * 50)
    
    # Test data
    test_symbol = "BTC/USDT"
    
    endpoints = [
        {
            'name': 'Technical Confluence Analysis',
            'url': 'http://localhost:5000/api/technical/analyze',
            'method': 'POST',
            'data': {
                'symbol': test_symbol,
                'timeframe': '1h',
                'exchange': 'binance',
                'limit': 200
            }
        },
        {
            'name': 'Multi-Timeframe Analysis', 
            'url': 'http://localhost:5000/api/technical/multi-timeframe',
            'method': 'POST',
            'data': {
                'symbol': test_symbol,
                'exchange': 'binance',
                'timeframes': ['1h', '4h', '1d']
            }
        },
        {
            'name': 'Confluence Details',
            'url': 'http://localhost:5000/api/technical/confluence-details',
            'method': 'GET',
            'params': {
                'symbol': test_symbol,
                'price': '50000',
                'exchange': 'binance',
                'timeframe': '1h'
            }
        }
    ]
    
    for endpoint in endpoints:
        try:
            print(f"\n🧪 Testing: {endpoint['name']}")
            print(f"📡 URL: {endpoint['url']}")
            
            if endpoint['method'] == 'POST':
                response = requests.post(
                    endpoint['url'],
                    json=endpoint['data'],
                    timeout=20
                )
            else:
                response = requests.get(
                    endpoint['url'],
                    params=endpoint.get('params', {}),
                    timeout=20
                )
            
            if response.status_code == 200:
                data = response.json()
                success = data.get('success', False)
                
                if success:
                    print(f"✅ {endpoint['name']}: SUCCESS")
                    
                    # Show key metrics
                    if 'target_zones' in data:
                        print(f"🎯 Target Zones: {len(data['target_zones'])}")
                    if 'timeframes' in data:
                        print(f"⏰ Timeframes Analyzed: {len(data['timeframes'])}")
                    if 'confluence_analysis' in data:
                        print(f"🧩 Confluence Analysis: Available")
                        
                else:
                    print(f"⚠️ {endpoint['name']}: API returned success=False")
                    if 'error' in data:
                        print(f"   Error: {data['error']}")
            else:
                print(f"❌ {endpoint['name']}: HTTP {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"❌ {endpoint['name']}: Connection failed")
            
        except Exception as e:
            print(f"❌ {endpoint['name']}: {e}")

if __name__ == "__main__":
    test_web_app_integration()
    test_new_api_endpoints()
    
    print("\n🎉 Integration Testing Complete!")
    print("💡 To start the web app: python web/app.py")
    print("🌐 Access at: http://localhost:5000")
