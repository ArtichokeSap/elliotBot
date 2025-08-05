#!/usr/bin/env python3
"""
Test script for the web API with comprehensive Elliott Wave validation
"""

import requests
import json
import time

def test_elliott_wave_api():
    """Test the comprehensive Elliott Wave API endpoint"""
    
    url = "http://localhost:5000/api/analyze"
    
    # Test different symbols and timeframes
    test_cases = [
        {"symbol": "AAPL", "timeframe": "1d"},
        {"symbol": "BTC-USD", "timeframe": "1d"},
        {"symbol": "EURUSD", "timeframe": "1h"},
    ]
    
    print("🧪 Testing Comprehensive Elliott Wave API...")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📊 Test {i}: {test_case['symbol']} ({test_case['timeframe']})")
        print("-" * 40)
        
        try:
            # Make API request
            response = requests.post(url, json=test_case, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Display results
                print(f"✅ Status: {data.get('status', 'unknown')}")
                print(f"📈 Symbol: {test_case['symbol']}")
                print(f"⏰ Timeframe: {test_case['timeframe']}")
                print(f"🌊 Waves Detected: {len(data.get('waves', []))}")
                
                # Validation results
                validation_results = data.get('validation_results', [])
                if validation_results:
                    for result in validation_results:
                        print(f"🎯 Pattern: {result.get('type', 'Unknown')}")
                        print(f"📊 Validation Score: {result.get('score', 0)}%")
                        print(f"🔥 Status: {result.get('status', 'Unknown')}")
                        
                        # Recommendations
                        recommendations = result.get('recommendations', [])
                        if recommendations:
                            print(f"💡 Recommendations: {recommendations[:2]}")  # Show first 2
                else:
                    print("⚠️  No validation results available")
                
                # Future predictions
                predictions = data.get('future_predictions', [])
                if predictions:
                    pred = predictions[0]  # Show first prediction
                    print(f"🔮 Prediction: {pred.get('pattern', 'Unknown')}")
                    print(f"🎲 Probability: {pred.get('probability', 'Unknown')}")
                    print(f"🎯 Expected Move: {pred.get('expected_move', 'Unknown')}")
                
                # Chart info
                chart_url = data.get('chart_url')
                if chart_url:
                    print(f"📈 Chart: {chart_url}")
                
                print(f"✨ Analysis completed successfully!")
                
            else:
                print(f"❌ API Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection Error: {e}")
        except Exception as e:
            print(f"❌ Unexpected Error: {e}")
        
        # Wait between requests
        if i < len(test_cases):
            print("\n⏳ Waiting 2 seconds...")
            time.sleep(2)
    
    print("\n" + "=" * 60)
    print("🎉 API Testing Complete!")
    print("📝 Check http://localhost:5000 for the web interface")

if __name__ == "__main__":
    test_elliott_wave_api()
