#!/usr/bin/env python3
"""
Test Improved ML Accuracy System
Tests the enhanced ML wave accuracy with auto-training
"""

import sys
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

def test_ml_accuracy_import():
    """Test ML accuracy module import and initialization"""
    print("🧪 Testing ML Accuracy Import...")
    
    try:
        from src.analysis.ml_wave_accuracy import MLWaveAccuracy
        print("✅ MLWaveAccuracy import successful")
        
        # Initialize ML accuracy (this should trigger auto-training)
        print("🤖 Initializing ML Accuracy system...")
        ml_accuracy = MLWaveAccuracy()
        
        print(f"✅ ML system initialized")
        print(f"📊 Training status: {'Trained' if ml_accuracy.is_trained else 'Not Trained'}")
        print(f"📈 Pattern database size: {len(ml_accuracy.pattern_database)}")
        
        return ml_accuracy
        
    except Exception as e:
        print(f"❌ ML Accuracy import failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_ml_prediction():
    """Test ML prediction functionality"""
    print("\n🔮 Testing ML Prediction...")
    
    try:
        ml_accuracy = test_ml_accuracy_import()
        if not ml_accuracy:
            return False
        
        # Create sample market data
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(42)  # For reproducible results
        
        # Generate realistic price data
        base_price = 100.0
        prices = [base_price]
        for i in range(99):
            change = np.random.normal(0.001, 0.02)  # 0.1% daily return, 2% volatility
            prices.append(prices[-1] * (1 + change))
        
        market_data = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'close': prices,
            'volume': [np.random.randint(1000, 10000) for _ in prices]
        }, index=dates)
        
        print(f"📊 Created market data: {len(market_data)} days")
        print(f"💰 Price range: ${market_data['close'].min():.2f} - ${market_data['close'].max():.2f}")
        
        # Test prediction
        print("🎯 Running ML prediction...")
        result = ml_accuracy.predict_wave_accuracy(market_data, "TEST-SYMBOL")
        
        print("✅ ML Prediction Results:")
        print(f"   📈 Accuracy Score: {result['accuracy_score']:.1%}")
        print(f"   🎯 Confidence Level: {result['confidence_level']}")
        print(f"   🌊 Pattern Match Score: {result['pattern_match_score']:.1%}")
        print(f"   📊 Features: {result['features']}")
        print(f"   🔍 Similar Patterns: {len(result['similar_patterns'])}")
        
        for pattern in result['similar_patterns'][:3]:
            print(f"      - {pattern}")
        
        return True
        
    except Exception as e:
        print(f"❌ ML Prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_web_ml_endpoint():
    """Test the web ML endpoint"""
    print("\n🌐 Testing Web ML Endpoint...")
    
    try:
        # Import web components
        from web.app import app
        
        print("✅ Web app import successful")
        
        # Test client
        with app.test_client() as client:
            # Test ML accuracy endpoint
            test_data = {
                'symbol': 'AAPL',
                'timeframe': '1d',
                'wave_data': [
                    {'type': '1', 'confidence': 0.8},
                    {'type': '2', 'confidence': 0.7},
                    {'type': '3', 'confidence': 0.9}
                ]
            }
            
            print("📡 Sending test request to /api/ml/accuracy...")
            response = client.post('/api/ml/accuracy', 
                                 json=test_data,
                                 content_type='application/json')
            
            print(f"📊 Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.get_json()
                print("✅ ML API Response:")
                print(f"   🎯 Success: {result.get('success', False)}")
                print(f"   📈 Accuracy: {result.get('accuracy_score', 0):.1%}")
                print(f"   🎯 Confidence: {result.get('confidence_level', 'Unknown')}")
                return True
            else:
                print(f"❌ ML API request failed: {response.status_code}")
                print(f"   Response: {response.get_data(as_text=True)}")
                return False
        
    except Exception as e:
        print(f"❌ Web ML endpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Elliott Bot - Improved ML Accuracy System Test")
    print("=" * 60)
    
    tests = [
        ("ML Import & Auto-Training", test_ml_accuracy_import),
        ("ML Prediction", test_ml_prediction),
        ("Web ML Endpoint", test_web_ml_endpoint)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📝 Running: {test_name}")
        try:
            result = test_func()
            if result:
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Improved ML system is working correctly.")
    else:
        print(f"⚠️ {total - passed} test(s) failed. Check the issues above.")
    
    print("✨ Test completed!")

if __name__ == "__main__":
    main()
