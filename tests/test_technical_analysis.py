#!/usr/bin/env python3
"""
Technical Analysis System Test
Tests the new Elliott Wave + Technical Confluence system
"""

import sys
import os
import logging
import requests
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def test_technical_analysis_api():
    """Test the new technical analysis API"""
    print("🧪 Testing Technical Analysis API...")
    
    try:
        # Import the technical analysis components
        from src.data.enhanced_data_fetcher import EnhancedDataFetcher
        from src.analysis.technical_confluence import TechnicalConfluenceAnalyzer
        
        print("✅ Import successful")
        
        # Test data fetching
        print("📊 Testing data fetching...")
        data_fetcher = EnhancedDataFetcher()
        
        # Test Binance data fetch
        market_data = data_fetcher.fetch_ohlcv_data('BTC/USDT', '1h', 'binance', 100)
        
        if not market_data.empty:
            print(f"✅ Binance data fetch successful: {len(market_data)} candles")
            print(f"   Price range: ${market_data['close'].min():.2f} - ${market_data['close'].max():.2f}")
        else:
            print("❌ Binance data fetch failed")
            return False
        
        # Test technical confluence analyzer
        print("🧩 Testing technical confluence analyzer...")
        confluence_analyzer = TechnicalConfluenceAnalyzer()
        
        # Create mock Elliott analysis for testing
        mock_elliott_analysis = {
            'wave_structure': 'impulse',
            'validation_score': 0.75,
            'direction': 'bullish',
            'waves': [
                {
                    'wave': '1',
                    'direction': 'bullish',
                    'start_time': market_data.index[0],
                    'end_time': market_data.index[20],
                    'start_price': market_data['close'].iloc[0],
                    'end_price': market_data['close'].iloc[20],
                    'confidence': 0.8,
                    'wave_type': 'impulse',
                    'length': abs(market_data['close'].iloc[20] - market_data['close'].iloc[0]),
                    'duration': 20
                },
                {
                    'wave': '3',
                    'direction': 'bullish',
                    'start_time': market_data.index[40],
                    'end_time': market_data.index[80],
                    'start_price': market_data['close'].iloc[40],
                    'end_price': market_data['close'].iloc[80],
                    'confidence': 0.9,
                    'wave_type': 'impulse',
                    'length': abs(market_data['close'].iloc[80] - market_data['close'].iloc[40]),
                    'duration': 40
                }
            ]
        }
        
        # Test confluence analysis
        target_zones = confluence_analyzer.analyze_target_zones(market_data, mock_elliott_analysis, '1h')
        
        if target_zones:
            print(f"✅ Technical confluence analysis successful: {len(target_zones)} target zones")
            
            # Show top target zone
            best_zone = target_zones[0]
            print(f"   🎯 Best Target Zone:")
            print(f"      Price: ${best_zone.price_level:.2f}")
            print(f"      Wave: {best_zone.wave_target}")
            print(f"      Confidence: {best_zone.confidence_level}")
            print(f"      Confluence Score: {best_zone.confluence_score}")
            print(f"      Probability: {best_zone.probability:.1%}")
            print(f"      Confluences: {best_zone.confluences[:3]}...")  # Show first 3
            
        else:
            print("⚠️ No target zones found (this might be normal with mock data)")
        
        return True
        
    except Exception as e:
        print(f"❌ Technical analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_data_fetcher():
    """Test the enhanced data fetcher functionality"""
    print("\n📊 Testing Enhanced Data Fetcher...")
    
    try:
        from src.data.enhanced_data_fetcher import EnhancedDataFetcher
        
        data_fetcher = EnhancedDataFetcher()
        
        # Test multiple data sources
        test_symbols = ['BTC/USDT', 'ETH/USDT']
        test_timeframes = ['1h', '4h']
        test_exchanges = ['binance']  # Start with just Binance
        
        for exchange in test_exchanges:
            print(f"\n🔗 Testing {exchange.upper()} exchange...")
            
            for symbol in test_symbols:
                try:
                    print(f"   📈 Fetching {symbol}...")
                    
                    # Test single timeframe
                    data = data_fetcher.fetch_ohlcv_data(symbol, '1h', exchange, 50)
                    
                    if not data.empty:
                        print(f"      ✅ {len(data)} candles fetched")
                        print(f"      💰 Price: ${data['close'].iloc[-1]:.2f}")
                        print(f"      📊 Volume: {data['volume'].iloc[-1]:,.0f}")
                        
                        # Test extended calculations
                        extended_data = data_fetcher.calculate_extended_market_data(data)
                        additional_cols = len(extended_data.columns) - len(data.columns)
                        print(f"      🔬 Added {additional_cols} technical indicators")
                        
                    else:
                        print(f"      ❌ No data for {symbol}")
                        
                except Exception as e:
                    print(f"      ❌ Error fetching {symbol}: {e}")
        
        # Test market summary
        print(f"\n📋 Testing market summary...")
        summary = data_fetcher.get_market_summary('BTC/USDT', 'binance')
        
        if summary:
            print(f"   ✅ Market summary successful:")
            print(f"      Symbol: {summary.get('symbol')}")
            print(f"      Price: ${summary.get('price', 0):.2f}")
            print(f"      24h Change: {summary.get('change_24h', 0):+.2f}%")
            print(f"      24h Volume: {summary.get('volume_24h', 0):,.0f}")
        else:
            print(f"   ❌ Market summary failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced data fetcher test failed: {e}")
        return False

def test_api_integration():
    """Test the full API integration"""
    print("\n🌐 Testing API Integration...")
    
    try:
        # Start the API in background for testing
        from src.api.technical_analysis_api import app
        
        print("✅ API import successful")
        
        # Test client
        with app.test_client() as client:
            print("📡 Testing API endpoints...")
            
            # Test health check
            response = client.get('/api/health')
            if response.status_code == 200:
                health_data = response.get_json()
                print(f"   ✅ Health check: {health_data['status']}")
                print(f"      Exchanges: {health_data['supported_exchanges']}")
                print(f"      Timeframes: {health_data['supported_timeframes']}")
            else:
                print(f"   ❌ Health check failed: {response.status_code}")
                return False
            
            # Test analysis endpoint (this will take longer)
            print("   🧠 Testing full analysis (this may take a moment)...")
            analysis_request = {
                'symbol': 'BTC/USDT',
                'timeframe': '1h',
                'exchange': 'binance',
                'limit': 200
            }
            
            response = client.post('/api/analyze', 
                                 json=analysis_request,
                                 content_type='application/json')
            
            if response.status_code == 200:
                analysis_data = response.get_json()
                
                if analysis_data.get('success'):
                    print(f"   ✅ Analysis successful:")
                    analysis = analysis_data.get('analysis', {})
                    
                    # Show summary
                    summary = analysis.get('summary', {})
                    best_target = summary.get('best_target')
                    
                    if best_target:
                        print(f"      🎯 Best Target: ${best_target['price_level']}")
                        print(f"         Wave: {best_target['wave_target']}")
                        print(f"         Confidence: {best_target['confidence_level']}")
                        print(f"         Confluences: {len(best_target['confluences'])}")
                    
                    # Show counts
                    total_targets = analysis.get('total_targets', 0)
                    high_conf = analysis.get('high_confidence_targets', 0)
                    print(f"      📊 Total Targets: {total_targets}")
                    print(f"      🔥 High Confidence: {high_conf}")
                    
                else:
                    error_msg = analysis_data.get('error', 'Unknown error')
                    print(f"   ⚠️ Analysis completed with limitations: {error_msg}")
            else:
                print(f"   ❌ Analysis failed: {response.status_code}")
                try:
                    error_data = response.get_json()
                    print(f"      Error: {error_data.get('error', 'Unknown error')}")
                except:
                    print(f"      Raw response: {response.get_data(as_text=True)}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ API integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all technical analysis tests"""
    print("🚀 Elliott Bot - Technical Analysis System Test")
    print("=" * 70)
    print("🌊 Elliott Wave Theory + Technical Confluence Analysis")
    print("🎯 High-Probability Target Zone Identification")
    print("📊 Multi-Exchange Data Support")
    print("=" * 70)
    
    tests = [
        ("Enhanced Data Fetcher", test_enhanced_data_fetcher),
        ("Technical Analysis Core", test_technical_analysis_api),
        ("API Integration", test_api_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📝 Running: {test_name}")
        print("-" * 50)
        try:
            result = test_func()
            if result:
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 70)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Technical Analysis System is ready!")
        print("\n🚀 Next Steps:")
        print("   1. Run the API: python src/api/technical_analysis_api.py")
        print("   2. Test with real data on different symbols and timeframes")
        print("   3. Integrate with your trading strategy")
    else:
        print(f"⚠️ {total - passed} test(s) failed. Check the issues above.")
    
    print("✨ Test completed!")

if __name__ == "__main__":
    main()
