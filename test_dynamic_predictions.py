"""
Dynamic Prediction Testing Script
Test the enhanced Elliott Wave predictions across different timeframes
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from web.app import analyze_wave_structure, calculate_dynamic_target, get_timeframe_multiplier
from src.data.data_loader import DataLoader
from src.analysis.wave_detector import WaveDetector

def test_dynamic_predictions():
    """Test dynamic predictions with different symbols and timeframes."""
    
    print("🧪 Testing Dynamic Elliott Wave Predictions")
    print("=" * 50)
    
    # Initialize components
    data_loader = DataLoader()
    wave_detector = WaveDetector()
    
    # Test symbols and timeframes
    test_cases = [
        ("AAPL", "1d"),
        ("AAPL", "1h"), 
        ("AAPL", "4h"),
        ("BTC-USD", "1d"),
        ("BTC-USD", "4h"),
        ("EURUSD=X", "1d")
    ]
    
    for symbol, timeframe in test_cases:
        print(f"\n📊 Testing {symbol} on {timeframe} timeframe")
        print("-" * 40)
        
        try:
            # Get timeframe configuration
            timeframes = {
                '1h': {'period': '5d', 'interval': '1h', 'label': '1 Hour'},
                '4h': {'period': '30d', 'interval': '4h', 'label': '4 Hours'},
                '1d': {'period': '1y', 'interval': '1d', 'label': '1 Day'}
            }
            
            tf_config = timeframes.get(timeframe, timeframes['1d'])
            
            # Load data
            market_data = data_loader.get_yahoo_data(
                symbol, 
                period=tf_config['period'],
                interval=tf_config['interval']
            )
            
            if market_data.empty:
                print(f"❌ No data available for {symbol}")
                continue
                
            print(f"✅ Loaded {len(market_data)} data points")
            
            # Detect waves
            waves = wave_detector.detect_waves(market_data)
            
            if not waves:
                print("❌ No waves detected")
                continue
                
            print(f"🌊 Detected {len(waves)} waves:")
            for wave in waves:
                print(f"   - {wave.wave_type.value}: {wave.start_point.price:.4f} → {wave.end_point.price:.4f} (conf: {wave.confidence:.2f})")
            
            # Test dynamic prediction
            current_price = market_data['close'].iloc[-1]
            price_changes = market_data['close'].pct_change().dropna()
            volatility = price_changes.std()
            recent_trend = (market_data['close'].iloc[-1] - market_data['close'].iloc[-20]) / market_data['close'].iloc[-20] if len(market_data) >= 20 else 0
            
            # Analyze wave structure
            wave_analysis = analyze_wave_structure(waves, market_data)
            print(f"\n🔍 Wave Analysis:")
            print(f"   Pattern: {wave_analysis['pattern']}")
            print(f"   Confidence: {wave_analysis['confidence']:.1%}")
            print(f"   Direction: {wave_analysis['direction']}")
            
            # Calculate dynamic target
            predicted_price = calculate_dynamic_target(
                current_price, waves, market_data, volatility, recent_trend, timeframe
            )
            
            if predicted_price:
                price_change = ((predicted_price - current_price) / current_price) * 100
                print(f"\n🎯 Prediction:")
                print(f"   Current Price: ${current_price:.4f}")
                print(f"   Target Price: ${predicted_price:.4f}")
                print(f"   Expected Move: {price_change:+.2f}%")
                print(f"   Timeframe Multiplier: {get_timeframe_multiplier(timeframe):.2f}")
                print(f"   Volatility: {volatility:.4f}")
                print(f"   Recent Trend: {recent_trend:+.2%}")
            else:
                print("❌ Could not generate prediction")
                
        except Exception as e:
            print(f"❌ Error testing {symbol} {timeframe}: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Dynamic Prediction Testing Complete")

if __name__ == "__main__":
    test_dynamic_predictions()
