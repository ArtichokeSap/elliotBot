#!/usr/bin/env python3
"""
Minimal test to verify Elliott Wave analysis works with synthetic data.
This bypasses the missing DataLoader and tests the core analysis components.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def create_sample_data():
    """Create synthetic OHLCV data for testing."""
    np.random.seed(42)  # For reproducible results

    # Create 200 days of data
    dates = pd.date_range('2023-01-01', periods=200, freq='D')

    # Generate realistic price data with trends and waves
    base_price = 100

    # Create a trending pattern with some waves
    trend = np.linspace(0, 30, 200)  # Upward trend
    waves = 5 * np.sin(np.linspace(0, 4*np.pi, 200))  # Wave pattern
    noise = np.random.randn(200) * 2  # Random noise

    close_prices = base_price + trend + waves + noise

    # Generate OHLC from close prices
    high_prices = close_prices + abs(np.random.randn(200) * 1.5)
    low_prices = close_prices - abs(np.random.randn(200) * 1.5)
    open_prices = close_prices + np.random.randn(200) * 0.5

    # Ensure OHLC relationships are correct
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))

    # Create DataFrame
    data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': np.random.randint(1000000, 5000000, 200)
    }, index=dates)

    return data

def test_wave_detection():
    """Test if wave detection works with synthetic data."""
    print("🧪 Testing Elliott Wave Detection with Synthetic Data")
    print("=" * 60)

    try:
        # Create sample data
        print("📊 Generating synthetic OHLCV data...")
        data = create_sample_data()
        print(f"✅ Created {len(data)} data points")
        print(f"   Price range: ${data['low'].min():.2f} - ${data['high'].max():.2f}")
        # Test imports
        print("\n🔍 Testing imports...")
        from src.analysis.wave_detector import WaveDetector
        from src.analysis.fibonacci import FibonacciAnalyzer
        print("✅ Core analysis modules imported successfully")

        # Initialize components
        print("\n🏗️  Initializing analysis components...")
        wave_detector = WaveDetector()
        fib_analyzer = FibonacciAnalyzer()
        print("✅ Components initialized")

        # Test wave detection
        print("\n🌊 Testing wave detection...")
        waves = wave_detector.detect_waves(data)
        print(f"✅ Detected {len(waves)} waves")

        if waves:
            print("\n📋 Wave Analysis Results:")
            for i, wave in enumerate(waves[:5]):  # Show first 5 waves
                print(f"  Wave {i+1}: {wave.wave_type.value} "
                      f"({wave.start_point.timestamp.strftime('%Y-%m-%d')} → "
                      f"{wave.end_point.timestamp.strftime('%Y-%m-%d')}) "
                      f"Change: {wave.price_change_pct:.1%} "
                      f"Confidence: {wave.confidence:.2f}")

        # Test Fibonacci analysis
        print("\n🔢 Testing Fibonacci analysis...")
        if len(data) > 50:
            high_price = data['high'].rolling(50).max().iloc[-1]
            low_price = data['low'].rolling(50).min().iloc[-1]
            current_price = data['close'].iloc[-1]

            fib_analysis = fib_analyzer.analyze_retracement(high_price, low_price, current_price, 'up')
            print(f"✅ Fibonacci analysis completed")
            print(f"   Retracement levels: {len(fib_analysis.retracements)}")
            print(f"   Extension levels: {len(fib_analysis.extensions)}")

            # Show key levels
            print("   Key Fibonacci levels:")
            for level in fib_analysis.key_levels[:3]:
                print(f"     {level.ratio:.3f}: ${level.price:.2f}")

        print("\n🎉 SUCCESS: Elliott Wave analysis components are functional!")
        print("📈 The core algorithms work with proper OHLCV data")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_wave_detection()
    if success:
        print("\n✅ CONCLUSION: The analysis engine works! The issue is only the missing data loading.")
        print("💡 RECOMMENDATION: Implement DataLoader or use synthetic data for development.")
    else:
        print("\n❌ CONCLUSION: The analysis engine has issues beyond just data loading.")
        print("🔧 RECOMMENDATION: Debug the core analysis components.")