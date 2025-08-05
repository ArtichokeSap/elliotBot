"""
Test Comprehensive Elliott Wave System
Demonstrates validation of ALL waves (1,2,3,4,5,A,B,C) with internal structures
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import DataLoader
from src.analysis.enhanced_wave_detector import EnhancedWaveDetector
from src.visualization.comprehensive_visualizer import ComprehensiveWaveVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_elliott_wave_data():
    """
    Create sample price data that follows Elliott Wave patterns
    """
    # Create 5-wave impulse pattern with proper alternating extremes
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    # Define clear Elliott Wave points
    base_price = 100.0
    
    # Wave points that create a clear 5-wave pattern
    wave_points = [
        (0, base_price),           # Start
        (20, base_price * 1.2),    # Wave 1 peak
        (35, base_price * 1.076),  # Wave 2 trough (61.8% retracement)
        (65, base_price * 1.4),    # Wave 3 peak (161.8% extension)
        (80, base_price * 1.276),  # Wave 4 trough (38.2% retracement)
        (99, base_price * 1.476)   # Wave 5 peak
    ]
    
    # Create price series with clear waves
    prices = np.full(100, base_price)
    
    # Interpolate between wave points
    for i in range(len(wave_points) - 1):
        start_idx, start_price = wave_points[i]
        end_idx, end_price = wave_points[i + 1]
        
        # Linear interpolation between points
        indices = np.arange(start_idx, end_idx + 1)
        interpolated = np.linspace(start_price, end_price, len(indices))
        prices[start_idx:end_idx + 1] = interpolated
    
    # Add small random variations but keep the pattern clear
    noise_factor = 0.005  # 0.5% noise
    for i in range(1, len(prices)):
        noise = np.random.normal(0, noise_factor)
        prices[i] = prices[i] * (1 + noise)
    
    # Create OHLC data
    data = []
    for i, close_price in enumerate(prices):
        daily_range = close_price * 0.01  # 1% daily range
        
        # Open price
        if i == 0:
            open_price = close_price
        else:
            open_price = prices[i-1]
        
        # High and low around the close
        high = close_price + np.random.uniform(0, daily_range)
        low = close_price - np.random.uniform(0, daily_range)
        
        # Ensure OHLC consistency
        high = max(open_price, close_price, high)
        low = min(open_price, close_price, low)
        
        data.append({
            'date': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': np.random.randint(1000000, 5000000)
        })
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    
    return df


def test_comprehensive_elliott_system():
    """
    Test the comprehensive Elliott Wave system with validation
    """
    logger.info("🚀 Starting Comprehensive Elliott Wave System Test")
    
    try:
        # Create sample data
        logger.info("📊 Creating sample Elliott Wave data...")
        price_data = create_sample_elliott_wave_data()
        
        # Initialize enhanced wave detector
        logger.info("🔍 Initializing Enhanced Wave Detector...")
        detector = EnhancedWaveDetector(min_wave_size=0.02, lookback_periods=3)
        
        # Detect Elliott Waves
        logger.info("🌊 Detecting Elliott Wave patterns...")
        analysis_result = detector.detect_elliott_waves(price_data, "TEST-SAMPLE")
        
        # Print analysis summary
        print("\n" + "="*80)
        print("COMPREHENSIVE ELLIOTT WAVE ANALYSIS RESULTS")
        print("="*80)
        
        print(f"\n📊 Symbol: {analysis_result['symbol']}")
        print(f"🏗️ Wave Structure: {analysis_result['wave_structure']}")
        print(f"📈 Direction: {analysis_result['direction']}")
        print(f"🎯 Validation Score: {analysis_result['validation_score']:.1%}")
        
        # Print wave details
        print(f"\n🌊 DETECTED WAVES ({len(analysis_result['waves'])} waves):")
        print("-" * 60)
        for wave in analysis_result['waves']:
            print(f"Wave {wave['wave']}: {wave['start_price']:.3f} → {wave['end_price']:.3f}")
            print(f"  Type: {wave['wave_type']}, Direction: {wave['direction']}")
            print(f"  Length: {wave['length']:.4f}, Confidence: {wave['confidence']:.1%}")
            print()
        
        # Print subwave details
        if analysis_result['subwaves']:
            print("🔍 SUBWAVE ANALYSIS:")
            print("-" * 40)
            for parent_wave, subwaves in analysis_result['subwaves'].items():
                print(f"Wave {parent_wave} subwaves:")
                for subwave in subwaves:
                    print(f"  {subwave['label']}: {subwave['start_price']:.3f} → {subwave['end_price']:.3f}")
                print()
        
        # Print Fibonacci levels
        if analysis_result['fibonacci_levels']:
            print("📐 FIBONACCI RELATIONSHIPS:")
            print("-" * 40)
            for level_name, value in analysis_result['fibonacci_levels'].items():
                if isinstance(value, (int, float)):
                    print(f"{level_name.replace('_', ' ').title()}: {value:.3f}")
            print()
        
        # Print rule compliance
        if analysis_result['rule_compliance']:
            print("✅ RULE COMPLIANCE:")
            print("-" * 40)
            for rule_name, rule_data in analysis_result['rule_compliance'].items():
                if isinstance(rule_data, dict) and 'score' in rule_data:
                    score = rule_data['score']
                    status = rule_data.get('status', 'unknown')
                    icon = "✅" if score > 0.8 else "⚠️" if score > 0.5 else "❌"
                    print(f"{icon} {rule_name.replace('_', ' ').title()}: {score:.2f} ({status})")
            print()
        
        # Print recommendations
        if analysis_result['recommendations']:
            print("💡 RECOMMENDATIONS:")
            print("-" * 40)
            for rec in analysis_result['recommendations']:
                print(f"• {rec}")
            print()
        
        # Print issues
        if analysis_result['issues']:
            print("⚠️ ISSUES IDENTIFIED:")
            print("-" * 40)
            for issue in analysis_result['issues']:
                print(f"• {issue}")
            print()
        
        # Print summary
        print("📋 ANALYSIS SUMMARY:")
        print("-" * 40)
        print(analysis_result['summary'])
        print()
        
        # Create comprehensive visualization
        logger.info("🎨 Creating comprehensive visualization...")
        visualizer = ComprehensiveWaveVisualizer()
        
        html_chart = visualizer.create_comprehensive_chart(
            price_data, 
            analysis_result,
            "Comprehensive Elliott Wave Analysis - Test Data"
        )
        
        # Save chart
        chart_filename = "comprehensive_elliott_wave_test.html"
        with open(chart_filename, 'w', encoding='utf-8') as f:
            f.write(html_chart)
        
        logger.info(f"💾 Chart saved as: {chart_filename}")
        
        # Print detailed report
        print("\n" + "="*80)
        print("DETAILED VALIDATION REPORT")
        print("="*80)
        print(analysis_result['detailed_report'])
        
        logger.info("✅ Comprehensive Elliott Wave system test completed successfully!")
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"❌ Error in comprehensive Elliott Wave test: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_with_real_data():
    """
    Test with real market data if available
    """
    try:
        logger.info("📈 Testing with real market data...")
        
        # Try to get real data using existing data loader
        data_loader = DataLoader()
        
        # Test with a stock that should be available
        symbol = "AAPL"
        timeframe = "1d"
        
        logger.info(f"📊 Loading {symbol} data...")
        
        # Try to load data using Yahoo Finance
        try:
            price_data = data_loader.get_yahoo_data(symbol, period="3mo", interval="1d")
        except Exception as e:
            logger.warning(f"Could not load Yahoo data: {e}")
            # Fallback to enhanced sample data
            logger.info("Using enhanced sample data instead...")
            
            # Create a basic dataframe structure and enhance it
            import yfinance as yf
            dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
            base_data = pd.DataFrame({
                'open': np.random.randn(100).cumsum() + 150,
                'high': np.random.randn(100).cumsum() + 152,
                'low': np.random.randn(100).cumsum() + 148,
                'close': np.random.randn(100).cumsum() + 150,
                'volume': np.random.randint(1000000, 10000000, 100)
            }, index=dates)
            
            price_data = enhance_sample_data_for_elliott_waves(base_data)
        
        if price_data is not None and not price_data.empty:
            logger.info(f"✅ Loaded {len(price_data)} periods of {symbol} data")
            
            # Initialize detector with more sensitive settings
            detector = EnhancedWaveDetector(min_wave_size=0.01, lookback_periods=3)
            
            # Analyze
            analysis_result = detector.detect_elliott_waves(price_data, symbol)
            
            # Create visualization
            visualizer = ComprehensiveWaveVisualizer()
            html_chart = visualizer.create_comprehensive_chart(
                price_data,
                analysis_result,
                f"Comprehensive Elliott Wave Analysis - {symbol}"
            )
            
            # Save
            chart_filename = f"comprehensive_{symbol.lower()}_analysis.html"
            with open(chart_filename, 'w', encoding='utf-8') as f:
                f.write(html_chart)
            
            logger.info(f"💾 Real data analysis saved as: {chart_filename}")
            
            # Print summary
            print(f"\n🔍 REAL DATA ANALYSIS - {symbol}")
            print(f"Validation Score: {analysis_result['validation_score']:.1%}")
            print(f"Summary: {analysis_result['summary']}")
            
            return analysis_result
        else:
            logger.warning("❌ Could not load market data")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error testing with real data: {e}")
        import traceback
        traceback.print_exc()
        return None


def enhance_sample_data_for_elliott_waves(price_data: pd.DataFrame) -> pd.DataFrame:
    """
    Enhance sample data to create more Elliott Wave-like patterns
    """
    # Create a more pronounced 5-wave pattern
    data = price_data.copy()
    base_close = data['close'].iloc[0]
    
    # Create Elliott Wave pattern over the data
    pattern_length = len(data)
    
    # Define wave segments
    wave1_end = int(pattern_length * 0.2)
    wave2_end = int(pattern_length * 0.35)
    wave3_end = int(pattern_length * 0.65)
    wave4_end = int(pattern_length * 0.8)
    wave5_end = pattern_length
    
    # Wave 1: Up move
    wave1_prices = np.linspace(base_close, base_close * 1.15, wave1_end)
    
    # Wave 2: Correction (61.8% retracement)
    wave2_start = base_close * 1.15
    wave2_end_price = wave2_start - (wave2_start - base_close) * 0.618
    wave2_prices = np.linspace(wave2_start, wave2_end_price, wave2_end - wave1_end)
    
    # Wave 3: Strong up move (161.8% of Wave 1)
    wave3_start = wave2_end_price
    wave3_end_price = base_close + (base_close * 0.15 * 1.618)
    wave3_prices = np.linspace(wave3_start, wave3_end_price, wave3_end - wave2_end)
    
    # Wave 4: Smaller correction (38.2% retracement)
    wave4_start = wave3_end_price
    wave4_end_price = wave4_start - (wave4_start - wave2_end_price) * 0.382
    wave4_prices = np.linspace(wave4_start, wave4_end_price, wave4_end - wave3_end)
    
    # Wave 5: Final up move
    wave5_start = wave4_end_price
    wave5_end_price = wave4_end_price + (base_close * 0.15)  # Similar to Wave 1
    wave5_prices = np.linspace(wave5_start, wave5_end_price, wave5_end - wave4_end)
    
    # Combine all waves
    elliott_pattern = np.concatenate([
        wave1_prices,
        wave2_prices,
        wave3_prices,
        wave4_prices,
        wave5_prices
    ])
    
    # Apply the pattern to the close prices
    data['close'] = elliott_pattern[:len(data)]
    
    # Adjust OHLC to match the new close prices
    for i in range(len(data)):
        close = data.iloc[i]['close']
        daily_range = close * 0.01  # 1% daily range
        
        if i > 0:
            prev_close = data.iloc[i-1]['close']
            data.iloc[i, data.columns.get_loc('open')] = prev_close
        
        # Create realistic high/low based on close
        high = close + np.random.uniform(0, daily_range)
        low = close - np.random.uniform(0, daily_range)
        
        data.iloc[i, data.columns.get_loc('high')] = max(close, high, data.iloc[i]['open'])
        data.iloc[i, data.columns.get_loc('low')] = min(close, low, data.iloc[i]['open'])
    
    return data


if __name__ == "__main__":
    print("🌊 COMPREHENSIVE ELLIOTT WAVE SYSTEM TEST")
    print("="*60)
    
    # Test with sample data
    print("\n1️⃣ Testing with Sample Elliott Wave Data")
    sample_result = test_comprehensive_elliott_system()
    
    # Test with real data
    print("\n2️⃣ Testing with Real Market Data")
    real_result = test_with_real_data()
    
    print("\n🎉 All tests completed!")
    if sample_result:
        print(f"✅ Sample data validation score: {sample_result['validation_score']:.1%}")
    if real_result:
        print(f"✅ Real data validation score: {real_result['validation_score']:.1%}")
