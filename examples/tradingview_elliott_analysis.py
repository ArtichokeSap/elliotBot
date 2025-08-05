"""
TradingView-Style Elliott Wave Analysis Example
Creates professional Elliott Wave charts matching TradingView appearance
"""

import sys
import os
import warnings
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_loader import DataLoader
from src.analysis.wave_detector import WaveDetector
from src.analysis.fibonacci import FibonacciAnalyzer
from src.visualization.visualizer import WaveVisualizer
from src.visualization.tradingview_style import create_tradingview_chart
import pandas as pd

warnings.filterwarnings('ignore')


def create_professional_analysis(symbol="AAPL", period="1y", save_chart=True):
    """
    Create professional TradingView-style Elliott Wave analysis
    
    Args:
        symbol: Stock symbol to analyze
        period: Time period for analysis
        save_chart: Whether to save the chart to HTML
    """
    print(f"\n🚀 Creating TradingView-style Elliott Wave Analysis for {symbol}")
    print("=" * 60)
    
    try:
        # 1. Load market data
        print(f"📊 Loading {symbol} data for period: {period}")
        loader = DataLoader()
        data = loader.get_yahoo_data(symbol, period=period)
        
        if data is None or len(data) < 50:
            print(f"❌ Insufficient data for {symbol}")
            return None
        
        print(f"✅ Loaded {len(data)} data points")
        print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"   Price range: ${data['low'].min():.2f} - ${data['high'].max():.2f}")
        
        # 2. Detect Elliott Waves
        print("\n🌊 Detecting Elliott Wave patterns...")
        detector = WaveDetector()
        waves = detector.detect_waves(data)
        
        if not waves:
            print("❌ No Elliott Wave patterns detected")
            return None
        
        print(f"✅ Detected {len(waves)} Elliott Wave patterns:")
        for i, wave in enumerate(waves):
            direction = "↗️" if wave.direction.value > 0 else "↘️"
            print(f"   Wave {i+1}: {wave.wave_type.value} {direction} "
                  f"(Confidence: {wave.confidence:.1%}, "
                  f"Change: {wave.price_change_pct:.1%})")
        
        # 3. Fibonacci Analysis
        print("\n📐 Calculating Fibonacci levels...")
        try:
            fib_analyzer = FibonacciAnalyzer()
            
            # Use the most significant wave for Fibonacci analysis
            if len(waves) >= 2:
                # Find the wave with largest price movement
                major_wave = max(waves, key=lambda w: abs(w.price_change))
                fib_analysis = fib_analyzer.analyze_wave(major_wave, data)
                
                if fib_analysis and fib_analysis.key_levels:
                    print(f"✅ Generated {len(fib_analysis.key_levels)} Fibonacci levels")
                    for level in fib_analysis.key_levels[:5]:  # Show top 5
                        print(f"   {level.ratio:.1%}: ${level.price:.2f} ({level.level_type})")
                else:
                    print("⚠️  No significant Fibonacci levels generated")
                    fib_analysis = None
            else:
                fib_analysis = None
        except Exception as e:
            print(f"⚠️  Fibonacci analysis failed: {e}")
            fib_analysis = None
        
        # 4. Create professional visualizations
        print("\n🎨 Creating professional visualizations...")
        
        # Create TradingView-style chart
        save_path = f"{symbol.lower()}_tradingview_elliott_waves.html" if save_chart else None
        
        # Create the professional chart
        fig = create_tradingview_chart(
            data=data,
            waves=waves,
            symbol=symbol,
            fibonacci_analysis=fib_analysis,
            degree="primary",  # Use primary degree for main chart
            save_path=save_path
        )
        
        if save_chart and save_path:
            print(f"✅ Professional chart saved as: {save_path}")
        
        # 5. Create alternative degree chart (intermediate)
        if len(waves) >= 3:
            save_path_alt = f"{symbol.lower()}_intermediate_waves.html" if save_chart else None
            
            fig_alt = create_tradingview_chart(
                data=data,
                waves=waves,
                symbol=f"{symbol} - Intermediate Degree",
                fibonacci_analysis=fib_analysis,
                degree="intermediate",
                save_path=save_path_alt
            )
            
            if save_chart and save_path_alt:
                print(f"✅ Intermediate degree chart saved as: {save_path_alt}")
        
        # 6. Display summary
        print("\n📈 Elliott Wave Analysis Summary:")
        print(f"   Symbol: {symbol}")
        print(f"   Analysis Period: {period}")
        print(f"   Total Waves Detected: {len(waves)}")
        print(f"   Date Range: {data.index[0].date()} to {data.index[-1].date()}")
        
        # Current position analysis
        if waves:
            latest_wave = waves[-1]
            current_price = data['close'].iloc[-1]
            print(f"   Current Price: ${current_price:.2f}")
            print(f"   Latest Wave: {latest_wave.wave_type.value}")
            print(f"   Wave Direction: {'Bullish' if latest_wave.direction.value > 0 else 'Bearish'}")
            
            # Price targets based on latest wave
            if latest_wave.wave_type.value in ['1', '3', '5']:
                print(f"   Pattern: Impulse wave in progress")
            elif latest_wave.wave_type.value in ['2', '4']:
                print(f"   Pattern: Corrective wave - expect continuation")
            elif latest_wave.wave_type.value in ['A', 'B', 'C']:
                print(f"   Pattern: Complex correction in progress")
        
        print("\n🎯 Professional Elliott Wave analysis complete!")
        return fig
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_multiple_symbols(symbols=["AAPL", "TSLA", "NVDA", "BTC-USD"], period="1y"):
    """Analyze multiple symbols with TradingView-style charts"""
    print("\n🚀 Multi-Symbol Elliott Wave Analysis")
    print("=" * 50)
    
    results = {}
    
    for symbol in symbols:
        print(f"\n📊 Analyzing {symbol}...")
        try:
            fig = create_professional_analysis(symbol, period, save_chart=True)
            if fig:
                results[symbol] = fig
                print(f"✅ {symbol} analysis complete")
            else:
                print(f"❌ {symbol} analysis failed")
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
    
    print(f"\n🎯 Multi-symbol analysis complete! Analyzed {len(results)}/{len(symbols)} symbols")
    return results


def create_bitcoin_analysis():
    """Create specialized Bitcoin Elliott Wave analysis"""
    print("\n₿ Bitcoin Elliott Wave Analysis")
    print("=" * 40)
    
    return create_professional_analysis("BTC-USD", period="1y", save_chart=True)


if __name__ == "__main__":
    print("🌊 TradingView-Style Elliott Wave Analysis")
    print("=" * 50)
    
    # Example 1: Single symbol analysis (AAPL)
    print("\n1️⃣  Apple Inc. (AAPL) Analysis")
    create_professional_analysis("AAPL", period="1y")
    
    # Example 2: Bitcoin analysis
    print("\n2️⃣  Bitcoin Analysis")
    create_bitcoin_analysis()
    
    # Example 3: Tesla analysis with shorter timeframe
    print("\n3️⃣  Tesla Inc. (TSLA) Analysis")
    create_professional_analysis("TSLA", period="6mo")
    
    # Example 4: NVIDIA analysis
    print("\n4️⃣  NVIDIA Corp. (NVDA) Analysis")
    create_professional_analysis("NVDA", period="1y")
    
    print("\n🎯 All analyses complete! Check the generated HTML files.")
    print("\nGenerated files:")
    print("  - aapl_tradingview_elliott_waves.html")
    print("  - aapl_intermediate_waves.html")
    print("  - btc-usd_tradingview_elliott_waves.html")
    print("  - tsla_tradingview_elliott_waves.html")
    print("  - nvda_tradingview_elliott_waves.html")
    
    print("\n💡 Open these HTML files in your browser to view the interactive charts!")
