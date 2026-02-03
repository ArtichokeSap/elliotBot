"""
Quick TradingView-Style Elliott Wave Demo
Showcases the professional Elliott Wave visualization features
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_loader import DataLoader
from src.analysis.wave_detector import WaveDetector
from src.visualization.tradingview_style import create_tradingview_chart
import warnings

warnings.filterwarnings('ignore')


def demo_tradingview_style():
    """Demonstrate TradingView-style Elliott Wave analysis"""
    print("🚀 TradingView-Style Elliott Wave Demo")
    print("=" * 50)
    
    # Load Apple data
    print("📊 Loading AAPL data...")
    loader = DataLoader()
    data = loader.get_yahoo_data("AAPL", period="1y")
    
    if data is None or len(data) < 50:
        print("❌ Failed to load data")
        return
    
    print(f"✅ Loaded {len(data)} data points")
    
    # Detect waves
    print("🌊 Detecting Elliott Waves...")
    detector = WaveDetector()
    waves = detector.detect_waves(data)
    
    if not waves:
        print("❌ No waves detected")
        return
    
    print(f"✅ Found {len(waves)} Elliott Wave patterns")
    
    # Create professional chart
    print("🎨 Creating TradingView-style chart...")
    
    fig = create_tradingview_chart(
        data=data,
        waves=waves,
        symbol="AAPL",
        degree="primary",
        save_path="aapl_demo_tradingview.html"
    )
    
    print("✅ Professional chart created: aapl_demo_tradingview.html")
    
    # Key features demonstrated:
    print("\n🎯 TradingView-Style Features:")
    print("  ✓ Professional wave labeling (1, 2, 3, 4, 5, A, B, C)")
    print("  ✓ Clean TradingView color scheme")
    print("  ✓ Professional candlestick styling")
    print("  ✓ Wave trend lines and annotations")
    print("  ✓ Volume analysis with color coding")
    print("  ✓ Interactive hover information")
    print("  ✓ Clean legend and layout")
    
    # Wave summary
    print(f"\n📈 Wave Analysis Summary:")
    for i, wave in enumerate(waves):
        direction = "↗️" if wave.direction.value > 0 else "↘️"
        print(f"  Wave {i+1}: {wave.wave_type.value} {direction} "
              f"({wave.confidence:.1%} confidence, {wave.price_change_pct:.1%} change)")
    
    print("\n💡 Open 'aapl_demo_tradingview.html' in your browser to see the professional chart!")


if __name__ == "__main__":
    demo_tradingview_style()
