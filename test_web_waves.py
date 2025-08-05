"""
Quick test for web wave detection
Test the improved wave detection logic
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_loader import DataLoader
from src.analysis.wave_detector import WaveDetector

def test_wave_detection():
    """Test wave detection with different symbols and parameters."""
    
    print("🔍 Testing Elliott Wave Detection for Web Interface")
    print("=" * 60)
    
    loader = DataLoader()
    
    # Test symbols
    test_symbols = ['AAPL', 'BTC-USD', 'EURUSD=X', 'TSLA']
    
    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol}:")
        
        try:
            # Load data
            data = loader.get_yahoo_data(symbol, period='1y')
            print(f"   ✅ Loaded {len(data)} data points")
            
            # Test different sensitivity levels
            sensitivity_levels = [0.02, 0.03, 0.04, 0.05, 0.06]
            
            for threshold in sensitivity_levels:
                detector = WaveDetector()
                detector.zigzag_threshold = threshold
                detector.min_wave_length = max(3, len(data) // 100)
                detector.confidence_threshold = 0.4
                
                waves = detector.detect_waves(data)
                
                if len(waves) > 0:
                    print(f"   🌊 Found {len(waves)} waves with threshold {threshold}")
                    
                    # Show wave details
                    for i, wave in enumerate(waves[:3]):  # Show first 3 waves
                        wave_type = wave.wave_type.value.split('_')[-1]
                        direction = "📈" if wave.direction.value == 1 else "📉"
                        print(f"      {i+1}. Wave {wave_type} {direction} - Confidence: {wave.confidence:.2f}")
                    break
            else:
                print(f"   ❌ No waves detected with any threshold")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 Test completed! Check results above.")

if __name__ == "__main__":
    test_wave_detection()
