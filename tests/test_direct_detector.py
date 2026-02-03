#!/usr/bin/env python3
"""
Direct test of the enhanced wave detector to debug the web API issue
"""

import sys
sys.path.append('.')

import pandas as pd
from src.data.data_loader import DataLoader
from src.analysis.enhanced_wave_detector import EnhancedWaveDetector

def test_direct_detection():
    print("🔍 Direct Enhanced Wave Detector Test")
    print("=" * 50)
    
    # Initialize components
    data_loader = DataLoader()
    enhanced_detector = EnhancedWaveDetector()
    
    # Load data
    print("📊 Loading AAPL data...")
    market_data = data_loader.get_yahoo_data('AAPL', period='3mo', interval='1d')
    print(f"✅ Loaded {len(market_data)} records")
    
    # Test detection
    print("\n🌊 Running enhanced wave detection...")
    analysis_result = enhanced_detector.detect_elliott_waves(market_data, 'AAPL')
    
    # Print results
    print(f"\n📈 Analysis Result Keys: {list(analysis_result.keys())}")
    print(f"🎯 Validation Score: {analysis_result.get('validation_score', 0):.1%}")
    print(f"🌊 Waves Detected: {len(analysis_result.get('waves', []))}")
    print(f"📊 Wave Structure: {analysis_result.get('wave_structure', 'Unknown')}")
    
    # Check wave data format
    waves = analysis_result.get('waves', [])
    if waves:
        print(f"\n🔍 First Wave Keys: {list(waves[0].keys())}")
        print(f"Wave Data: {waves[0]}")
    
    print("\n✅ Direct test completed!")
    return analysis_result

if __name__ == "__main__":
    result = test_direct_detection()
