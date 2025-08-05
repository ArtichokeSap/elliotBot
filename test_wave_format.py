#!/usr/bin/env python3
"""
Quick test to see the format of the enhanced detector output
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.data_loader import DataLoader
from src.analysis.enhanced_wave_detector import EnhancedWaveDetector
import json

def test_wave_format():
    """Test the format of waves returned by enhanced detector"""
    
    data_loader = DataLoader()
    enhanced_detector = EnhancedWaveDetector(min_wave_size=0.02, lookback_periods=5)
    
    # Load AAPL data
    market_data = data_loader.get_yahoo_data('AAPL', period='1y', interval='1d')
    print(f"Loaded {len(market_data)} data points for AAPL")
    
    # Analyze with enhanced detector
    analysis_result = enhanced_detector.detect_elliott_waves(market_data, 'AAPL')
    
    print(f"\nAnalysis result keys: {list(analysis_result.keys())}")
    print(f"Number of waves: {len(analysis_result['waves'])}")
    
    # Show first wave structure
    if analysis_result['waves']:
        first_wave = analysis_result['waves'][0]
        print(f"\nFirst wave structure:")
        print(json.dumps(first_wave, indent=2, default=str))
    
    return analysis_result

if __name__ == '__main__':
    result = test_wave_format()
