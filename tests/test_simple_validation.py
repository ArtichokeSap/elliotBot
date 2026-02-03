"""
Simple Elliott Wave Comprehensive Validation Test
Using existing project structure
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing components
from src.data.data_loader import DataLoader
from src.analysis.wave_detector import WaveDetector
from src.analysis.fibonacci import FibonacciAnalyzer
from src.visualization.tradingview_style import TradingViewStyleVisualizer

# Import our new comprehensive validator
from src.analysis.comprehensive_elliott_validator import ComprehensiveElliottValidator, WavePoint


def create_manual_elliott_wave_points():
    """
    Create manual Elliott Wave points for testing
    """
    # Define clear 5-wave pattern points
    wave_points = [
        WavePoint(0, 100.0, '2024-01-01'),     # Start
        WavePoint(20, 120.0, '2024-01-21'),    # Wave 1 peak
        WavePoint(35, 107.6, '2024-02-05'),    # Wave 2 trough (61.8% retracement)
        WavePoint(65, 140.0, '2024-03-06'),    # Wave 3 peak (161.8% extension)
        WavePoint(80, 127.6, '2024-03-21'),    # Wave 4 trough (38.2% retracement) 
        WavePoint(99, 147.6, '2024-04-09')     # Wave 5 peak
    ]
    
    return wave_points


def test_comprehensive_validation():
    """
    Test comprehensive Elliott Wave validation with manual points
    """
    print("🌊 COMPREHENSIVE ELLIOTT WAVE VALIDATION TEST")
    print("=" * 60)
    
    # Create sample price data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    # Create Elliott Wave pattern
    base_price = 100.0
    prices = []
    
    # Create realistic price movements
    for i in range(100):
        if i <= 20:  # Wave 1
            price = base_price + (20.0 * i / 20)
        elif i <= 35:  # Wave 2
            wave1_size = 20.0
            retrace = wave1_size * 0.618
            price = 120.0 - (retrace * (i - 20) / 15)
        elif i <= 65:  # Wave 3
            price = 107.6 + (32.4 * (i - 35) / 30)
        elif i <= 80:  # Wave 4
            wave3_size = 32.4
            retrace = wave3_size * 0.382
            price = 140.0 - (retrace * (i - 65) / 15)
        else:  # Wave 5
            price = 127.6 + (20.0 * (i - 80) / 19)
        
        prices.append(price)
    
    # Create OHLC data
    price_data = pd.DataFrame({
        'open': [prices[i-1] if i > 0 else prices[i] for i in range(len(prices))],
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': [1000000] * len(prices)
    }, index=dates)
    
    print(f"📊 Created sample data with {len(price_data)} periods")
    
    # Create manual wave points
    wave_points = create_manual_elliott_wave_points()
    print(f"🌊 Created {len(wave_points)} Elliott Wave points")
    
    # Initialize comprehensive validator
    validator = ComprehensiveElliottValidator()
    print("🔍 Initialized Comprehensive Elliott Wave Validator")
    
    # Validate the wave structure
    print("\n🎯 VALIDATING ELLIOTT WAVE STRUCTURE...")
    structure = validator.validate_complete_structure(wave_points, price_data)
    
    print(f"\n📊 VALIDATION RESULTS:")
    print(f"Wave Structure: {structure.wave_type.value}")
    print(f"Direction: {structure.direction.value}")
    print(f"Validation Score: {structure.validation_score:.1%}")
    print(f"Main Waves: {len(structure.main_waves)}")
    
    # Display main waves
    print(f"\n🌊 MAIN WAVES:")
    for wave in structure.main_waves:
        print(f"Wave {wave.label}: {wave.start.price:.2f} → {wave.end.price:.2f}")
        print(f"  Type: {wave.wave_type.value}, Direction: {wave.direction.value}")
        print(f"  Length: {wave.length:.2f}, Duration: {wave.duration}")
    
    # Display rule compliance
    print(f"\n✅ RULE COMPLIANCE:")
    for rule_name, rule_data in structure.rule_compliance.items():
        if isinstance(rule_data, dict) and 'score' in rule_data:
            score = rule_data['score']
            status = rule_data.get('status', 'unknown')
            icon = "✅" if score > 0.8 else "⚠️" if score > 0.5 else "❌"
            print(f"{icon} {rule_name.replace('_', ' ').title()}: {score:.2f} ({status})")
    
    # Display Fibonacci levels
    print(f"\n📐 FIBONACCI LEVELS:")
    for level_name, value in structure.fibonacci_levels.items():
        if isinstance(value, (int, float)):
            print(f"{level_name.replace('_', ' ').title()}: {value:.3f}")
    
    # Display recommendations and issues
    if structure.recommendations:
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in structure.recommendations:
            print(f"• {rec}")
    
    if structure.issues:
        print(f"\n⚠️ ISSUES:")
        for issue in structure.issues:
            print(f"• {issue}")
    
    # Generate detailed report
    print(f"\n📋 DETAILED REPORT:")
    print("-" * 60)
    detailed_report = validator.generate_detailed_report(structure)
    print(detailed_report)
    
    # Create visualization using existing TradingView style
    try:
        print(f"\n🎨 Creating visualization...")
        
        # Use existing wave detector to get waves in the right format
        wave_detector = WaveDetector()
        detected_waves = []
        
        # Convert our validated waves to the expected format
        for wave in structure.main_waves:
            detected_waves.append({
                'wave': wave.label,
                'start_price': wave.start.price,
                'end_price': wave.end.price,
                'start_time': wave.start.time,
                'end_time': wave.end.time,
                'direction': wave.direction.value,
                'confidence': wave.validation_score
            })
        
        # Create chart
        visualizer = TradingViewStyleVisualizer()
        chart_html = visualizer.create_professional_chart(
            price_data,
            detected_waves,
            [],  # fibonacci levels
            "Comprehensive Elliott Wave Validation Test"
        )
        
        # Save chart
        chart_file = "comprehensive_validation_test.html"
        with open(chart_file, 'w', encoding='utf-8') as f:
            f.write(chart_html)
        
        print(f"💾 Chart saved as: {chart_file}")
        
    except Exception as e:
        print(f"❌ Error creating chart: {e}")
    
    return structure


if __name__ == "__main__":
    result = test_comprehensive_validation()
    
    if result.validation_score > 0.7:
        print(f"\n🎉 EXCELLENT! Validation score: {result.validation_score:.1%}")
    elif result.validation_score > 0.5:
        print(f"\n👍 GOOD! Validation score: {result.validation_score:.1%}")
    else:
        print(f"\n🤔 NEEDS WORK! Validation score: {result.validation_score:.1%}")
    
    print("\n✅ Comprehensive Elliott Wave validation test completed!")
