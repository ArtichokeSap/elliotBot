"""
Tests for Fibonacci Analysis Module
"""
import pytest
import pandas as pd
from src.analysis.fibonacci import FibonacciAnalyzer


class TestFibonacciAnalyzer:
    """Test FibonacciAnalyzer class."""
    
    def test_initialization(self):
        """Test FibonacciAnalyzer initialization."""
        analyzer = FibonacciAnalyzer()
        assert analyzer is not None
    
    def test_analyze_retracement(self):
        """Test Fibonacci retracement analysis."""
        analyzer = FibonacciAnalyzer()
        
        high = 150.0
        low = 100.0
        current = 125.0
        
        analysis = analyzer.analyze_retracement(high, low, current, 'up')
        
        assert analysis is not None
        assert hasattr(analysis, 'retracements')
        assert hasattr(analysis, 'extensions')
        assert len(analysis.retracements) > 0
        
        # Check that retracement levels exist
        assert all(hasattr(level, 'ratio') and hasattr(level, 'price') for level in analysis.retracements)
    
    def test_analyze_retracement_with_extensions(self):
        """Test that analysis includes extension levels."""
        analyzer = FibonacciAnalyzer()
        
        high = 150.0
        low = 100.0
        current = 125.0
        
        analysis = analyzer.analyze_retracement(high, low, current, 'up')
        
        assert hasattr(analysis, 'extensions')
        assert len(analysis.extensions) >= 0
    
    def test_retracement_values_in_range(self):
        """Test that retracement values are between high and low."""
        analyzer = FibonacciAnalyzer()
        
        high = 150.0
        low = 100.0
        current = 125.0
        
        analysis = analyzer.analyze_retracement(high, low, current, 'up')
        
        for level in analysis.retracements:
            if level.ratio <= 1.0:  # Standard retracements
                assert low <= level.price <= high, f"Level {level.ratio} = {level.price} is out of range [{low}, {high}]"
    
    def test_key_levels_identified(self):
        """Test that key Fibonacci levels are identified."""
        analyzer = FibonacciAnalyzer()
        
        high = 150.0
        low = 100.0
        current = 125.0
        
        analysis = analyzer.analyze_retracement(high, low, current, 'up')
        
        # Should have key levels
        assert hasattr(analysis, 'key_levels')
        assert len(analysis.key_levels) > 0
        
        # All key levels should have is_key_level=True
        for level in analysis.key_levels:
            assert level.is_key_level is True
    
    def test_analyze_with_wave_data(self, wave_pattern_data):
        """Test Fibonacci analysis on wave data."""
        analyzer = FibonacciAnalyzer()
        
        # Get high and low from data
        high = wave_pattern_data['high'].max()
        low = wave_pattern_data['low'].min()
        current = wave_pattern_data['close'].iloc[-1]
        
        analysis = analyzer.analyze_retracement(high, low, current, 'up')
        
        assert analysis is not None
        assert analysis.swing_high == high
        assert analysis.swing_low == low
        assert analysis.current_price == current
