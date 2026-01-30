"""
Tests for Visualization Module
"""
import pytest
import pandas as pd
from src.visualization.visualizer import WaveVisualizer
from src.analysis.wave_detector import WaveDetector


class TestWaveVisualizer:
    """Test WaveVisualizer class."""
    
    def test_initialization(self):
        """Test WaveVisualizer initialization."""
        visualizer = WaveVisualizer()
        assert visualizer is not None
    
    def test_plot_waves_returns_figure(self, sample_data):
        """Test that plot_waves returns a Plotly figure."""
        detector = WaveDetector()
        waves = detector.detect_waves(sample_data)
        
        visualizer = WaveVisualizer()
        fig = visualizer.plot_waves(sample_data, waves)
        
        assert fig is not None
        assert hasattr(fig, 'data')  # Plotly figure has data attribute
    
    def test_plot_with_empty_waves(self, sample_data):
        """Test plotting with no waves detected."""
        visualizer = WaveVisualizer()
        fig = visualizer.plot_waves(sample_data, [])
        
        assert fig is not None
    
    def test_plot_with_fibonacci_analysis(self, wave_pattern_data):
        """Test plotting with Fibonacci levels."""
        detector = WaveDetector()
        waves = detector.detect_waves(wave_pattern_data)
        
        visualizer = WaveVisualizer()
        fig = visualizer.plot_waves(
            wave_pattern_data, 
            waves,
            fibonacci_analysis={'levels': {0.618: 125.0}}
        )
        
        assert fig is not None
    
    def test_custom_title(self, sample_data):
        """Test custom title setting."""
        visualizer = WaveVisualizer()
        title = "Custom Test Title"
        fig = visualizer.plot_waves(sample_data, [], title=title)
        
        assert fig is not None
        # Check if title is in layout
        if hasattr(fig, 'layout') and hasattr(fig.layout, 'title'):
            assert title in str(fig.layout.title)
    
    def test_candlestick_data(self, sample_data):
        """Test that candlestick data is properly formatted."""
        visualizer = WaveVisualizer()
        fig = visualizer.plot_waves(sample_data, [])
        
        # Should have traces
        assert len(fig.data) > 0
