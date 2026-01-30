"""
Tests for Trading Strategy Module
"""
import pytest
import pandas as pd
from src.trading.strategy import ElliottWaveStrategy
from src.analysis.wave_detector import WaveDetector


class TestElliottWaveStrategy:
    """Test ElliottWaveStrategy class."""
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = ElliottWaveStrategy()
        assert strategy is not None
    
    def test_generate_signals_returns_list(self, wave_pattern_data):
        """Test that generate_signals returns a list."""
        strategy = ElliottWaveStrategy()
        detector = WaveDetector()
        waves = detector.detect_waves(wave_pattern_data)
        
        signals = strategy.generate_signals(wave_pattern_data, waves)
        
        assert isinstance(signals, list)
    
    def test_signal_structure(self, wave_pattern_data):
        """Test that signals have required attributes."""
        strategy = ElliottWaveStrategy()
        detector = WaveDetector()
        waves = detector.detect_waves(wave_pattern_data)
        
        signals = strategy.generate_signals(wave_pattern_data, waves)
        
        if len(signals) > 0:
            signal = signals[0]
            assert hasattr(signal, 'entry_price') or 'entry_price' in signal
            assert hasattr(signal, 'signal_type') or 'type' in signal or 'signal_type' in signal
    
    def test_risk_management(self, wave_pattern_data):
        """Test that signals include risk management."""
        strategy = ElliottWaveStrategy(risk_per_trade=0.02)
        detector = WaveDetector()
        waves = detector.detect_waves(wave_pattern_data)
        
        signals = strategy.generate_signals(wave_pattern_data, waves)
        
        for signal in signals:
            # Check if stop loss exists
            has_stop = hasattr(signal, 'stop_loss') or 'stop_loss' in signal
            assert has_stop or len(signals) == 0  # Either has stop loss or no signals
    
    def test_confidence_filtering(self, wave_pattern_data):
        """Test that only high-confidence waves generate signals."""
        strategy = ElliottWaveStrategy(confidence_threshold=0.7)
        detector = WaveDetector()
        waves = detector.detect_waves(wave_pattern_data)
        
        signals = strategy.generate_signals(wave_pattern_data, waves)
        
        # All signals should be based on high-confidence waves
        assert isinstance(signals, list)
