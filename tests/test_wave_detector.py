"""
Tests for Wave Detection Module
"""
import pytest
import pandas as pd
import numpy as np
from src.analysis.wave_detector import WaveDetector, WaveType, TrendDirection


class TestWaveDetector:
    """Test WaveDetector class."""
    
    def test_initialization(self):
        """Test WaveDetector initialization."""
        detector = WaveDetector()
        assert detector is not None
    
    def test_detect_waves_returns_list(self, sample_data):
        """Test that detect_waves returns a list."""
        detector = WaveDetector()
        waves = detector.detect_waves(sample_data)
        
        assert isinstance(waves, list)
    
    def test_detect_waves_with_pattern(self, wave_pattern_data):
        """Test wave detection with clear pattern."""
        detector = WaveDetector()
        waves = detector.detect_waves(wave_pattern_data)
        
        # Should detect some waves in patterned data
        assert len(waves) > 0
        
        # Check wave structure
        for wave in waves:
            assert hasattr(wave, 'start_point')
            assert hasattr(wave, 'end_point')
            assert hasattr(wave, 'wave_type')
            assert hasattr(wave, 'confidence')
            assert 0 <= wave.confidence <= 1
    
    def test_get_current_wave_count(self, sample_data):
        """Test getting current wave count."""
        detector = WaveDetector()
        detector.detect_waves(sample_data)
        
        result = detector.get_current_wave_count(sample_data)
        
        assert isinstance(result, dict)
        assert 'current_wave' in result
        assert 'confidence' in result
    
    def test_wave_confidence_in_range(self, wave_pattern_data):
        """Test that wave confidence is between 0 and 1."""
        detector = WaveDetector()
        waves = detector.detect_waves(wave_pattern_data)
        
        for wave in waves:
            assert 0 <= wave.confidence <= 1
    
    def test_wave_types_are_valid(self, wave_pattern_data):
        """Test that detected wave types are valid enums."""
        detector = WaveDetector()
        waves = detector.detect_waves(wave_pattern_data)
        
        valid_types = [wt for wt in WaveType]
        
        for wave in waves:
            assert wave.wave_type in valid_types
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        detector = WaveDetector()
        empty_data = pd.DataFrame()
        
        with pytest.raises(Exception):
            detector.detect_waves(empty_data)
    
    def test_zigzag_detection(self, sample_data):
        """Test zigzag pivot point detection."""
        detector = WaveDetector(zigzag_threshold=0.05)
        pivots = detector._detect_zigzag_pivots(sample_data)
        
        assert len(pivots) > 0
    
    def test_trend_direction_classification(self, wave_pattern_data):
        """Test trend direction classification."""
        detector = WaveDetector()
        waves = detector.detect_waves(wave_pattern_data)
        
        for wave in waves:
            assert wave.direction in [TrendDirection.UP, TrendDirection.DOWN, TrendDirection.SIDEWAYS]
