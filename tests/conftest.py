"""
Pytest Configuration and Fixtures
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def sample_data():
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Generate realistic price data with upward trend
    np.random.seed(42)
    base_price = 100
    trend = np.linspace(0, 20, 100)
    noise = np.random.randn(100) * 2
    close = base_price + trend + noise
    
    data = pd.DataFrame({
        'open': close + np.random.randn(100) * 0.5,
        'high': close + abs(np.random.randn(100) * 1.5),
        'low': close - abs(np.random.randn(100) * 1.5),
        'close': close,
        'volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    
    # Ensure high is highest and low is lowest
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data


@pytest.fixture
def wave_pattern_data():
    """Generate data with a clear Elliott Wave pattern."""
    dates = pd.date_range(start='2023-01-01', periods=150, freq='D')
    
    # Create 5-wave impulse pattern
    wave_1 = np.linspace(100, 120, 30)  # Up
    wave_2 = np.linspace(120, 112, 15)  # Down (retracement)
    wave_3 = np.linspace(112, 140, 40)  # Up (strongest)
    wave_4 = np.linspace(140, 135, 20)  # Down (retracement)
    wave_5 = np.linspace(135, 150, 45)  # Up (final)
    
    close = np.concatenate([wave_1, wave_2, wave_3, wave_4, wave_5])
    
    data = pd.DataFrame({
        'open': close + np.random.randn(150) * 0.3,
        'high': close + abs(np.random.randn(150) * 0.5),
        'low': close - abs(np.random.randn(150) * 0.5),
        'close': close,
        'volume': np.random.randint(1000000, 5000000, 150)
    }, index=dates)
    
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data


@pytest.fixture
def config_dict():
    """Return a test configuration dictionary."""
    return {
        'wave_detection': {
            'zigzag_threshold': 0.05,
            'min_wave_length': 5,
            'confidence_threshold': 0.7
        },
        'fibonacci': {
            'retracement_levels': [0.236, 0.382, 0.5, 0.618, 0.786],
            'extension_levels': [1.272, 1.618, 2.618]
        },
        'trading': {
            'risk_per_trade': 0.02,
            'max_positions': 3
        }
    }
