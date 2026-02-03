import pandas as pd
import numpy as np
from src.analysis.wave_detector import WaveDetector


def create_sample_data(n=200):
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n, freq='D')
    base_price = 100
    trend = np.linspace(0, 30, n)
    waves = 5 * np.sin(np.linspace(0, 4 * np.pi, n))
    noise = np.random.randn(n) * 2
    close_prices = base_price + trend + waves + noise
    high_prices = close_prices + np.abs(np.random.randn(n) * 1.5)
    low_prices = close_prices - np.abs(np.random.randn(n) * 1.5)
    open_prices = close_prices + np.random.randn(n) * 0.5
    volume = np.random.randint(100000, 500000, n)

    data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)
    return data


def test_wave_detector_get_swing_points():
    data = create_sample_data()
    detector = WaveDetector()
    swings = detector._get_swing_points(data)

    # With synthetic data, expect at least 2 swing points (realistic for zigzag)
    assert len(swings) >= 2
