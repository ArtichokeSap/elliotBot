import pandas as pd
import numpy as np
from src.data.indicators import TechnicalIndicators


def test_zigzag_basic():
    # Construct a simple series with clear peaks and troughs
    idx = pd.date_range('2024-01-01', periods=9, freq='D')
    close = pd.Series([100, 102, 105, 103, 99, 101, 104, 98, 95], index=idx)
    data = pd.DataFrame({'close': close})

    zigzag, direction = TechnicalIndicators.zigzag(data, threshold=0.04, min_distance=1)

    # Expect at least one pivot (peak or trough)
    assert (direction != 0).sum() >= 1


def test_zigzag_threshold_sensitivity():
    idx = pd.date_range('2024-01-01', periods=9, freq='D')
    close = pd.Series([100, 101, 102, 103, 104, 103, 102, 101, 100], index=idx)
    data = pd.DataFrame({'close': close})

    # Small threshold should detect more pivots than large threshold
    zigzag_small, direction_small = TechnicalIndicators.zigzag(data, threshold=0.01, min_distance=1)
    zigzag_large, direction_large = TechnicalIndicators.zigzag(data, threshold=0.05, min_distance=1)

    small_count = (direction_small != 0).sum()
    large_count = (direction_large != 0).sum()

    assert small_count >= large_count  # More sensitive threshold detects more pivots


def test_zigzag_flat_and_nan():
    idx = pd.date_range('2024-01-01', periods=6, freq='D')
    close = pd.Series([100, 100, np.nan, 100, 100, 100], index=idx)
    data = pd.DataFrame({'close': close})

    zigzag, direction = TechnicalIndicators.zigzag(data, threshold=0.02)
    assert (direction != 0).sum() == 0


def test_zigzag_min_distance():
    # Rapid oscillation: with min_distance large, should reduce pivots
    idx = pd.date_range('2024-01-01', periods=10, freq='D')
    close = pd.Series([100, 104, 99, 105, 98, 106, 97, 107, 96, 108], index=idx)
    data = pd.DataFrame({'close': close})

    zigzag_small, direction_small = TechnicalIndicators.zigzag(data, threshold=0.03, min_distance=1)
    zigzag_large, direction_large = TechnicalIndicators.zigzag(data, threshold=0.03, min_distance=3)

    small_count = (direction_small != 0).sum()
    large_count = (direction_large != 0).sum()

    assert large_count < small_count  # Larger min_distance reduces pivots