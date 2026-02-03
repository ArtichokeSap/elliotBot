# Technical Indicators (ZigZag)

This document describes the ZigZag implementation used by the Elliott Wave Bot.

## ZigZag (percent-change method)
- Default algorithm: `percent` (threshold as fractional value, e.g., `0.05` = 5%)
- Function: `TechnicalIndicators.zigzag(data, threshold=0.05, min_distance=3, method='percent')`
- Returns: `(zigzag_series, direction_series)` where `zigzag_series` has pivot prices and `direction_series` has `1` for peaks and `-1` for troughs.

## Configuration
Add the following keys to control behavior:
- `wave_detection.zigzag_algorithm` (default: `'percent'`)
- `wave_detection.zigzag_threshold` (default: `0.05`)
- `wave_detection.zigzag_min_distance` (default: `3`)

## Notes
- Designed to be simple and testable. ATR and hybrid methods can be added without changing the public API.
- Tests: `tests/test_indicators.py`, `tests/test_wave_detector_integration.py`