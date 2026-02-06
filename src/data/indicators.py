"""
Technical indicators for the Elliott Wave project

Provides a simple, well-tested percent-change ZigZag implementation and helper
functions used by the analysis modules and tests.
"""
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np


class TechnicalIndicators:
    @staticmethod
    def zigzag(data: pd.DataFrame, threshold: float = 0.05, min_distance: int = 3, method: str = 'percent') -> Tuple[pd.Series, pd.Series]:
        """Compute a percent-change ZigZag on the close price.

        Args:
            data: DataFrame with a `close` column.
            threshold: Percent threshold for detecting pivots (e.g. 0.05 for 5%).
            min_distance: Minimum number of periods between pivots.
            method: Currently only 'percent' is supported.

        Returns:
            zigzag_series: Series with pivot prices (NaN elsewhere)
            direction_series: Series with 1 for highs, -1 for lows, 0 for others
        """
        if 'close' not in data.columns:
            raise ValueError("DataFrame must contain a 'close' column")

        close = data['close'].astype(float).copy()
        zigzag = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(0, index=close.index, dtype=int)

        # Handle trivial cases and NaNs
        if close.isnull().all() or close.dropna().nunique() == 1:
            return zigzag, direction

        # Initialize
        idxs = close.dropna().index
        last_pivot_idx = idxs[0]
        last_pivot_price = close.loc[last_pivot_idx]
        last_pivot_type = 0  # 1: high, -1: low, 0: unknown

        for ts in idxs[1:]:
            price = close.loc[ts]
            if pd.isna(price) or pd.isna(last_pivot_price):
                continue

            change = (price - last_pivot_price) / last_pivot_price

            # Check for new high
            if last_pivot_type <= 0 and change >= threshold:
                # Enforce min_distance
                if (close.index.get_loc(ts) - close.index.get_loc(last_pivot_idx)) >= min_distance:
                    zigzag.loc[ts] = price
                    direction.loc[ts] = 1
                    last_pivot_idx = ts
                    last_pivot_price = price
                    last_pivot_type = 1
                else:
                    # If within min_distance, keep the more extreme pivot
                    if price > last_pivot_price:
                        # remove previous pivot if it was recorded
                        zigzag.loc[last_pivot_idx] = np.nan
                        direction.loc[last_pivot_idx] = 0
                        zigzag.loc[ts] = price
                        direction.loc[ts] = 1
                        last_pivot_idx = ts
                        last_pivot_price = price
                        last_pivot_type = 1

            # Check for new low
            elif last_pivot_type >= 0 and change <= -threshold:
                if (close.index.get_loc(ts) - close.index.get_loc(last_pivot_idx)) >= min_distance:
                    zigzag.loc[ts] = price
                    direction.loc[ts] = -1
                    last_pivot_idx = ts
                    last_pivot_price = price
                    last_pivot_type = -1
                else:
                    if price < last_pivot_price:
                        zigzag.loc[last_pivot_idx] = np.nan
                        direction.loc[last_pivot_idx] = 0
                        zigzag.loc[ts] = price
                        direction.loc[ts] = -1
                        last_pivot_idx = ts
                        last_pivot_price = price
                        last_pivot_type = -1

            # No pivot detected; continue scanning

        return zigzag, direction

    @staticmethod
    def swing_points(data: pd.DataFrame, threshold: float = 0.05, min_distance: int = 3) -> List[Dict[str, object]]:
        """Return a list of swing points (dicts) derived from zigzag output.

        Each swing point is a dict: {'timestamp': Timestamp, 'price': float, 'type': 'high'|'low'}
        """
        zigzag, direction = TechnicalIndicators.zigzag(data, threshold=threshold, min_distance=min_distance)
        swings = []
        for ts, dir_val in direction.items():
            if dir_val == 0:
                continue
            price = float(zigzag.loc[ts])
            swings.append({'timestamp': ts, 'price': price, 'type': 'high' if dir_val == 1 else 'low'})
        return swings
