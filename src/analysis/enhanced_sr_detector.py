"""
Enhanced Support & Resistance Detection Module
Advanced S/R level detection using multiple methods including volume analysis, 
swing highs/lows, candle clusters, and reversal points.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

@dataclass
class SRLevel:
    """Enhanced Support/Resistance Level"""
    price: float
    type: str  # "support", "resistance", "supply", "demand"
    strength: int  # number of touches/tests
    zone_width: float  # width of the zone (±%)
    volume_confirmation: bool
    age: int  # periods since first identified
    last_test: int  # periods since last test
    conviction: float  # 0-1 score combining strength, volume, age
    formation_method: str  # how it was detected
    additional_context: Dict[str, Any]

@dataclass
class SRZone:
    """Support/Resistance Zone (range rather than single level)"""
    lower_bound: float
    upper_bound: float
    center_price: float
    type: str
    strength: int
    volume_profile: List[float]  # volume at different price levels
    conviction: float
    formation_periods: List[int]  # when it was formed/tested

class EnhancedSRDetector:
    """
    Enhanced Support & Resistance Detection System
    Combines multiple detection methods for robust S/R identification
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        
        # Configuration parameters
        self.config = config or {}
        self.zone_tolerance = self.config.get('zone_tolerance', 0.015)  # 1.5% zone width
        self.min_touches = self.config.get('min_touches', 2)
        self.lookback_periods = self.config.get('lookback_periods', 200)
        self.volume_threshold = self.config.get('volume_threshold', 1.2)  # 20% above average
        self.wick_threshold = self.config.get('wick_threshold', 0.6)  # 60% of candle range
        
    def detect_sr_levels(self, market_data: pd.DataFrame) -> Dict[str, List[SRLevel]]:
        """
        Main detection function combining all S/R methods
        
        Returns:
            Dict with 'support' and 'resistance' keys containing lists of SRLevel objects
        """
        try:
            self.logger.info("🔍 Starting Enhanced S/R Detection...")
            
            # Limit data to lookback period
            data = market_data.tail(self.lookback_periods).copy()
            
            # Initialize results
            all_support_levels = []
            all_resistance_levels = []
            
            # Method 1: Swing High/Low Detection
            swing_levels = self._detect_swing_levels(data)
            all_support_levels.extend(swing_levels['support'])
            all_resistance_levels.extend(swing_levels['resistance'])
            
            # Method 2: Volume-Weighted Levels
            volume_levels = self._detect_volume_levels(data)
            all_support_levels.extend(volume_levels['support'])
            all_resistance_levels.extend(volume_levels['resistance'])
            
            # Method 3: Rejection/Wick Analysis
            wick_levels = self._detect_wick_levels(data)
            all_support_levels.extend(wick_levels['support'])
            all_resistance_levels.extend(wick_levels['resistance'])
            
            # Method 4: Clustering Analysis
            cluster_levels = self._detect_clustered_levels(data)
            all_support_levels.extend(cluster_levels['support'])
            all_resistance_levels.extend(cluster_levels['resistance'])
            
            # Method 5: Round Number Levels
            round_levels = self._detect_round_number_levels(data)
            all_support_levels.extend(round_levels['support'])
            all_resistance_levels.extend(round_levels['resistance'])
            
            # Consolidate and rank levels
            consolidated_support = self._consolidate_levels(all_support_levels, data)
            consolidated_resistance = self._consolidate_levels(all_resistance_levels, data)
            
            # Filter and rank by conviction
            final_support = self._filter_and_rank_levels(consolidated_support, 'support')
            final_resistance = self._filter_and_rank_levels(consolidated_resistance, 'resistance')
            
            result = {
                'support_levels': final_support,
                'resistance_levels': final_resistance
            }
            
            self.logger.info(f"✅ S/R Detection Complete: {len(final_support)} support, {len(final_resistance)} resistance")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in S/R detection: {e}")
            return {'support_levels': [], 'resistance_levels': []}
    
    def _detect_swing_levels(self, data: pd.DataFrame) -> Dict[str, List[SRLevel]]:
        """Detect S/R from swing highs and lows"""
        try:
            support_levels = []
            resistance_levels = []
            
            # Calculate swing points
            swing_window = 10
            highs = data['high'].rolling(window=swing_window, center=True).max()
            lows = data['low'].rolling(window=swing_window, center=True).min()
            
            # Find swing highs (resistance)
            swing_highs = data[data['high'] == highs]['high'].dropna()
            for idx, price in swing_highs.items():
                # Count how many times this level was tested
                touches = self._count_level_touches(data, price, 'resistance')
                if touches >= self.min_touches:
                    age = len(data) - data.index.get_loc(idx)
                    last_test = self._get_last_test_age(data, price, 'resistance')
                    
                    level = SRLevel(
                        price=price,
                        type='resistance',
                        strength=touches,
                        zone_width=self.zone_tolerance,
                        volume_confirmation=self._check_volume_confirmation(data, idx, price),
                        age=age,
                        last_test=last_test,
                        conviction=self._calculate_conviction(touches, age, last_test),
                        formation_method='swing_high',
                        additional_context={'swing_index': idx}
                    )
                    resistance_levels.append(level)
            
            # Find swing lows (support)
            swing_lows = data[data['low'] == lows]['low'].dropna()
            for idx, price in swing_lows.items():
                touches = self._count_level_touches(data, price, 'support')
                if touches >= self.min_touches:
                    age = len(data) - data.index.get_loc(idx)
                    last_test = self._get_last_test_age(data, price, 'support')
                    
                    level = SRLevel(
                        price=price,
                        type='support',
                        strength=touches,
                        zone_width=self.zone_tolerance,
                        volume_confirmation=self._check_volume_confirmation(data, idx, price),
                        age=age,
                        last_test=last_test,
                        conviction=self._calculate_conviction(touches, age, last_test),
                        formation_method='swing_low',
                        additional_context={'swing_index': idx}
                    )
                    support_levels.append(level)
            
            return {'support': support_levels, 'resistance': resistance_levels}
            
        except Exception as e:
            self.logger.error(f"Error detecting swing levels: {e}")
            return {'support': [], 'resistance': []}
    
    def _detect_volume_levels(self, data: pd.DataFrame) -> Dict[str, List[SRLevel]]:
        """Detect S/R from high-volume price areas using volume profile"""
        try:
            support_levels = []
            resistance_levels = []
            
            # Create price bins for volume profile
            price_range = data['high'].max() - data['low'].min()
            num_bins = 50
            bin_size = price_range / num_bins
            
            volume_profile = {}
            
            # Calculate volume at each price level
            for idx, row in data.iterrows():
                # For each candle, distribute volume across its price range
                candle_range = row['high'] - row['low']
                if candle_range > 0:
                    price_levels = np.linspace(row['low'], row['high'], 10)
                    volume_per_level = row['volume'] / len(price_levels)
                    
                    for price in price_levels:
                        bin_key = round(price / bin_size) * bin_size
                        volume_profile[bin_key] = volume_profile.get(bin_key, 0) + volume_per_level
            
            # Find high-volume areas
            avg_volume = np.mean(list(volume_profile.values()))
            high_volume_threshold = avg_volume * self.volume_threshold
            
            for price, volume in volume_profile.items():
                if volume >= high_volume_threshold:
                    # Determine if support or resistance based on current price position
                    current_price = data['close'].iloc[-1]
                    
                    if price < current_price:
                        level_type = 'support'
                    else:
                        level_type = 'resistance'
                    
                    # Check for actual price interaction
                    touches = self._count_level_touches(data, price, level_type)
                    if touches >= 1:  # More lenient for volume-based levels
                        level = SRLevel(
                            price=price,
                            type=level_type,
                            strength=touches,
                            zone_width=self.zone_tolerance,
                            volume_confirmation=True,
                            age=self._estimate_level_age(data, price),
                            last_test=self._get_last_test_age(data, price, level_type),
                            conviction=self._calculate_volume_conviction(volume, avg_volume, touches),
                            formation_method='volume_profile',
                            additional_context={'volume': volume, 'avg_volume': avg_volume}
                        )
                        
                        if level_type == 'support':
                            support_levels.append(level)
                        else:
                            resistance_levels.append(level)
            
            return {'support': support_levels, 'resistance': resistance_levels}
            
        except Exception as e:
            self.logger.error(f"Error detecting volume levels: {e}")
            return {'support': [], 'resistance': []}
    
    def _detect_wick_levels(self, data: pd.DataFrame) -> Dict[str, List[SRLevel]]:
        """Detect S/R from candle wicks (rejection levels)"""
        try:
            support_levels = []
            resistance_levels = []
            
            for idx, row in data.iterrows():
                # Calculate wick sizes
                body_size = abs(row['close'] - row['open'])
                upper_wick = row['high'] - max(row['close'], row['open'])
                lower_wick = min(row['close'], row['open']) - row['low']
                total_range = row['high'] - row['low']
                
                if total_range == 0:
                    continue
                
                # Upper wick rejection (resistance)
                if upper_wick / total_range >= self.wick_threshold and upper_wick > body_size:
                    price = row['high']
                    touches = self._count_level_touches(data, price, 'resistance')
                    
                    level = SRLevel(
                        price=price,
                        type='resistance',
                        strength=max(1, touches),
                        zone_width=self.zone_tolerance * 0.5,  # Tighter zones for wick levels
                        volume_confirmation=self._check_volume_confirmation(data, idx, price),
                        age=len(data) - data.index.get_loc(idx),
                        last_test=0,  # Just formed
                        conviction=self._calculate_wick_conviction(upper_wick, total_range, row['volume']),
                        formation_method='upper_wick_rejection',
                        additional_context={
                            'wick_ratio': upper_wick / total_range,
                            'volume': row['volume']
                        }
                    )
                    resistance_levels.append(level)
                
                # Lower wick rejection (support)
                if lower_wick / total_range >= self.wick_threshold and lower_wick > body_size:
                    price = row['low']
                    touches = self._count_level_touches(data, price, 'support')
                    
                    level = SRLevel(
                        price=price,
                        type='support',
                        strength=max(1, touches),
                        zone_width=self.zone_tolerance * 0.5,
                        volume_confirmation=self._check_volume_confirmation(data, idx, price),
                        age=len(data) - data.index.get_loc(idx),
                        last_test=0,
                        conviction=self._calculate_wick_conviction(lower_wick, total_range, row['volume']),
                        formation_method='lower_wick_rejection',
                        additional_context={
                            'wick_ratio': lower_wick / total_range,
                            'volume': row['volume']
                        }
                    )
                    support_levels.append(level)
            
            return {'support': support_levels, 'resistance': resistance_levels}
            
        except Exception as e:
            self.logger.error(f"Error detecting wick levels: {e}")
            return {'support': [], 'resistance': []}
    
    def _detect_clustered_levels(self, data: pd.DataFrame) -> Dict[str, List[SRLevel]]:
        """Detect S/R from clustered highs/lows within narrow price bands"""
        try:
            support_levels = []
            resistance_levels = []
            
            # Collect all highs and lows
            all_highs = data['high'].values
            all_lows = data['low'].values
            
            # Find clusters in highs (resistance)
            high_clusters = self._find_price_clusters(all_highs)
            for cluster in high_clusters:
                if len(cluster['prices']) >= 2:  # At least 2 candles in cluster
                    avg_price = np.mean(cluster['prices'])
                    touches = len(cluster['prices'])
                    
                    level = SRLevel(
                        price=avg_price,
                        type='resistance',
                        strength=touches,
                        zone_width=self.zone_tolerance,
                        volume_confirmation=False,  # Would need additional analysis
                        age=self._estimate_cluster_age(data, cluster),
                        last_test=self._estimate_cluster_last_test(data, cluster),
                        conviction=self._calculate_cluster_conviction(touches, cluster['density']),
                        formation_method='clustered_highs',
                        additional_context={
                            'cluster_size': len(cluster['prices']),
                            'price_range': max(cluster['prices']) - min(cluster['prices'])
                        }
                    )
                    resistance_levels.append(level)
            
            # Find clusters in lows (support)
            low_clusters = self._find_price_clusters(all_lows)
            for cluster in low_clusters:
                if len(cluster['prices']) >= 2:
                    avg_price = np.mean(cluster['prices'])
                    touches = len(cluster['prices'])
                    
                    level = SRLevel(
                        price=avg_price,
                        type='support',
                        strength=touches,
                        zone_width=self.zone_tolerance,
                        volume_confirmation=False,
                        age=self._estimate_cluster_age(data, cluster),
                        last_test=self._estimate_cluster_last_test(data, cluster),
                        conviction=self._calculate_cluster_conviction(touches, cluster['density']),
                        formation_method='clustered_lows',
                        additional_context={
                            'cluster_size': len(cluster['prices']),
                            'price_range': max(cluster['prices']) - min(cluster['prices'])
                        }
                    )
                    support_levels.append(level)
            
            return {'support': support_levels, 'resistance': resistance_levels}
            
        except Exception as e:
            self.logger.error(f"Error detecting clustered levels: {e}")
            return {'support': [], 'resistance': []}
    
    def _detect_round_number_levels(self, data: pd.DataFrame) -> Dict[str, List[SRLevel]]:
        """Detect psychological S/R at round numbers"""
        try:
            support_levels = []
            resistance_levels = []
            
            current_price = data['close'].iloc[-1]
            price_range = data['high'].max() - data['low'].min()
            
            # Determine round number intervals based on price level
            if current_price < 1:
                intervals = [0.01, 0.05, 0.1]
            elif current_price < 10:
                intervals = [0.1, 0.5, 1.0]
            elif current_price < 100:
                intervals = [1, 5, 10]
            elif current_price < 1000:
                intervals = [10, 50, 100]
            else:
                intervals = [100, 500, 1000]
            
            min_price = data['low'].min()
            max_price = data['high'].max()
            
            for interval in intervals:
                # Find round numbers in the price range
                start_round = (min_price // interval) * interval
                end_round = ((max_price // interval) + 1) * interval
                
                round_price = start_round
                while round_price <= end_round:
                    if min_price <= round_price <= max_price:
                        # Check if this round number was actually tested
                        touches_resistance = self._count_level_touches(data, round_price, 'resistance')
                        touches_support = self._count_level_touches(data, round_price, 'support')
                        
                        if touches_resistance >= 1:
                            level = SRLevel(
                                price=round_price,
                                type='resistance',
                                strength=touches_resistance,
                                zone_width=self.zone_tolerance,
                                volume_confirmation=False,
                                age=self._estimate_level_age(data, round_price),
                                last_test=self._get_last_test_age(data, round_price, 'resistance'),
                                conviction=self._calculate_round_number_conviction(round_price, interval, touches_resistance),
                                formation_method='round_number',
                                additional_context={'interval': interval, 'round_number': True}
                            )
                            resistance_levels.append(level)
                        
                        if touches_support >= 1:
                            level = SRLevel(
                                price=round_price,
                                type='support',
                                strength=touches_support,
                                zone_width=self.zone_tolerance,
                                volume_confirmation=False,
                                age=self._estimate_level_age(data, round_price),
                                last_test=self._get_last_test_age(data, round_price, 'support'),
                                conviction=self._calculate_round_number_conviction(round_price, interval, touches_support),
                                formation_method='round_number',
                                additional_context={'interval': interval, 'round_number': True}
                            )
                            support_levels.append(level)
                    
                    round_price += interval
            
            return {'support': support_levels, 'resistance': resistance_levels}
            
        except Exception as e:
            self.logger.error(f"Error detecting round number levels: {e}")
            return {'support': [], 'resistance': []}
    
    def _count_level_touches(self, data: pd.DataFrame, price: float, level_type: str) -> int:
        """Count how many times a price level was touched/tested"""
        tolerance = price * self.zone_tolerance
        touches = 0
        
        for _, row in data.iterrows():
            if level_type == 'resistance':
                # Check if high came close to the level
                if abs(row['high'] - price) <= tolerance:
                    touches += 1
            else:  # support
                # Check if low came close to the level
                if abs(row['low'] - price) <= tolerance:
                    touches += 1
        
        return touches
    
    def _get_last_test_age(self, data: pd.DataFrame, price: float, level_type: str) -> int:
        """Get periods since last test of the level"""
        tolerance = price * self.zone_tolerance
        last_test_idx = -1
        
        for i, (_, row) in enumerate(data.iterrows()):
            if level_type == 'resistance':
                if abs(row['high'] - price) <= tolerance:
                    last_test_idx = i
            else:
                if abs(row['low'] - price) <= tolerance:
                    last_test_idx = i
        
        if last_test_idx == -1:
            return len(data)  # Never tested
        
        return len(data) - 1 - last_test_idx
    
    def _check_volume_confirmation(self, data: pd.DataFrame, idx: pd.Timestamp, price: float) -> bool:
        """Check if the level formation had above-average volume"""
        try:
            if idx not in data.index:
                return False
            
            volume = data.loc[idx, 'volume']
            avg_volume = data['volume'].rolling(20).mean().loc[idx]
            
            return volume > avg_volume * self.volume_threshold
        except:
            return False
    
    def _calculate_conviction(self, touches: int, age: int, last_test: int) -> float:
        """Calculate conviction score based on touches, age, and recency"""
        # Base score from touches
        touch_score = min(touches / 5.0, 1.0)  # Max at 5 touches
        
        # Age factor (prefer established levels but not too old)
        if age < 10:
            age_factor = age / 10.0  # Building conviction
        elif age < 50:
            age_factor = 1.0  # Optimal age
        else:
            age_factor = max(0.5, 1.0 - (age - 50) / 100.0)  # Decay over time
        
        # Recency factor (recent tests are more relevant)
        if last_test < 5:
            recency_factor = 1.0
        elif last_test < 20:
            recency_factor = 0.8
        else:
            recency_factor = max(0.3, 1.0 - last_test / 100.0)
        
        return touch_score * age_factor * recency_factor
    
    def _calculate_volume_conviction(self, volume: float, avg_volume: float, touches: int) -> float:
        """Calculate conviction for volume-based levels"""
        volume_ratio = min(volume / avg_volume / 2.0, 1.0)  # Max at 2x average
        touch_factor = min(touches / 3.0, 1.0)  # Max at 3 touches
        return volume_ratio * touch_factor
    
    def _calculate_wick_conviction(self, wick_size: float, total_range: float, volume: float) -> float:
        """Calculate conviction for wick-based levels"""
        wick_ratio = wick_size / total_range
        # Volume factor could be added here if available
        return min(wick_ratio * 1.5, 1.0)  # Max conviction at strong rejection
    
    def _calculate_cluster_conviction(self, cluster_size: int, density: float) -> float:
        """Calculate conviction for clustered levels"""
        size_factor = min(cluster_size / 5.0, 1.0)  # Max at 5 occurrences
        density_factor = min(density, 1.0)
        return size_factor * density_factor
    
    def _calculate_round_number_conviction(self, price: float, interval: float, touches: int) -> float:
        """Calculate conviction for round number levels"""
        # Bigger round numbers get higher conviction
        significance = min(interval / price * 100, 1.0)  # As percentage of price
        touch_factor = min(touches / 3.0, 1.0)
        return significance * touch_factor * 0.7  # Cap at 70% for psychological levels
    
    def _find_price_clusters(self, prices: np.ndarray) -> List[Dict]:
        """Find clusters of prices within tolerance"""
        clusters = []
        sorted_prices = np.sort(prices)
        
        i = 0
        while i < len(sorted_prices):
            cluster_prices = [sorted_prices[i]]
            j = i + 1
            
            # Find all prices within tolerance of the first price
            while j < len(sorted_prices):
                if abs(sorted_prices[j] - sorted_prices[i]) / sorted_prices[i] <= self.zone_tolerance:
                    cluster_prices.append(sorted_prices[j])
                    j += 1
                else:
                    break
            
            if len(cluster_prices) >= 2:
                price_range = max(cluster_prices) - min(cluster_prices)
                density = len(cluster_prices) / (price_range + 1e-8)  # Avoid division by zero
                
                clusters.append({
                    'prices': cluster_prices,
                    'center': np.mean(cluster_prices),
                    'density': density
                })
            
            i = j
        
        return clusters
    
    def _estimate_level_age(self, data: pd.DataFrame, price: float) -> int:
        """Estimate how long ago a level was first established"""
        tolerance = price * self.zone_tolerance
        
        for i, (_, row) in enumerate(data.iterrows()):
            if (abs(row['high'] - price) <= tolerance or 
                abs(row['low'] - price) <= tolerance):
                return len(data) - i
        
        return len(data)  # If not found, assume it's old
    
    def _estimate_cluster_age(self, data: pd.DataFrame, cluster: Dict) -> int:
        """Estimate age of a price cluster"""
        # Find earliest occurrence of any price in the cluster
        min_age = len(data)
        
        for price in cluster['prices']:
            age = self._estimate_level_age(data, price)
            min_age = min(min_age, age)
        
        return min_age
    
    def _estimate_cluster_last_test(self, data: pd.DataFrame, cluster: Dict) -> int:
        """Estimate last test of a price cluster"""
        # Find most recent occurrence of any price in the cluster
        max_recency = 0
        
        for price in cluster['prices']:
            last_test = self._get_last_test_age(data, price, 'resistance')  # Type doesn't matter for this calculation
            max_recency = max(max_recency, len(data) - last_test)
        
        return len(data) - max_recency
    
    def _consolidate_levels(self, levels: List[SRLevel], data: pd.DataFrame) -> List[SRLevel]:
        """Consolidate overlapping levels by merging similar ones"""
        if not levels:
            return []
        
        # Sort by price
        sorted_levels = sorted(levels, key=lambda x: x.price)
        consolidated = []
        
        i = 0
        while i < len(sorted_levels):
            current_level = sorted_levels[i]
            similar_levels = [current_level]
            
            # Find all levels within tolerance
            j = i + 1
            while j < len(sorted_levels):
                if (abs(sorted_levels[j].price - current_level.price) / current_level.price <= 
                    self.zone_tolerance):
                    similar_levels.append(sorted_levels[j])
                    j += 1
                else:
                    break
            
            # Merge similar levels
            if len(similar_levels) > 1:
                merged_level = self._merge_levels(similar_levels)
                consolidated.append(merged_level)
            else:
                consolidated.append(current_level)
            
            i = j
        
        return consolidated
    
    def _merge_levels(self, levels: List[SRLevel]) -> SRLevel:
        """Merge multiple similar levels into one"""
        # Weighted average price by conviction
        total_weight = sum(level.conviction for level in levels)
        if total_weight == 0:
            avg_price = np.mean([level.price for level in levels])
        else:
            avg_price = sum(level.price * level.conviction for level in levels) / total_weight
        
        # Combine strengths
        total_strength = sum(level.strength for level in levels)
        
        # Best conviction
        best_conviction = max(level.conviction for level in levels)
        
        # Combine formation methods
        methods = list(set(level.formation_method for level in levels))
        
        # Take properties from the highest conviction level
        best_level = max(levels, key=lambda x: x.conviction)
        
        return SRLevel(
            price=avg_price,
            type=best_level.type,
            strength=total_strength,
            zone_width=best_level.zone_width,
            volume_confirmation=any(level.volume_confirmation for level in levels),
            age=min(level.age for level in levels),  # Oldest age
            last_test=min(level.last_test for level in levels),  # Most recent test
            conviction=best_conviction,
            formation_method='+'.join(methods[:3]),  # Combine up to 3 methods
            additional_context={
                'merged_from': len(levels),
                'methods': methods
            }
        )
    
    def _filter_and_rank_levels(self, levels: List[SRLevel], level_type: str) -> List[SRLevel]:
        """Filter and rank levels by conviction"""
        # Filter by minimum conviction
        min_conviction = 0.3
        filtered = [level for level in levels if level.conviction >= min_conviction]
        
        # Sort by conviction (highest first)
        ranked = sorted(filtered, key=lambda x: x.conviction, reverse=True)
        
        # Limit to top levels to avoid clutter
        max_levels = 10
        return ranked[:max_levels]
    
    def format_sr_output(self, sr_results: Dict[str, List[SRLevel]]) -> Dict[str, List[float]]:
        """
        Format S/R results for integration with confluence analyzer
        Returns simple price lists as required by the user specification
        """
        support_prices = [level.price for level in sr_results['support_levels']]
        resistance_prices = [level.price for level in sr_results['resistance_levels']]
        
        return {
            'support_levels': support_prices,
            'resistance_levels': resistance_prices
        }
    
    def get_strong_levels_only(self, sr_results: Dict[str, List[SRLevel]]) -> Dict[str, List[float]]:
        """
        Get only strong S/R levels (touched 3+ times) as specified in requirements
        """
        strong_support = [
            level.price for level in sr_results['support_levels'] 
            if level.strength >= 3
        ]
        strong_resistance = [
            level.price for level in sr_results['resistance_levels'] 
            if level.strength >= 3
        ]
        
        return {
            'support_levels': strong_support,
            'resistance_levels': strong_resistance
        }
