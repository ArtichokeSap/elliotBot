"""
Enhanced Elliott Wave Detector with Comprehensive Validation
Detects ALL wave types and validates internal structures
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from scipy.signal import argrelextrema
import logging

from .comprehensive_elliott_validator import (
    ComprehensiveElliottValidator, 
    WavePoint, 
    WaveStructure,
    WaveType,
    WaveDirection
)

logger = logging.getLogger(__name__)


class EnhancedWaveDetector:
    """
    Enhanced Elliott Wave Detector that finds and validates complete wave structures
    """
    
    def __init__(self, min_wave_size: float = 0.02, lookback_periods: int = 5):
        """
        Initialize enhanced wave detector
        
        Args:
            min_wave_size: Minimum wave size as percentage of price range
            lookback_periods: Periods to look back for extrema detection
        """
        self.min_wave_size = min_wave_size
        self.lookback_periods = lookback_periods
        self.validator = ComprehensiveElliottValidator()
        
    def detect_elliott_waves(self, price_data: pd.DataFrame, 
                           symbol: str = "Unknown") -> Dict[str, Any]:
        """
        Detect and validate complete Elliott Wave structures
        """
        try:
            logger.info(f"Starting Elliott Wave detection for {symbol}")
            
            # Find significant price extremes
            extremes = self._find_significant_extremes(price_data)
            
            if len(extremes) < 4:
                logger.warning(f"Insufficient extremes found: {len(extremes)}")
                return self._create_empty_result("Insufficient price extremes for wave analysis")
            
            # Try to identify wave structures
            wave_structures = []
            
            # Look for 5-wave impulse patterns
            impulse_structures = self._find_impulse_patterns(extremes, price_data)
            wave_structures.extend(impulse_structures)
            
            # Look for 3-wave corrective patterns
            corrective_structures = self._find_corrective_patterns(extremes, price_data)
            wave_structures.extend(corrective_structures)
            
            if not wave_structures:
                logger.warning("No valid wave structures detected")
                return self._create_empty_result("No valid Elliott Wave patterns detected")
            
            # Select best structure based on validation score
            best_structure = max(wave_structures, key=lambda x: x.validation_score)
            
            # Generate analysis results
            return self._create_analysis_result(best_structure, price_data, symbol)
            
        except Exception as e:
            logger.error(f"Error in Elliott Wave detection: {e}")
            return self._create_empty_result(f"Detection error: {e}")
    
    def _find_significant_extremes(self, price_data: pd.DataFrame) -> List[WavePoint]:
        """
        Find significant price extremes that could be Elliott Wave points
        """
        highs = price_data['high'].values
        lows = price_data['low'].values
        times = price_data.index
        
        # Find local maxima and minima
        high_indices = argrelextrema(highs, np.greater, order=self.lookback_periods)[0]
        low_indices = argrelextrema(lows, np.less, order=self.lookback_periods)[0]
        
        # Combine and sort by time
        all_extremes = []
        
        for idx in high_indices:
            if self._is_significant_extreme(idx, highs, 'high'):
                all_extremes.append(WavePoint(idx, highs[idx], times[idx]))
        
        for idx in low_indices:
            if self._is_significant_extreme(idx, lows, 'low'):
                all_extremes.append(WavePoint(idx, lows[idx], times[idx]))
        
        # Sort by index (time)
        all_extremes.sort(key=lambda x: x.index)
        
        # Filter out extremes that are too close together
        filtered_extremes = self._filter_close_extremes(all_extremes, price_data)
        
        logger.info(f"Found {len(filtered_extremes)} significant extremes")
        return filtered_extremes
    
    def _is_significant_extreme(self, index: int, prices: np.ndarray, 
                              extreme_type: str) -> bool:
        """
        Check if an extreme is significant enough to be a wave point
        """
        if index < self.lookback_periods or index >= len(prices) - self.lookback_periods:
            return False
        
        price_range = np.max(prices) - np.min(prices)
        min_move = price_range * self.min_wave_size
        
        if extreme_type == 'high':
            # Check if this high is significantly higher than surrounding lows
            surrounding_lows = prices[index-self.lookback_periods:index+self.lookback_periods+1]
            min_surrounding = np.min(surrounding_lows)
            return (prices[index] - min_surrounding) >= min_move
        else:
            # Check if this low is significantly lower than surrounding highs
            surrounding_highs = prices[index-self.lookback_periods:index+self.lookback_periods+1]
            max_surrounding = np.max(surrounding_highs)
            return (max_surrounding - prices[index]) >= min_move
    
    def _filter_close_extremes(self, extremes: List[WavePoint], 
                             price_data: pd.DataFrame) -> List[WavePoint]:
        """
        Filter out extremes that are too close together
        """
        if len(extremes) < 2:
            return extremes
        
        filtered = [extremes[0]]
        
        for i in range(1, len(extremes)):
            current = extremes[i]
            last = filtered[-1]
            
            # Check if alternating high/low pattern
            current_is_high = self._is_local_high(current, price_data)
            last_is_high = self._is_local_high(last, price_data)
            
            # Keep if alternating or significant time/price distance
            time_distance = current.index - last.index
            price_distance = abs(current.price - last.price)
            
            if (current_is_high != last_is_high or 
                time_distance >= 5 or 
                price_distance >= np.std(price_data['close']) * 2):
                filtered.append(current)
        
        return filtered
    
    def _is_local_high(self, point: WavePoint, price_data: pd.DataFrame) -> bool:
        """
        Check if a point is a local high
        """
        idx = point.index
        if idx == 0 or idx >= len(price_data) - 1:
            return False
        
        return (price_data.iloc[idx]['high'] > price_data.iloc[idx-1]['high'] and
                price_data.iloc[idx]['high'] > price_data.iloc[idx+1]['high'])
    
    def _find_impulse_patterns(self, extremes: List[WavePoint], 
                             price_data: pd.DataFrame) -> List[WaveStructure]:
        """
        Find 5-wave impulse patterns from extremes
        """
        structures = []
        
        # Need at least 6 points for 5-wave structure
        if len(extremes) < 6:
            logger.warning(f"Not enough extremes for impulse pattern: {len(extremes)}")
            return structures
        
        # Try different 6-point combinations
        for i in range(len(extremes) - 5):
            wave_points = extremes[i:i+6]
            
            # Check if pattern alternates correctly (up-down-up-down-up or vice versa)
            if self._is_valid_impulse_pattern(wave_points, price_data):
                try:
                    logger.info(f"Attempting to validate impulse pattern starting at index {i}")
                    structure = self.validator.validate_complete_structure(wave_points, price_data)
                    logger.info(f"Impulse validation score: {structure.validation_score:.2f}")
                    
                    if structure.validation_score > 0.2:  # Lower threshold for testing
                        structures.append(structure)
                        logger.info(f"Found impulse pattern with score: {structure.validation_score:.2f}")
                except Exception as e:
                    logger.warning(f"Error validating impulse pattern: {e}")
        
        return structures
    
    def _find_corrective_patterns(self, extremes: List[WavePoint], 
                                price_data: pd.DataFrame) -> List[WaveStructure]:
        """
        Find 3-wave corrective patterns from extremes
        """
        structures = []
        
        # Need at least 4 points for 3-wave structure
        if len(extremes) < 4:
            logger.warning(f"Not enough extremes for corrective pattern: {len(extremes)}")
            return structures
        
        # Try different 4-point combinations
        for i in range(len(extremes) - 3):
            wave_points = extremes[i:i+4]
            
            # Check if pattern alternates correctly
            if self._is_valid_corrective_pattern(wave_points, price_data):
                try:
                    logger.info(f"Attempting to validate corrective pattern starting at index {i}")
                    structure = self.validator.validate_complete_structure(wave_points, price_data)
                    logger.info(f"Corrective validation score: {structure.validation_score:.2f}")
                    
                    if structure.validation_score > 0.2:  # Lower threshold for testing
                        structures.append(structure)
                        logger.info(f"Found corrective pattern with score: {structure.validation_score:.2f}")
                except Exception as e:
                    logger.warning(f"Error validating corrective pattern: {e}")
        
        return structures
    
    def _is_valid_impulse_pattern(self, points: List[WavePoint], 
                                price_data: pd.DataFrame) -> bool:
        """
        Check if 6 points form a valid impulse pattern
        """
        if len(points) != 6:
            return False
        
        # Check alternating high-low pattern
        is_highs = [self._is_local_high(point, price_data) for point in points]
        
        # Should be either L-H-L-H-L-H or H-L-H-L-H-L
        pattern1 = [False, True, False, True, False, True]   # Bullish
        pattern2 = [True, False, True, False, True, False]   # Bearish
        
        return is_highs == pattern1 or is_highs == pattern2
    
    def _is_valid_corrective_pattern(self, points: List[WavePoint], 
                                   price_data: pd.DataFrame) -> bool:
        """
        Check if 4 points form a valid corrective pattern
        """
        if len(points) != 4:
            return False
        
        # Check alternating high-low pattern
        is_highs = [self._is_local_high(point, price_data) for point in points]
        
        # Should be either L-H-L-H or H-L-H-L
        pattern1 = [False, True, False, True]   # Up correction
        pattern2 = [True, False, True, False]   # Down correction
        
        return is_highs == pattern1 or is_highs == pattern2
    
    def _create_analysis_result(self, structure: WaveStructure, 
                              price_data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Create comprehensive analysis result
        """
        # Convert waves to standard format
        waves = []
        for wave in structure.main_waves:
            waves.append({
                'wave': wave.label,
                'start_price': wave.start.price,
                'end_price': wave.end.price,
                'start_time': wave.start.time,
                'end_time': wave.end.time,
                'direction': wave.direction.value,
                'wave_type': wave.wave_type.value,
                'length': wave.length,
                'duration': wave.duration,
                'confidence': wave.validation_score
            })
        
        # Add subwave information
        subwave_data = {}
        for parent_label, subwaves in structure.subwaves.items():
            subwave_data[parent_label] = []
            for subwave in subwaves:
                subwave_data[parent_label].append({
                    'label': subwave.label,
                    'start_price': subwave.start.price,
                    'end_price': subwave.end.price,
                    'wave_type': subwave.wave_type.value,
                    'direction': subwave.direction.value,
                    'confidence': subwave.validation_score
                })
        
        # Create comprehensive result
        result = {
            'symbol': symbol,
            'wave_structure': structure.wave_type.value,
            'direction': structure.direction.value,
            'validation_score': structure.validation_score,
            'waves': waves,
            'subwaves': subwave_data,
            'fibonacci_levels': structure.fibonacci_levels,
            'rule_compliance': structure.rule_compliance,
            'recommendations': structure.recommendations,
            'issues': structure.issues,
            'summary': self._generate_summary(structure),
            'detailed_report': self.validator.generate_detailed_report(structure)
        }
        
        return result
    
    def _generate_summary(self, structure: WaveStructure) -> str:
        """
        Generate concise summary of the analysis
        """
        score_pct = structure.validation_score * 100
        wave_count = len(structure.main_waves)
        
        if structure.wave_type == WaveType.IMPULSE:
            pattern_desc = f"{wave_count}-wave impulse"
        else:
            pattern_desc = f"{wave_count}-wave {structure.wave_type.value}"
        
        direction_desc = "bullish" if structure.direction == WaveDirection.BULLISH else "bearish"
        
        if score_pct >= 80:
            quality = "excellent"
        elif score_pct >= 60:
            quality = "good"
        else:
            quality = "questionable"
        
        summary = f"Detected {quality} {direction_desc} {pattern_desc} structure "
        summary += f"with {score_pct:.0f}% validation confidence. "
        
        if structure.issues:
            summary += f"Note: {len(structure.issues)} validation issues identified."
        
        return summary
    
    def _create_empty_result(self, message: str) -> Dict[str, Any]:
        """
        Create empty result when no waves are detected
        """
        return {
            'symbol': 'Unknown',
            'wave_structure': 'none',
            'direction': 'unknown',
            'validation_score': 0.0,
            'waves': [],
            'subwaves': {},
            'fibonacci_levels': {},
            'rule_compliance': {},
            'recommendations': ['Insufficient price data or no clear Elliott Wave patterns'],
            'issues': [message],
            'summary': f"No Elliott Wave patterns detected: {message}",
            'detailed_report': f"Elliott Wave Analysis Failed\n{'-' * 40}\n{message}"
        }
