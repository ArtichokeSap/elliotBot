"""
Elliott Wave detection and labeling module.
Implements pattern recognition for identifying Elliott Wave structures.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass
import logging
from scipy.signal import find_peaks

from ..utils.logger import get_logger
from ..utils.config import get_config
from ..utils.helpers import calculate_fibonacci_levels, find_peaks_and_troughs

logger = get_logger(__name__)


class WaveType(Enum):
    """Elliott Wave types with enhanced labeling support."""
    # Primary degree waves (main trend)
    IMPULSE_1 = "1"
    IMPULSE_2 = "2"
    IMPULSE_3 = "3"
    IMPULSE_4 = "4"
    IMPULSE_5 = "5"
    
    # Corrective waves
    CORRECTIVE_A = "A"
    CORRECTIVE_B = "B"
    CORRECTIVE_C = "C"
    
    # Intermediate degree (sub-waves)
    MINOR_1 = "(1)"
    MINOR_2 = "(2)"
    MINOR_3 = "(3)"
    MINOR_4 = "(4)"
    MINOR_5 = "(5)"
    
    MINOR_A = "(A)"
    MINOR_B = "(B)"
    MINOR_C = "(C)"
    
    # Minute degree (sub-sub-waves)
    MINUTE_1 = "(i)"
    MINUTE_2 = "(ii)"
    MINUTE_3 = "(iii)"
    MINUTE_4 = "(iv)"
    MINUTE_5 = "(v)"
    
    MINUTE_A = "(a)"
    MINUTE_B = "(b)"
    MINUTE_C = "(c)"
    
    UNKNOWN = "?"


class WaveDegree(Enum):
    """Elliott Wave degrees for proper hierarchy."""
    SUPERCYCLE = "SUPERCYCLE"
    CYCLE = "CYCLE"
    PRIMARY = "PRIMARY"
    INTERMEDIATE = "INTERMEDIATE"
    MINOR = "MINOR"
    MINUTE = "MINUTE"
    MINUETTE = "MINUETTE"


class TrendDirection(Enum):
    """Trend direction."""
    UP = 1
    DOWN = -1
    SIDEWAYS = 0

# Alias for backward compatibility
Direction = TrendDirection


@dataclass
class WavePoint:
    """Represents a point in Elliott Wave analysis."""
    timestamp: pd.Timestamp
    price: float
    index: int
    wave_type: WaveType
    confidence: float = 0.0
    degree: WaveDegree = WaveDegree.PRIMARY
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = pd.Timestamp(self.timestamp)


@dataclass
class Wave:
    """Represents an Elliott Wave with enhanced properties."""
    start_point: WavePoint
    end_point: WavePoint
    wave_type: WaveType
    direction: TrendDirection
    confidence: float
    degree: WaveDegree = WaveDegree.PRIMARY
    fibonacci_ratios: Dict[str, float] = None
    invalidation_level: Optional[float] = None
    target_levels: List[float] = None
    
    def __post_init__(self):
        if self.fibonacci_ratios is None:
            self.fibonacci_ratios = {}
        if self.target_levels is None:
            self.target_levels = []
    
    @property
    def duration(self) -> int:
        """Get wave duration in periods."""
        return self.end_point.index - self.start_point.index
    
    @property
    def price_change(self) -> float:
        """Get wave price change."""
        return self.end_point.price - self.start_point.price
    
    @property
    def price_change_pct(self) -> float:
        """Get wave price change percentage."""
        return (self.end_point.price - self.start_point.price) / self.start_point.price


class WaveDetector:
    """
    Main Elliott Wave detection and analysis class.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize WaveDetector.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.zigzag_threshold = self.config.get('wave_detection.zigzag_threshold', 0.05)
        self.min_wave_length = self.config.get('wave_detection.min_wave_length', 5)
        self.fibonacci_levels = self.config.get('wave_detection.fibonacci_levels', 
                                               [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618])
        self.confidence_threshold = self.config.get('wave_detection.confidence_threshold', 0.7)
        
        logger.info(f"WaveDetector initialized with threshold: {self.zigzag_threshold}")
    
    def determine_wave_degrees(self, waves: List[Wave], data: pd.DataFrame) -> List[Wave]:
        """
        Determine appropriate wave degrees based on timeframe and wave magnitude
        
        Args:
            waves: List of detected waves
            data: OHLCV DataFrame
            
        Returns:
            List of waves with appropriate degrees assigned
        """
        if not waves:
            return waves
        
        try:
            # Calculate timeframe (days between data points)
            if len(data) > 1:
                time_diff = (data.index[1] - data.index[0]).days
                if time_diff == 0:  # Intraday data
                    time_diff = (data.index[1] - data.index[0]).seconds / 3600 / 24  # Convert to days
            else:
                time_diff = 1
            
            # Calculate average wave duration and magnitude
            durations = [wave.duration for wave in waves]
            magnitudes = [abs(wave.price_change_pct) for wave in waves]
            
            avg_duration = np.mean(durations) if durations else 10
            avg_magnitude = np.mean(magnitudes) if magnitudes else 0.05
            
            # Assign degrees based on duration and magnitude
            for wave in waves:
                wave_duration_days = wave.duration * time_diff
                wave_magnitude = abs(wave.price_change_pct)
                
                # Determine degree based on duration and magnitude
                if wave_duration_days > 365 or wave_magnitude > 1.0:  # > 1 year or > 100% move
                    wave.degree = WaveDegree.CYCLE
                elif wave_duration_days > 90 or wave_magnitude > 0.3:   # > 3 months or > 30% move
                    wave.degree = WaveDegree.PRIMARY
                elif wave_duration_days > 21 or wave_magnitude > 0.1:   # > 3 weeks or > 10% move
                    wave.degree = WaveDegree.INTERMEDIATE
                elif wave_duration_days > 5 or wave_magnitude > 0.05:   # > 1 week or > 5% move
                    wave.degree = WaveDegree.MINOR
                else:
                    wave.degree = WaveDegree.MINUTE
            
            return waves
            
        except Exception as e:
            logger.warning(f"Error determining wave degrees: {e}")
            # Default to PRIMARY degree
            for wave in waves:
                wave.degree = WaveDegree.PRIMARY
            return waves
    
    def detect_waves(self, data: pd.DataFrame) -> List[Wave]:
        """
        Detect Elliott Waves in price data.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            List of detected waves
        """
        try:
            # Step 1: Identify swing points using ZigZag
            swing_points = self._get_swing_points(data)
            
            if len(swing_points) < 5:
                logger.warning("Insufficient swing points for wave analysis")
                return []
            
            # Step 2: Analyze wave patterns
            waves = self._analyze_wave_patterns(swing_points, data)
            
            # Step 3: Validate and score waves
            validated_waves = self._validate_waves(waves, data)
            
            # Step 4: Determine appropriate wave degrees
            final_waves = self.determine_wave_degrees(validated_waves, data)
            
            logger.info(f"Detected {len(final_waves)} valid Elliott Waves")
            return final_waves
            
        except Exception as e:
            logger.error(f"Error detecting waves: {e}")
            return []
    
    def detect_multi_degree_waves(self, data: pd.DataFrame, max_degree: int = 3) -> List[Wave]:
        """
        Detect waves at multiple degrees recursively.
        Args:
            data: OHLCV DataFrame
            max_degree: How many degrees deep to analyze (e.g., 3 = PRIMARY, INTERMEDIATE, MINOR)
        Returns:
            List of all detected waves with degree assigned
        """
        all_waves = []
        degree_order = [WaveDegree.PRIMARY, WaveDegree.INTERMEDIATE, WaveDegree.MINOR, WaveDegree.MINUTE, WaveDegree.MINUETTE]
        def _recursive_detect(data, current_degree_idx):
            if current_degree_idx >= max_degree:
                return []
            degree = degree_order[current_degree_idx]
            waves = self.detect_waves(data)
            for wave in waves:
                wave.degree = degree
            all_waves.extend(waves)
            # Recursively detect subwaves within each wave
            for wave in waves:
                sub_data = data.iloc[wave.start_point.index:wave.end_point.index+1]
                _recursive_detect(sub_data, current_degree_idx + 1)
        _recursive_detect(data, 0)
        return all_waves

    def validate_time_symmetry(self, waves: List[Wave]) -> List[Dict[str, float]]:
        """
        Validate time symmetry (duration ratios) for impulse and corrective patterns.
        Returns a list of dicts with ratio info for each pattern found.
        """
        results = []
        # Impulse: check 1:3:5 and 2:4 ratios
        impulse_waves = [w for w in waves if w.wave_type in [WaveType.IMPULSE_1, WaveType.IMPULSE_2, WaveType.IMPULSE_3, WaveType.IMPULSE_4, WaveType.IMPULSE_5]]
        if len(impulse_waves) == 5:
            durs = [w.duration for w in impulse_waves]
            ratios = {
                '1:3': durs[0]/durs[2] if durs[2] else None,
                '3:5': durs[2]/durs[4] if durs[4] else None,
                '2:4': durs[1]/durs[3] if durs[3] else None,
            }
            results.append({'pattern': 'impulse', 'ratios': ratios})
        # Corrective: check A:B:C ratios
        corr_waves = [w for w in waves if w.wave_type in [WaveType.CORRECTIVE_A, WaveType.CORRECTIVE_B, WaveType.CORRECTIVE_C]]
        if len(corr_waves) == 3:
            durs = [w.duration for w in corr_waves]
            ratios = {
                'A:B': durs[0]/durs[1] if durs[1] else None,
                'B:C': durs[1]/durs[2] if durs[2] else None,
            }
            results.append({'pattern': 'corrective', 'ratios': ratios})
        return results
    
    def _get_swing_points(self, data: pd.DataFrame) -> List[WavePoint]:
        """
        Identify swing points using ZigZag indicator.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            List of swing points
        """
        from ..data.indicators import TechnicalIndicators
        
        # Calculate ZigZag
        zigzag, direction = TechnicalIndicators.zigzag(data, self.zigzag_threshold)
        
        swing_points = []
        
        for timestamp, price in zigzag.dropna().items():
            if pd.isna(price):
                continue
                
            index = data.index.get_loc(timestamp)
            wave_direction = direction.loc[timestamp]
            
            # Create wave point
            point = WavePoint(
                timestamp=timestamp,
                price=price,
                index=index,
                wave_type=WaveType.UNKNOWN,
                confidence=1.0  # Initial confidence
            )
            
            swing_points.append(point)
        
        # Sort by timestamp
        swing_points.sort(key=lambda x: x.timestamp)
        
        logger.debug(f"Found {len(swing_points)} swing points")
        return swing_points
    
    def _analyze_wave_patterns(self, swing_points: List[WavePoint], data: pd.DataFrame) -> List[Wave]:
        """
        Analyze swing points to identify Elliott Wave patterns.
        
        Args:
            swing_points: List of swing points
            data: OHLCV DataFrame
            
        Returns:
            List of identified waves
        """
        waves = []
        
        # Look for 5-wave impulse patterns
        impulse_waves = self._find_impulse_patterns(swing_points, data)
        waves.extend(impulse_waves)
        
        # Look for 3-wave corrective patterns
        corrective_waves = self._find_corrective_patterns(swing_points, data)
        waves.extend(corrective_waves)
        
        return waves
    
    def _find_impulse_patterns(self, swing_points: List[WavePoint], data: pd.DataFrame) -> List[Wave]:
        """
        Find 5-wave impulse patterns (1-2-3-4-5).
        
        Args:
            swing_points: List of swing points
            data: OHLCV DataFrame
            
        Returns:
            List of impulse waves
        """
        impulse_waves = []
        
        # Need at least 6 points for a complete 5-wave pattern
        if len(swing_points) < 6:
            return impulse_waves
        
        # Sliding window to find 5-wave patterns
        for i in range(len(swing_points) - 5):
            pattern_points = swing_points[i:i+6]  # 6 points define 5 waves
            
            # Check if this could be an impulse pattern
            if self._validate_impulse_pattern(pattern_points, data):
                # Create individual waves
                wave_types = [WaveType.IMPULSE_1, WaveType.IMPULSE_2, WaveType.IMPULSE_3, 
                             WaveType.IMPULSE_4, WaveType.IMPULSE_5]
                
                for j in range(5):
                    start_point = pattern_points[j]
                    end_point = pattern_points[j+1]
                    
                    # Determine direction
                    direction = TrendDirection.UP if end_point.price > start_point.price else TrendDirection.DOWN
                    
                    # Calculate confidence
                    confidence = self._calculate_wave_confidence(start_point, end_point, wave_types[j], data)
                    
                    # Create wave
                    wave = Wave(
                        start_point=start_point,
                        end_point=end_point,
                        wave_type=wave_types[j],
                        direction=direction,
                        confidence=confidence
                    )
                    
                    # Add Fibonacci analysis
                    wave.fibonacci_ratios = self._calculate_fibonacci_ratios(wave, pattern_points)
                    
                    # Calculate invalidation level
                    wave.invalidation_level = self._calculate_invalidation_level(wave, pattern_points, j)
                    
                    impulse_waves.append(wave)
        
        return impulse_waves
    
    def _find_corrective_patterns(self, swing_points: List[WavePoint], data: pd.DataFrame) -> List[Wave]:
        """
        Find 3-wave corrective patterns (A-B-C).
        
        Args:
            swing_points: List of swing points
            data: OHLCV DataFrame
            
        Returns:
            List of corrective waves
        """
        corrective_waves = []
        
        # Need at least 4 points for a complete 3-wave pattern
        if len(swing_points) < 4:
            return corrective_waves
        
        # Sliding window to find 3-wave patterns
        for i in range(len(swing_points) - 3):
            pattern_points = swing_points[i:i+4]  # 4 points define 3 waves
            
            # Check if this could be a corrective pattern
            if self._validate_corrective_pattern(pattern_points, data):
                # Create individual waves
                wave_types = [WaveType.CORRECTIVE_A, WaveType.CORRECTIVE_B, WaveType.CORRECTIVE_C]
                
                for j in range(3):
                    start_point = pattern_points[j]
                    end_point = pattern_points[j+1]
                    
                    # Determine direction
                    direction = TrendDirection.UP if end_point.price > start_point.price else TrendDirection.DOWN
                    
                    # Calculate confidence
                    confidence = self._calculate_wave_confidence(start_point, end_point, wave_types[j], data)
                    
                    # Create wave
                    wave = Wave(
                        start_point=start_point,
                        end_point=end_point,
                        wave_type=wave_types[j],
                        direction=direction,
                        confidence=confidence
                    )
                    
                    # Add Fibonacci analysis
                    wave.fibonacci_ratios = self._calculate_fibonacci_ratios(wave, pattern_points)
                    
                    # Calculate invalidation level (simpler for corrective waves)
                    if wave_types[j] == WaveType.CORRECTIVE_A and j > 0:
                        # A wave invalidation is typically the start of the preceding trend
                        wave.invalidation_level = pattern_points[0].price
                    
                    corrective_waves.append(wave)
        
        return corrective_waves
    
    def _validate_impulse_pattern(self, points: List[WavePoint], data: pd.DataFrame) -> bool:
        """
        Validate if points form a valid 5-wave impulse pattern using strict Elliott Wave rules.
        
        Args:
            points: List of 6 points defining 5 waves
            data: OHLCV DataFrame
            
        Returns:
            True if valid impulse pattern
        """
        if len(points) != 6:
            return False
        
        try:
            # Import validator here to avoid circular imports
            from .elliott_wave_validator import ElliottWaveValidator
            
            # Create temporary waves for validation
            temp_waves = []
            wave_types = [WaveType.IMPULSE_1, WaveType.IMPULSE_2, WaveType.IMPULSE_3, 
                         WaveType.IMPULSE_4, WaveType.IMPULSE_5]
            
            for i in range(5):
                start_point = points[i]
                end_point = points[i+1]
                direction = TrendDirection.UP if end_point.price > start_point.price else TrendDirection.DOWN
                
                temp_wave = Wave(
                    start_point=start_point,
                    end_point=end_point,
                    wave_type=wave_types[i],
                    direction=direction,
                    confidence=0.5  # Temporary confidence
                )
                temp_waves.append(temp_wave)
            
            # Use strict validator
            validator = ElliottWaveValidator()
            structure = validator.validate_impulse_structure(temp_waves, data)
            
            # Consider valid if score > 0.6 and no INVALID violations
            has_invalid_violations = any(v.severity.value == "INVALID" for v in structure.violations)
            
            return structure.validation_score > 0.6 and not has_invalid_violations
            
        except Exception as e:
            logger.debug(f"Error in strict impulse validation: {e}")
            # Fallback to original validation logic
            return self._validate_impulse_pattern_legacy(points, data)
    
    def _validate_impulse_pattern_legacy(self, points: List[WavePoint], data: pd.DataFrame) -> bool:
        """
        Legacy validation method (kept as fallback).
        """
        if len(points) != 6:
            return False
        
        # Elliott Wave rules for impulse patterns:
        # 1. Wave 2 cannot retrace more than 100% of wave 1
        # 2. Wave 3 cannot be the shortest wave among waves 1, 3, and 5
        # 3. Wave 4 cannot overlap with wave 1 (except in diagonal patterns)
        
        try:
            # Calculate wave lengths
            wave1_len = abs(points[1].price - points[0].price)
            wave2_len = abs(points[2].price - points[1].price)
            wave3_len = abs(points[3].price - points[2].price)
            wave4_len = abs(points[4].price - points[3].price)
            wave5_len = abs(points[5].price - points[4].price)
            
            # Rule 1: Wave 2 retracement
            wave2_retracement = wave2_len / wave1_len
            if wave2_retracement > 1.0:
                return False
            
            # Rule 2: Wave 3 length
            if wave3_len <= min(wave1_len, wave5_len):
                return False
            
            # Rule 3: Wave 4 overlap (simplified check)
            if points[0].price < points[4].price < points[1].price or points[1].price < points[4].price < points[0].price:
                return False
            
            # Additional checks
            # Minimum wave length
            for i in range(1, 6):
                wave_duration = points[i].index - points[i-1].index
                if wave_duration < self.min_wave_length:
                    return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error validating impulse pattern: {e}")
            return False
    
    def _validate_corrective_pattern(self, points: List[WavePoint], data: pd.DataFrame) -> bool:
        """
        Validate if points form a valid 3-wave corrective pattern using strict rules.
        
        Args:
            points: List of 4 points defining 3 waves
            data: OHLCV DataFrame
            
        Returns:
            True if valid corrective pattern
        """
        if len(points) != 4:
            return False
        
        try:
            # Import validator here to avoid circular imports
            from .elliott_wave_validator import ElliottWaveValidator
            
            # Create temporary waves for validation
            temp_waves = []
            wave_types = [WaveType.CORRECTIVE_A, WaveType.CORRECTIVE_B, WaveType.CORRECTIVE_C]
            
            for i in range(3):
                start_point = points[i]
                end_point = points[i+1]
                direction = TrendDirection.UP if end_point.price > start_point.price else TrendDirection.DOWN
                
                temp_wave = Wave(
                    start_point=start_point,
                    end_point=end_point,
                    wave_type=wave_types[i],
                    direction=direction,
                    confidence=0.5  # Temporary confidence
                )
                temp_waves.append(temp_wave)
            
            # Use strict validator
            validator = ElliottWaveValidator()
            structure = validator.validate_corrective_structure(temp_waves, data)
            
            # Consider valid if score > 0.5 and no INVALID violations
            has_invalid_violations = any(v.severity.value == "INVALID" for v in structure.violations)
            
            return structure.validation_score > 0.5 and not has_invalid_violations
            
        except Exception as e:
            logger.debug(f"Error in strict corrective validation: {e}")
            # Fallback to original validation logic
            return self._validate_corrective_pattern_legacy(points, data)
    
    def _validate_corrective_pattern_legacy(self, points: List[WavePoint], data: pd.DataFrame) -> bool:
        """
        Legacy corrective pattern validation (kept as fallback).
        """
        if len(points) != 4:
            return False
        
        try:
            # Basic validation for corrective patterns
            # Minimum wave length
            for i in range(1, 4):
                wave_duration = points[i].index - points[i-1].index
                if wave_duration < self.min_wave_length:
                    return False
            
            # Wave C should typically be related to wave A by Fibonacci ratios
            wave_a_len = abs(points[1].price - points[0].price)
            wave_c_len = abs(points[3].price - points[2].price)
            
            if wave_a_len > 0:
                c_to_a_ratio = wave_c_len / wave_a_len
                # Common Fibonacci ratios for wave C
                valid_ratios = [0.618, 1.0, 1.618, 2.618]
                tolerance = 0.1
                
                ratio_valid = any(abs(c_to_a_ratio - ratio) / ratio <= tolerance for ratio in valid_ratios)
                if not ratio_valid:
                    return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error validating corrective pattern: {e}")
            return False
    
    def _calculate_invalidation_level(self, wave: Wave, pattern_points: List[WavePoint], wave_index: int) -> Optional[float]:
        """
        Calculate invalidation level for Elliott Wave rules
        
        Args:
            wave: Current wave
            pattern_points: All points in the pattern
            wave_index: Index of current wave in pattern
            
        Returns:
            Invalidation price level or None
        """
        try:
            if wave.wave_type == WaveType.IMPULSE_2 and wave_index >= 1:
                # Wave 2 cannot retrace below the start of wave 1
                return pattern_points[0].price
            
            elif wave.wave_type == WaveType.IMPULSE_4 and wave_index >= 3:
                # Wave 4 cannot overlap with wave 1 territory (in most patterns)
                return pattern_points[1].price  # End of wave 1
            
            elif wave.wave_type == WaveType.IMPULSE_5 and wave_index >= 4:
                # Wave 5 should not fail to exceed wave 3 end
                return pattern_points[3].price  # End of wave 3
            
            elif wave.wave_type in [WaveType.CORRECTIVE_A, WaveType.CORRECTIVE_C]:
                # Corrective waves typically have invalidation at trend start
                return pattern_points[0].price
            
            return None
            
        except Exception as e:
            logger.debug(f"Error calculating invalidation level: {e}")
            return None
    
    def _calculate_wave_confidence(
        self, 
        start_point: WavePoint, 
        end_point: WavePoint, 
        wave_type: WaveType, 
        data: pd.DataFrame
    ) -> float:
        """
        Calculate confidence score for a wave.
        
        Args:
            start_point: Wave start point
            end_point: Wave end point
            wave_type: Type of wave
            data: OHLCV DataFrame
            
        Returns:
            Confidence score (0-1)
        """
        confidence_factors = []
        
        try:
            # Factor 1: Wave length relative to threshold
            price_change_pct = abs(end_point.price - start_point.price) / start_point.price
            length_factor = min(price_change_pct / self.zigzag_threshold, 1.0)
            confidence_factors.append(length_factor)
            
            # Factor 2: Volume confirmation
            wave_data = data.iloc[start_point.index:end_point.index+1]
            if len(wave_data) > 1:
                avg_volume = wave_data['volume'].mean()
                baseline_volume = data['volume'].rolling(20).mean().iloc[end_point.index]
                volume_factor = min(avg_volume / baseline_volume, 1.5) / 1.5
                confidence_factors.append(volume_factor)
            
            # Factor 3: Wave type specific rules
            type_factor = self._get_wave_type_confidence(wave_type, start_point, end_point)
            confidence_factors.append(type_factor)
            
            # Factor 4: Duration factor
            duration = end_point.index - start_point.index
            duration_factor = min(duration / (self.min_wave_length * 2), 1.0)
            confidence_factors.append(duration_factor)
            
            # Calculate weighted average
            weights = [0.3, 0.25, 0.25, 0.2]  # Adjust weights as needed
            confidence = sum(factor * weight for factor, weight in zip(confidence_factors, weights))
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.debug(f"Error calculating wave confidence: {e}")
            return 0.5  # Default confidence
    
    def _get_wave_type_confidence(self, wave_type: WaveType, start_point: WavePoint, end_point: WavePoint) -> float:
        """
        Get confidence factor based on wave type specific characteristics.
        
        Args:
            wave_type: Type of wave
            start_point: Wave start point
            end_point: Wave end point
            
        Returns:
            Type-specific confidence factor
        """
        # Simplified type-specific confidence
        if wave_type in [WaveType.IMPULSE_1, WaveType.IMPULSE_3, WaveType.IMPULSE_5]:
            # Impulse waves should be strong moves
            return 0.8
        elif wave_type in [WaveType.IMPULSE_2, WaveType.IMPULSE_4]:
            # Corrective waves within impulse should be smaller
            return 0.7
        elif wave_type in [WaveType.CORRECTIVE_A, WaveType.CORRECTIVE_C]:
            # Corrective waves
            return 0.6
        elif wave_type == WaveType.CORRECTIVE_B:
            # B waves are often irregular
            return 0.5
        else:
            return 0.5
    
    def _calculate_fibonacci_ratios(self, wave: Wave, pattern_points: List[WavePoint]) -> Dict[str, float]:
        """
        Calculate Fibonacci ratios for the wave relative to the pattern.
        
        Args:
            wave: Wave to analyze
            pattern_points: Points defining the pattern
            
        Returns:
            Dictionary of Fibonacci ratios
        """
        ratios = {}
        
        try:
            if len(pattern_points) >= 3:
                # Calculate retracement levels
                if wave.wave_type in [WaveType.IMPULSE_2, WaveType.IMPULSE_4, WaveType.CORRECTIVE_B]:
                    # For retracement waves
                    previous_wave_start = pattern_points[0].price
                    previous_wave_end = pattern_points[1].price
                    current_level = wave.end_point.price
                    
                    if previous_wave_end != previous_wave_start:
                        retracement = abs(current_level - previous_wave_end) / abs(previous_wave_end - previous_wave_start)
                        ratios['retracement'] = retracement
                        
                        # Find closest Fibonacci level
                        for fib_level in self.fibonacci_levels:
                            if abs(retracement - fib_level) < 0.05:  # 5% tolerance
                                ratios['fibonacci_match'] = fib_level
                                break
                
                # Calculate extension levels for impulse waves
                elif wave.wave_type in [WaveType.IMPULSE_3, WaveType.IMPULSE_5]:
                    if len(pattern_points) >= 4:
                        wave1_length = abs(pattern_points[1].price - pattern_points[0].price)
                        current_wave_length = abs(wave.end_point.price - wave.start_point.price)
                        
                        if wave1_length > 0:
                            extension_ratio = current_wave_length / wave1_length
                            ratios['extension'] = extension_ratio
                            
                            # Find closest Fibonacci extension
                            extension_levels = [1.0, 1.618, 2.618, 4.236]
                            for ext_level in extension_levels:
                                if abs(extension_ratio - ext_level) < 0.1:  # 10% tolerance
                                    ratios['fibonacci_extension'] = ext_level
                                    break
        
        except Exception as e:
            logger.debug(f"Error calculating Fibonacci ratios: {e}")
        
        return ratios
    
    def _validate_waves(self, waves: List[Wave], data: pd.DataFrame) -> List[Wave]:
        """
        Validate and filter waves based on confidence and rules.
        
        Args:
            waves: List of detected waves
            data: OHLCV DataFrame
            
        Returns:
            List of validated waves
        """
        validated_waves = []
        
        for wave in waves:
            # Filter by confidence threshold
            if wave.confidence >= self.confidence_threshold:
                validated_waves.append(wave)
        
        # Remove overlapping waves (keep higher confidence)
        validated_waves = self._remove_overlapping_waves(validated_waves)
        
        # Sort by timestamp
        validated_waves.sort(key=lambda w: w.start_point.timestamp)
        
        return validated_waves
    
    def _remove_overlapping_waves(self, waves: List[Wave]) -> List[Wave]:
        """
        Remove overlapping waves, keeping those with higher confidence.
        
        Args:
            waves: List of waves
            
        Returns:
            List of non-overlapping waves
        """
        if not waves:
            return waves
        
        # Sort by confidence (descending)
        sorted_waves = sorted(waves, key=lambda w: w.confidence, reverse=True)
        
        non_overlapping = []
        
        for wave in sorted_waves:
            overlaps = False
            
            for existing_wave in non_overlapping:
                # Check for overlap
                if (wave.start_point.index <= existing_wave.end_point.index and 
                    wave.end_point.index >= existing_wave.start_point.index):
                    overlaps = True
                    break
            
            if not overlaps:
                non_overlapping.append(wave)
        
        return non_overlapping
    
    def get_current_wave_count(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get the current wave count and analysis.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Dictionary with current wave analysis
        """
        waves = self.detect_waves(data)
        
        if not waves:
            return {
                'current_wave': 'Unknown',
                'confidence': 0.0,
                'next_target': None,
                'analysis': 'Insufficient data for wave analysis'
            }
        
        # Get the most recent wave
        latest_wave = max(waves, key=lambda w: w.end_point.timestamp)
        
        # Determine what might come next
        next_target = self._predict_next_wave(latest_wave, waves)
        
        return {
            'current_wave': latest_wave.wave_type.value,
            'confidence': latest_wave.confidence,
            'direction': latest_wave.direction.name,
            'next_target': next_target,
            'fibonacci_ratios': latest_wave.fibonacci_ratios,
            'analysis': f"Currently in {latest_wave.wave_type.value} wave with {latest_wave.confidence:.2f} confidence"
        }
    
    def generate_future_wave_scenarios(self, current_wave: Wave, all_waves: List[Wave], data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate multiple future wave scenarios with different paths and confidence levels.
        
        Args:
            current_wave: Current wave
            all_waves: All detected waves
            data: OHLCV DataFrame
            
        Returns:
            List of future wave scenarios
        """
        scenarios = []
        
        try:
            if current_wave.wave_type == WaveType.IMPULSE_1:
                # Scenario A: Standard Wave 2 correction
                scenarios.append({
                    'scenario': 'A',
                    'name': 'Standard Wave 2 Correction',
                    'next_wave': 'Wave 2',
                    'direction': 'Opposite to Wave 1',
                    'fibonacci_targets': [
                        current_wave.start_point.price * 0.618,  # 61.8% retracement
                        current_wave.start_point.price * 0.5,    # 50% retracement
                        current_wave.start_point.price * 0.382   # 38.2% retracement
                    ],
                    'likelihood': 0.8,
                    'description': 'Standard corrective wave 2, typically retraces 50-61.8% of wave 1'
                })
                
                # Scenario B: Shallow Wave 2
                scenarios.append({
                    'scenario': 'B',
                    'name': 'Shallow Wave 2',
                    'next_wave': 'Wave 2 (Shallow)',
                    'direction': 'Opposite to Wave 1',
                    'fibonacci_targets': [
                        current_wave.start_point.price * 0.382,  # 38.2% retracement
                        current_wave.start_point.price * 0.236   # 23.6% retracement
                    ],
                    'likelihood': 0.6,
                    'description': 'Shallow correction, often in strong trends'
                })
                
            elif current_wave.wave_type == WaveType.IMPULSE_2:
                # Scenario A: Extended Wave 3
                scenarios.append({
                    'scenario': 'A',
                    'name': 'Extended Wave 3',
                    'next_wave': 'Wave 3 (Extended)',
                    'direction': 'Same as Wave 1',
                    'fibonacci_targets': [
                        current_wave.end_point.price + (current_wave.price_change * 1.618),  # 161.8% extension
                        current_wave.end_point.price + (current_wave.price_change * 2.618),  # 261.8% extension
                        current_wave.end_point.price + (current_wave.price_change * 4.236)   # 423.6% extension
                    ],
                    'likelihood': 0.7,
                    'description': 'Wave 3 is often the longest and strongest wave'
                })
                
                # Scenario B: Normal Wave 3
                scenarios.append({
                    'scenario': 'B',
                    'name': 'Normal Wave 3',
                    'next_wave': 'Wave 3',
                    'direction': 'Same as Wave 1',
                    'fibonacci_targets': [
                        current_wave.end_point.price + (current_wave.price_change * 1.0),   # 100% of wave 1
                        current_wave.end_point.price + (current_wave.price_change * 1.272), # 127.2% extension
                        current_wave.end_point.price + (current_wave.price_change * 1.618)  # 161.8% extension
                    ],
                    'likelihood': 0.8,
                    'description': 'Standard wave 3, typically 100-161.8% of wave 1'
                })
                
            elif current_wave.wave_type == WaveType.IMPULSE_3:
                # Scenario A: Standard Wave 4 correction
                scenarios.append({
                    'scenario': 'A',
                    'name': 'Standard Wave 4 Correction',
                    'next_wave': 'Wave 4',
                    'direction': 'Opposite to Wave 3',
                    'fibonacci_targets': [
                        current_wave.start_point.price * 0.382,  # 38.2% retracement
                        current_wave.start_point.price * 0.236   # 23.6% retracement
                    ],
                    'likelihood': 0.8,
                    'description': 'Wave 4 typically retraces 23.6-38.2% of wave 3'
                })
                
                # Scenario B: Deep Wave 4
                scenarios.append({
                    'scenario': 'B',
                    'name': 'Deep Wave 4',
                    'next_wave': 'Wave 4 (Deep)',
                    'direction': 'Opposite to Wave 3',
                    'fibonacci_targets': [
                        current_wave.start_point.price * 0.5,    # 50% retracement
                        current_wave.start_point.price * 0.618   # 61.8% retracement
                    ],
                    'likelihood': 0.4,
                    'description': 'Deep correction, less common but possible'
                })
                
            elif current_wave.wave_type == WaveType.IMPULSE_4:
                # Scenario A: Final Wave 5
                scenarios.append({
                    'scenario': 'A',
                    'name': 'Final Wave 5',
                    'next_wave': 'Wave 5',
                    'direction': 'Same as Wave 3',
                    'fibonacci_targets': [
                        current_wave.end_point.price + (current_wave.price_change * 0.618),  # 61.8% of wave 1
                        current_wave.end_point.price + (current_wave.price_change * 1.0),   # 100% of wave 1
                        current_wave.end_point.price + (current_wave.price_change * 1.618)  # 161.8% of wave 1
                    ],
                    'likelihood': 0.9,
                    'description': 'Final impulse wave, typically 61.8-100% of wave 1'
                })
                
            elif current_wave.wave_type == WaveType.IMPULSE_5:
                # Scenario A: ABC Correction
                scenarios.append({
                    'scenario': 'A',
                    'name': 'ABC Correction',
                    'next_wave': 'Wave A',
                    'direction': 'Opposite to Wave 5',
                    'fibonacci_targets': [
                        current_wave.start_point.price * 0.618,  # 61.8% retracement
                        current_wave.start_point.price * 0.5,    # 50% retracement
                        current_wave.start_point.price * 0.382   # 38.2% retracement
                    ],
                    'likelihood': 0.8,
                    'description': 'Standard ABC corrective pattern'
                })
                
                # Scenario B: Complex Correction (WXY)
                scenarios.append({
                    'scenario': 'B',
                    'name': 'Complex Correction (WXY)',
                    'next_wave': 'Wave W',
                    'direction': 'Opposite to Wave 5',
                    'fibonacci_targets': [
                        current_wave.start_point.price * 0.786,  # 78.6% retracement
                        current_wave.start_point.price * 0.618   # 61.8% retracement
                    ],
                    'likelihood': 0.6,
                    'description': 'Complex correction with multiple sub-waves'
                })
                
            elif current_wave.wave_type == WaveType.CORRECTIVE_A:
                # Scenario A: Standard Wave B
                scenarios.append({
                    'scenario': 'A',
                    'name': 'Standard Wave B',
                    'next_wave': 'Wave B',
                    'direction': 'Partial retracement of A',
                    'fibonacci_targets': [
                        current_wave.end_point.price * 0.5,      # 50% retracement
                        current_wave.end_point.price * 0.618,    # 61.8% retracement
                        current_wave.end_point.price * 0.786     # 78.6% retracement
                    ],
                    'likelihood': 0.8,
                    'description': 'Wave B typically retraces 50-78.6% of wave A'
                })
                
            elif current_wave.wave_type == WaveType.CORRECTIVE_B:
                # Scenario A: Final Wave C
                scenarios.append({
                    'scenario': 'A',
                    'name': 'Final Wave C',
                    'next_wave': 'Wave C',
                    'direction': 'Same as Wave A',
                    'fibonacci_targets': [
                        current_wave.end_point.price - (current_wave.price_change * 1.0),   # 100% of wave A
                        current_wave.end_point.price - (current_wave.price_change * 1.272), # 127.2% of wave A
                        current_wave.end_point.price - (current_wave.price_change * 1.618)  # 161.8% of wave A
                    ],
                    'likelihood': 0.9,
                    'description': 'Wave C typically equals or extends wave A'
                })
                
            elif current_wave.wave_type == WaveType.CORRECTIVE_C:
                # Scenario A: New Impulse Cycle
                scenarios.append({
                    'scenario': 'A',
                    'name': 'New Impulse Cycle',
                    'next_wave': 'New Wave 1',
                    'direction': 'New trend direction',
                    'fibonacci_targets': [
                        current_wave.end_point.price * 1.272,    # 127.2% extension
                        current_wave.end_point.price * 1.618,    # 161.8% extension
                        current_wave.end_point.price * 2.618     # 261.8% extension
                    ],
                    'likelihood': 0.7,
                    'description': 'New impulse cycle beginning'
                })
                
        except Exception as e:
            logger.error(f"Error generating future scenarios: {e}")
            
        return scenarios

    def generate_advanced_future_scenarios(
        self, 
        current_wave: Wave, 
        all_waves: List[Wave], 
        data: pd.DataFrame,
        include_time_projections: bool = True,
        include_invalidation_levels: bool = True,
        include_complex_patterns: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate advanced future wave scenarios with time projections, invalidation levels, and complex patterns.
        
        Args:
            current_wave: Current wave
            all_waves: All detected waves
            data: OHLCV DataFrame
            include_time_projections: Whether to include time-based projections
            include_invalidation_levels: Whether to include invalidation levels
            include_complex_patterns: Whether to include complex correction patterns
            
        Returns:
            List of advanced future wave scenarios
        """
        scenarios = []
        
        try:
            # Get base scenarios
            base_scenarios = self.generate_future_wave_scenarios(current_wave, all_waves, data)
            
            for base_scenario in base_scenarios:
                # Enhance with advanced features
                enhanced_scenario = base_scenario.copy()
                
                # Add time-based projections
                if include_time_projections:
                    enhanced_scenario.update(self._add_time_projections(current_wave, base_scenario, data))
                
                # Add invalidation levels
                if include_invalidation_levels:
                    enhanced_scenario.update(self._add_invalidation_levels(current_wave, base_scenario, data))
                
                # Add complex pattern scenarios
                if include_complex_patterns:
                    complex_scenarios = self._generate_complex_pattern_scenarios(current_wave, base_scenario, data)
                    scenarios.extend(complex_scenarios)
                
                # Add historical pattern matching
                enhanced_scenario.update(self._add_historical_pattern_matching(current_wave, all_waves, data))
                
                scenarios.append(enhanced_scenario)
            
            # Add triangle and diagonal scenarios
            if include_complex_patterns:
                triangle_scenarios = self._generate_triangle_scenarios(current_wave, data)
                diagonal_scenarios = self._generate_diagonal_scenarios(current_wave, data)
                scenarios.extend(triangle_scenarios)
                scenarios.extend(diagonal_scenarios)
            
            return scenarios
            
        except Exception as e:
            logger.error(f"Error generating advanced scenarios: {e}")
            return []

    def _add_time_projections(self, current_wave: Wave, scenario: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Add time-based projections to a scenario."""
        time_projections = {}
        
        try:
            # Calculate average wave duration from historical data
            if len(data) > 20:
                recent_waves = [w for w in all_waves if w.end_point.timestamp > data.index[-20]]
                if recent_waves:
                    avg_duration = np.mean([w.duration for w in recent_waves])
                    time_projections['estimated_duration'] = int(avg_duration)
                    time_projections['estimated_completion'] = current_wave.end_point.timestamp + pd.Timedelta(hours=avg_duration)
            
            # Add Fibonacci time ratios
            if current_wave.duration > 0:
                time_projections['fibonacci_time_targets'] = [
                    current_wave.end_point.timestamp + pd.Timedelta(hours=int(current_wave.duration * 0.618)),
                    current_wave.end_point.timestamp + pd.Timedelta(hours=int(current_wave.duration * 1.0)),
                    current_wave.end_point.timestamp + pd.Timedelta(hours=int(current_wave.duration * 1.618))
                ]
            
            return {'time_projections': time_projections}
            
        except Exception as e:
            logger.error(f"Error adding time projections: {e}")
            return {}

    def _add_invalidation_levels(self, current_wave: Wave, scenario: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Add invalidation levels to a scenario."""
        invalidation_levels = {}
        
        try:
            if current_wave.wave_type in [WaveType.IMPULSE_1, WaveType.IMPULSE_3, WaveType.IMPULSE_5]:
                # For impulse waves, invalidation is typically the start of the previous wave
                invalidation_levels['primary'] = current_wave.start_point.price
                invalidation_levels['secondary'] = current_wave.start_point.price * 0.95  # 5% buffer
                
            elif current_wave.wave_type in [WaveType.CORRECTIVE_A, WaveType.CORRECTIVE_B, WaveType.CORRECTIVE_C]:
                # For corrective waves, invalidation is typically the end of the previous impulse
                invalidation_levels['primary'] = current_wave.end_point.price
                invalidation_levels['secondary'] = current_wave.end_point.price * 1.05  # 5% buffer
                
            # Add scenario-specific invalidation
            if scenario.get('fibonacci_targets'):
                invalidation_levels['scenario_specific'] = scenario['fibonacci_targets'][0] * 0.9  # 10% below target
                
            return {'invalidation_levels': invalidation_levels}
            
        except Exception as e:
            logger.error(f"Error adding invalidation levels: {e}")
            return {}

    def _generate_complex_pattern_scenarios(self, current_wave: Wave, base_scenario: Dict[str, Any], data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate complex correction pattern scenarios."""
        complex_scenarios = []
        
        try:
            if current_wave.wave_type == WaveType.IMPULSE_5:
                # WXY Complex Correction
                wxy_scenario = base_scenario.copy()
                wxy_scenario.update({
                    'scenario': 'WXY',
                    'name': 'WXY Complex Correction',
                    'next_wave': 'Wave W',
                    'pattern_type': 'complex_correction',
                    'sub_waves': ['W', 'X', 'Y'],
                    'fibonacci_targets': [
                        current_wave.start_point.price * 0.786,  # 78.6% retracement
                        current_wave.start_point.price * 0.618,  # 61.8% retracement
                        current_wave.start_point.price * 0.5     # 50% retracement
                    ],
                    'likelihood': 0.4,
                    'description': 'Complex WXY correction with multiple sub-waves'
                })
                complex_scenarios.append(wxy_scenario)
                
                # WXYXZ Complex Correction
                wxyxz_scenario = base_scenario.copy()
                wxyxz_scenario.update({
                    'scenario': 'WXYXZ',
                    'name': 'WXYXZ Complex Correction',
                    'next_wave': 'Wave W',
                    'pattern_type': 'complex_correction',
                    'sub_waves': ['W', 'X', 'Y', 'X2', 'Z'],
                    'fibonacci_targets': [
                        current_wave.start_point.price * 0.886,  # 88.6% retracement
                        current_wave.start_point.price * 0.786,  # 78.6% retracement
                        current_wave.start_point.price * 0.618   # 61.8% retracement
                    ],
                    'likelihood': 0.2,
                    'description': 'Extended complex correction with five sub-waves'
                })
                complex_scenarios.append(wxyxz_scenario)
                
        except Exception as e:
            logger.error(f"Error generating complex pattern scenarios: {e}")
            
        return complex_scenarios

    def _generate_triangle_scenarios(self, current_wave: Wave, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate triangle pattern scenarios."""
        triangle_scenarios = []
        
        try:
            if current_wave.wave_type in [WaveType.IMPULSE_3, WaveType.IMPULSE_4]:
                # Contracting Triangle
                contracting_scenario = {
                    'scenario': 'TRIANGLE_CONTRACTING',
                    'name': 'Contracting Triangle',
                    'next_wave': 'Triangle A',
                    'pattern_type': 'triangle',
                    'triangle_type': 'contracting',
                    'sub_waves': ['A', 'B', 'C', 'D', 'E'],
                    'fibonacci_targets': [
                        current_wave.end_point.price * 0.618,  # 61.8% retracement
                        current_wave.end_point.price * 0.5,    # 50% retracement
                        current_wave.end_point.price * 0.382   # 38.2% retracement
                    ],
                    'likelihood': 0.3,
                    'description': 'Contracting triangle pattern in wave 4 or B',
                    'time_projections': {
                        'estimated_duration': int(current_wave.duration * 1.5),
                        'breakout_target': current_wave.end_point.price * 0.618
                    }
                }
                triangle_scenarios.append(contracting_scenario)
                
                # Expanding Triangle
                expanding_scenario = {
                    'scenario': 'TRIANGLE_EXPANDING',
                    'name': 'Expanding Triangle',
                    'next_wave': 'Triangle A',
                    'pattern_type': 'triangle',
                    'triangle_type': 'expanding',
                    'sub_waves': ['A', 'B', 'C', 'D', 'E'],
                    'fibonacci_targets': [
                        current_wave.end_point.price * 0.786,  # 78.6% retracement
                        current_wave.end_point.price * 0.886,  # 88.6% retracement
                        current_wave.end_point.price * 1.0     # 100% retracement
                    ],
                    'likelihood': 0.2,
                    'description': 'Expanding triangle pattern, less common',
                    'time_projections': {
                        'estimated_duration': int(current_wave.duration * 2.0),
                        'breakout_target': current_wave.end_point.price * 0.786
                    }
                }
                triangle_scenarios.append(expanding_scenario)
                
        except Exception as e:
            logger.error(f"Error generating triangle scenarios: {e}")
            
        return triangle_scenarios

    def _generate_diagonal_scenarios(self, current_wave: Wave, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate diagonal pattern scenarios."""
        diagonal_scenarios = []
        
        try:
            if current_wave.wave_type == WaveType.IMPULSE_1:
                # Leading Diagonal
                leading_scenario = {
                    'scenario': 'DIAGONAL_LEADING',
                    'name': 'Leading Diagonal',
                    'next_wave': 'Diagonal 1',
                    'pattern_type': 'diagonal',
                    'diagonal_type': 'leading',
                    'sub_waves': ['1', '2', '3', '4', '5'],
                    'fibonacci_targets': [
                        current_wave.end_point.price * 1.272,  # 127.2% extension
                        current_wave.end_point.price * 1.618,  # 161.8% extension
                        current_wave.end_point.price * 2.0     # 200% extension
                    ],
                    'likelihood': 0.1,
                    'description': 'Leading diagonal in wave 1 or A',
                    'characteristics': {
                        'wave_4_overlaps_wave_1': True,
                        'wedge_shaped': True,
                        'subwaves_are_3s_or_5s': True
                    }
                }
                diagonal_scenarios.append(leading_scenario)
                
            elif current_wave.wave_type == WaveType.IMPULSE_5:
                # Ending Diagonal
                ending_scenario = {
                    'scenario': 'DIAGONAL_ENDING',
                    'name': 'Ending Diagonal',
                    'next_wave': 'Diagonal 5',
                    'pattern_type': 'diagonal',
                    'diagonal_type': 'ending',
                    'sub_waves': ['1', '2', '3', '4', '5'],
                    'fibonacci_targets': [
                        current_wave.end_point.price * 0.618,  # 61.8% extension
                        current_wave.end_point.price * 1.0,    # 100% extension
                        current_wave.end_point.price * 1.272   # 127.2% extension
                    ],
                    'likelihood': 0.15,
                    'description': 'Ending diagonal in wave 5 or C',
                    'characteristics': {
                        'wave_4_overlaps_wave_1': True,
                        'wedge_shaped': True,
                        'subwaves_are_3s': True
                    }
                }
                diagonal_scenarios.append(ending_scenario)
                
        except Exception as e:
            logger.error(f"Error generating diagonal scenarios: {e}")
            
        return diagonal_scenarios

    def _add_historical_pattern_matching(self, current_wave: Wave, all_waves: List[Wave], data: pd.DataFrame) -> Dict[str, Any]:
        """Add historical pattern matching for likelihood scoring."""
        pattern_matching = {}
        
        try:
            # Find similar historical patterns
            similar_patterns = self._find_similar_patterns(current_wave, all_waves, data)
            
            if similar_patterns:
                # Calculate pattern similarity score
                avg_similarity = np.mean([p['similarity'] for p in similar_patterns])
                pattern_matching['historical_similarity'] = avg_similarity
                pattern_matching['similar_patterns_count'] = len(similar_patterns)
                pattern_matching['most_similar_pattern'] = max(similar_patterns, key=lambda x: x['similarity'])
                
                # Adjust likelihood based on historical patterns
                pattern_matching['likelihood_adjustment'] = avg_similarity * 0.2  # Up to 20% adjustment
            else:
                pattern_matching['historical_similarity'] = 0.0
                pattern_matching['similar_patterns_count'] = 0
                pattern_matching['likelihood_adjustment'] = 0.0
                
            return {'pattern_matching': pattern_matching}
            
        except Exception as e:
            logger.error(f"Error adding historical pattern matching: {e}")
            return {}

    def _find_similar_patterns(self, current_wave: Wave, all_waves: List[Wave], data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find similar historical patterns for likelihood scoring."""
        similar_patterns = []
        
        try:
            # Look for waves with similar characteristics
            for wave in all_waves:
                if wave.wave_type == current_wave.wave_type and wave != current_wave:
                    # Calculate similarity based on price change percentage and duration
                    price_similarity = 1 - abs(wave.price_change_pct - current_wave.price_change_pct) / max(abs(wave.price_change_pct), abs(current_wave.price_change_pct), 0.01)
                    duration_similarity = 1 - abs(wave.duration - current_wave.duration) / max(wave.duration, current_wave.duration, 1)
                    
                    overall_similarity = (price_similarity + duration_similarity) / 2
                    
                    if overall_similarity > 0.7:  # Only include highly similar patterns
                        similar_patterns.append({
                            'wave': wave,
                            'similarity': overall_similarity,
                            'price_similarity': price_similarity,
                            'duration_similarity': duration_similarity,
                            'timestamp': wave.end_point.timestamp
                        })
            
            # Sort by similarity
            similar_patterns.sort(key=lambda x: x['similarity'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error finding similar patterns: {e}")
            
        return similar_patterns

    def _predict_next_wave(self, current_wave: Wave, all_waves: List[Wave]) -> Optional[Dict[str, Any]]:
        """
        Predict the next likely wave based on Elliott Wave theory.
        
        Args:
            current_wave: Current wave
            all_waves: All detected waves
            
        Returns:
            Dictionary with next wave prediction
        """
        try:
            if current_wave.wave_type == WaveType.IMPULSE_1:
                return {'type': 'Corrective Wave 2', 'direction': 'Opposite to Wave 1'}
            elif current_wave.wave_type == WaveType.IMPULSE_2:
                return {'type': 'Impulse Wave 3', 'direction': 'Same as Wave 1, likely extended'}
            elif current_wave.wave_type == WaveType.IMPULSE_3:
                return {'type': 'Corrective Wave 4', 'direction': 'Opposite to Wave 3'}
            elif current_wave.wave_type == WaveType.IMPULSE_4:
                return {'type': 'Impulse Wave 5', 'direction': 'Same as Wave 3'}
            elif current_wave.wave_type == WaveType.IMPULSE_5:
                return {'type': 'Corrective Wave A', 'direction': 'Opposite to Wave 5'}
            elif current_wave.wave_type == WaveType.CORRECTIVE_A:
                return {'type': 'Corrective Wave B', 'direction': 'Partial retracement of A'}
            elif current_wave.wave_type == WaveType.CORRECTIVE_B:
                return {'type': 'Corrective Wave C', 'direction': 'Same as Wave A'}
            elif current_wave.wave_type == WaveType.CORRECTIVE_C:
                return {'type': 'New Impulse Cycle', 'direction': 'New trend direction'}
            else:
                return None
        
        except Exception as e:
            logger.debug(f"Error predicting next wave: {e}")
            return None


if __name__ == "__main__":
    # Example usage
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.data.data_loader import DataLoader
    
    # Load sample data
    loader = DataLoader()
    data = loader.get_yahoo_data("AAPL", period="1y")
    
    # Detect waves
    detector = WaveDetector()
    waves = detector.detect_waves(data)
    
    print(f"Detected {len(waves)} Elliott Waves")
    
    # Print wave details
    for i, wave in enumerate(waves[:5]):  # Show first 5 waves
        print(f"Wave {i+1}: {wave.wave_type.value} "
              f"({wave.start_point.timestamp.strftime('%Y-%m-%d')} -> "
              f"{wave.end_point.timestamp.strftime('%Y-%m-%d')}) "
              f"Confidence: {wave.confidence:.2f}")
    
    # Get current wave count
    current_analysis = detector.get_current_wave_count(data)
    print(f"\nCurrent Analysis: {current_analysis}")
