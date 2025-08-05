"""
Comprehensive Elliott Wave Validator
Validates ALL waves (1, 2, 3, 4, 5, A, B, C) with internal structure analysis
Implements strict Elliott Wave rules with subwave validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class WaveType(Enum):
    IMPULSE = "impulse"
    CORRECTIVE = "corrective"
    ZIGZAG = "zigzag"
    FLAT = "flat"
    TRIANGLE = "triangle"
    DIAGONAL = "diagonal"


class WaveDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"


@dataclass
class WavePoint:
    """Represents a wave point with price and time data"""
    index: int
    price: float
    time: Any
    
    
@dataclass
class SubWave:
    """Represents a subwave with validation data"""
    label: str
    start: WavePoint
    end: WavePoint
    wave_type: WaveType
    direction: WaveDirection
    length: float
    duration: int
    validation_score: float
    fibonacci_ratio: Optional[float] = None
    issues: List[str] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []


@dataclass
class WaveStructure:
    """Complete wave structure with all validation data"""
    main_waves: List[SubWave]
    subwaves: Dict[str, List[SubWave]]
    wave_type: WaveType
    direction: WaveDirection
    validation_score: float
    fibonacci_levels: Dict[str, float]
    rule_compliance: Dict[str, Dict[str, Any]]
    recommendations: List[str]
    issues: List[str]


class ComprehensiveElliottValidator:
    """
    Comprehensive Elliott Wave Validator that validates ALL waves
    including internal structures and subwaves
    """
    
    def __init__(self):
        self.fibonacci_ratios = {
            'retracement': [0.236, 0.382, 0.5, 0.618, 0.786],
            'extension': [1.0, 1.272, 1.414, 1.618, 2.0, 2.618]
        }
        
        self.wave_2_retracement_range = (0.382, 0.786)  # Typical range
        self.wave_4_retracement_range = (0.236, 0.5)   # Typical range
        self.wave_b_retracement_range = (0.5, 1.0)     # For flats
        
    def validate_complete_structure(self, wave_points: List[WavePoint], 
                                  price_data: pd.DataFrame) -> WaveStructure:
        """
        Validate complete Elliott Wave structure with all waves and subwaves
        """
        try:
            # Determine if this is impulse or corrective
            if len(wave_points) == 6:  # 5-wave structure (6 points)
                return self._validate_impulse_structure(wave_points, price_data)
            elif len(wave_points) == 4:  # 3-wave structure (4 points)
                return self._validate_corrective_structure(wave_points, price_data)
            else:
                raise ValueError(f"Invalid wave structure: {len(wave_points)} points")
                
        except Exception as e:
            logger.error(f"Error validating wave structure: {e}")
            return self._create_failed_structure(wave_points, str(e))
    
    def _validate_impulse_structure(self, wave_points: List[WavePoint], 
                                  price_data: pd.DataFrame) -> WaveStructure:
        """
        Validate 5-wave impulse structure (1-2-3-4-5)
        ALL waves must be validated including internal structures
        """
        direction = self._determine_direction(wave_points[0], wave_points[-1])
        main_waves = []
        subwaves = {}
        rule_compliance = {}
        issues = []
        recommendations = []
        
        # Create main waves 1, 2, 3, 4, 5
        for i in range(5):
            wave_label = str(i + 1)
            start_point = wave_points[i]
            end_point = wave_points[i + 1]
            
            wave = SubWave(
                label=wave_label,
                start=start_point,
                end=end_point,
                wave_type=WaveType.IMPULSE if i in [0, 2, 4] else WaveType.CORRECTIVE,
                direction=direction if i in [0, 2, 4] else self._opposite_direction(direction),
                length=abs(end_point.price - start_point.price),
                duration=abs(end_point.index - start_point.index),
                validation_score=0.5  # Initial score
            )
            
            main_waves.append(wave)
            
            # Validate internal structure and get subwaves
            internal_validation = self._validate_internal_structure(
                wave, price_data, wave_label
            )
            subwaves[wave_label] = internal_validation['subwaves']
            rule_compliance[wave_label] = internal_validation['rules']
            issues.extend(internal_validation['issues'])
            recommendations.extend(internal_validation['recommendations'])
        
        # Validate impulse-specific rules
        impulse_rules = self._validate_impulse_rules(main_waves, direction)
        rule_compliance.update(impulse_rules['rules'])
        issues.extend(impulse_rules['issues'])
        recommendations.extend(impulse_rules['recommendations'])
        
        # Calculate Fibonacci levels
        fibonacci_levels = self._calculate_impulse_fibonacci_levels(main_waves)
        
        # Calculate overall validation score
        validation_score = self._calculate_overall_score(rule_compliance)
        
        return WaveStructure(
            main_waves=main_waves,
            subwaves=subwaves,
            wave_type=WaveType.IMPULSE,
            direction=direction,
            validation_score=validation_score,
            fibonacci_levels=fibonacci_levels,
            rule_compliance=rule_compliance,
            recommendations=recommendations,
            issues=issues
        )
    
    def _validate_corrective_structure(self, wave_points: List[WavePoint], 
                                     price_data: pd.DataFrame) -> WaveStructure:
        """
        Validate 3-wave corrective structure (A-B-C)
        ALL waves must be validated including internal structures
        """
        direction = self._determine_direction(wave_points[0], wave_points[-1])
        main_waves = []
        subwaves = {}
        rule_compliance = {}
        issues = []
        recommendations = []
        
        # Create main waves A, B, C
        wave_labels = ['A', 'B', 'C']
        for i, label in enumerate(wave_labels):
            start_point = wave_points[i]
            end_point = wave_points[i + 1]
            
            # A and C are typically impulse/diagonal, B is corrective
            wave_type = WaveType.IMPULSE if label in ['A', 'C'] else WaveType.CORRECTIVE
            wave_direction = direction if label in ['A', 'C'] else self._opposite_direction(direction)
            
            wave = SubWave(
                label=label,
                start=start_point,
                end=end_point,
                wave_type=wave_type,
                direction=wave_direction,
                length=abs(end_point.price - start_point.price),
                duration=abs(end_point.index - start_point.index),
                validation_score=0.5  # Initial score
            )
            
            main_waves.append(wave)
            
            # Validate internal structure and get subwaves
            internal_validation = self._validate_internal_structure(
                wave, price_data, label
            )
            subwaves[label] = internal_validation['subwaves']
            rule_compliance[label] = internal_validation['rules']
            issues.extend(internal_validation['issues'])
            recommendations.extend(internal_validation['recommendations'])
        
        # Determine corrective pattern type and validate specific rules
        corrective_type = self._determine_corrective_type(main_waves)
        corrective_rules = self._validate_corrective_rules(main_waves, corrective_type, direction)
        rule_compliance.update(corrective_rules['rules'])
        issues.extend(corrective_rules['issues'])
        recommendations.extend(corrective_rules['recommendations'])
        
        # Calculate Fibonacci levels
        fibonacci_levels = self._calculate_corrective_fibonacci_levels(main_waves)
        
        # Calculate overall validation score
        validation_score = self._calculate_overall_score(rule_compliance)
        
        return WaveStructure(
            main_waves=main_waves,
            subwaves=subwaves,
            wave_type=corrective_type,
            direction=direction,
            validation_score=validation_score,
            fibonacci_levels=fibonacci_levels,
            rule_compliance=rule_compliance,
            recommendations=recommendations,
            issues=issues
        )
    
    def _validate_internal_structure(self, wave: SubWave, price_data: pd.DataFrame, 
                                   wave_label: str) -> Dict[str, Any]:
        """
        Validate internal structure of a wave and extract subwaves
        """
        try:
            # Extract price data for this wave
            wave_data = price_data.iloc[wave.start.index:wave.end.index + 1].copy()
            
            if len(wave_data) < 5:
                return {
                    'subwaves': [],
                    'rules': {f'{wave_label}_internal': {'score': 0.3, 'status': 'insufficient_data'}},
                    'issues': [f"Wave {wave_label}: Insufficient data for internal analysis"],
                    'recommendations': [f"Wave {wave_label}: Need more price data for subwave analysis"]
                }
            
            # Detect internal waves based on wave type
            if wave.wave_type == WaveType.IMPULSE:
                return self._detect_impulse_subwaves(wave_data, wave_label, wave.direction)
            else:
                return self._detect_corrective_subwaves(wave_data, wave_label, wave.direction)
                
        except Exception as e:
            logger.error(f"Error validating internal structure for wave {wave_label}: {e}")
            return {
                'subwaves': [],
                'rules': {f'{wave_label}_internal': {'score': 0.0, 'status': 'error'}},
                'issues': [f"Wave {wave_label}: Error in internal analysis"],
                'recommendations': []
            }
    
    def _detect_impulse_subwaves(self, wave_data: pd.DataFrame, 
                               parent_label: str, direction: WaveDirection) -> Dict[str, Any]:
        """
        Detect 5-wave impulse subwaves (i, ii, iii, iv, v)
        """
        subwaves = []
        issues = []
        recommendations = []
        
        # Simple peak/trough detection for subwaves
        highs = wave_data['high'].values
        lows = wave_data['low'].values
        closes = wave_data['close'].values
        
        # Find potential subwave points
        if direction == WaveDirection.BULLISH:
            # Look for pattern: low-high-low-high-low-high
            extremes = self._find_alternating_extremes(lows, highs, 6, 'bullish')
        else:
            # Look for pattern: high-low-high-low-high-low
            extremes = self._find_alternating_extremes(highs, lows, 6, 'bearish')
        
        if len(extremes) >= 6:
            # Create subwaves i, ii, iii, iv, v
            subwave_labels = ['i', 'ii', 'iii', 'iv', 'v']
            for i, label in enumerate(subwave_labels):
                start_idx = extremes[i]
                end_idx = extremes[i + 1]
                
                subwave = SubWave(
                    label=f"{parent_label}.{label}",
                    start=WavePoint(start_idx, extremes[i], wave_data.index[start_idx]),
                    end=WavePoint(end_idx, extremes[i + 1], wave_data.index[end_idx]),
                    wave_type=WaveType.IMPULSE if i in [0, 2, 4] else WaveType.CORRECTIVE,
                    direction=direction if i in [0, 2, 4] else self._opposite_direction(direction),
                    length=abs(extremes[i + 1] - extremes[i]),
                    duration=end_idx - start_idx,
                    validation_score=0.8  # Basic score
                )
                subwaves.append(subwave)
                
            # Validate subwave impulse rules
            subwave_validation = self._validate_subwave_impulse_rules(subwaves)
            validation_score = subwave_validation['score']
            issues.extend(subwave_validation['issues'])
            recommendations.extend(subwave_validation['recommendations'])
        else:
            validation_score = 0.4
            issues.append(f"Wave {parent_label}: Could not detect clear 5-wave impulse structure")
            recommendations.append(f"Wave {parent_label}: Consider alternative wave counting")
        
        return {
            'subwaves': subwaves,
            'rules': {f'{parent_label}_impulse_structure': {
                'score': validation_score,
                'status': 'valid' if validation_score > 0.6 else 'questionable'
            }},
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _detect_corrective_subwaves(self, wave_data: pd.DataFrame, 
                                  parent_label: str, direction: WaveDirection) -> Dict[str, Any]:
        """
        Detect corrective subwaves (typically 3-wave: a, b, c)
        """
        subwaves = []
        issues = []
        recommendations = []
        
        # Simple 3-wave detection for corrective patterns
        highs = wave_data['high'].values
        lows = wave_data['low'].values
        
        # Find potential 3-wave structure
        if direction == WaveDirection.BEARISH:  # Corrective down
            extremes = self._find_alternating_extremes(highs, lows, 4, 'bearish')
        else:  # Corrective up
            extremes = self._find_alternating_extremes(lows, highs, 4, 'bullish')
        
        if len(extremes) >= 4:
            # Create subwaves a, b, c
            subwave_labels = ['a', 'b', 'c']
            for i, label in enumerate(subwave_labels):
                start_idx = extremes[i]
                end_idx = extremes[i + 1]
                
                subwave = SubWave(
                    label=f"{parent_label}.{label}",
                    start=WavePoint(start_idx, extremes[i], wave_data.index[start_idx]),
                    end=WavePoint(end_idx, extremes[i + 1], wave_data.index[end_idx]),
                    wave_type=WaveType.IMPULSE if i in [0, 2] else WaveType.CORRECTIVE,
                    direction=direction if i in [0, 2] else self._opposite_direction(direction),
                    length=abs(extremes[i + 1] - extremes[i]),
                    duration=end_idx - start_idx,
                    validation_score=0.7  # Basic score
                )
                subwaves.append(subwave)
                
            validation_score = 0.7
        else:
            validation_score = 0.4
            issues.append(f"Wave {parent_label}: Could not detect clear corrective structure")
            recommendations.append(f"Wave {parent_label}: May be a complex correction")
        
        return {
            'subwaves': subwaves,
            'rules': {f'{parent_label}_corrective_structure': {
                'score': validation_score,
                'status': 'valid' if validation_score > 0.6 else 'questionable'
            }},
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _validate_impulse_rules(self, waves: List[SubWave], 
                              direction: WaveDirection) -> Dict[str, Any]:
        """
        Validate all impulse wave rules (1-2-3-4-5)
        """
        rules = {}
        issues = []
        recommendations = []
        
        wave1, wave2, wave3, wave4, wave5 = waves
        
        # Rule 1: Wave 2 cannot retrace more than 100% of Wave 1
        wave2_retracement = self._calculate_retracement(wave1, wave2)
        wave2_valid = wave2_retracement < 1.0
        
        rules['wave2_retracement'] = {
            'score': 1.0 if wave2_valid else 0.0,
            'value': wave2_retracement,
            'threshold': 1.0,
            'status': 'valid' if wave2_valid else 'invalid',
            'description': f"Wave 2 retracement: {wave2_retracement:.1%}"
        }
        
        if not wave2_valid:
            issues.append("Wave 2 retraces more than 100% of Wave 1 - Invalid impulse")
        
        # Rule 2: Wave 3 cannot be the shortest among 1, 3, 5
        lengths = [wave1.length, wave3.length, wave5.length]
        wave3_shortest = wave3.length == min(lengths)
        
        rules['wave3_not_shortest'] = {
            'score': 0.0 if wave3_shortest else 1.0,
            'lengths': {
                'wave1': wave1.length,
                'wave3': wave3.length,
                'wave5': wave5.length
            },
            'status': 'invalid' if wave3_shortest else 'valid',
            'description': f"Wave 3 length rank: {sorted(lengths, reverse=True).index(wave3.length) + 1}/3"
        }
        
        if wave3_shortest:
            issues.append("Wave 3 is the shortest wave - Invalid impulse")
        
        # Rule 3: Wave 4 must not overlap with Wave 1
        overlap = self._check_wave_overlap(wave1, wave4, direction)
        
        rules['wave4_no_overlap'] = {
            'score': 0.0 if overlap else 1.0,
            'overlap_detected': overlap,
            'status': 'invalid' if overlap else 'valid',
            'description': f"Wave 4 overlap with Wave 1: {'Yes' if overlap else 'No'}"
        }
        
        if overlap:
            issues.append("Wave 4 overlaps with Wave 1 - Invalid impulse")
        
        # Rule 4: Fibonacci relationships
        fib_score = self._validate_impulse_fibonacci(waves)
        rules['fibonacci_compliance'] = fib_score
        
        # Rule 5: Wave proportionality
        prop_score = self._validate_wave_proportionality(waves)
        rules['wave_proportionality'] = prop_score
        
        # Add recommendations
        if wave2_retracement > 0.786:
            recommendations.append("Wave 2 retracement is deep - monitor for potential invalidation")
        
        if wave3.length < 1.618 * wave1.length:
            recommendations.append("Wave 3 could be more extended - typical target is 161.8% of Wave 1")
        
        return {
            'rules': rules,
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _validate_corrective_rules(self, waves: List[SubWave], 
                                 corrective_type: WaveType, 
                                 direction: WaveDirection) -> Dict[str, Any]:
        """
        Validate corrective wave rules (A-B-C)
        """
        rules = {}
        issues = []
        recommendations = []
        
        waveA, waveB, waveC = waves
        
        # Wave B retracement validation
        waveB_retracement = self._calculate_retracement(waveA, waveB)
        
        if corrective_type == WaveType.ZIGZAG:
            # Zigzag: Wave B should retrace < 100% of A, ideally 50-61.8%
            b_valid = waveB_retracement < 1.0
            ideal_range = 0.5 <= waveB_retracement <= 0.618
            
            rules['waveB_zigzag_retracement'] = {
                'score': 1.0 if b_valid else 0.0,
                'bonus': 0.2 if ideal_range else 0.0,
                'value': waveB_retracement,
                'status': 'valid' if b_valid else 'invalid',
                'description': f"Wave B retracement (zigzag): {waveB_retracement:.1%}"
            }
            
        elif corrective_type == WaveType.FLAT:
            # Flat: Wave B should retrace ~90-100% or more of A
            b_valid = waveB_retracement >= 0.9
            ideal_range = 0.9 <= waveB_retracement <= 1.1
            
            rules['waveB_flat_retracement'] = {
                'score': 1.0 if b_valid else 0.5,
                'bonus': 0.2 if ideal_range else 0.0,
                'value': waveB_retracement,
                'status': 'valid' if b_valid else 'questionable',
                'description': f"Wave B retracement (flat): {waveB_retracement:.1%}"
            }
        
        # Wave C length validation
        c_to_a_ratio = waveC.length / waveA.length
        
        # Common C wave relationships: 100% or 161.8% of A
        c_relationships = [1.0, 1.618]
        c_score = max([1.0 - abs(c_to_a_ratio - target) / target 
                      for target in c_relationships])
        c_score = max(0.0, c_score)
        
        rules['waveC_length_relationship'] = {
            'score': c_score,
            'ratio': c_to_a_ratio,
            'targets': c_relationships,
            'status': 'valid' if c_score > 0.7 else 'questionable',
            'description': f"Wave C = {c_to_a_ratio:.1%} of Wave A"
        }
        
        # Validate internal structures
        for i, (wave, label) in enumerate(zip(waves, ['A', 'B', 'C'])):
            expected_type = WaveType.IMPULSE if label in ['A', 'C'] else WaveType.CORRECTIVE
            if wave.wave_type != expected_type:
                issues.append(f"Wave {label} should be {expected_type.value}")
        
        return {
            'rules': rules,
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _find_alternating_extremes(self, primary: np.ndarray, secondary: np.ndarray, 
                                 count: int, direction: str) -> List[float]:
        """
        Find alternating extremes for wave detection
        """
        extremes = []
        
        if direction == 'bullish':
            # Start with low, then alternate high-low
            extremes.append(np.min(primary[:len(primary)//3]))  # Starting low
            for i in range((count - 1) // 2):
                # Add high
                start_idx = len(extremes) * len(primary) // count
                end_idx = min((len(extremes) + 1) * len(primary) // count, len(secondary))
                if start_idx < len(secondary) and end_idx <= len(secondary):
                    extremes.append(np.max(secondary[start_idx:end_idx]))
                
                # Add low
                start_idx = len(extremes) * len(primary) // count
                end_idx = min((len(extremes) + 1) * len(primary) // count, len(primary))
                if start_idx < len(primary) and end_idx <= len(primary):
                    extremes.append(np.min(primary[start_idx:end_idx]))
        else:
            # Start with high, then alternate low-high
            extremes.append(np.max(primary[:len(primary)//3]))  # Starting high
            for i in range((count - 1) // 2):
                # Add low
                start_idx = len(extremes) * len(secondary) // count
                end_idx = min((len(extremes) + 1) * len(secondary) // count, len(secondary))
                if start_idx < len(secondary) and end_idx <= len(secondary):
                    extremes.append(np.min(secondary[start_idx:end_idx]))
                
                # Add high
                start_idx = len(extremes) * len(primary) // count
                end_idx = min((len(extremes) + 1) * len(primary) // count, len(primary))
                if start_idx < len(primary) and end_idx <= len(primary):
                    extremes.append(np.max(primary[start_idx:end_idx]))
        
        return extremes[:count]
    
    def _calculate_retracement(self, wave1: SubWave, wave2: SubWave) -> float:
        """Calculate retracement percentage of wave2 relative to wave1"""
        return abs(wave2.length) / abs(wave1.length)
    
    def _check_wave_overlap(self, wave1: SubWave, wave4: SubWave, 
                          direction: WaveDirection) -> bool:
        """Check if Wave 4 overlaps with Wave 1"""
        if direction == WaveDirection.BULLISH:
            return wave4.start.price <= wave1.end.price
        else:
            return wave4.start.price >= wave1.end.price
    
    def _determine_direction(self, start: WavePoint, end: WavePoint) -> WaveDirection:
        """Determine wave direction"""
        return WaveDirection.BULLISH if end.price > start.price else WaveDirection.BEARISH
    
    def _opposite_direction(self, direction: WaveDirection) -> WaveDirection:
        """Get opposite direction"""
        return WaveDirection.BEARISH if direction == WaveDirection.BULLISH else WaveDirection.BULLISH
    
    def _determine_corrective_type(self, waves: List[SubWave]) -> WaveType:
        """Determine type of corrective pattern"""
        waveA, waveB, waveC = waves
        b_retracement = self._calculate_retracement(waveA, waveB)
        
        if b_retracement < 0.9:
            return WaveType.ZIGZAG
        else:
            return WaveType.FLAT
    
    def _validate_impulse_fibonacci(self, waves: List[SubWave]) -> Dict[str, Any]:
        """Validate Fibonacci relationships in impulse waves"""
        wave1, wave2, wave3, wave4, wave5 = waves
        
        # Wave 2 should be 50-61.8% of Wave 1
        wave2_ratio = self._calculate_retracement(wave1, wave2)
        wave2_fib_score = 1.0 if 0.5 <= wave2_ratio <= 0.618 else 0.5
        
        # Wave 3 often 161.8% of Wave 1
        wave3_ratio = wave3.length / wave1.length
        wave3_fib_score = 1.0 - abs(wave3_ratio - 1.618) / 1.618
        wave3_fib_score = max(0.0, min(1.0, wave3_fib_score))
        
        # Wave 4 should be 23.6-38.2% of Wave 3
        wave4_ratio = self._calculate_retracement(wave3, wave4)
        wave4_fib_score = 1.0 if 0.236 <= wave4_ratio <= 0.382 else 0.5
        
        overall_score = (wave2_fib_score + wave3_fib_score + wave4_fib_score) / 3
        
        return {
            'score': overall_score,
            'wave2_fib': wave2_fib_score,
            'wave3_fib': wave3_fib_score,
            'wave4_fib': wave4_fib_score,
            'status': 'excellent' if overall_score > 0.8 else 'good' if overall_score > 0.6 else 'questionable'
        }
    
    def _validate_wave_proportionality(self, waves: List[SubWave]) -> Dict[str, Any]:
        """Validate wave proportionality and natural progression"""
        durations = [wave.duration for wave in waves]
        lengths = [wave.length for wave in waves]
        
        # Check for natural progression in durations
        duration_score = 0.8  # Default good score
        
        # Check for extreme duration differences
        max_duration = max(durations)
        min_duration = min(durations)
        if max_duration > 5 * min_duration:
            duration_score *= 0.7
        
        # Check for natural length progression
        length_score = 0.8  # Default good score
        
        overall_score = (duration_score + length_score) / 2
        
        return {
            'score': overall_score,
            'duration_score': duration_score,
            'length_score': length_score,
            'status': 'natural' if overall_score > 0.7 else 'questionable'
        }
    
    def _validate_subwave_impulse_rules(self, subwaves: List[SubWave]) -> Dict[str, Any]:
        """Validate impulse rules for subwaves"""
        if len(subwaves) != 5:
            return {'score': 0.0, 'issues': ['Incomplete subwave structure'], 'recommendations': []}
        
        issues = []
        recommendations = []
        scores = []
        
        # Basic subwave validation
        for i, subwave in enumerate(subwaves):
            expected_type = WaveType.IMPULSE if i in [0, 2, 4] else WaveType.CORRECTIVE
            if subwave.wave_type == expected_type:
                scores.append(1.0)
            else:
                scores.append(0.5)
                issues.append(f"Subwave {subwave.label} type mismatch")
        
        overall_score = np.mean(scores) if scores else 0.0
        
        return {
            'score': overall_score,
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _calculate_impulse_fibonacci_levels(self, waves: List[SubWave]) -> Dict[str, float]:
        """Calculate Fibonacci levels for impulse waves"""
        wave1, wave2, wave3, wave4, wave5 = waves
        
        return {
            'wave1_length': wave1.length,
            'wave2_retracement': self._calculate_retracement(wave1, wave2),
            'wave3_extension': wave3.length / wave1.length,
            'wave4_retracement': self._calculate_retracement(wave3, wave4),
            'wave5_projection': wave5.length / wave1.length
        }
    
    def _calculate_corrective_fibonacci_levels(self, waves: List[SubWave]) -> Dict[str, float]:
        """Calculate Fibonacci levels for corrective waves"""
        waveA, waveB, waveC = waves
        
        return {
            'waveA_length': waveA.length,
            'waveB_retracement': self._calculate_retracement(waveA, waveB),
            'waveC_projection': waveC.length / waveA.length
        }
    
    def _calculate_overall_score(self, rule_compliance: Dict[str, Dict[str, Any]]) -> float:
        """Calculate overall validation score from all rules"""
        scores = []
        weights = []
        
        for rule_name, rule_data in rule_compliance.items():
            if isinstance(rule_data, dict) and 'score' in rule_data:
                score = rule_data['score']
                
                # Add bonus if present
                if 'bonus' in rule_data:
                    score += rule_data['bonus']
                
                scores.append(min(1.0, score))  # Cap at 1.0
                
                # Weight critical rules higher
                if any(critical in rule_name.lower() for critical in 
                       ['retracement', 'overlap', 'shortest']):
                    weights.append(2.0)
                else:
                    weights.append(1.0)
        
        if not scores:
            return 0.0
        
        # Weighted average
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        return min(1.0, weighted_score)
    
    def _create_failed_structure(self, wave_points: List[WavePoint], 
                               error_msg: str) -> WaveStructure:
        """Create a failed validation structure"""
        return WaveStructure(
            main_waves=[],
            subwaves={},
            wave_type=WaveType.IMPULSE,
            direction=WaveDirection.BULLISH,
            validation_score=0.0,
            fibonacci_levels={},
            rule_compliance={},
            recommendations=[],
            issues=[f"Validation failed: {error_msg}"]
        )
    
    def generate_detailed_report(self, structure: WaveStructure) -> str:
        """Generate detailed validation report"""
        report = []
        report.append("=" * 60)
        report.append("COMPREHENSIVE ELLIOTT WAVE VALIDATION REPORT")
        report.append("=" * 60)
        
        # Overall score
        score_pct = structure.validation_score * 100
        report.append(f"\n🎯 OVERALL VALIDATION SCORE: {score_pct:.1f}%")
        
        if score_pct >= 80:
            report.append("✅ EXCELLENT - Strong Elliott Wave structure")
        elif score_pct >= 60:
            report.append("⚠️ GOOD - Acceptable Elliott Wave structure")
        else:
            report.append("❌ QUESTIONABLE - Weak Elliott Wave structure")
        
        # Main waves summary
        report.append(f"\n📊 WAVE STRUCTURE: {structure.wave_type.value.upper()}")
        report.append(f"📈 DIRECTION: {structure.direction.value.upper()}")
        report.append(f"🌊 MAIN WAVES: {len(structure.main_waves)}")
        
        # Main waves details
        report.append("\n" + "─" * 40)
        report.append("MAIN WAVES ANALYSIS")
        report.append("─" * 40)
        
        for wave in structure.main_waves:
            report.append(f"\nWave {wave.label}:")
            report.append(f"  Type: {wave.wave_type.value}")
            report.append(f"  Length: {wave.length:.4f}")
            report.append(f"  Duration: {wave.duration} periods")
            report.append(f"  Direction: {wave.direction.value}")
            
            # Subwaves
            if wave.label in structure.subwaves:
                subwaves = structure.subwaves[wave.label]
                if subwaves:
                    report.append(f"  Subwaves: {', '.join([sw.label for sw in subwaves])}")
        
        # Rule compliance
        report.append("\n" + "─" * 40)
        report.append("RULE COMPLIANCE DETAILS")
        report.append("─" * 40)
        
        for rule_name, rule_data in structure.rule_compliance.items():
            if isinstance(rule_data, dict):
                score = rule_data.get('score', 0)
                status = rule_data.get('status', 'unknown')
                
                icon = "✅" if score > 0.8 else "⚠️" if score > 0.5 else "❌"
                report.append(f"\n{icon} {rule_name.replace('_', ' ').title()}")
                report.append(f"   Score: {score:.2f}")
                report.append(f"   Status: {status}")
                
                if 'description' in rule_data:
                    report.append(f"   Details: {rule_data['description']}")
        
        # Fibonacci levels
        if structure.fibonacci_levels:
            report.append("\n" + "─" * 40)
            report.append("FIBONACCI RELATIONSHIPS")
            report.append("─" * 40)
            
            for level_name, value in structure.fibonacci_levels.items():
                if isinstance(value, (int, float)):
                    report.append(f"{level_name.replace('_', ' ').title()}: {value:.3f}")
        
        # Issues and recommendations
        if structure.issues:
            report.append("\n" + "─" * 40)
            report.append("⚠️ ISSUES IDENTIFIED")
            report.append("─" * 40)
            for issue in structure.issues:
                report.append(f"• {issue}")
        
        if structure.recommendations:
            report.append("\n" + "─" * 40)
            report.append("💡 RECOMMENDATIONS")
            report.append("─" * 40)
            for rec in structure.recommendations:
                report.append(f"• {rec}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
