"""
Strict Elliott Wave rule validation module.
Enforces all Elliott Wave Theory rules and guidelines.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from ..utils.logger import get_logger
from ..utils.config import get_config
from .wave_detector import Wave, WaveType, WaveDegree, TrendDirection, WavePoint

logger = get_logger(__name__)


class ValidationRule(Enum):
    """Elliott Wave validation rules."""
    # Impulse wave rules
    IMPULSE_WAVE_2_RETRACE = "Wave 2 must not retrace past Wave 1 start"
    IMPULSE_WAVE_3_LENGTH = "Wave 3 cannot be the shortest among 1, 3, 5"
    IMPULSE_WAVE_4_OVERLAP = "Wave 4 must not overlap Wave 1 territory"
    IMPULSE_SUBWAVES = "Wave 1, 3, 5 must subdivide into 5 smaller waves"
    
    # Fibonacci rules
    FIBONACCI_WAVE_2_RETRACE = "Wave 2 should retrace 50-61.8% of Wave 1"
    FIBONACCI_WAVE_3_EXTENSION = "Wave 3 should extend 161.8% of Wave 1"
    FIBONACCI_WAVE_5_LENGTH = "Wave 5 should be 61.8-100% of Wave 1 or 3"
    
    # Corrective wave rules
    CORRECTIVE_WAVE_B_RETRACE = "Wave B should not exceed 100% of Wave A"
    CORRECTIVE_WAVE_C_LENGTH = "Wave C should equal or extend Wave A"
    CORRECTIVE_SUBWAVES = "Corrective waves should subdivide into 3 waves"
    
    # Time symmetry rules
    TIME_SYMMETRY_WAVE_2_4 = "Wave 2 and 4 should have similar duration"
    TIME_SYMMETRY_WAVE_1_3_5 = "Wave 1, 3, 5 should show time symmetry"
    
    # Diagonal rules
    DIAGONAL_WAVE_4_OVERLAP = "Wave 4 must overlap Wave 1 in diagonals"
    DIAGONAL_SUBWAVES = "Diagonal subwaves should be 3s or 5s"
    DIAGONAL_WEDGE_SHAPE = "Diagonal should form wedge shape"


class ValidationSeverity(Enum):
    """Validation rule severity levels."""
    CRITICAL = "critical"      # Must pass for valid pattern
    WARNING = "warning"        # Should pass, but not required
    INFO = "info"             # Informational only


@dataclass
class ValidationResult:
    """Result of a single validation rule check."""
    rule: ValidationRule
    severity: ValidationSeverity
    passed: bool
    actual_value: Optional[float] = None
    expected_range: Optional[Tuple[float, float]] = None
    message: str = ""
    confidence_impact: float = 0.0


@dataclass
class WaveValidation:
    """Complete validation result for a wave or pattern."""
    wave: Wave
    pattern_type: str
    validation_results: List[ValidationResult]
    overall_score: float
    passed_critical_rules: bool
    warnings: List[str]
    recommendations: List[str]


class WaveValidator:
    """
    Strict Elliott Wave rule validator.
    Enforces all Elliott Wave Theory rules and guidelines.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize WaveValidator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.fibonacci_tolerance = self.config.get('validation.fibonacci_tolerance', 0.1)
        self.time_symmetry_tolerance = self.config.get('validation.time_symmetry_tolerance', 0.2)
        self.overlap_tolerance = self.config.get('validation.overlap_tolerance', 0.01)
        
        logger.info("WaveValidator initialized")
    
    def validate_wave_pattern(self, waves: List[Wave], data: pd.DataFrame) -> WaveValidation:
        """
        Validate a complete Elliott Wave pattern.
        
        Args:
            waves: List of waves forming the pattern
            data: OHLCV DataFrame
            
        Returns:
            WaveValidation object with detailed results
        """
        if not waves:
            return self._create_empty_validation()
        
        try:
            # Determine pattern type
            pattern_type = self._determine_pattern_type(waves)
            
            # Validate based on pattern type
            if pattern_type == "impulse":
                validation_results = self._validate_impulse_pattern(waves, data)
            elif pattern_type == "corrective":
                validation_results = self._validate_corrective_pattern(waves, data)
            elif pattern_type == "diagonal":
                validation_results = self._validate_diagonal_pattern(waves, data)
            elif pattern_type == "triangle":
                validation_results = self._validate_triangle_pattern(waves, data)
            else:
                validation_results = self._validate_general_pattern(waves, data)
            
            # Calculate overall score
            overall_score = self._calculate_validation_score(validation_results)
            
            # Check critical rules
            passed_critical = all(
                result.passed for result in validation_results 
                if result.severity == ValidationSeverity.CRITICAL
            )
            
            # Generate warnings and recommendations
            warnings, recommendations = self._generate_feedback(validation_results)
            
            return WaveValidation(
                wave=waves[0],  # Use first wave as representative
                pattern_type=pattern_type,
                validation_results=validation_results,
                overall_score=overall_score,
                passed_critical_rules=passed_critical,
                warnings=warnings,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error validating wave pattern: {e}")
            return self._create_empty_validation()
    
    def _validate_impulse_pattern(self, waves: List[Wave], data: pd.DataFrame) -> List[ValidationResult]:
        """Validate impulse wave pattern (1-2-3-4-5)."""
        results = []
        
        if len(waves) < 5:
            results.append(ValidationResult(
                rule=ValidationRule.IMPULSE_SUBWAVES,
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                message=f"Impulse pattern requires 5 waves, found {len(waves)}"
            ))
            return results
        
        # Validate wave 2 retracement
        if len(waves) >= 2:
            wave1_length = abs(waves[0].price_change)
            wave2_retrace = abs(waves[1].price_change) / wave1_length
            
            results.append(ValidationResult(
                rule=ValidationRule.IMPULSE_WAVE_2_RETRACE,
                severity=ValidationSeverity.CRITICAL,
                passed=wave2_retrace < 1.0,
                actual_value=wave2_retrace,
                expected_range=(0.0, 1.0),
                message=f"Wave 2 retraces {wave2_retrace:.1%} of Wave 1"
            ))
            
            # Fibonacci validation for wave 2
            results.append(ValidationResult(
                rule=ValidationRule.FIBONACCI_WAVE_2_RETRACE,
                severity=ValidationSeverity.WARNING,
                passed=0.5 <= wave2_retrace <= 0.618,
                actual_value=wave2_retrace,
                expected_range=(0.5, 0.618),
                message=f"Wave 2 retracement should be 50-61.8%, actual: {wave2_retrace:.1%}"
            ))
        
        # Validate wave 3 length
        if len(waves) >= 3:
            wave_lengths = [abs(w.price_change) for w in waves[:3]]
            wave3_is_longest = wave_lengths[2] == max(wave_lengths)
            
            results.append(ValidationResult(
                rule=ValidationRule.IMPULSE_WAVE_3_LENGTH,
                severity=ValidationSeverity.CRITICAL,
                passed=wave3_is_longest,
                actual_value=wave_lengths[2],
                message=f"Wave 3 length: {wave_lengths[2]:.2f}, should be longest"
            ))
            
            # Fibonacci validation for wave 3
            wave1_length = wave_lengths[0]
            wave3_extension = wave_lengths[2] / wave1_length
            
            results.append(ValidationResult(
                rule=ValidationRule.FIBONACCI_WAVE_3_EXTENSION,
                severity=ValidationSeverity.WARNING,
                passed=wave3_extension >= 1.618,
                actual_value=wave3_extension,
                expected_range=(1.618, float('inf')),
                message=f"Wave 3 should extend 161.8% of Wave 1, actual: {wave3_extension:.1%}"
            ))
        
        # Validate wave 4 overlap
        if len(waves) >= 4:
            wave1_end = waves[0].end_point.price
            wave4_low = min(waves[3].start_point.price, waves[3].end_point.price)
            wave4_high = max(waves[3].start_point.price, waves[3].end_point.price)
            
            # Check if wave 4 overlaps wave 1 territory
            if waves[0].direction == TrendDirection.UP:
                overlap = wave4_low < wave1_end
            else:
                overlap = wave4_high > wave1_end
            
            results.append(ValidationResult(
                rule=ValidationRule.IMPULSE_WAVE_4_OVERLAP,
                severity=ValidationSeverity.CRITICAL,
                passed=not overlap,
                actual_value=1.0 if overlap else 0.0,
                expected_range=(0.0, 0.0),
                message=f"Wave 4 {'overlaps' if overlap else 'does not overlap'} Wave 1 territory"
            ))
        
        # Validate wave 5 length
        if len(waves) >= 5:
            wave1_length = abs(waves[0].price_change)
            wave5_length = abs(waves[4].price_change)
            wave5_ratio = wave5_length / wave1_length
            
            results.append(ValidationResult(
                rule=ValidationRule.FIBONACCI_WAVE_5_LENGTH,
                severity=ValidationSeverity.WARNING,
                passed=0.618 <= wave5_ratio <= 1.0,
                actual_value=wave5_ratio,
                expected_range=(0.618, 1.0),
                message=f"Wave 5 should be 61.8-100% of Wave 1, actual: {wave5_ratio:.1%}"
            ))
        
        # Validate subwaves
        results.extend(self._validate_impulse_subwaves(waves, data))
        
        # Validate time symmetry
        results.extend(self._validate_time_symmetry(waves))
        
        return results
    
    def _validate_corrective_pattern(self, waves: List[Wave], data: pd.DataFrame) -> List[ValidationResult]:
        """Validate corrective wave pattern (A-B-C)."""
        results = []
        
        if len(waves) < 3:
            results.append(ValidationResult(
                rule=ValidationRule.CORRECTIVE_SUBWAVES,
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                message=f"Corrective pattern requires 3 waves, found {len(waves)}"
            ))
            return results
        
        # Validate wave B retracement
        if len(waves) >= 2:
            waveA_length = abs(waves[0].price_change)
            waveB_retrace = abs(waves[1].price_change) / waveA_length
            
            results.append(ValidationResult(
                rule=ValidationRule.CORRECTIVE_WAVE_B_RETRACE,
                severity=ValidationSeverity.CRITICAL,
                passed=waveB_retrace <= 1.0,
                actual_value=waveB_retrace,
                expected_range=(0.0, 1.0),
                message=f"Wave B retraces {waveB_retrace:.1%} of Wave A"
            ))
        
        # Validate wave C length
        if len(waves) >= 3:
            waveA_length = abs(waves[0].price_change)
            waveC_length = abs(waves[2].price_change)
            waveC_ratio = waveC_length / waveA_length
            
            results.append(ValidationResult(
                rule=ValidationRule.CORRECTIVE_WAVE_C_LENGTH,
                severity=ValidationSeverity.WARNING,
                passed=0.8 <= waveC_ratio <= 1.618,
                actual_value=waveC_ratio,
                expected_range=(0.8, 1.618),
                message=f"Wave C should equal or extend Wave A, ratio: {waveC_ratio:.1%}"
            ))
        
        # Validate subwaves
        results.extend(self._validate_corrective_subwaves(waves, data))
        
        return results
    
    def _validate_diagonal_pattern(self, waves: List[Wave], data: pd.DataFrame) -> List[ValidationResult]:
        """Validate diagonal pattern."""
        results = []
        
        if len(waves) < 5:
            results.append(ValidationResult(
                rule=ValidationRule.DIAGONAL_SUBWAVES,
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                message=f"Diagonal pattern requires 5 waves, found {len(waves)}"
            ))
            return results
        
        # Validate wave 4 overlap with wave 1
        if len(waves) >= 4:
            wave1_end = waves[0].end_point.price
            wave4_low = min(waves[3].start_point.price, waves[3].end_point.price)
            wave4_high = max(waves[3].start_point.price, waves[3].end_point.price)
            
            if waves[0].direction == TrendDirection.UP:
                overlap = wave4_low < wave1_end
            else:
                overlap = wave4_high > wave1_end
            
            results.append(ValidationResult(
                rule=ValidationRule.DIAGONAL_WAVE_4_OVERLAP,
                severity=ValidationSeverity.CRITICAL,
                passed=overlap,
                actual_value=1.0 if overlap else 0.0,
                expected_range=(1.0, 1.0),
                message=f"Wave 4 {'overlaps' if overlap else 'does not overlap'} Wave 1 (required for diagonal)"
            ))
        
        # Validate wedge shape
        results.append(ValidationResult(
            rule=ValidationRule.DIAGONAL_WEDGE_SHAPE,
            severity=ValidationSeverity.WARNING,
            passed=self._check_wedge_shape(waves),
            message="Diagonal should form wedge shape"
        ))
        
        return results
    
    def _validate_triangle_pattern(self, waves: List[Wave], data: pd.DataFrame) -> List[ValidationResult]:
        """Validate triangle pattern (A-B-C-D-E)."""
        results = []
        
        if len(waves) < 5:
            results.append(ValidationResult(
                rule=ValidationRule.CORRECTIVE_SUBWAVES,
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                message=f"Triangle pattern requires 5 waves, found {len(waves)}"
            ))
            return results
        
        # Validate triangle shape (converging or expanding)
        is_contracting = self._check_triangle_contracting(waves)
        is_expanding = self._check_triangle_expanding(waves)
        
        results.append(ValidationResult(
            rule=ValidationRule.DIAGONAL_WEDGE_SHAPE,
            severity=ValidationSeverity.WARNING,
            passed=is_contracting or is_expanding,
            message=f"Triangle is {'contracting' if is_contracting else 'expanding' if is_expanding else 'not properly shaped'}"
        ))
        
        return results
    
    def _validate_general_pattern(self, waves: List[Wave], data: pd.DataFrame) -> List[ValidationResult]:
        """Validate general wave pattern."""
        results = []
        
        # Basic wave count validation
        if len(waves) < 3:
            results.append(ValidationResult(
                rule=ValidationRule.IMPULSE_SUBWAVES,
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                message=f"Pattern requires at least 3 waves, found {len(waves)}"
            ))
        
        # Time symmetry validation
        results.extend(self._validate_time_symmetry(waves))
        
        return results
    
    def _validate_impulse_subwaves(self, waves: List[Wave], data: pd.DataFrame) -> List[ValidationResult]:
        """Validate that impulse waves subdivide into 5 smaller waves."""
        results = []
        
        for i, wave in enumerate(waves):
            if wave.wave_type in [WaveType.IMPULSE_1, WaveType.IMPULSE_3, WaveType.IMPULSE_5]:
                # Check if wave has sufficient subwaves (simplified check)
                wave_duration = wave.duration
                if wave_duration < 5:  # Minimum duration for 5 subwaves
                    results.append(ValidationResult(
                        rule=ValidationRule.IMPULSE_SUBWAVES,
                        severity=ValidationSeverity.WARNING,
                        passed=False,
                        actual_value=wave_duration,
                        expected_range=(5, float('inf')),
                        message=f"Wave {i+1} duration ({wave_duration}) may be too short for 5 subwaves"
                    ))
        
        return results
    
    def _validate_corrective_subwaves(self, waves: List[Wave], data: pd.DataFrame) -> List[ValidationResult]:
        """Validate that corrective waves subdivide into 3 smaller waves."""
        results = []
        
        for i, wave in enumerate(waves):
            if wave.wave_type in [WaveType.CORRECTIVE_A, WaveType.CORRECTIVE_B, WaveType.CORRECTIVE_C]:
                # Check if wave has sufficient subwaves (simplified check)
                wave_duration = wave.duration
                if wave_duration < 3:  # Minimum duration for 3 subwaves
                    results.append(ValidationResult(
                        rule=ValidationRule.CORRECTIVE_SUBWAVES,
                        severity=ValidationSeverity.WARNING,
                        passed=False,
                        actual_value=wave_duration,
                        expected_range=(3, float('inf')),
                        message=f"Wave {chr(65+i)} duration ({wave_duration}) may be too short for 3 subwaves"
                    ))
        
        return results
    
    def _validate_time_symmetry(self, waves: List[Wave]) -> List[ValidationResult]:
        """Validate time symmetry between waves."""
        results = []
        
        if len(waves) >= 4:
            # Check wave 2 vs wave 4 duration
            wave2_duration = waves[1].duration
            wave4_duration = waves[3].duration
            duration_ratio = min(wave2_duration, wave4_duration) / max(wave2_duration, wave4_duration)
            
            results.append(ValidationResult(
                rule=ValidationRule.TIME_SYMMETRY_WAVE_2_4,
                severity=ValidationSeverity.WARNING,
                passed=duration_ratio >= (1 - self.time_symmetry_tolerance),
                actual_value=duration_ratio,
                expected_range=(0.8, 1.0),
                message=f"Wave 2/4 duration ratio: {duration_ratio:.2f}"
            ))
        
        if len(waves) >= 5:
            # Check wave 1, 3, 5 duration symmetry
            durations = [w.duration for w in waves[:5]]
            avg_duration = np.mean(durations)
            duration_variance = np.var(durations) / avg_duration
            
            results.append(ValidationResult(
                rule=ValidationRule.TIME_SYMMETRY_WAVE_1_3_5,
                severity=ValidationSeverity.INFO,
                passed=duration_variance <= self.time_symmetry_tolerance,
                actual_value=duration_variance,
                expected_range=(0.0, self.time_symmetry_tolerance),
                message=f"Wave 1-3-5 duration variance: {duration_variance:.2f}"
            ))
        
        return results
    
    def _check_wedge_shape(self, waves: List[Wave]) -> bool:
        """Check if waves form a wedge shape."""
        if len(waves) < 3:
            return False
        
        # Simple wedge check: check if highs/lows are converging or diverging
        highs = [max(w.start_point.price, w.end_point.price) for w in waves]
        lows = [min(w.start_point.price, w.end_point.price) for w in waves]
        
        # Check if highs are converging/diverging
        high_trend = np.polyfit(range(len(highs)), highs, 1)[0]
        low_trend = np.polyfit(range(len(lows)), lows, 1)[0]
        
        return abs(high_trend) > 0.001 or abs(low_trend) > 0.001
    
    def _check_triangle_contracting(self, waves: List[Wave]) -> bool:
        """Check if triangle is contracting."""
        if len(waves) < 3:
            return False
        
        highs = [max(w.start_point.price, w.end_point.price) for w in waves]
        lows = [min(w.start_point.price, w.end_point.price) for w in waves]
        
        # Check if range is decreasing
        ranges = [h - l for h, l in zip(highs, lows)]
        return all(ranges[i] > ranges[i+1] for i in range(len(ranges)-1))
    
    def _check_triangle_expanding(self, waves: List[Wave]) -> bool:
        """Check if triangle is expanding."""
        if len(waves) < 3:
            return False
        
        highs = [max(w.start_point.price, w.end_point.price) for w in waves]
        lows = [min(w.start_point.price, w.end_point.price) for w in waves]
        
        # Check if range is increasing
        ranges = [h - l for h, l in zip(highs, lows)]
        return all(ranges[i] < ranges[i+1] for i in range(len(ranges)-1))
    
    def _determine_pattern_type(self, waves: List[Wave]) -> str:
        """Determine the type of Elliott Wave pattern."""
        if len(waves) == 5:
            wave_types = [w.wave_type for w in waves]
            if all(wt in [WaveType.IMPULSE_1, WaveType.IMPULSE_2, WaveType.IMPULSE_3, WaveType.IMPULSE_4, WaveType.IMPULSE_5] for wt in wave_types):
                return "impulse"
            elif all(wt in [WaveType.CORRECTIVE_A, WaveType.CORRECTIVE_B, WaveType.CORRECTIVE_C] for wt in wave_types):
                return "corrective"
        
        elif len(waves) == 3:
            wave_types = [w.wave_type for w in waves]
            if all(wt in [WaveType.CORRECTIVE_A, WaveType.CORRECTIVE_B, WaveType.CORRECTIVE_C] for wt in wave_types):
                return "corrective"
        
        # Check for diagonal characteristics
        if len(waves) >= 3 and self._check_wedge_shape(waves):
            return "diagonal"
        
        # Check for triangle characteristics
        if len(waves) >= 5 and (self._check_triangle_contracting(waves) or self._check_triangle_expanding(waves)):
            return "triangle"
        
        return "unknown"
    
    def _calculate_validation_score(self, results: List[ValidationResult]) -> float:
        """Calculate overall validation score."""
        if not results:
            return 0.0
        
        critical_weight = 0.5
        warning_weight = 0.3
        info_weight = 0.2
        
        critical_results = [r for r in results if r.severity == ValidationSeverity.CRITICAL]
        warning_results = [r for r in results if r.severity == ValidationSeverity.WARNING]
        info_results = [r for r in results if r.severity == ValidationSeverity.INFO]
        
        critical_score = np.mean([r.passed for r in critical_results]) if critical_results else 1.0
        warning_score = np.mean([r.passed for r in warning_results]) if warning_results else 1.0
        info_score = np.mean([r.passed for r in info_results]) if info_results else 1.0
        
        overall_score = (
            critical_score * critical_weight +
            warning_score * warning_weight +
            info_score * info_weight
        )
        
        return overall_score
    
    def _generate_feedback(self, results: List[ValidationResult]) -> Tuple[List[str], List[str]]:
        """Generate warnings and recommendations from validation results."""
        warnings = []
        recommendations = []
        
        for result in results:
            if not result.passed:
                if result.severity == ValidationSeverity.CRITICAL:
                    warnings.append(f"CRITICAL: {result.message}")
                elif result.severity == ValidationSeverity.WARNING:
                    warnings.append(f"WARNING: {result.message}")
                
                # Generate recommendations
                if result.rule == ValidationRule.IMPULSE_WAVE_2_RETRACE:
                    recommendations.append("Consider adjusting Wave 2 to not retrace past Wave 1 start")
                elif result.rule == ValidationRule.IMPULSE_WAVE_3_LENGTH:
                    recommendations.append("Wave 3 should be the longest wave in the impulse")
                elif result.rule == ValidationRule.IMPULSE_WAVE_4_OVERLAP:
                    recommendations.append("Wave 4 should not overlap Wave 1 territory")
                elif result.rule == ValidationRule.FIBONACCI_WAVE_2_RETRACE:
                    recommendations.append("Wave 2 typically retraces 50-61.8% of Wave 1")
                elif result.rule == ValidationRule.FIBONACCI_WAVE_3_EXTENSION:
                    recommendations.append("Wave 3 typically extends 161.8% of Wave 1")
        
        return warnings, recommendations
    
    def _create_empty_validation(self) -> WaveValidation:
        """Create empty validation result."""
        return WaveValidation(
            wave=Wave(
                start_point=WavePoint(pd.Timestamp.now(), 0.0, 0, WaveType.UNKNOWN),
                end_point=WavePoint(pd.Timestamp.now(), 0.0, 0, WaveType.UNKNOWN),
                wave_type=WaveType.UNKNOWN,
                direction=TrendDirection.SIDEWAYS,
                confidence=0.0
            ),
            pattern_type="unknown",
            validation_results=[],
            overall_score=0.0,
            passed_critical_rules=False,
            warnings=["No waves to validate"],
            recommendations=["Provide valid wave data"]
        ) 