"""
Elliott Wave Rule Validator
Implements strict Elliott Wave theory rules and validation logic.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass
import logging

from .wave_detector import Wave, WavePoint, WaveType, TrendDirection, WaveDegree
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ValidationResult(Enum):
    """Validation result types."""
    VALID = "VALID"
    INVALID = "INVALID"
    WARNING = "WARNING"


@dataclass
class RuleViolation:
    """Represents a rule violation."""
    rule_name: str
    description: str
    severity: ValidationResult
    violation_data: Dict[str, Any]


@dataclass
class WaveStructure:
    """Represents a complete wave structure (5-wave impulse or 3-wave corrective)."""
    waves: List[Wave]
    structure_type: str  # "IMPULSE" or "CORRECTIVE"
    validation_score: float
    violations: List[RuleViolation]
    fibonacci_scores: Dict[str, float]


class ElliottWaveValidator:
    """
    Comprehensive Elliott Wave rule validator implementing all core Elliott Wave principles.
    """
    
    def __init__(self):
        """Initialize the validator."""
        self.core_rules = {
            'wave_2_retracement': self._validate_wave_2_retracement,
            'wave_3_not_shortest': self._validate_wave_3_not_shortest,
            'wave_4_no_overlap': self._validate_wave_4_no_overlap,
            'fibonacci_ratios': self._validate_fibonacci_ratios,
            'alternation': self._validate_alternation,
            'momentum_divergence': self._validate_momentum_divergence,
            'volume_confirmation': self._validate_volume_confirmation
        }
        
        # Fibonacci ratio tolerances
        self.fib_tolerance = 0.05  # 5% tolerance for Fibonacci matches
        
        # Standard Fibonacci levels
        self.retracement_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        self.extension_levels = [1.0, 1.272, 1.618, 2.618, 4.236]
    
    def validate_impulse_structure(self, waves: List[Wave], data: pd.DataFrame) -> WaveStructure:
        """
        Validate a 5-wave impulse structure according to Elliott Wave rules.
        
        Args:
            waves: List of 5 waves forming potential impulse
            data: OHLCV DataFrame
            
        Returns:
            WaveStructure with validation results
        """
        if len(waves) != 5:
            return WaveStructure(
                waves=waves,
                structure_type="IMPULSE",
                validation_score=0.0,
                violations=[RuleViolation(
                    rule_name="structure_length",
                    description=f"Impulse must have exactly 5 waves, got {len(waves)}",
                    severity=ValidationResult.INVALID,
                    violation_data={"expected": 5, "actual": len(waves)}
                )],
                fibonacci_scores={}
            )
        
        violations = []
        fibonacci_scores = {}
        
        # Extract wave points for easier analysis
        points = [waves[0].start_point] + [w.end_point for w in waves]
        
        # Core Rule 1: Wave 2 cannot retrace more than 100% of Wave 1
        wave_2_violation = self._validate_wave_2_retracement(waves, points, data)
        if wave_2_violation:
            violations.append(wave_2_violation)
        
        # Core Rule 2: Wave 3 cannot be the shortest among waves 1, 3, and 5
        wave_3_violation = self._validate_wave_3_not_shortest(waves, points, data)
        if wave_3_violation:
            violations.append(wave_3_violation)
        
        # Core Rule 3: Wave 4 cannot overlap with Wave 1
        wave_4_violation = self._validate_wave_4_no_overlap(waves, points, data)
        if wave_4_violation:
            violations.append(wave_4_violation)
        
        # Additional validations
        fib_violations, fib_scores = self._validate_fibonacci_ratios(waves, points, data)
        violations.extend(fib_violations)
        fibonacci_scores.update(fib_scores)
        
        # Alternation principle
        alt_violation = self._validate_alternation(waves, points, data)
        if alt_violation:
            violations.append(alt_violation)
        
        # Calculate overall validation score
        validation_score = self._calculate_validation_score(violations, fibonacci_scores)
        
        return WaveStructure(
            waves=waves,
            structure_type="IMPULSE",
            validation_score=validation_score,
            violations=violations,
            fibonacci_scores=fibonacci_scores
        )
    
    def validate_corrective_structure(self, waves: List[Wave], data: pd.DataFrame) -> WaveStructure:
        """
        Validate a 3-wave corrective structure (ABC).
        
        Args:
            waves: List of 3 waves forming potential correction
            data: OHLCV DataFrame
            
        Returns:
            WaveStructure with validation results
        """
        if len(waves) != 3:
            return WaveStructure(
                waves=waves,
                structure_type="CORRECTIVE",
                validation_score=0.0,
                violations=[RuleViolation(
                    rule_name="structure_length",
                    description=f"Corrective structure must have exactly 3 waves, got {len(waves)}",
                    severity=ValidationResult.INVALID,
                    violation_data={"expected": 3, "actual": len(waves)}
                )],
                fibonacci_scores={}
            )
        
        violations = []
        fibonacci_scores = {}
        
        # Extract wave points
        points = [waves[0].start_point] + [w.end_point for w in waves]
        
        # Validate corrective pattern rules
        corrective_violations, corrective_scores = self._validate_corrective_pattern_rules(waves, points, data)
        violations.extend(corrective_violations)
        fibonacci_scores.update(corrective_scores)
        
        # Calculate validation score
        validation_score = self._calculate_validation_score(violations, fibonacci_scores)
        
        return WaveStructure(
            waves=waves,
            structure_type="CORRECTIVE",
            validation_score=validation_score,
            violations=violations,
            fibonacci_scores=fibonacci_scores
        )
    
    def _validate_wave_2_retracement(self, waves: List[Wave], points: List[WavePoint], data: pd.DataFrame) -> Optional[RuleViolation]:
        """
        Core Rule 1: Wave 2 cannot retrace more than 100% of Wave 1.
        Formula: Wave2.Low > Wave1.Start (in bullish trend)
        """
        try:
            wave_1 = waves[0]
            wave_2 = waves[1]
            
            # Determine trend direction
            if wave_1.direction == TrendDirection.UP:
                # Bullish trend: Wave 2 low must be above Wave 1 start
                wave_1_start = points[0].price
                wave_2_end = points[2].price
                
                if wave_2_end <= wave_1_start:
                    retracement_pct = abs(wave_2_end - points[1].price) / abs(points[1].price - wave_1_start) * 100
                    return RuleViolation(
                        rule_name="wave_2_retracement",
                        description=f"Wave 2 retraces {retracement_pct:.1f}% of Wave 1 (>100% invalid)",
                        severity=ValidationResult.INVALID,
                        violation_data={
                            "wave_1_start": wave_1_start,
                            "wave_1_end": points[1].price,
                            "wave_2_end": wave_2_end,
                            "retracement_percent": retracement_pct
                        }
                    )
            else:
                # Bearish trend: Wave 2 high must be below Wave 1 start
                wave_1_start = points[0].price
                wave_2_end = points[2].price
                
                if wave_2_end >= wave_1_start:
                    retracement_pct = abs(wave_2_end - points[1].price) / abs(points[1].price - wave_1_start) * 100
                    return RuleViolation(
                        rule_name="wave_2_retracement",
                        description=f"Wave 2 retraces {retracement_pct:.1f}% of Wave 1 (>100% invalid)",
                        severity=ValidationResult.INVALID,
                        violation_data={
                            "wave_1_start": wave_1_start,
                            "wave_1_end": points[1].price,
                            "wave_2_end": wave_2_end,
                            "retracement_percent": retracement_pct
                        }
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error validating Wave 2 retracement: {e}")
            return None
    
    def _validate_wave_3_not_shortest(self, waves: List[Wave], points: List[WavePoint], data: pd.DataFrame) -> Optional[RuleViolation]:
        """
        Core Rule 2: Wave 3 cannot be the shortest among Waves 1, 3, and 5.
        Formula: Length(Wave3) > min(Length(Wave1), Length(Wave5))
        """
        try:
            wave_1_length = abs(points[1].price - points[0].price)
            wave_3_length = abs(points[3].price - points[2].price)
            wave_5_length = abs(points[5].price - points[4].price)
            
            min_length = min(wave_1_length, wave_5_length)
            
            if wave_3_length <= min_length:
                return RuleViolation(
                    rule_name="wave_3_not_shortest",
                    description=f"Wave 3 ({wave_3_length:.2f}) is shortest among waves 1,3,5",
                    severity=ValidationResult.INVALID,
                    violation_data={
                        "wave_1_length": wave_1_length,
                        "wave_3_length": wave_3_length,
                        "wave_5_length": wave_5_length,
                        "min_length": min_length
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error validating Wave 3 length: {e}")
            return None
    
    def _validate_wave_4_no_overlap(self, waves: List[Wave], points: List[WavePoint], data: pd.DataFrame) -> Optional[RuleViolation]:
        """
        Core Rule 3: Wave 4 cannot enter the price territory of Wave 1.
        Formula: Wave4.Low > Wave1.High (in bullish trend)
        """
        try:
            wave_1 = waves[0]
            wave_4 = waves[3]
            
            if wave_1.direction == TrendDirection.UP:
                # Bullish trend: Wave 4 low must be above Wave 1 high
                wave_1_high = max(points[0].price, points[1].price)
                wave_4_low = min(points[4].price, points[3].price)
                
                if wave_4_low <= wave_1_high:
                    overlap_amount = wave_1_high - wave_4_low
                    return RuleViolation(
                        rule_name="wave_4_no_overlap",
                        description=f"Wave 4 overlaps Wave 1 territory by {overlap_amount:.2f}",
                        severity=ValidationResult.INVALID,
                        violation_data={
                            "wave_1_high": wave_1_high,
                            "wave_4_low": wave_4_low,
                            "overlap_amount": overlap_amount
                        }
                    )
            else:
                # Bearish trend: Wave 4 high must be below Wave 1 low
                wave_1_low = min(points[0].price, points[1].price)
                wave_4_high = max(points[4].price, points[3].price)
                
                if wave_4_high >= wave_1_low:
                    overlap_amount = wave_4_high - wave_1_low
                    return RuleViolation(
                        rule_name="wave_4_no_overlap",
                        description=f"Wave 4 overlaps Wave 1 territory by {overlap_amount:.2f}",
                        severity=ValidationResult.INVALID,
                        violation_data={
                            "wave_1_low": wave_1_low,
                            "wave_4_high": wave_4_high,
                            "overlap_amount": overlap_amount
                        }
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error validating Wave 4 overlap: {e}")
            return None
    
    def _validate_fibonacci_ratios(self, waves: List[Wave], points: List[WavePoint], data: pd.DataFrame) -> Tuple[List[RuleViolation], Dict[str, float]]:
        """
        Validate Fibonacci ratio relationships between waves.
        """
        violations = []
        scores = {}
        
        try:
            # Wave 2 retracement (should be 50-61.8% of Wave 1)
            wave_1_length = abs(points[1].price - points[0].price)
            wave_2_length = abs(points[2].price - points[1].price)
            
            if wave_1_length > 0:
                wave_2_ratio = wave_2_length / wave_1_length
                scores['wave_2_retracement'] = self._score_fibonacci_match(wave_2_ratio, [0.382, 0.5, 0.618])
                
                if not self._is_fibonacci_match(wave_2_ratio, [0.382, 0.5, 0.618]):
                    violations.append(RuleViolation(
                        rule_name="wave_2_fibonacci",
                        description=f"Wave 2 retracement ({wave_2_ratio:.3f}) not at ideal Fibonacci level",
                        severity=ValidationResult.WARNING,
                        violation_data={"ratio": wave_2_ratio, "expected": [0.382, 0.5, 0.618]}
                    ))
            
            # Wave 3 extension (often 161.8% of Wave 1)
            wave_3_length = abs(points[3].price - points[2].price)
            if wave_1_length > 0:
                wave_3_ratio = wave_3_length / wave_1_length
                scores['wave_3_extension'] = self._score_fibonacci_match(wave_3_ratio, [1.618, 2.618])
                
                if not self._is_fibonacci_match(wave_3_ratio, [1.0, 1.618, 2.618]):
                    violations.append(RuleViolation(
                        rule_name="wave_3_fibonacci",
                        description=f"Wave 3 extension ({wave_3_ratio:.3f}) not at ideal Fibonacci level",
                        severity=ValidationResult.WARNING,
                        violation_data={"ratio": wave_3_ratio, "expected": [1.0, 1.618, 2.618]}
                    ))
            
            # Wave 4 retracement (should be 23.6-38.2% of Wave 3)
            wave_4_length = abs(points[4].price - points[3].price)
            if wave_3_length > 0:
                wave_4_ratio = wave_4_length / wave_3_length
                scores['wave_4_retracement'] = self._score_fibonacci_match(wave_4_ratio, [0.236, 0.382])
                
                if not self._is_fibonacci_match(wave_4_ratio, [0.236, 0.382]):
                    violations.append(RuleViolation(
                        rule_name="wave_4_fibonacci",
                        description=f"Wave 4 retracement ({wave_4_ratio:.3f}) not at ideal Fibonacci level",
                        severity=ValidationResult.WARNING,
                        violation_data={"ratio": wave_4_ratio, "expected": [0.236, 0.382]}
                    ))
            
            # Wave 5 projection (often 61.8% or 100% of Wave 1)
            wave_5_length = abs(points[5].price - points[4].price)
            if wave_1_length > 0:
                wave_5_ratio = wave_5_length / wave_1_length
                scores['wave_5_projection'] = self._score_fibonacci_match(wave_5_ratio, [0.618, 1.0])
                
                if not self._is_fibonacci_match(wave_5_ratio, [0.618, 1.0, 1.618]):
                    violations.append(RuleViolation(
                        rule_name="wave_5_fibonacci",
                        description=f"Wave 5 projection ({wave_5_ratio:.3f}) not at ideal Fibonacci level",
                        severity=ValidationResult.WARNING,
                        violation_data={"ratio": wave_5_ratio, "expected": [0.618, 1.0, 1.618]}
                    ))
        
        except Exception as e:
            logger.error(f"Error validating Fibonacci ratios: {e}")
        
        return violations, scores
    
    def _validate_corrective_pattern_rules(self, waves: List[Wave], points: List[WavePoint], data: pd.DataFrame) -> Tuple[List[RuleViolation], Dict[str, float]]:
        """
        Validate rules specific to corrective patterns (ABC).
        """
        violations = []
        scores = {}
        
        try:
            # Wave C often equals Wave A or is 61.8%, 100%, or 161.8% of Wave A
            wave_a_length = abs(points[1].price - points[0].price)
            wave_c_length = abs(points[3].price - points[2].price)
            
            if wave_a_length > 0:
                c_to_a_ratio = wave_c_length / wave_a_length
                scores['wave_c_projection'] = self._score_fibonacci_match(c_to_a_ratio, [0.618, 1.0, 1.618])
                
                if not self._is_fibonacci_match(c_to_a_ratio, [0.618, 1.0, 1.618]):
                    violations.append(RuleViolation(
                        rule_name="wave_c_fibonacci",
                        description=f"Wave C ({c_to_a_ratio:.3f} of A) not at ideal Fibonacci ratio",
                        severity=ValidationResult.WARNING,
                        violation_data={"ratio": c_to_a_ratio, "expected": [0.618, 1.0, 1.618]}
                    ))
            
            # Wave B should not exceed 100% of Wave A (for zigzag patterns)
            wave_b_length = abs(points[2].price - points[1].price)
            if wave_a_length > 0:
                b_to_a_ratio = wave_b_length / wave_a_length
                
                if b_to_a_ratio > 1.0:
                    violations.append(RuleViolation(
                        rule_name="wave_b_retracement",
                        description=f"Wave B ({b_to_a_ratio:.3f} of A) exceeds 100% retracement",
                        severity=ValidationResult.WARNING,
                        violation_data={"ratio": b_to_a_ratio, "limit": 1.0}
                    ))
        
        except Exception as e:
            logger.error(f"Error validating corrective pattern rules: {e}")
        
        return violations, scores
    
    def _validate_alternation(self, waves: List[Wave], points: List[WavePoint], data: pd.DataFrame) -> Optional[RuleViolation]:
        """
        Validate alternation principle: Waves 2 and 4 should alternate in character.
        """
        try:
            # This is a guideline, not a strict rule, so only generate warnings
            wave_2_duration = waves[1].duration
            wave_4_duration = waves[3].duration
            
            # Check for significant difference in duration (alternation in time)
            duration_ratio = max(wave_2_duration, wave_4_duration) / max(min(wave_2_duration, wave_4_duration), 1)
            
            if duration_ratio < 1.5:  # Less than 50% difference
                return RuleViolation(
                    rule_name="alternation",
                    description="Waves 2 and 4 lack alternation in duration",
                    severity=ValidationResult.WARNING,
                    violation_data={
                        "wave_2_duration": wave_2_duration,
                        "wave_4_duration": wave_4_duration,
                        "duration_ratio": duration_ratio
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error validating alternation: {e}")
            return None
    
    def _validate_momentum_divergence(self, waves: List[Wave], points: List[WavePoint], data: pd.DataFrame) -> Optional[RuleViolation]:
        """
        Validate momentum characteristics (Wave 3 should show strong momentum).
        """
        # This would require momentum indicators like RSI, MACD
        # For now, return None (placeholder for future implementation)
        return None
    
    def _validate_volume_confirmation(self, waves: List[Wave], points: List[WavePoint], data: pd.DataFrame) -> Optional[RuleViolation]:
        """
        Validate volume characteristics (Wave 3 should show high volume).
        """
        try:
            if 'volume' not in data.columns:
                return None
            
            wave_3 = waves[2]
            wave_3_data = data.iloc[wave_3.start_point.index:wave_3.end_point.index+1]
            
            if len(wave_3_data) == 0:
                return None
            
            wave_3_avg_volume = wave_3_data['volume'].mean()
            baseline_volume = data['volume'].rolling(50).mean().iloc[wave_3.end_point.index]
            
            volume_ratio = wave_3_avg_volume / baseline_volume if baseline_volume > 0 else 1.0
            
            if volume_ratio < 1.2:  # Wave 3 should have at least 20% higher volume
                return RuleViolation(
                    rule_name="volume_confirmation",
                    description=f"Wave 3 lacks volume confirmation (ratio: {volume_ratio:.2f})",
                    severity=ValidationResult.WARNING,
                    violation_data={
                        "wave_3_volume": wave_3_avg_volume,
                        "baseline_volume": baseline_volume,
                        "volume_ratio": volume_ratio
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error validating volume confirmation: {e}")
            return None
    
    def _is_fibonacci_match(self, ratio: float, targets: List[float]) -> bool:
        """Check if ratio matches any Fibonacci target within tolerance."""
        return any(abs(ratio - target) / target <= self.fib_tolerance for target in targets)
    
    def _score_fibonacci_match(self, ratio: float, targets: List[float]) -> float:
        """Score how well ratio matches Fibonacci targets (0-1)."""
        if not targets:
            return 0.0
        
        best_match = min(abs(ratio - target) / target for target in targets)
        return max(0.0, 1.0 - (best_match / self.fib_tolerance))
    
    def _calculate_validation_score(self, violations: List[RuleViolation], fibonacci_scores: Dict[str, float]) -> float:
        """
        Calculate overall validation score based on violations and Fibonacci matches.
        
        Returns:
            Score from 0.0 (invalid) to 1.0 (perfect)
        """
        # Start with base score
        score = 1.0
        
        # Penalize violations
        for violation in violations:
            if violation.severity == ValidationResult.INVALID:
                score -= 0.3  # Major penalty for invalid structures
            elif violation.severity == ValidationResult.WARNING:
                score -= 0.1  # Minor penalty for warnings
        
        # Bonus for Fibonacci matches
        if fibonacci_scores:
            fib_bonus = np.mean(list(fibonacci_scores.values())) * 0.2
            score += fib_bonus
        
        return max(0.0, min(1.0, score))
    
    def get_validation_summary(self, structure: WaveStructure) -> str:
        """
        Get human-readable validation summary.
        
        Args:
            structure: WaveStructure to summarize
            
        Returns:
            Formatted validation summary
        """
        summary = []
        summary.append(f"=== {structure.structure_type} WAVE VALIDATION ===")
        summary.append(f"Overall Score: {structure.validation_score:.2f}/1.00")
        
        if structure.validation_score >= 0.8:
            summary.append("✅ HIGH CONFIDENCE - Strong Elliott Wave pattern")
        elif structure.validation_score >= 0.6:
            summary.append("⚠️  MODERATE CONFIDENCE - Some rule violations")
        else:
            summary.append("❌ LOW CONFIDENCE - Significant issues detected")
        
        if structure.violations:
            summary.append(f"\n📋 RULE VIOLATIONS ({len(structure.violations)}):")
            for i, violation in enumerate(structure.violations, 1):
                severity_icon = "❌" if violation.severity == ValidationResult.INVALID else "⚠️"
                summary.append(f"{i}. {severity_icon} {violation.rule_name}: {violation.description}")
        
        if structure.fibonacci_scores:
            summary.append(f"\n📊 FIBONACCI ANALYSIS:")
            for ratio_name, score in structure.fibonacci_scores.items():
                score_icon = "✅" if score > 0.8 else "⚠️" if score > 0.5 else "❌"
                summary.append(f"   {score_icon} {ratio_name}: {score:.2f}")
        
        return "\n".join(summary)
