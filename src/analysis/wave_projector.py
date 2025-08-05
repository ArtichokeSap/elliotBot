"""
Advanced Elliott Wave projection module.
Generates future wave scenarios with likelihoods and risk/reward analysis.
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
from .wave_validator import WaveValidator, WaveValidation
from .fibonacci import FibonacciAnalysis

logger = get_logger(__name__)


class ProjectionType(Enum):
    """Types of wave projections."""
    IMPULSE_CONTINUATION = "impulse_continuation"
    COMPLEX_CORRECTION = "complex_correction"
    TRIANGLE = "triangle"
    DIAGONAL = "diagonal"
    FLAT_CORRECTION = "flat_correction"
    ZIGZAG_CORRECTION = "zigzag_correction"


class ProjectionConfidence(Enum):
    """Confidence levels for projections."""
    HIGH = "high"      # 80-100% confidence
    MEDIUM = "medium"  # 60-79% confidence
    LOW = "low"        # 40-59% confidence
    VERY_LOW = "very_low"  # < 40% confidence


@dataclass
class WaveProjection:
    """A single wave projection scenario."""
    projection_type: ProjectionType
    confidence: ProjectionConfidence
    likelihood: float
    description: str
    fibonacci_targets: List[float]
    invalidation_levels: List[float]
    time_targets: List[pd.Timestamp]
    risk_reward_ratios: List[float]
    sub_waves: List[str]
    pattern_characteristics: Dict[str, Any]


@dataclass
class ProjectionScenario:
    """Complete projection scenario with multiple paths."""
    scenario_name: str
    primary_projection: WaveProjection
    alternative_projections: List[WaveProjection]
    overall_confidence: float
    market_context: Dict[str, Any]
    recommendations: List[str]


class WaveProjector:
    """
    Advanced Elliott Wave projection engine.
    Generates future wave scenarios with likelihoods and risk analysis.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize WaveProjector.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.validator = WaveValidator(config_path)
        self.historical_pattern_weight = self.config.get('projection.historical_pattern_weight', 0.3)
        self.rule_compliance_weight = self.config.get('projection.rule_compliance_weight', 0.4)
        self.market_context_weight = self.config.get('projection.market_context_weight', 0.3)
        
        logger.info("WaveProjector initialized")
    
    def generate_comprehensive_projections(
        self, 
        current_wave: Wave, 
        all_waves: List[Wave], 
        data: pd.DataFrame,
        include_alternatives: bool = True,
        max_scenarios: int = 5
    ) -> List[ProjectionScenario]:
        """
        Generate comprehensive future wave projections.
        
        Args:
            current_wave: Current wave for projection
            all_waves: All detected waves for context
            data: OHLCV DataFrame
            include_alternatives: Whether to include alternative scenarios
            max_scenarios: Maximum number of scenarios to generate
            
        Returns:
            List of projection scenarios
        """
        scenarios = []
        
        try:
            # Generate primary projection based on current wave type
            primary_projection = self._generate_primary_projection(current_wave, all_waves, data)
            
            if primary_projection:
                # Create main scenario
                scenario = ProjectionScenario(
                    scenario_name=f"Primary {current_wave.wave_type.value} Projection",
                    primary_projection=primary_projection,
                    alternative_projections=[],
                    overall_confidence=primary_projection.likelihood,
                    market_context=self._analyze_market_context(data, current_wave),
                    recommendations=self._generate_recommendations(primary_projection, data)
                )
                
                scenarios.append(scenario)
                
                # Generate alternative scenarios if requested
                if include_alternatives:
                    alternatives = self._generate_alternative_projections(current_wave, all_waves, data, max_scenarios - 1)
                    scenario.alternative_projections = alternatives
                    
                    # Update overall confidence
                    if alternatives:
                        scenario.overall_confidence = np.mean([p.likelihood for p in [primary_projection] + alternatives])
            
            # Generate additional scenarios based on pattern types
            additional_scenarios = self._generate_pattern_based_scenarios(current_wave, all_waves, data, max_scenarios - len(scenarios))
            scenarios.extend(additional_scenarios)
            
            return scenarios[:max_scenarios]
            
        except Exception as e:
            logger.error(f"Error generating comprehensive projections: {e}")
            return []
    
    def _generate_primary_projection(self, current_wave: Wave, all_waves: List[Wave], data: pd.DataFrame) -> Optional[WaveProjection]:
        """Generate primary projection based on current wave type."""
        try:
            if current_wave.wave_type in [WaveType.IMPULSE_1, WaveType.IMPULSE_2, WaveType.IMPULSE_3, WaveType.IMPULSE_4]:
                return self._project_impulse_continuation(current_wave, all_waves, data)
            elif current_wave.wave_type == WaveType.IMPULSE_5:
                return self._project_correction_after_impulse(current_wave, all_waves, data)
            elif current_wave.wave_type in [WaveType.CORRECTIVE_A, WaveType.CORRECTIVE_B]:
                return self._project_correction_continuation(current_wave, all_waves, data)
            elif current_wave.wave_type == WaveType.CORRECTIVE_C:
                return self._project_new_impulse_cycle(current_wave, all_waves, data)
            else:
                return self._project_general_continuation(current_wave, all_waves, data)
                
        except Exception as e:
            logger.error(f"Error generating primary projection: {e}")
            return None
    
    def _project_impulse_continuation(self, current_wave: Wave, all_waves: List[Wave], data: pd.DataFrame) -> WaveProjection:
        """Project impulse wave continuation."""
        next_wave_type = self._get_next_impulse_wave(current_wave.wave_type)
        
        # Calculate Fibonacci targets based on current wave
        wave1_length = abs(current_wave.price_change)
        current_price = current_wave.end_point.price
        
        if next_wave_type == WaveType.IMPULSE_3:
            # Wave 3 is often the longest and strongest
            targets = [
                current_price + (wave1_length * 1.618),  # 161.8% extension
                current_price + (wave1_length * 2.618),  # 261.8% extension
                current_price + (wave1_length * 4.236)   # 423.6% extension
            ]
        elif next_wave_type == WaveType.IMPULSE_5:
            # Wave 5 is typically shorter than wave 3
            targets = [
                current_price + (wave1_length * 0.618),  # 61.8% of wave 1
                current_price + (wave1_length * 1.0),    # 100% of wave 1
                current_price + (wave1_length * 1.272)   # 127.2% extension
            ]
        else:
            # Standard extension
            targets = [
                current_price + (wave1_length * 1.0),    # 100% extension
                current_price + (wave1_length * 1.272),  # 127.2% extension
                current_price + (wave1_length * 1.618)   # 161.8% extension
            ]
        
        # Calculate invalidation levels
        invalidation_levels = [
            current_wave.start_point.price,  # Wave start
            current_wave.end_point.price * 0.95  # 5% below current
        ]
        
        # Calculate time targets
        estimated_duration = current_wave.duration * 1.5  # Conservative estimate
        time_targets = [
            current_wave.end_point.timestamp + pd.Timedelta(hours=int(estimated_duration * 0.618)),
            current_wave.end_point.timestamp + pd.Timedelta(hours=int(estimated_duration)),
            current_wave.end_point.timestamp + pd.Timedelta(hours=int(estimated_duration * 1.618))
        ]
        
        # Calculate risk/reward ratios
        risk_reward_ratios = []
        for target in targets:
            risk = abs(current_price - invalidation_levels[0])
            reward = abs(target - current_price)
            if risk > 0:
                risk_reward_ratios.append(reward / risk)
            else:
                risk_reward_ratios.append(0)
        
        # Determine confidence based on historical patterns and rule compliance
        confidence = self._calculate_projection_confidence(current_wave, all_waves, data)
        
        return WaveProjection(
            projection_type=ProjectionType.IMPULSE_CONTINUATION,
            confidence=confidence,
            likelihood=self._calculate_likelihood(confidence),
            description=f"Impulse continuation: {next_wave_type.value} wave",
            fibonacci_targets=targets,
            invalidation_levels=invalidation_levels,
            time_targets=time_targets,
            risk_reward_ratios=risk_reward_ratios,
            sub_waves=self._get_impulse_subwaves(next_wave_type),
            pattern_characteristics={
                'direction': current_wave.direction.value,
                'wave_degree': current_wave.degree.value,
                'expected_strength': 'strong' if next_wave_type == WaveType.IMPULSE_3 else 'moderate'
            }
        )
    
    def _project_correction_after_impulse(self, current_wave: Wave, all_waves: List[Wave], data: pd.DataFrame) -> WaveProjection:
        """Project correction after completed impulse wave."""
        # Calculate correction targets
        impulse_start = current_wave.start_point.price
        impulse_end = current_wave.end_point.price
        impulse_length = abs(current_wave.price_change)
        
        # Standard correction targets
        targets = [
            impulse_end - (impulse_length * 0.382),  # 38.2% retracement
            impulse_end - (impulse_length * 0.5),    # 50% retracement
            impulse_end - (impulse_length * 0.618)   # 61.8% retracement
        ]
        
        # Invalidation levels
        invalidation_levels = [
            impulse_end,  # Don't exceed impulse end
            impulse_end + (impulse_length * 0.1)  # 10% above impulse end
        ]
        
        # Time targets
        estimated_duration = current_wave.duration * 0.8  # Corrections typically shorter
        time_targets = [
            current_wave.end_point.timestamp + pd.Timedelta(hours=int(estimated_duration * 0.5)),
            current_wave.end_point.timestamp + pd.Timedelta(hours=int(estimated_duration)),
            current_wave.end_point.timestamp + pd.Timedelta(hours=int(estimated_duration * 1.5))
        ]
        
        # Risk/reward ratios
        risk_reward_ratios = []
        for target in targets:
            risk = abs(current_wave.end_point.price - invalidation_levels[0])
            reward = abs(current_wave.end_point.price - target)
            if risk > 0:
                risk_reward_ratios.append(reward / risk)
            else:
                risk_reward_ratios.append(0)
        
        confidence = self._calculate_projection_confidence(current_wave, all_waves, data)
        
        return WaveProjection(
            projection_type=ProjectionType.COMPLEX_CORRECTION,
            confidence=confidence,
            likelihood=self._calculate_likelihood(confidence),
            description="ABC correction after completed impulse",
            fibonacci_targets=targets,
            invalidation_levels=invalidation_levels,
            time_targets=time_targets,
            risk_reward_ratios=risk_reward_ratios,
            sub_waves=['A', 'B', 'C'],
            pattern_characteristics={
                'correction_type': 'ABC',
                'expected_depth': 'moderate',
                'time_symmetry': 'likely'
            }
        )
    
    def _project_correction_continuation(self, current_wave: Wave, all_waves: List[Wave], data: pd.DataFrame) -> WaveProjection:
        """Project continuation of corrective pattern."""
        if current_wave.wave_type == WaveType.CORRECTIVE_A:
            # Project Wave B
            wave_a_length = abs(current_wave.price_change)
            targets = [
                current_wave.end_point.price + (wave_a_length * 0.5),   # 50% retracement
                current_wave.end_point.price + (wave_a_length * 0.618), # 61.8% retracement
                current_wave.end_point.price + (wave_a_length * 0.786)  # 78.6% retracement
            ]
            sub_waves = ['B']
            description = "Wave B retracement"
        else:
            # Project Wave C
            wave_a = self._find_wave_by_type(all_waves, WaveType.CORRECTIVE_A)
            if wave_a:
                wave_a_length = abs(wave_a.price_change)
                targets = [
                    current_wave.end_point.price - (wave_a_length * 1.0),   # 100% of wave A
                    current_wave.end_point.price - (wave_a_length * 1.272), # 127.2% of wave A
                    current_wave.end_point.price - (wave_a_length * 1.618)  # 161.8% of wave A
                ]
            else:
                targets = [
                    current_wave.end_point.price * 0.9,
                    current_wave.end_point.price * 0.8,
                    current_wave.end_point.price * 0.7
                ]
            sub_waves = ['C']
            description = "Wave C completion"
        
        # Invalidation levels
        invalidation_levels = [
            current_wave.end_point.price,
            current_wave.end_point.price * 1.05
        ]
        
        # Time targets
        estimated_duration = current_wave.duration * 0.6
        time_targets = [
            current_wave.end_point.timestamp + pd.Timedelta(hours=int(estimated_duration * 0.5)),
            current_wave.end_point.timestamp + pd.Timedelta(hours=int(estimated_duration)),
            current_wave.end_point.timestamp + pd.Timedelta(hours=int(estimated_duration * 1.2))
        ]
        
        # Risk/reward ratios
        risk_reward_ratios = []
        for target in targets:
            risk = abs(current_wave.end_point.price - invalidation_levels[0])
            reward = abs(current_wave.end_point.price - target)
            if risk > 0:
                risk_reward_ratios.append(reward / risk)
            else:
                risk_reward_ratios.append(0)
        
        confidence = self._calculate_projection_confidence(current_wave, all_waves, data)
        
        return WaveProjection(
            projection_type=ProjectionType.ZIGZAG_CORRECTION,
            confidence=confidence,
            likelihood=self._calculate_likelihood(confidence),
            description=description,
            fibonacci_targets=targets,
            invalidation_levels=invalidation_levels,
            time_targets=time_targets,
            risk_reward_ratios=risk_reward_ratios,
            sub_waves=sub_waves,
            pattern_characteristics={
                'correction_type': 'zigzag',
                'expected_depth': 'moderate',
                'time_symmetry': 'likely'
            }
        )
    
    def _project_new_impulse_cycle(self, current_wave: Wave, all_waves: List[Wave], data: pd.DataFrame) -> WaveProjection:
        """Project new impulse cycle after completed correction."""
        # Calculate targets for new impulse wave 1
        correction_depth = abs(current_wave.price_change)
        targets = [
            current_wave.end_point.price + (correction_depth * 1.272),  # 127.2% extension
            current_wave.end_point.price + (correction_depth * 1.618),  # 161.8% extension
            current_wave.end_point.price + (correction_depth * 2.618)   # 261.8% extension
        ]
        
        # Invalidation levels
        invalidation_levels = [
            current_wave.end_point.price,  # Don't go below correction end
            current_wave.end_point.price * 0.95  # 5% below correction end
        ]
        
        # Time targets
        estimated_duration = current_wave.duration * 1.2
        time_targets = [
            current_wave.end_point.timestamp + pd.Timedelta(hours=int(estimated_duration * 0.618)),
            current_wave.end_point.timestamp + pd.Timedelta(hours=int(estimated_duration)),
            current_wave.end_point.timestamp + pd.Timedelta(hours=int(estimated_duration * 1.618))
        ]
        
        # Risk/reward ratios
        risk_reward_ratios = []
        for target in targets:
            risk = abs(current_wave.end_point.price - invalidation_levels[0])
            reward = abs(target - current_wave.end_point.price)
            if risk > 0:
                risk_reward_ratios.append(reward / risk)
            else:
                risk_reward_ratios.append(0)
        
        confidence = self._calculate_projection_confidence(current_wave, all_waves, data)
        
        return WaveProjection(
            projection_type=ProjectionType.IMPULSE_CONTINUATION,
            confidence=confidence,
            likelihood=self._calculate_likelihood(confidence),
            description="New impulse cycle: Wave 1",
            fibonacci_targets=targets,
            invalidation_levels=invalidation_levels,
            time_targets=time_targets,
            risk_reward_ratios=risk_reward_ratios,
            sub_waves=['1', '2', '3', '4', '5'],
            pattern_characteristics={
                'cycle_type': 'new_impulse',
                'expected_strength': 'strong',
                'direction': 'opposite_to_correction'
            }
        )
    
    def _generate_alternative_projections(self, current_wave: Wave, all_waves: List[Wave], data: pd.DataFrame, max_alternatives: int) -> List[WaveProjection]:
        """Generate alternative projection scenarios."""
        alternatives = []
        
        try:
            # Alternative 1: Triangle formation
            if len(all_waves) >= 3:
                triangle_projection = self._project_triangle_formation(current_wave, all_waves, data)
                if triangle_projection:
                    alternatives.append(triangle_projection)
            
            # Alternative 2: Complex correction
            complex_projection = self._project_complex_correction(current_wave, all_waves, data)
            if complex_projection:
                alternatives.append(complex_projection)
            
            # Alternative 3: Diagonal formation
            diagonal_projection = self._project_diagonal_formation(current_wave, all_waves, data)
            if diagonal_projection:
                alternatives.append(diagonal_projection)
            
            # Alternative 4: Flat correction
            flat_projection = self._project_flat_correction(current_wave, all_waves, data)
            if flat_projection:
                alternatives.append(flat_projection)
            
            # Sort by likelihood and return top alternatives
            alternatives.sort(key=lambda x: x.likelihood, reverse=True)
            return alternatives[:max_alternatives]
            
        except Exception as e:
            logger.error(f"Error generating alternative projections: {e}")
            return []
    
    def _project_triangle_formation(self, current_wave: Wave, all_waves: List[Wave], data: pd.DataFrame) -> Optional[WaveProjection]:
        """Project triangle pattern formation."""
        # Triangle targets (converging)
        current_price = current_wave.end_point.price
        targets = [
            current_price * 1.05,  # Upper boundary
            current_price * 0.95,  # Lower boundary
            current_price * 1.02,  # Converging upper
            current_price * 0.98,  # Converging lower
            current_price * 1.0    # Final convergence
        ]
        
        # Invalidation levels
        invalidation_levels = [
            current_price * 1.1,   # Upper invalidation
            current_price * 0.9    # Lower invalidation
        ]
        
        # Time targets (triangles take time to form)
        estimated_duration = current_wave.duration * 2.0
        time_targets = [
            current_wave.end_point.timestamp + pd.Timedelta(hours=int(estimated_duration * 0.25)),
            current_wave.end_point.timestamp + pd.Timedelta(hours=int(estimated_duration * 0.5)),
            current_wave.end_point.timestamp + pd.Timedelta(hours=int(estimated_duration * 0.75)),
            current_wave.end_point.timestamp + pd.Timedelta(hours=int(estimated_duration)),
            current_wave.end_point.timestamp + pd.Timedelta(hours=int(estimated_duration * 1.25))
        ]
        
        confidence = self._calculate_projection_confidence(current_wave, all_waves, data) * 0.8  # Lower confidence for triangles
        
        return WaveProjection(
            projection_type=ProjectionType.TRIANGLE,
            confidence=ProjectionConfidence.MEDIUM if confidence == ProjectionConfidence.HIGH else confidence,
            likelihood=self._calculate_likelihood(confidence) * 0.7,  # Lower likelihood
            description="Contracting triangle formation",
            fibonacci_targets=targets,
            invalidation_levels=invalidation_levels,
            time_targets=time_targets,
            risk_reward_ratios=[1.5, 2.0, 2.5, 3.0, 3.5],  # Good risk/reward for triangles
            sub_waves=['A', 'B', 'C', 'D', 'E'],
            pattern_characteristics={
                'triangle_type': 'contracting',
                'formation_time': 'extended',
                'breakout_direction': 'uncertain'
            }
        )
    
    def _project_complex_correction(self, current_wave: Wave, all_waves: List[Wave], data: pd.DataFrame) -> Optional[WaveProjection]:
        """Project complex correction (WXY or WXYXZ)."""
        # Complex correction targets
        correction_depth = abs(current_wave.price_change)
        targets = [
            current_wave.end_point.price - (correction_depth * 0.786),  # 78.6% retracement
            current_wave.end_point.price - (correction_depth * 0.886),  # 88.6% retracement
            current_wave.end_point.price - (correction_depth * 1.0)     # 100% retracement
        ]
        
        # Invalidation levels
        invalidation_levels = [
            current_wave.end_point.price,
            current_wave.end_point.price * 1.05
        ]
        
        # Time targets (complex corrections take longer)
        estimated_duration = current_wave.duration * 1.5
        time_targets = [
            current_wave.end_point.timestamp + pd.Timedelta(hours=int(estimated_duration * 0.5)),
            current_wave.end_point.timestamp + pd.Timedelta(hours=int(estimated_duration)),
            current_wave.end_point.timestamp + pd.Timedelta(hours=int(estimated_duration * 1.5))
        ]
        
        confidence = self._calculate_projection_confidence(current_wave, all_waves, data) * 0.7
        
        return WaveProjection(
            projection_type=ProjectionType.COMPLEX_CORRECTION,
            confidence=ProjectionConfidence.MEDIUM if confidence == ProjectionConfidence.HIGH else confidence,
            likelihood=self._calculate_likelihood(confidence) * 0.6,
            description="Complex WXY correction",
            fibonacci_targets=targets,
            invalidation_levels=invalidation_levels,
            time_targets=time_targets,
            risk_reward_ratios=[1.2, 1.5, 1.8],
            sub_waves=['W', 'X', 'Y'],
            pattern_characteristics={
                'correction_type': 'complex',
                'formation_time': 'extended',
                'sub_wave_count': 3
            }
        )
    
    def _project_diagonal_formation(self, current_wave: Wave, all_waves: List[Wave], data: pd.DataFrame) -> Optional[WaveProjection]:
        """Project diagonal pattern formation."""
        # Diagonal targets (wedge-shaped)
        current_price = current_wave.end_point.price
        targets = [
            current_price * 1.1,   # Upper boundary
            current_price * 0.9,   # Lower boundary
            current_price * 1.05,  # Converging upper
            current_price * 0.95,  # Converging lower
            current_price * 1.0    # Final convergence
        ]
        
        # Invalidation levels
        invalidation_levels = [
            current_price * 1.15,  # Upper invalidation
            current_price * 0.85   # Lower invalidation
        ]
        
        # Time targets
        estimated_duration = current_wave.duration * 1.8
        time_targets = [
            current_wave.end_point.timestamp + pd.Timedelta(hours=int(estimated_duration * 0.2)),
            current_wave.end_point.timestamp + pd.Timedelta(hours=int(estimated_duration * 0.4)),
            current_wave.end_point.timestamp + pd.Timedelta(hours=int(estimated_duration * 0.6)),
            current_wave.end_point.timestamp + pd.Timedelta(hours=int(estimated_duration * 0.8)),
            current_wave.end_point.timestamp + pd.Timedelta(hours=int(estimated_duration))
        ]
        
        confidence = self._calculate_projection_confidence(current_wave, all_waves, data) * 0.6
        
        return WaveProjection(
            projection_type=ProjectionType.DIAGONAL,
            confidence=ProjectionConfidence.LOW if confidence == ProjectionConfidence.HIGH else confidence,
            likelihood=self._calculate_likelihood(confidence) * 0.5,
            description="Diagonal pattern formation",
            fibonacci_targets=targets,
            invalidation_levels=invalidation_levels,
            time_targets=time_targets,
            risk_reward_ratios=[1.3, 1.6, 1.9, 2.2, 2.5],
            sub_waves=['1', '2', '3', '4', '5'],
            pattern_characteristics={
                'diagonal_type': 'ending',
                'wave_4_overlap': True,
                'wedge_shape': True
            }
        )
    
    def _project_flat_correction(self, current_wave: Wave, all_waves: List[Wave], data: pd.DataFrame) -> Optional[WaveProjection]:
        """Project flat correction pattern."""
        # Flat correction targets (sideways)
        current_price = current_wave.end_point.price
        targets = [
            current_price * 0.95,  # Slight retracement
            current_price * 1.0,   # No movement
            current_price * 1.05   # Slight extension
        ]
        
        # Invalidation levels
        invalidation_levels = [
            current_price * 0.9,   # Lower invalidation
            current_price * 1.1    # Upper invalidation
        ]
        
        # Time targets
        estimated_duration = current_wave.duration * 1.0
        time_targets = [
            current_wave.end_point.timestamp + pd.Timedelta(hours=int(estimated_duration * 0.5)),
            current_wave.end_point.timestamp + pd.Timedelta(hours=int(estimated_duration)),
            current_wave.end_point.timestamp + pd.Timedelta(hours=int(estimated_duration * 1.5))
        ]
        
        confidence = self._calculate_projection_confidence(current_wave, all_waves, data) * 0.8
        
        return WaveProjection(
            projection_type=ProjectionType.FLAT_CORRECTION,
            confidence=ProjectionConfidence.MEDIUM if confidence == ProjectionConfidence.HIGH else confidence,
            likelihood=self._calculate_likelihood(confidence) * 0.7,
            description="Flat correction pattern",
            fibonacci_targets=targets,
            invalidation_levels=invalidation_levels,
            time_targets=time_targets,
            risk_reward_ratios=[1.1, 1.2, 1.3],
            sub_waves=['A', 'B', 'C'],
            pattern_characteristics={
                'correction_type': 'flat',
                'movement': 'sideways',
                'depth': 'shallow'
            }
        )
    
    def _generate_pattern_based_scenarios(self, current_wave: Wave, all_waves: List[Wave], data: pd.DataFrame, max_scenarios: int) -> List[ProjectionScenario]:
        """Generate scenarios based on different pattern types."""
        scenarios = []
        
        # Scenario 1: Strong trend continuation
        if self._is_strong_trend(data, current_wave):
            strong_trend_projection = self._project_strong_trend_continuation(current_wave, all_waves, data)
            if strong_trend_projection:
                scenarios.append(ProjectionScenario(
                    scenario_name="Strong Trend Continuation",
                    primary_projection=strong_trend_projection,
                    alternative_projections=[],
                    overall_confidence=strong_trend_projection.likelihood,
                    market_context={'trend_strength': 'strong'},
                    recommendations=["Follow the trend", "Use trailing stops"]
                ))
        
        # Scenario 2: Consolidation pattern
        if self._is_consolidation(data, current_wave):
            consolidation_projection = self._project_consolidation_breakout(current_wave, all_waves, data)
            if consolidation_projection:
                scenarios.append(ProjectionScenario(
                    scenario_name="Consolidation Breakout",
                    primary_projection=consolidation_projection,
                    alternative_projections=[],
                    overall_confidence=consolidation_projection.likelihood,
                    market_context={'pattern_type': 'consolidation'},
                    recommendations=["Wait for breakout", "Use breakout strategies"]
                ))
        
        return scenarios[:max_scenarios]
    
    def _calculate_projection_confidence(self, current_wave: Wave, all_waves: List[Wave], data: pd.DataFrame) -> ProjectionConfidence:
        """Calculate confidence level for projection."""
        # Historical pattern matching
        historical_score = self._calculate_historical_pattern_score(current_wave, all_waves)
        
        # Rule compliance
        rule_score = self._calculate_rule_compliance_score(current_wave, all_waves)
        
        # Market context
        market_score = self._calculate_market_context_score(data, current_wave)
        
        # Weighted average
        overall_score = (
            historical_score * self.historical_pattern_weight +
            rule_score * self.rule_compliance_weight +
            market_score * self.market_context_weight
        )
        
        if overall_score >= 0.8:
            return ProjectionConfidence.HIGH
        elif overall_score >= 0.6:
            return ProjectionConfidence.MEDIUM
        elif overall_score >= 0.4:
            return ProjectionConfidence.LOW
        else:
            return ProjectionConfidence.VERY_LOW
    
    def _calculate_likelihood(self, confidence: ProjectionConfidence) -> float:
        """Convert confidence to likelihood percentage."""
        confidence_map = {
            ProjectionConfidence.HIGH: 0.85,
            ProjectionConfidence.MEDIUM: 0.65,
            ProjectionConfidence.LOW: 0.45,
            ProjectionConfidence.VERY_LOW: 0.25
        }
        return confidence_map.get(confidence, 0.5)
    
    def _get_next_impulse_wave(self, current_wave_type: WaveType) -> WaveType:
        """Get the next impulse wave type."""
        wave_sequence = {
            WaveType.IMPULSE_1: WaveType.IMPULSE_2,
            WaveType.IMPULSE_2: WaveType.IMPULSE_3,
            WaveType.IMPULSE_3: WaveType.IMPULSE_4,
            WaveType.IMPULSE_4: WaveType.IMPULSE_5
        }
        return wave_sequence.get(current_wave_type, WaveType.UNKNOWN)
    
    def _get_impulse_subwaves(self, wave_type: WaveType) -> List[str]:
        """Get subwaves for impulse wave."""
        subwave_map = {
            WaveType.IMPULSE_1: ['i', 'ii', 'iii', 'iv', 'v'],
            WaveType.IMPULSE_2: ['a', 'b', 'c'],
            WaveType.IMPULSE_3: ['i', 'ii', 'iii', 'iv', 'v'],
            WaveType.IMPULSE_4: ['a', 'b', 'c'],
            WaveType.IMPULSE_5: ['i', 'ii', 'iii', 'iv', 'v']
        }
        return subwave_map.get(wave_type, [])
    
    def _find_wave_by_type(self, waves: List[Wave], wave_type: WaveType) -> Optional[Wave]:
        """Find wave by type."""
        for wave in waves:
            if wave.wave_type == wave_type:
                return wave
        return None
    
    def _calculate_historical_pattern_score(self, current_wave: Wave, all_waves: List[Wave]) -> float:
        """Calculate score based on historical pattern matching."""
        similar_patterns = 0
        total_patterns = len(all_waves)
        
        for wave in all_waves:
            if wave.wave_type == current_wave.wave_type and wave != current_wave:
                # Check similarity in price change and duration
                price_similarity = 1 - abs(wave.price_change_pct - current_wave.price_change_pct) / max(abs(wave.price_change_pct), abs(current_wave.price_change_pct), 0.01)
                duration_similarity = 1 - abs(wave.duration - current_wave.duration) / max(wave.duration, current_wave.duration, 1)
                
                if (price_similarity + duration_similarity) / 2 > 0.7:
                    similar_patterns += 1
        
        return similar_patterns / max(total_patterns, 1)
    
    def _calculate_rule_compliance_score(self, current_wave: Wave, all_waves: List[Wave]) -> float:
        """Calculate score based on Elliott Wave rule compliance."""
        # Validate the pattern
        validation_result = self.validator.validate_wave_pattern(all_waves, pd.DataFrame())
        return validation_result.overall_score
    
    def _calculate_market_context_score(self, data: pd.DataFrame, current_wave: Wave) -> float:
        """Calculate score based on market context."""
        # Simple market context analysis
        recent_volatility = data['close'].pct_change().std()
        trend_strength = abs(data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20]
        
        # Higher score for moderate volatility and strong trends
        volatility_score = 1.0 if 0.01 <= recent_volatility <= 0.05 else 0.5
        trend_score = min(trend_strength * 10, 1.0)
        
        return (volatility_score + trend_score) / 2
    
    def _analyze_market_context(self, data: pd.DataFrame, current_wave: Wave) -> Dict[str, Any]:
        """Analyze market context for projections."""
        return {
            'volatility': data['close'].pct_change().std(),
            'trend_strength': abs(data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20],
            'volume_trend': data['volume'].iloc[-10:].mean() / data['volume'].iloc[-20:-10].mean(),
            'price_momentum': (data['close'].iloc[-1] - data['close'].iloc[-5]) / data['close'].iloc[-5]
        }
    
    def _generate_recommendations(self, projection: WaveProjection, data: pd.DataFrame) -> List[str]:
        """Generate trading recommendations based on projection."""
        recommendations = []
        
        if projection.projection_type == ProjectionType.IMPULSE_CONTINUATION:
            recommendations.extend([
                "Follow the trend direction",
                "Use trailing stops to protect profits",
                "Consider Fibonacci retracements for entries"
            ])
        elif projection.projection_type == ProjectionType.COMPLEX_CORRECTION:
            recommendations.extend([
                "Wait for correction completion",
                "Use Fibonacci retracements for entries",
                "Monitor invalidation levels closely"
            ])
        elif projection.projection_type == ProjectionType.TRIANGLE:
            recommendations.extend([
                "Wait for triangle breakout",
                "Prepare for breakout in either direction",
                "Use tight stops during triangle formation"
            ])
        
        # Add confidence-based recommendations
        if projection.confidence == ProjectionConfidence.HIGH:
            recommendations.append("High confidence projection - consider larger position sizes")
        elif projection.confidence == ProjectionConfidence.LOW:
            recommendations.append("Low confidence projection - use smaller position sizes and tight stops")
        
        return recommendations
    
    def _is_strong_trend(self, data: pd.DataFrame, current_wave: Wave) -> bool:
        """Check if market is in strong trend."""
        recent_change = (data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20]
        return abs(recent_change) > 0.1  # 10% change in 20 periods
    
    def _is_consolidation(self, data: pd.DataFrame, current_wave: Wave) -> bool:
        """Check if market is in consolidation."""
        recent_volatility = data['close'].pct_change().std()
        recent_range = (data['high'].iloc[-20:].max() - data['low'].iloc[-20:].min()) / data['close'].iloc[-20]
        return recent_volatility < 0.02 and recent_range < 0.05  # Low volatility and range
    
    def _project_strong_trend_continuation(self, current_wave: Wave, all_waves: List[Wave], data: pd.DataFrame) -> Optional[WaveProjection]:
        """Project strong trend continuation."""
        # Enhanced targets for strong trends
        current_price = current_wave.end_point.price
        trend_strength = abs(data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20]
        
        targets = [
            current_price * (1 + trend_strength * 1.618),
            current_price * (1 + trend_strength * 2.618),
            current_price * (1 + trend_strength * 4.236)
        ]
        
        confidence = ProjectionConfidence.HIGH
        likelihood = 0.9
        
        return WaveProjection(
            projection_type=ProjectionType.IMPULSE_CONTINUATION,
            confidence=confidence,
            likelihood=likelihood,
            description="Strong trend continuation",
            fibonacci_targets=targets,
            invalidation_levels=[current_price * 0.95],
            time_targets=[current_wave.end_point.timestamp + pd.Timedelta(hours=24)],
            risk_reward_ratios=[2.0, 3.0, 4.0],
            sub_waves=['1', '2', '3', '4', '5'],
            pattern_characteristics={'trend_strength': 'strong', 'momentum': 'high'}
        )
    
    def _project_consolidation_breakout(self, current_wave: Wave, all_waves: List[Wave], data: pd.DataFrame) -> Optional[WaveProjection]:
        """Project consolidation breakout."""
        # Breakout targets
        consolidation_range = data['high'].iloc[-20:].max() - data['low'].iloc[-20:].min()
        current_price = current_wave.end_point.price
        
        targets = [
            current_price + consolidation_range,
            current_price + consolidation_range * 1.618,
            current_price + consolidation_range * 2.618
        ]
        
        confidence = ProjectionConfidence.MEDIUM
        likelihood = 0.7
        
        return WaveProjection(
            projection_type=ProjectionType.IMPULSE_CONTINUATION,
            confidence=confidence,
            likelihood=likelihood,
            description="Consolidation breakout",
            fibonacci_targets=targets,
            invalidation_levels=[current_price - consolidation_range * 0.5],
            time_targets=[current_wave.end_point.timestamp + pd.Timedelta(hours=12)],
            risk_reward_ratios=[1.5, 2.0, 2.5],
            sub_waves=['1', '2', '3', '4', '5'],
            pattern_characteristics={'breakout_type': 'consolidation', 'range': consolidation_range}
        ) 