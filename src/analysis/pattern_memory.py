import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.config import get_config
from .wave_detector import Wave, WaveType, WaveDegree, TrendDirection, WavePoint
from .wave_validator import WaveValidator, WaveValidation
from .fibonacci import FibonacciAnalysis

logger = get_logger(__name__)


class PatternCategory(Enum):
    """Categories of Elliott Wave patterns."""
    IMPULSE = "impulse"
    CORRECTIVE = "corrective"
    TRIANGLE = "triangle"
    DIAGONAL = "diagonal"
    COMPLEX_CORRECTION = "complex_correction"
    FLAT = "flat"
    ZIGZAG = "zigzag"


class PatternSimilarityMetric(Enum):
    """Metrics for calculating pattern similarity."""
    STRUCTURAL = "structural"      # Wave structure similarity
    FIBONACCI = "fibonacci"        # Fibonacci ratio similarity
    TEMPORAL = "temporal"          # Time duration similarity
    PRICE_ACTION = "price_action"  # Price movement similarity
    COMPOSITE = "composite"        # Weighted combination of all metrics


@dataclass
class HistoricalPattern:
    """Represents a historical Elliott Wave pattern."""
    pattern_id: str
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    pattern_category: PatternCategory
    waves: List[Wave]
    validation_result: WaveValidation
    fibonacci_analysis: Optional[FibonacciAnalysis]
    
    # Pattern characteristics
    total_duration: int
    total_price_change: float
    price_change_pct: float
    wave_count: int
    confidence_score: float
    
    # Structural features
    wave_sequence: List[str]
    wave_durations: List[int]
    wave_price_changes: List[float]
    fibonacci_ratios: Dict[str, float]
    
    # Market context
    market_condition: str
    volume_profile: Dict[str, float]
    volatility_level: float
    
    # Outcome tracking
    outcome_known: bool = False
    outcome_price_target: Optional[float] = None
    outcome_time_target: Optional[datetime] = None
    actual_outcome: Optional[str] = None
    outcome_accuracy: Optional[float] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class PatternMatch:
    """Represents a match between current and historical patterns."""
    historical_pattern: HistoricalPattern
    similarity_score: float
    similarity_breakdown: Dict[str, float]
    confidence_adjustment: float
    match_quality: str  # "excellent", "good", "fair", "poor"
    outcome_prediction: Optional[str] = None
    predicted_targets: List[float] = field(default_factory=list)
    predicted_timeframe: Optional[str] = None


@dataclass
class PatternMemoryStats:
    """Statistics about the pattern memory database."""
    total_patterns: int
    patterns_by_category: Dict[str, int]
    patterns_by_symbol: Dict[str, int]
    patterns_by_timeframe: Dict[str, int]
    average_confidence: float
    average_similarity: float
    most_common_patterns: List[Tuple[str, int]]
    recent_patterns: List[HistoricalPattern]


class PatternMemory:
    """
    Advanced pattern memory system for Elliott Wave analysis.
    Stores, compares, and learns from historical wave patterns.
    """
    
    def __init__(self, config_path: Optional[str] = None, memory_file: Optional[str] = None):
        """
        Initialize PatternMemory.
        
        Args:
            config_path: Path to configuration file
            memory_file: Path to persistent memory file
        """
        self.config = get_config(config_path)
        self.memory_file = memory_file or "pattern_memory.pkl"
        self.validator = WaveValidator(config_path)
        
        # Similarity weights
        self.structural_weight = self.config.get('pattern_memory.structural_weight', 0.3)
        self.fibonacci_weight = self.config.get('pattern_memory.fibonacci_weight', 0.25)
        self.temporal_weight = self.config.get('pattern_memory.temporal_weight', 0.2)
        self.price_action_weight = self.config.get('pattern_memory.price_action_weight', 0.25)
        
        # Pattern storage
        self.patterns: List[HistoricalPattern] = []
        self.pattern_index: Dict[str, HistoricalPattern] = {}
        
        # Load existing patterns
        self._load_patterns()
        
        logger.info(f"PatternMemory initialized with {len(self.patterns)} patterns")

    def store_pattern(
        self,
        symbol: str,
        timeframe: str,
        waves: List[Wave],
        validation_result: WaveValidation,
        fibonacci_analysis: Optional[FibonacciAnalysis] = None,
        market_condition: str = "unknown",
        volume_profile: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None,
        notes: str = ""
    ) -> str:
        """
        Store a new pattern in memory.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe of the pattern
            waves: List of detected waves
            validation_result: Validation result
            fibonacci_analysis: Optional Fibonacci analysis
            market_condition: Market condition description
            volume_profile: Volume profile data
            tags: Pattern tags
            notes: Additional notes
            
        Returns:
            Pattern ID
        """
        try:
            if not waves:
                raise ValueError("No waves provided for pattern storage")
            
            # Generate pattern ID
            pattern_id = self._generate_pattern_id(symbol, timeframe, waves)
            
            # Determine pattern category
            pattern_category = self._categorize_pattern(waves)
            
            # Calculate pattern characteristics
            start_date = waves[0].start_point.timestamp
            end_date = waves[-1].end_point.timestamp
            total_duration = (end_date - start_date).total_seconds() / 3600  # hours
            total_price_change = waves[-1].end_point.price - waves[0].start_point.price
            price_change_pct = total_price_change / waves[0].start_point.price
            
            # Extract structural features
            wave_sequence = [wave.wave_type.value for wave in waves]
            wave_durations = [wave.duration for wave in waves]
            wave_price_changes = [wave.price_change for wave in waves]
            fibonacci_ratios = self._extract_fibonacci_ratios(waves, fibonacci_analysis)
            
            # Create historical pattern
            pattern = HistoricalPattern(
                pattern_id=pattern_id,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                pattern_category=pattern_category,
                waves=waves,
                validation_result=validation_result,
                fibonacci_analysis=fibonacci_analysis,
                total_duration=total_duration,
                total_price_change=total_price_change,
                price_change_pct=price_change_pct,
                wave_count=len(waves),
                confidence_score=validation_result.overall_score,
                wave_sequence=wave_sequence,
                wave_durations=wave_durations,
                wave_price_changes=wave_price_changes,
                fibonacci_ratios=fibonacci_ratios,
                market_condition=market_condition,
                volume_profile=volume_profile or {},
                volatility_level=self._calculate_volatility(waves),
                tags=tags or [],
                notes=notes
            )
            
            # Store pattern
            self.patterns.append(pattern)
            self.pattern_index[pattern_id] = pattern
            
            # Save to file
            self._save_patterns()
            
            logger.info(f"Stored pattern {pattern_id} with {len(waves)} waves")
            return pattern_id
            
        except Exception as e:
            logger.error(f"Error storing pattern: {e}")
            raise

    def find_similar_patterns(
        self,
        current_waves: List[Wave],
        current_validation: WaveValidation,
        current_fibonacci: Optional[FibonacciAnalysis] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        min_similarity: float = 0.6,
        max_results: int = 10,
        category_filter: Optional[PatternCategory] = None
    ) -> List[PatternMatch]:
        """
        Find similar historical patterns.
        
        Args:
            current_waves: Current wave pattern
            current_validation: Current validation result
            current_fibonacci: Current Fibonacci analysis
            symbol: Filter by symbol
            timeframe: Filter by timeframe
            min_similarity: Minimum similarity score
            max_results: Maximum number of results
            category_filter: Filter by pattern category
            
        Returns:
            List of pattern matches
        """
        try:
            if not current_waves:
                return []
            
            # Filter patterns
            candidate_patterns = self._filter_patterns(symbol, timeframe, category_filter)
            
            # Calculate similarities
            matches = []
            for pattern in candidate_patterns:
                similarity_score, breakdown = self._calculate_similarity(
                    current_waves, current_validation, current_fibonacci,
                    pattern
                )
                
                if similarity_score >= min_similarity:
                    confidence_adjustment = self._calculate_confidence_adjustment(
                        similarity_score, pattern
                    )
                    
                    match = PatternMatch(
                        historical_pattern=pattern,
                        similarity_score=similarity_score,
                        similarity_breakdown=breakdown,
                        confidence_adjustment=confidence_adjustment,
                        outcome_prediction=self._predict_outcome(pattern),
                        predicted_targets=self._predict_targets(pattern, current_waves),
                        predicted_timeframe=self._predict_timeframe(pattern),
                        match_quality=self._assess_match_quality(similarity_score, pattern)
                    )
                    matches.append(match)
            
            # Sort by similarity score
            matches.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Limit results
            matches = matches[:max_results]
            
            logger.info(f"Found {len(matches)} similar patterns")
            return matches
            
        except Exception as e:
            logger.error(f"Error finding similar patterns: {e}")
            return []

    def update_pattern_outcome(
        self,
        pattern_id: str,
        actual_outcome: str,
        outcome_price_target: Optional[float] = None,
        outcome_time_target: Optional[datetime] = None,
        outcome_accuracy: Optional[float] = None
    ) -> bool:
        """
        Update a pattern with its actual outcome.
        
        Args:
            pattern_id: Pattern ID to update
            actual_outcome: Actual outcome description
            outcome_price_target: Actual price target reached
            outcome_time_target: Actual time target reached
            outcome_accuracy: Accuracy score (0-1)
            
        Returns:
            True if updated successfully
        """
        try:
            if pattern_id not in self.pattern_index:
                logger.warning(f"Pattern {pattern_id} not found")
                return False
            
            pattern = self.pattern_index[pattern_id]
            pattern.outcome_known = True
            pattern.actual_outcome = actual_outcome
            pattern.outcome_price_target = outcome_price_target
            pattern.outcome_time_target = outcome_time_target
            pattern.outcome_accuracy = outcome_accuracy
            
            # Save updated patterns
            self._save_patterns()
            
            logger.info(f"Updated outcome for pattern {pattern_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating pattern outcome: {e}")
            return False

    def get_pattern_statistics(self) -> PatternMemoryStats:
        """Get statistics about stored patterns."""
        try:
            # Count by category
            patterns_by_category = {}
            for pattern in self.patterns:
                category = pattern.pattern_category.value
                patterns_by_category[category] = patterns_by_category.get(category, 0) + 1
            
            # Count by symbol
            patterns_by_symbol = {}
            for pattern in self.patterns:
                patterns_by_symbol[pattern.symbol] = patterns_by_symbol.get(pattern.symbol, 0) + 1
            
            # Count by timeframe
            patterns_by_timeframe = {}
            for pattern in self.patterns:
                patterns_by_timeframe[pattern.timeframe] = patterns_by_timeframe.get(pattern.timeframe, 0) + 1
            
            # Calculate averages
            avg_confidence = np.mean([p.confidence_score for p in self.patterns]) if self.patterns else 0
            avg_similarity = np.mean([p.confidence_score for p in self.patterns]) if self.patterns else 0
            
            # Most common patterns
            pattern_counts = {}
            for pattern in self.patterns:
                sequence = '-'.join(pattern.wave_sequence)
                pattern_counts[sequence] = pattern_counts.get(sequence, 0) + 1
            
            most_common = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Recent patterns
            recent_patterns = sorted(self.patterns, key=lambda x: x.created_at, reverse=True)[:10]
            
            return PatternMemoryStats(
                total_patterns=len(self.patterns),
                patterns_by_category=patterns_by_category,
                patterns_by_symbol=patterns_by_symbol,
                patterns_by_timeframe=patterns_by_timeframe,
                average_confidence=avg_confidence,
                average_similarity=avg_similarity,
                most_common_patterns=most_common,
                recent_patterns=recent_patterns
            )
            
        except Exception as e:
            logger.error(f"Error calculating pattern statistics: {e}")
            return PatternMemoryStats(
                total_patterns=0,
                patterns_by_category={},
                patterns_by_symbol={},
                patterns_by_timeframe={},
                average_confidence=0,
                average_similarity=0,
                most_common_patterns=[],
                recent_patterns=[]
            )

    def export_patterns(self, file_path: str, format: str = "json") -> bool:
        """
        Export patterns to file.
        
        Args:
            file_path: Path to export file
            format: Export format ("json" or "csv")
            
        Returns:
            True if exported successfully
        """
        try:
            if format.lower() == "json":
                # Convert patterns to JSON-serializable format
                export_data = []
                for pattern in self.patterns:
                    pattern_dict = {
                        'pattern_id': pattern.pattern_id,
                        'symbol': pattern.symbol,
                        'timeframe': pattern.timeframe,
                        'start_date': pattern.start_date.isoformat(),
                        'end_date': pattern.end_date.isoformat(),
                        'pattern_category': pattern.pattern_category.value,
                        'total_duration': pattern.total_duration,
                        'total_price_change': pattern.total_price_change,
                        'price_change_pct': pattern.price_change_pct,
                        'wave_count': pattern.wave_count,
                        'confidence_score': pattern.confidence_score,
                        'wave_sequence': pattern.wave_sequence,
                        'wave_durations': pattern.wave_durations,
                        'wave_price_changes': pattern.wave_price_changes,
                        'fibonacci_ratios': pattern.fibonacci_ratios,
                        'market_condition': pattern.market_condition,
                        'volatility_level': pattern.volatility_level,
                        'outcome_known': pattern.outcome_known,
                        'actual_outcome': pattern.actual_outcome,
                        'outcome_accuracy': pattern.outcome_accuracy,
                        'created_at': pattern.created_at.isoformat(),
                        'tags': pattern.tags,
                        'notes': pattern.notes
                    }
                    export_data.append(pattern_dict)
                
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                    
            elif format.lower() == "csv":
                # Create DataFrame and export to CSV
                export_data = []
                for pattern in self.patterns:
                    export_data.append({
                        'pattern_id': pattern.pattern_id,
                        'symbol': pattern.symbol,
                        'timeframe': pattern.timeframe,
                        'start_date': pattern.start_date,
                        'end_date': pattern.end_date,
                        'pattern_category': pattern.pattern_category.value,
                        'total_duration': pattern.total_duration,
                        'total_price_change': pattern.total_price_change,
                        'price_change_pct': pattern.price_change_pct,
                        'wave_count': pattern.wave_count,
                        'confidence_score': pattern.confidence_score,
                        'wave_sequence': '-'.join(pattern.wave_sequence),
                        'market_condition': pattern.market_condition,
                        'volatility_level': pattern.volatility_level,
                        'outcome_known': pattern.outcome_known,
                        'actual_outcome': pattern.actual_outcome,
                        'outcome_accuracy': pattern.outcome_accuracy,
                        'created_at': pattern.created_at,
                        'tags': ','.join(pattern.tags),
                        'notes': pattern.notes
                    })
                
                df = pd.DataFrame(export_data)
                df.to_csv(file_path, index=False)
            
            logger.info(f"Exported {len(self.patterns)} patterns to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting patterns: {e}")
            return False

    def _generate_pattern_id(self, symbol: str, timeframe: str, waves: List[Wave]) -> str:
        """Generate unique pattern ID."""
        start_time = waves[0].start_point.timestamp.strftime("%Y%m%d_%H%M")
        wave_sequence = '-'.join([w.wave_type.value for w in waves])
        return f"{symbol}_{timeframe}_{start_time}_{wave_sequence}"

    def _categorize_pattern(self, waves: List[Wave]) -> PatternCategory:
        """Categorize pattern based on wave structure."""
        wave_types = [w.wave_type for w in waves]
        
        # Check for impulse pattern
        if all(wt in [WaveType.IMPULSE_1, WaveType.IMPULSE_2, WaveType.IMPULSE_3, 
                     WaveType.IMPULSE_4, WaveType.IMPULSE_5] for wt in wave_types):
            return PatternCategory.IMPULSE
        
        # Check for corrective pattern
        if all(wt in [WaveType.CORRECTIVE_A, WaveType.CORRECTIVE_B, WaveType.CORRECTIVE_C] for wt in wave_types):
            return PatternCategory.CORRECTIVE
        
        # Check for triangle (5 waves)
        if len(waves) == 5 and all(wt in [WaveType.CORRECTIVE_A, WaveType.CORRECTIVE_B, 
                                         WaveType.CORRECTIVE_C, WaveType.CORRECTIVE_D, WaveType.CORRECTIVE_E] for wt in wave_types):
            return PatternCategory.TRIANGLE
        
        # Check for diagonal
        if len(waves) == 5 and any(wt in [WaveType.IMPULSE_1, WaveType.IMPULSE_2, WaveType.IMPULSE_3, 
                                         WaveType.IMPULSE_4, WaveType.IMPULSE_5] for wt in wave_types):
            return PatternCategory.DIAGONAL
        
        # Default to complex correction
        return PatternCategory.COMPLEX_CORRECTION

    def _extract_fibonacci_ratios(self, waves: List[Wave], fibonacci_analysis: Optional[FibonacciAnalysis]) -> Dict[str, float]:
        """Extract Fibonacci ratios from waves."""
        ratios = {}
        
        if len(waves) >= 3:
            # Wave 2 retracement of Wave 1
            if len(waves) >= 2:
                wave1_change = abs(waves[0].price_change)
                wave2_change = abs(waves[1].price_change)
                if wave1_change > 0:
                    ratios['wave2_retrace'] = wave2_change / wave1_change
            
            # Wave 3 extension of Wave 1
            if len(waves) >= 3:
                wave3_change = abs(waves[2].price_change)
                if wave1_change > 0:
                    ratios['wave3_extension'] = wave3_change / wave1_change
        
        return ratios

    def _calculate_volatility(self, waves: List[Wave]) -> float:
        """Calculate volatility level from waves."""
        if not waves:
            return 0.0
        
        price_changes = [abs(w.price_change) for w in waves]
        return np.std(price_changes) / np.mean(price_changes) if np.mean(price_changes) > 0 else 0.0

    def _filter_patterns(
        self,
        symbol: Optional[str],
        timeframe: Optional[str],
        category_filter: Optional[PatternCategory]
    ) -> List[HistoricalPattern]:
        """Filter patterns based on criteria."""
        filtered = self.patterns
        
        if symbol:
            filtered = [p for p in filtered if p.symbol == symbol]
        
        if timeframe:
            filtered = [p for p in filtered if p.timeframe == timeframe]
        
        if category_filter:
            filtered = [p for p in filtered if p.pattern_category == category_filter]
        
        return filtered

    def _calculate_similarity(
        self,
        current_waves: List[Wave],
        current_validation: WaveValidation,
        current_fibonacci: Optional[FibonacciAnalysis],
        historical_pattern: HistoricalPattern
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate similarity between current and historical patterns."""
        try:
            # Structural similarity
            structural_sim = self._calculate_structural_similarity(current_waves, historical_pattern)
            
            # Fibonacci similarity
            fibonacci_sim = self._calculate_fibonacci_similarity(current_fibonacci, historical_pattern)
            
            # Temporal similarity
            temporal_sim = self._calculate_temporal_similarity(current_waves, historical_pattern)
            
            # Price action similarity
            price_action_sim = self._calculate_price_action_similarity(current_waves, historical_pattern)
            
            # Composite similarity
            composite_sim = (
                self.structural_weight * structural_sim +
                self.fibonacci_weight * fibonacci_sim +
                self.temporal_weight * temporal_sim +
                self.price_action_weight * price_action_sim
            )
            
            breakdown = {
                'structural': structural_sim,
                'fibonacci': fibonacci_sim,
                'temporal': temporal_sim,
                'price_action': price_action_sim,
                'composite': composite_sim
            }
            
            return composite_sim, breakdown
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0, {}

    def _calculate_structural_similarity(self, current_waves: List[Wave], historical_pattern: HistoricalPattern) -> float:
        """Calculate structural similarity between patterns."""
        try:
            current_sequence = [w.wave_type.value for w in current_waves]
            historical_sequence = historical_pattern.wave_sequence
            
            # Sequence similarity
            if len(current_sequence) == len(historical_sequence):
                sequence_match = sum(1 for c, h in zip(current_sequence, historical_sequence) if c == h)
                sequence_sim = sequence_match / len(current_sequence)
            else:
                sequence_sim = 0.0
            
            # Wave count similarity
            count_sim = 1.0 - abs(len(current_waves) - len(historical_pattern.waves)) / max(len(current_waves), len(historical_pattern.waves))
            
            # Pattern category similarity
            current_category = self._categorize_pattern(current_waves)
            category_sim = 1.0 if current_category == historical_pattern.pattern_category else 0.0
            
            return (sequence_sim + count_sim + category_sim) / 3
            
        except Exception as e:
            logger.error(f"Error calculating structural similarity: {e}")
            return 0.0

    def _calculate_fibonacci_similarity(self, current_fibonacci: Optional[FibonacciAnalysis], historical_pattern: HistoricalPattern) -> float:
        """Calculate Fibonacci similarity between patterns."""
        try:
            if not current_fibonacci or not historical_pattern.fibonacci_ratios:
                return 0.5  # Neutral score if no Fibonacci data
            
            # Compare Fibonacci ratios
            similarities = []
            for ratio_name, historical_ratio in historical_pattern.fibonacci_ratios.items():
                if hasattr(current_fibonacci, ratio_name):
                    current_ratio = getattr(current_fibonacci, ratio_name)
                    if current_ratio and historical_ratio:
                        # Calculate similarity based on ratio difference
                        diff = abs(current_ratio - historical_ratio) / max(current_ratio, historical_ratio)
                        similarity = max(0, 1 - diff)
                        similarities.append(similarity)
            
            return np.mean(similarities) if similarities else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating Fibonacci similarity: {e}")
            return 0.5

    def _calculate_temporal_similarity(self, current_waves: List[Wave], historical_pattern: HistoricalPattern) -> float:
        """Calculate temporal similarity between patterns."""
        try:
            current_durations = [w.duration for w in current_waves]
            historical_durations = historical_pattern.wave_durations
            
            if len(current_durations) != len(historical_durations):
                return 0.0
            
            # Calculate duration similarities
            duration_similarities = []
            for current_dur, historical_dur in zip(current_durations, historical_durations):
                if historical_dur > 0:
                    diff = abs(current_dur - historical_dur) / historical_dur
                    similarity = max(0, 1 - diff)
                    duration_similarities.append(similarity)
            
            return np.mean(duration_similarities) if duration_similarities else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating temporal similarity: {e}")
            return 0.0

    def _calculate_price_action_similarity(self, current_waves: List[Wave], historical_pattern: HistoricalPattern) -> float:
        """Calculate price action similarity between patterns."""
        try:
            current_changes = [abs(w.price_change) for w in current_waves]
            historical_changes = [abs(change) for change in historical_pattern.wave_price_changes]
            
            if len(current_changes) != len(historical_changes):
                return 0.0
            
            # Calculate price change similarities
            change_similarities = []
            for current_change, historical_change in zip(current_changes, historical_changes):
                if historical_change > 0:
                    diff = abs(current_change - historical_change) / historical_change
                    similarity = max(0, 1 - diff)
                    change_similarities.append(similarity)
            
            return np.mean(change_similarities) if change_similarities else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating price action similarity: {e}")
            return 0.0

    def _calculate_confidence_adjustment(self, similarity_score: float, historical_pattern: HistoricalPattern) -> float:
        """Calculate confidence adjustment based on historical pattern performance."""
        try:
            # Base adjustment from similarity
            base_adjustment = similarity_score - 0.5  # Range: -0.5 to 0.5
            
            # Adjust based on historical pattern accuracy
            if historical_pattern.outcome_known and historical_pattern.outcome_accuracy is not None:
                accuracy_adjustment = (historical_pattern.outcome_accuracy - 0.5) * 0.2  # Range: -0.1 to 0.1
                base_adjustment += accuracy_adjustment
            
            # Adjust based on pattern confidence
            confidence_adjustment = (historical_pattern.confidence_score - 0.5) * 0.1  # Range: -0.05 to 0.05
            base_adjustment += confidence_adjustment
            
            return max(-0.3, min(0.3, base_adjustment))  # Clamp to reasonable range
            
        except Exception as e:
            logger.error(f"Error calculating confidence adjustment: {e}")
            return 0.0

    def _predict_outcome(self, historical_pattern: HistoricalPattern) -> Optional[str]:
        """Predict outcome based on historical pattern."""
        if not historical_pattern.outcome_known:
            return None
        
        return historical_pattern.actual_outcome

    def _predict_targets(self, historical_pattern: HistoricalPattern, current_waves: List[Wave]) -> List[float]:
        """Predict price targets based on historical pattern."""
        try:
            if not current_waves or not historical_pattern.outcome_known:
                return []
            
            current_price = current_waves[-1].end_point.price
            historical_start = historical_pattern.waves[0].start_point.price
            historical_end = historical_pattern.waves[-1].end_point.price
            historical_outcome = historical_pattern.outcome_price_target
            
            if historical_outcome is None:
                return []
            
            # Calculate price movement ratio
            historical_movement = historical_end - historical_start
            if historical_movement == 0:
                return []
            
            outcome_movement = historical_outcome - historical_end
            movement_ratio = outcome_movement / historical_movement
            
            # Apply ratio to current pattern
            current_movement = current_waves[-1].end_point.price - current_waves[0].start_point.price
            predicted_target = current_waves[-1].end_point.price + (current_movement * movement_ratio)
            
            return [predicted_target]
            
        except Exception as e:
            logger.error(f"Error predicting targets: {e}")
            return []

    def _predict_timeframe(self, historical_pattern: HistoricalPattern) -> Optional[str]:
        """Predict timeframe based on historical pattern."""
        if not historical_pattern.outcome_known or not historical_pattern.outcome_time_target:
            return None
        
        # Calculate time from pattern end to outcome
        time_to_outcome = historical_pattern.outcome_time_target - historical_pattern.end_date
        hours_to_outcome = time_to_outcome.total_seconds() / 3600
        
        if hours_to_outcome < 24:
            return "within 24 hours"
        elif hours_to_outcome < 168:  # 7 days
            return "within 1 week"
        elif hours_to_outcome < 720:  # 30 days
            return "within 1 month"
        else:
            return "long-term"

    def _assess_match_quality(self, similarity_score: float, historical_pattern: HistoricalPattern) -> str:
        """Assess the quality of a pattern match."""
        if similarity_score >= 0.9:
            return "excellent"
        elif similarity_score >= 0.8:
            return "good"
        elif similarity_score >= 0.7:
            return "fair"
        else:
            return "poor"

    def _load_patterns(self):
        """Load patterns from file."""
        try:
            if Path(self.memory_file).exists():
                with open(self.memory_file, 'rb') as f:
                    self.patterns = pickle.load(f)
                    self.pattern_index = {p.pattern_id: p for p in self.patterns}
                logger.info(f"Loaded {len(self.patterns)} patterns from {self.memory_file}")
        except Exception as e:
            logger.warning(f"Could not load patterns from {self.memory_file}: {e}")

    def _save_patterns(self):
        """Save patterns to file."""
        try:
            with open(self.memory_file, 'wb') as f:
                pickle.dump(self.patterns, f)
            logger.debug(f"Saved {len(self.patterns)} patterns to {self.memory_file}")
        except Exception as e:
            logger.error(f"Error saving patterns: {e}") 