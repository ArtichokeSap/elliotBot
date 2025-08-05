"""
Elliott Wave Auto-Tuning Module
Automatically optimizes pivot sensitivity and wave depth detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class TuningResult:
    """Results from auto-tuning optimization"""
    optimal_threshold: float
    optimal_min_wave_length: int
    optimal_lookback_periods: int
    confidence_score: float
    validation_accuracy: float
    multi_timeframe_confirmed: bool

@dataclass
class TimeframeConfig:
    """Configuration for multi-timeframe analysis"""
    primary_tf: str      # Main timeframe (e.g., '1h')
    confirmation_tf: str # Higher timeframe for confirmation (e.g., '4h')
    sensitivity_ratio: float  # How much more sensitive lower TF should be

class WaveAutoTuner:
    """Automatically tunes Elliott Wave detection parameters"""
    
    def __init__(self, config_path: str = "config/auto_tuning.json"):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.optimal_params = {}
        self.timeframe_configs = self._initialize_timeframe_configs()
        
        # Ensure config directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Load existing configuration
        self._load_config()
    
    def _initialize_timeframe_configs(self) -> Dict[str, TimeframeConfig]:
        """Initialize multi-timeframe configurations"""
        return {
            '1m': TimeframeConfig('1m', '5m', 1.5),
            '5m': TimeframeConfig('5m', '15m', 1.4),
            '15m': TimeframeConfig('15m', '1h', 1.3),
            '30m': TimeframeConfig('30m', '2h', 1.3),
            '1h': TimeframeConfig('1h', '4h', 1.2),
            '4h': TimeframeConfig('4h', '1d', 1.2),
            '1d': TimeframeConfig('1d', '1wk', 1.1),
            '1wk': TimeframeConfig('1wk', '1mo', 1.1)
        }
    
    def optimize_parameters(self, market_data: pd.DataFrame, 
                          symbol: str, timeframe: str,
                          target_accuracy: float = 0.75) -> TuningResult:
        """Optimize Elliott Wave detection parameters for given market data"""
        try:
            self.logger.info(f"Auto-tuning Elliott Wave parameters for {symbol} on {timeframe}")
            
            # Parameter ranges to test
            threshold_range = np.arange(0.01, 0.10, 0.005)  # Zigzag threshold
            min_wave_range = range(3, 15, 2)                # Minimum wave length
            lookback_range = range(3, 20, 2)               # Lookback periods
            
            best_score = 0.0
            best_params = None
            results = []
            
            # Grid search optimization
            total_combinations = len(threshold_range) * len(min_wave_range) * len(lookback_range)
            tested = 0
            
            for threshold in threshold_range:
                for min_wave_length in min_wave_range:
                    for lookback_periods in lookback_range:
                        tested += 1
                        if tested % 50 == 0:
                            self.logger.info(f"Testing combination {tested}/{total_combinations}")
                        
                        # Test this parameter combination
                        score = self._evaluate_parameters(
                            market_data, threshold, min_wave_length, lookback_periods
                        )
                        
                        results.append({
                            'threshold': threshold,
                            'min_wave_length': min_wave_length,
                            'lookback_periods': lookback_periods,
                            'score': score
                        })
                        
                        if score > best_score:
                            best_score = score
                            best_params = {
                                'threshold': threshold,
                                'min_wave_length': min_wave_length,
                                'lookback_periods': lookback_periods
                            }
            
            if best_params is None:
                # Fallback to default parameters
                best_params = {'threshold': 0.03, 'min_wave_length': 5, 'lookback_periods': 10}
                best_score = 0.5
            
            # Test multi-timeframe confirmation
            multi_tf_confirmed = self._test_multi_timeframe_confirmation(
                market_data, symbol, timeframe, best_params
            )
            
            # Create tuning result
            result = TuningResult(
                optimal_threshold=best_params['threshold'],
                optimal_min_wave_length=best_params['min_wave_length'],
                optimal_lookback_periods=best_params['lookback_periods'],
                confidence_score=best_score,
                validation_accuracy=best_score,
                multi_timeframe_confirmed=multi_tf_confirmed
            )
            
            # Store optimal parameters
            self.optimal_params[f"{symbol}_{timeframe}"] = best_params
            self._save_config()
            
            self.logger.info(f"Optimization complete. Best score: {best_score:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in parameter optimization: {e}")
            # Return default result
            return TuningResult(0.03, 5, 10, 0.5, 0.5, False)
    
    def _evaluate_parameters(self, market_data: pd.DataFrame, 
                           threshold: float, min_wave_length: int, 
                           lookback_periods: int) -> float:
        """Evaluate parameter combination using historical validation"""
        try:
            # Import here to avoid circular imports
            from ..analysis.enhanced_wave_detector import EnhancedWaveDetector
            
            # Create detector with test parameters
            detector = EnhancedWaveDetector(
                min_wave_size=threshold,
                lookback_periods=lookback_periods,
                min_wave_length=min_wave_length
            )
            
            # Split data for validation
            split_point = int(len(market_data) * 0.8)
            train_data = market_data.iloc[:split_point]
            test_data = market_data.iloc[split_point:]
            
            if len(train_data) < 50 or len(test_data) < 20:
                return 0.0  # Insufficient data
            
            # Detect waves on training data
            try:
                analysis_result = detector.detect_elliott_waves(train_data, "TEST_SYMBOL")
                waves = analysis_result.get('waves', [])
                
                if len(waves) < 2:
                    return 0.0  # No meaningful waves detected
                
                # Calculate score based on multiple criteria
                score = 0.0
                
                # 1. Wave detection success (found reasonable number of waves)
                wave_count_score = min(len(waves) / 10.0, 1.0)  # Normalize to 0-1
                score += 0.3 * wave_count_score
                
                # 2. Validation score from analysis
                validation_score = analysis_result.get('validation_score', 0.0)
                score += 0.4 * validation_score
                
                # 3. Fibonacci compliance
                fib_score = self._calculate_fibonacci_compliance_score(waves, train_data)
                score += 0.2 * fib_score
                
                # 4. Prediction accuracy on test data
                prediction_score = self._test_prediction_accuracy(waves, train_data, test_data)
                score += 0.1 * prediction_score
                
                return min(score, 1.0)
                
            except Exception as e:
                self.logger.debug(f"Error in wave detection during evaluation: {e}")
                return 0.0
            
        except Exception as e:
            self.logger.debug(f"Error evaluating parameters: {e}")
            return 0.0
    
    def _calculate_fibonacci_compliance_score(self, waves: List[Dict], 
                                            market_data: pd.DataFrame) -> float:
        """Calculate how well waves comply with Fibonacci ratios"""
        try:
            if len(waves) < 3:
                return 0.5
            
            compliance_scores = []
            
            for i in range(len(waves) - 1):
                wave = waves[i]
                start_price = wave.get('start_price', 0)
                end_price = wave.get('end_price', 0)
                
                if start_price == 0 or end_price == 0:
                    continue
                
                # Calculate wave magnitude
                wave_magnitude = abs(end_price - start_price) / start_price
                
                # Check against common Fibonacci ratios
                fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618]
                min_distance = min(abs(wave_magnitude - ratio) for ratio in fib_ratios)
                
                # Convert distance to compliance score
                compliance = max(0, 1.0 - min_distance * 5)  # Scale appropriately
                compliance_scores.append(compliance)
            
            return np.mean(compliance_scores) if compliance_scores else 0.5
            
        except Exception as e:
            self.logger.debug(f"Error calculating Fibonacci compliance: {e}")
            return 0.5
    
    def _test_prediction_accuracy(self, waves: List[Dict], 
                                train_data: pd.DataFrame, 
                                test_data: pd.DataFrame) -> float:
        """Test how well detected waves predict future price movement"""
        try:
            if not waves or len(test_data) < 5:
                return 0.5
            
            # Get last wave from training data
            last_wave = waves[-1]
            last_direction = last_wave.get('direction', 'unknown')
            
            # Check if prediction matches actual movement in test data
            test_start_price = test_data['close'].iloc[0]
            test_end_price = test_data['close'].iloc[min(10, len(test_data)-1)]  # Look ahead 10 periods
            
            actual_direction = 'bullish' if test_end_price > test_start_price else 'bearish'
            
            # Score based on direction accuracy
            if last_direction.lower() == actual_direction:
                return 1.0
            elif last_direction == 'unknown':
                return 0.5
            else:
                return 0.0
                
        except Exception as e:
            self.logger.debug(f"Error testing prediction accuracy: {e}")
            return 0.5
    
    def _test_multi_timeframe_confirmation(self, market_data: pd.DataFrame,
                                         symbol: str, timeframe: str,
                                         best_params: Dict) -> bool:
        """Test if waves are confirmed on higher timeframe"""
        try:
            # Get timeframe configuration
            tf_config = self.timeframe_configs.get(timeframe)
            if not tf_config:
                return False
            
            # For demo purposes, we'll simulate higher timeframe confirmation
            # In practice, you'd fetch actual higher timeframe data
            
            # Resample data to simulate higher timeframe
            higher_tf_data = self._resample_to_higher_timeframe(market_data, timeframe)
            
            if len(higher_tf_data) < 30:
                return False
            
            # Import here to avoid circular imports
            from ..analysis.enhanced_wave_detector import EnhancedWaveDetector
            
            # Adjust parameters for higher timeframe (less sensitive)
            higher_tf_threshold = best_params['threshold'] * tf_config.sensitivity_ratio
            higher_tf_detector = EnhancedWaveDetector(
                min_wave_size=higher_tf_threshold,
                lookback_periods=best_params['lookback_periods'],
                min_wave_length=best_params['min_wave_length']
            )
            
            # Detect waves on higher timeframe
            higher_tf_result = higher_tf_detector.detect_elliott_waves(higher_tf_data, symbol)
            higher_tf_waves = higher_tf_result.get('waves', [])
            
            # Confirmation if higher timeframe also shows significant waves
            return len(higher_tf_waves) >= 2 and higher_tf_result.get('validation_score', 0) > 0.6
            
        except Exception as e:
            self.logger.debug(f"Error in multi-timeframe confirmation: {e}")
            return False
    
    def _resample_to_higher_timeframe(self, data: pd.DataFrame, current_tf: str) -> pd.DataFrame:
        """Resample data to higher timeframe for confirmation"""
        try:
            # Mapping of timeframes to resampling rules
            resample_map = {
                '1m': '5min',   '5m': '15min',  '15m': '1h',    '30m': '2h',
                '1h': '4h',     '4h': '1D',     '1d': '1W',     '1wk': '1M'
            }
            
            resample_rule = resample_map.get(current_tf, '1h')
            
            # Ensure we have a datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                # Create a synthetic datetime index if needed
                data = data.copy()
                data.index = pd.date_range(start='2023-01-01', periods=len(data), freq='h')
            
            # Resample OHLCV data
            resampled = data.resample(resample_rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum' if 'volume' in data.columns else lambda x: 0
            }).dropna()
            
            return resampled
            
        except Exception as e:
            self.logger.error(f"Error resampling data: {e}")
            return data.iloc[::4]  # Simple downsampling fallback
    
    def get_optimal_parameters(self, symbol: str, timeframe: str) -> Dict:
        """Get optimal parameters for a symbol/timeframe combination"""
        key = f"{symbol}_{timeframe}"
        return self.optimal_params.get(key, {
            'threshold': 0.03,
            'min_wave_length': 5,
            'lookback_periods': 10
        })
    
    def auto_configure_detector(self, detector, symbol: str, timeframe: str):
        """Automatically configure a wave detector with optimal parameters"""
        params = self.get_optimal_parameters(symbol, timeframe)
        
        detector.min_wave_size = params['threshold']
        detector.lookback_periods = params['lookback_periods']
        if hasattr(detector, 'min_wave_length'):
            detector.min_wave_length = params['min_wave_length']
        
        self.logger.info(f"Auto-configured detector for {symbol} {timeframe}: {params}")
    
    def _save_config(self):
        """Save optimal parameters to configuration file"""
        try:
            config_data = {
                'optimal_params': self.optimal_params,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error saving auto-tuning config: {e}")
    
    def _load_config(self):
        """Load optimal parameters from configuration file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                
                self.optimal_params = config_data.get('optimal_params', {})
                self.logger.info("Auto-tuning configuration loaded")
            
        except Exception as e:
            self.logger.warning(f"Could not load auto-tuning config: {e}")
    
    def run_adaptive_tuning(self, market_data: pd.DataFrame, 
                          symbol: str, timeframe: str,
                          retune_threshold: float = 0.6) -> bool:
        """Run adaptive tuning that re-optimizes if performance drops"""
        try:
            # Get current parameters
            current_params = self.get_optimal_parameters(symbol, timeframe)
            
            # Test current performance
            current_score = self._evaluate_parameters(
                market_data,
                current_params['threshold'],
                current_params['min_wave_length'],
                current_params['lookback_periods']
            )
            
            self.logger.info(f"Current performance score: {current_score:.3f}")
            
            # Re-tune if performance is below threshold
            if current_score < retune_threshold:
                self.logger.info("Performance below threshold, starting re-optimization...")
                result = self.optimize_parameters(market_data, symbol, timeframe)
                return result.confidence_score > current_score
            
            return True  # No retuning needed
            
        except Exception as e:
            self.logger.error(f"Error in adaptive tuning: {e}")
            return False
