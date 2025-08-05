"""
Dataset Generation Pipeline for Elliott Wave ML Training
Generates comprehensive training datasets through historical backtesting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class BacktestResult:
    """Single backtest result for training data generation"""
    # Market Context
    symbol: str
    timestamp: datetime
    timeframe: str
    current_price: float
    
    # Elliott Wave Analysis
    wave_count: str
    projected_wave: str
    wave_structure: str
    validation_score: float
    
    # Target Zone Details
    target_price: float
    target_zone_low: float
    target_zone_high: float
    confluence_score: int
    confluences: List[str]
    
    # Market Features
    rsi: float
    macd_signal: float
    volume_ratio: float
    volatility: float
    trend_strength: float
    
    # Elliott Wave Quality
    fibonacci_alignment: float
    rule_compliance_score: float
    pattern_clarity: float
    wave_proportions: Dict[str, float]
    
    # Confluence Breakdown
    fibonacci_confluences: int
    sr_confluences: int
    momentum_confluences: int
    pattern_confluences: int
    volume_confluences: int
    harmonic_confluences: int
    
    # Outcome (Labels)
    hit: bool
    hit_accuracy: float
    time_to_hit: Optional[int]
    max_adverse_move: float
    max_favorable_move: float
    final_outcome: str  # "hit", "miss", "partial"
    
    # Risk Management
    risk_reward_ratio: float
    maximum_drawdown: float
    confidence_level: str

class DatasetGenerator:
    """
    Comprehensive dataset generation through systematic backtesting
    """
    
    def __init__(self, output_dir: str = "datasets"):
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Backtesting parameters
        self.lookback_window = 200  # Minimum data for analysis
        self.prediction_horizon = 50  # Days to check for target hits
        self.step_size = 5  # Days between analysis points
        self.target_tolerance = 0.02  # 2% tolerance for target hits
        
        self.logger.info("📊 Dataset Generator initialized")
    
    def generate_comprehensive_dataset(self, symbols: List[str], 
                                     start_date: str, end_date: str,
                                     timeframes: List[str] = ['1d']) -> str:
        """
        Generate comprehensive dataset for multiple symbols and timeframes
        
        Returns:
            Path to generated dataset file
        """
        try:
            self.logger.info(f"🚀 Starting dataset generation for {len(symbols)} symbols")
            
            all_results = []
            total_combinations = len(symbols) * len(timeframes)
            completed = 0
            
            # Use thread pool for parallel processing
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit all symbol/timeframe combinations
                futures = []
                for symbol in symbols:
                    for timeframe in timeframes:
                        future = executor.submit(
                            self._backtest_symbol_timeframe,
                            symbol, timeframe, start_date, end_date
                        )
                        futures.append((future, symbol, timeframe))
                
                # Collect results as they complete
                for future, symbol, timeframe in futures:
                    try:
                        results = future.result(timeout=300)  # 5 minute timeout
                        all_results.extend(results)
                        completed += 1
                        
                        self.logger.info(f"✅ Completed {symbol} {timeframe}: "
                                       f"{len(results)} samples ({completed}/{total_combinations})")
                        
                    except Exception as e:
                        self.logger.error(f"❌ Error processing {symbol} {timeframe}: {e}")
                        completed += 1
            
            # Save comprehensive dataset
            dataset_path = self._save_dataset(all_results, start_date, end_date)
            
            # Generate summary statistics
            self._generate_dataset_summary(all_results, dataset_path)
            
            self.logger.info(f"🎯 Dataset generation complete: {len(all_results)} total samples")
            return dataset_path
            
        except Exception as e:
            self.logger.error(f"Error generating dataset: {e}")
            return ""
    
    def _backtest_symbol_timeframe(self, symbol: str, timeframe: str, 
                                 start_date: str, end_date: str) -> List[BacktestResult]:
        """Backtest a single symbol/timeframe combination"""
        try:
            # Import here to avoid circular imports
            from ..data.data_loader import DataLoader
            from ..analysis.enhanced_wave_detector import EnhancedWaveDetector
            from ..analysis.technical_confluence import TechnicalConfluenceAnalyzer
            from ..analysis.enhanced_sr_detector import EnhancedSRDetector
            
            data_loader = DataLoader()
            wave_detector = EnhancedWaveDetector()
            confluence_analyzer = TechnicalConfluenceAnalyzer()
            sr_detector = EnhancedSRDetector()
            
            # Load market data
            market_data = data_loader.get_yahoo_data(symbol, period='5y', interval=timeframe)
            
            if market_data.empty or len(market_data) < self.lookback_window + self.prediction_horizon:
                self.logger.warning(f"Insufficient data for {symbol} {timeframe}")
                return []
            
            # Filter to date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Find valid analysis periods
            valid_data = market_data[(market_data.index >= start_dt) & 
                                   (market_data.index <= end_dt)]
            
            if len(valid_data) < self.lookback_window:
                return []
            
            results = []
            
            # Rolling window analysis
            for i in range(self.lookback_window, len(market_data) - self.prediction_horizon, self.step_size):
                analysis_data = market_data.iloc[:i]
                future_data = market_data.iloc[i:i + self.prediction_horizon]
                
                current_timestamp = market_data.index[i]
                
                # Skip if outside date range
                if current_timestamp < start_dt or current_timestamp > end_dt:
                    continue
                
                # Run Elliott Wave analysis
                elliott_result = wave_detector.detect_elliott_waves(analysis_data, symbol)
                
                if not elliott_result or elliott_result.get('validation_score', 0) < 0.2:
                    continue
                
                # Run confluence analysis
                target_zones = confluence_analyzer.analyze_target_zones(
                    analysis_data, elliott_result, timeframe
                )
                
                if not target_zones:
                    continue
                
                # Run S/R analysis for additional context
                sr_levels = sr_detector.detect_sr_levels(analysis_data)
                
                # Process each target zone
                for target_zone in target_zones[:5]:  # Top 5 targets
                    backtest_result = self._create_backtest_result(
                        symbol, timeframe, current_timestamp, analysis_data, 
                        future_data, elliott_result, target_zone, sr_levels
                    )
                    
                    if backtest_result:
                        results.append(backtest_result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error backtesting {symbol} {timeframe}: {e}")
            return []
    
    def _create_backtest_result(self, symbol: str, timeframe: str, timestamp: datetime,
                              analysis_data: pd.DataFrame, future_data: pd.DataFrame,
                              elliott_result: Dict, target_zone, sr_levels: Dict) -> Optional[BacktestResult]:
        """Create a single backtest result with all features and labels"""
        try:
            current_price = analysis_data['close'].iloc[-1]
            target_price = target_zone.price_level
            
            # Extract market features
            market_features = self._extract_market_features(analysis_data)
            
            # Extract Elliott Wave quality features
            ew_features = self._extract_elliott_wave_features(elliott_result, analysis_data)
            
            # Extract confluence breakdown
            confluence_breakdown = self._extract_confluence_breakdown(target_zone)
            
            # Calculate outcome labels
            outcome_labels = self._calculate_outcome_labels(
                current_price, target_price, future_data
            )
            
            # Create target zone bounds
            zone_width = abs(target_price - current_price) * 0.01  # 1% of distance
            target_zone_low = target_price - zone_width
            target_zone_high = target_price + zone_width
            
            result = BacktestResult(
                # Market Context
                symbol=symbol,
                timestamp=timestamp,
                timeframe=timeframe,
                current_price=current_price,
                
                # Elliott Wave Analysis
                wave_count=elliott_result.get('wave_structure', 'unknown'),
                projected_wave=target_zone.wave_target,
                wave_structure=elliott_result.get('pattern_type', 'unknown'),
                validation_score=elliott_result.get('validation_score', 0),
                
                # Target Zone Details
                target_price=target_price,
                target_zone_low=target_zone_low,
                target_zone_high=target_zone_high,
                confluence_score=target_zone.confluence_score,
                confluences=target_zone.confluences[:10],  # Top 10 confluences
                
                # Market Features
                **market_features,
                
                # Elliott Wave Quality
                **ew_features,
                
                # Confluence Breakdown
                **confluence_breakdown,
                
                # Outcome Labels
                **outcome_labels,
                
                # Risk Management
                risk_reward_ratio=target_zone.risk_reward_ratio,
                maximum_drawdown=outcome_labels.get('max_adverse_move', 0),
                confidence_level=target_zone.confidence_level
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error creating backtest result: {e}")
            return None
    
    def _extract_market_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract market context features"""
        try:
            closes = data['close']
            highs = data['high']
            lows = data['low']
            volumes = data['volume']
            
            # Technical indicators
            rsi = self._calculate_rsi(closes).iloc[-1]
            macd_signal = self._get_macd_signal(closes)
            
            # Volume analysis
            avg_volume = volumes.rolling(20).mean().iloc[-1]
            volume_ratio = volumes.iloc[-1] / avg_volume if avg_volume > 0 else 1.0
            
            # Volatility
            volatility = closes.pct_change().rolling(20).std().iloc[-1]
            
            # Trend strength
            sma_20 = closes.rolling(20).mean().iloc[-1]
            sma_50 = closes.rolling(50).mean().iloc[-1]
            trend_strength = abs(sma_20 - sma_50) / sma_50 if sma_50 > 0 else 0
            
            return {
                'rsi': rsi,
                'macd_signal': macd_signal,
                'volume_ratio': volume_ratio,
                'volatility': volatility,
                'trend_strength': trend_strength
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting market features: {e}")
            return {
                'rsi': 50.0,
                'macd_signal': 0.0,
                'volume_ratio': 1.0,
                'volatility': 0.02,
                'trend_strength': 0.1
            }
    
    def _extract_elliott_wave_features(self, elliott_result: Dict, data: pd.DataFrame) -> Dict[str, float]:
        """Extract Elliott Wave quality features"""
        try:
            waves = elliott_result.get('waves', [])
            
            # Fibonacci alignment
            fibonacci_alignment = elliott_result.get('fibonacci_alignment', 0.6)
            
            # Rule compliance
            rule_compliance = elliott_result.get('rule_compliance', {})
            if isinstance(rule_compliance, dict):
                compliance_values = [v for v in rule_compliance.values() if isinstance(v, (int, float))]
                rule_compliance_score = np.mean(compliance_values) if compliance_values else 0.6
            else:
                rule_compliance_score = 0.6
            
            # Pattern clarity (based on validation score and wave count)
            pattern_clarity = elliott_result.get('validation_score', 0) * len(waves) / 5.0
            pattern_clarity = min(pattern_clarity, 1.0)
            
            # Wave proportions
            wave_proportions = self._calculate_wave_proportions(waves)
            
            return {
                'fibonacci_alignment': fibonacci_alignment,
                'rule_compliance_score': rule_compliance_score,
                'pattern_clarity': pattern_clarity,
                'wave_proportions': wave_proportions
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting Elliott Wave features: {e}")
            return {
                'fibonacci_alignment': 0.6,
                'rule_compliance_score': 0.6,
                'pattern_clarity': 0.6,
                'wave_proportions': {}
            }
    
    def _extract_confluence_breakdown(self, target_zone) -> Dict[str, int]:
        """Extract detailed confluence breakdown"""
        try:
            confluences = target_zone.confluences
            
            # Count different types of confluences
            fibonacci_count = sum(1 for c in confluences if 'fibonacci' in c.lower() or 'fib' in c.lower())
            sr_count = sum(1 for c in confluences if any(term in c.lower() for term in ['support', 'resistance', 'level']))
            momentum_count = sum(1 for c in confluences if any(term in c.lower() for term in ['rsi', 'macd', 'momentum']))
            pattern_count = sum(1 for c in confluences if any(term in c.lower() for term in ['pattern', 'triangle', 'channel']))
            volume_count = sum(1 for c in confluences if 'volume' in c.lower())
            harmonic_count = sum(1 for c in confluences if any(term in c.lower() for term in ['harmonic', 'gartley', 'butterfly']))
            
            return {
                'fibonacci_confluences': fibonacci_count,
                'sr_confluences': sr_count,
                'momentum_confluences': momentum_count,
                'pattern_confluences': pattern_count,
                'volume_confluences': volume_count,
                'harmonic_confluences': harmonic_count
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting confluence breakdown: {e}")
            return {
                'fibonacci_confluences': 0,
                'sr_confluences': 0,
                'momentum_confluences': 0,
                'pattern_confluences': 0,
                'volume_confluences': 0,
                'harmonic_confluences': 0
            }
    
    def _calculate_outcome_labels(self, current_price: float, target_price: float, 
                                future_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate ground truth labels for the prediction"""
        try:
            tolerance = abs(target_price - current_price) * self.target_tolerance
            
            hit = False
            hit_accuracy = 0.0
            time_to_hit = None
            max_adverse_move = 0.0
            max_favorable_move = 0.0
            final_outcome = "miss"
            
            for i, (_, row) in enumerate(future_data.iterrows()):
                # Calculate moves
                price_change = row['close'] - current_price
                adverse_move = abs(price_change) / current_price
                
                if (target_price > current_price and price_change < 0) or \
                   (target_price < current_price and price_change > 0):
                    max_adverse_move = max(max_adverse_move, adverse_move)
                else:
                    favorable_move = abs(price_change) / current_price
                    max_favorable_move = max(max_favorable_move, favorable_move)
                
                # Check for target hit
                if not hit:
                    hit_condition = False
                    actual_hit_price = target_price
                    
                    # Check close price
                    if abs(row['close'] - target_price) <= tolerance:
                        hit_condition = True
                        actual_hit_price = row['close']
                    
                    # Check high/low for wick touches
                    elif target_price > current_price and row['high'] >= target_price - tolerance:
                        hit_condition = True
                        actual_hit_price = min(row['high'], target_price + tolerance)
                    elif target_price < current_price and row['low'] <= target_price + tolerance:
                        hit_condition = True
                        actual_hit_price = max(row['low'], target_price - tolerance)
                    
                    if hit_condition:
                        hit = True
                        time_to_hit = i + 1
                        hit_accuracy = 1.0 - (abs(actual_hit_price - target_price) / tolerance)
                        hit_accuracy = max(0, min(1, hit_accuracy))
                        final_outcome = "hit"
                        break
            
            # Check for partial hit (came close but didn't quite reach)
            if not hit:
                closest_distance = float('inf')
                for _, row in future_data.iterrows():
                    distance = abs(row['close'] - target_price)
                    closest_distance = min(closest_distance, distance)
                
                if closest_distance <= tolerance * 2:  # Within 2x tolerance
                    final_outcome = "partial"
                    hit_accuracy = 0.5 * (1.0 - (closest_distance - tolerance) / tolerance)
                    hit_accuracy = max(0, min(0.5, hit_accuracy))
            
            return {
                'hit': hit,
                'hit_accuracy': hit_accuracy,
                'time_to_hit': time_to_hit,
                'max_adverse_move': max_adverse_move,
                'max_favorable_move': max_favorable_move,
                'final_outcome': final_outcome
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating outcome labels: {e}")
            return {
                'hit': False,
                'hit_accuracy': 0.0,
                'time_to_hit': None,
                'max_adverse_move': 0.0,
                'max_favorable_move': 0.0,
                'final_outcome': "miss"
            }
    
    def _calculate_wave_proportions(self, waves: List[Dict]) -> Dict[str, float]:
        """Calculate wave proportion relationships"""
        try:
            if len(waves) < 3:
                return {}
            
            proportions = {}
            
            # Calculate wave lengths
            wave_lengths = []
            for wave in waves:
                length = abs(wave.get('end_price', 0) - wave.get('start_price', 0))
                wave_lengths.append(length)
            
            # Calculate common ratios
            if len(wave_lengths) >= 3:
                if wave_lengths[0] > 0:
                    proportions['wave_3_to_1_ratio'] = wave_lengths[2] / wave_lengths[0]
                if len(wave_lengths) >= 5 and wave_lengths[0] > 0:
                    proportions['wave_5_to_1_ratio'] = wave_lengths[4] / wave_lengths[0]
                if len(wave_lengths) >= 4 and wave_lengths[2] > 0:
                    proportions['wave_4_to_3_ratio'] = wave_lengths[3] / wave_lengths[2]
            
            return proportions
            
        except Exception as e:
            self.logger.error(f"Error calculating wave proportions: {e}")
            return {}
    
    def _save_dataset(self, results: List[BacktestResult], start_date: str, end_date: str) -> str:
        """Save the complete dataset"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"elliott_wave_dataset_{start_date}_{end_date}_{timestamp}.json"
            filepath = self.output_dir / filename
            
            # Convert to serializable format
            serializable_data = []
            for result in results:
                data_dict = asdict(result)
                # Convert datetime to string
                if isinstance(data_dict['timestamp'], datetime):
                    data_dict['timestamp'] = data_dict['timestamp'].isoformat()
                serializable_data.append(data_dict)
            
            # Save as JSON
            with open(filepath, 'w') as f:
                json.dump(serializable_data, f, indent=2, default=str)
            
            self.logger.info(f"💾 Dataset saved: {filepath}")
            
            # Also save as CSV for easier analysis
            csv_filepath = filepath.with_suffix('.csv')
            df = pd.DataFrame(serializable_data)
            df.to_csv(csv_filepath, index=False)
            self.logger.info(f"💾 CSV dataset saved: {csv_filepath}")
            
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error saving dataset: {e}")
            return ""
    
    def _generate_dataset_summary(self, results: List[BacktestResult], dataset_path: str) -> None:
        """Generate summary statistics for the dataset"""
        try:
            if not results:
                return
            
            # Create summary statistics
            df = pd.DataFrame([asdict(result) for result in results])
            
            summary = {
                'dataset_info': {
                    'total_samples': len(results),
                    'symbols': df['symbol'].nunique(),
                    'symbol_list': df['symbol'].unique().tolist(),
                    'timeframes': df['timeframe'].unique().tolist(),
                    'date_range': {
                        'start': df['timestamp'].min(),
                        'end': df['timestamp'].max()
                    }
                },
                'target_performance': {
                    'hit_rate': df['hit'].mean(),
                    'average_hit_accuracy': df[df['hit']]['hit_accuracy'].mean() if df['hit'].any() else 0,
                    'average_time_to_hit': df[df['hit']]['time_to_hit'].mean() if df['hit'].any() else 0,
                    'outcome_distribution': df['final_outcome'].value_counts().to_dict()
                },
                'confluence_analysis': {
                    'average_confluence_score': df['confluence_score'].mean(),
                    'confluence_score_distribution': df['confluence_score'].value_counts().sort_index().to_dict(),
                    'high_confidence_hit_rate': df[df['confidence_level'] == 'HIGH']['hit'].mean() if (df['confidence_level'] == 'HIGH').any() else 0,
                    'medium_confidence_hit_rate': df[df['confidence_level'] == 'MEDIUM']['hit'].mean() if (df['confidence_level'] == 'MEDIUM').any() else 0
                },
                'elliott_wave_quality': {
                    'average_validation_score': df['validation_score'].mean(),
                    'wave_structure_distribution': df['wave_structure'].value_counts().to_dict(),
                    'fibonacci_alignment_avg': df['fibonacci_alignment'].mean()
                },
                'risk_metrics': {
                    'average_max_adverse_move': df['max_adverse_move'].mean(),
                    'average_risk_reward_ratio': df['risk_reward_ratio'].mean(),
                    'maximum_drawdown_avg': df['maximum_drawdown'].mean()
                }
            }
            
            # Save summary
            summary_path = Path(dataset_path).with_suffix('.summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.info(f"📊 Dataset summary saved: {summary_path}")
            
            # Log key metrics
            self.logger.info(f"📈 Hit Rate: {summary['target_performance']['hit_rate']:.3f}")
            self.logger.info(f"🎯 Avg Confluence Score: {summary['confluence_analysis']['average_confluence_score']:.1f}")
            self.logger.info(f"⚡ High Confidence Hit Rate: {summary['confluence_analysis']['high_confidence_hit_rate']:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error generating dataset summary: {e}")
    
    def load_dataset(self, dataset_path: str) -> List[BacktestResult]:
        """Load a previously generated dataset"""
        try:
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            
            results = []
            for item in data:
                # Convert timestamp back to datetime
                if isinstance(item['timestamp'], str):
                    item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                
                result = BacktestResult(**item)
                results.append(result)
            
            self.logger.info(f"📂 Loaded {len(results)} samples from {dataset_path}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            return []
    
    # Helper methods (same as in training framework)
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _get_macd_signal(self, prices: pd.Series) -> float:
        """Get MACD signal as numeric value"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        return 1.0 if macd.iloc[-1] > signal.iloc[-1] else 0.0
