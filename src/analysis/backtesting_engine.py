"""
Elliott Wave Backtesting Engine
Provides comprehensive backtesting with forward-walking validation and confidence scoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
import os
from enum import Enum

logger = logging.getLogger(__name__)

class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class WavePosition(Enum):
    WAVE_1 = "wave_1"
    WAVE_2 = "wave_2"
    WAVE_3 = "wave_3"
    WAVE_4 = "wave_4"
    WAVE_5 = "wave_5"
    WAVE_A = "wave_a"
    WAVE_B = "wave_b"
    WAVE_C = "wave_c"

@dataclass
class Trade:
    """Represents a single trade"""
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    signal_type: SignalType = SignalType.HOLD
    wave_position: Optional[WavePosition] = None
    confidence: float = 0.0
    size: float = 1.0
    pnl: float = 0.0
    pnl_percent: float = 0.0
    max_drawdown: float = 0.0
    max_runup: float = 0.0
    trade_duration: Optional[timedelta] = None
    elliott_pattern: str = ""
    
    def calculate_pnl(self):
        """Calculate trade P&L"""
        if self.exit_price > 0 and self.entry_price > 0:
            if self.signal_type == SignalType.BUY:
                self.pnl = (self.exit_price - self.entry_price) * self.size
                self.pnl_percent = (self.exit_price / self.entry_price - 1) * 100
            elif self.signal_type == SignalType.SELL:
                self.pnl = (self.entry_price - self.exit_price) * self.size
                self.pnl_percent = (1 - self.exit_price / self.entry_price) * 100
        
        if self.exit_time and self.entry_time:
            self.trade_duration = self.exit_time - self.entry_time

@dataclass
class BacktestMetrics:
    """Comprehensive backtesting metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_percent: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    average_trade_duration: Optional[timedelta] = None
    total_return: float = 0.0
    annual_return: float = 0.0
    volatility: float = 0.0
    confidence_score: float = 0.0
    
    # Elliott Wave specific metrics
    wave_accuracy: Dict[str, float] = field(default_factory=dict)
    pattern_performance: Dict[str, float] = field(default_factory=dict)
    best_performing_waves: List[str] = field(default_factory=list)
    worst_performing_waves: List[str] = field(default_factory=list)

class BacktestingEngine:
    """Comprehensive Elliott Wave backtesting engine"""
    
    def __init__(self, initial_capital: float = 100000.0,
                 commission: float = 0.001,  # 0.1% commission
                 slippage: float = 0.0005):  # 0.05% slippage
        self.logger = logging.getLogger(__name__)
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.signals: List[Dict] = []
        
    def run_backtest(self, market_data: pd.DataFrame, 
                     wave_signals: List[Dict],
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None) -> BacktestMetrics:
        """Run comprehensive backtest with Elliott Wave signals"""
        try:
            self.logger.info("Starting Elliott Wave backtest")
            
            # Filter data by date range if specified
            if start_date or end_date:
                market_data = self._filter_data_by_date(market_data, start_date, end_date)
                wave_signals = self._filter_signals_by_date(wave_signals, start_date, end_date)
            
            # Initialize backtest state
            current_capital = self.initial_capital
            current_position = None
            position_size = 0.0
            
            # Track equity curve
            self.equity_curve = []
            self.trades = []
            
            # Process each signal
            for signal in wave_signals:
                # Get market data for signal time
                signal_time = signal.get('timestamp')
                if signal_time is None:
                    continue
                
                # Find corresponding market data
                market_row = self._get_market_data_at_time(market_data, signal_time)
                if market_row is None:
                    continue
                
                current_price = market_row['close']
                
                # Process signal
                trade = self._process_signal(signal, current_price, current_capital)
                
                if trade:
                    self.trades.append(trade)
                    
                    # Update capital and position
                    if trade.signal_type in [SignalType.BUY, SignalType.SELL]:
                        current_position = trade
                        position_size = trade.size
                    elif current_position and trade.exit_price > 0:
                        # Close position
                        current_position.exit_time = trade.entry_time
                        current_position.exit_price = trade.exit_price
                        current_position.calculate_pnl()
                        
                        # Update capital
                        current_capital += current_position.pnl
                        current_position = None
                        position_size = 0.0
                
                # Record equity curve point
                self.equity_curve.append((signal_time, current_capital))
            
            # Calculate comprehensive metrics
            metrics = self._calculate_metrics()
            
            self.logger.info(f"Backtest complete. Total trades: {metrics.total_trades}, "
                           f"Win rate: {metrics.win_rate:.2f}%, Total return: {metrics.total_return:.2f}%")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in backtesting: {e}")
            return BacktestMetrics()
    
    def run_forward_walking_validation(self, market_data: pd.DataFrame,
                                     window_size: int = 252,  # 1 year
                                     step_size: int = 21) -> Dict:
        """Run forward-walking validation for robust testing"""
        try:
            self.logger.info("Starting forward-walking validation")
            
            results = []
            start_idx = 0
            
            while start_idx + window_size < len(market_data):
                # Define training and testing windows
                train_end_idx = start_idx + window_size
                test_end_idx = min(train_end_idx + step_size, len(market_data))
                
                train_data = market_data.iloc[start_idx:train_end_idx]
                test_data = market_data.iloc[train_end_idx:test_end_idx]
                
                # Generate signals for training period (for pattern learning)
                train_signals = self._generate_signals_for_period(train_data)
                
                # Generate signals for test period
                test_signals = self._generate_signals_for_period(test_data)
                
                # Run backtest on test period
                test_metrics = self.run_backtest(test_data, test_signals)
                
                results.append({
                    'period_start': train_data.index[0],
                    'period_end': test_data.index[-1],
                    'train_size': len(train_data),
                    'test_size': len(test_data),
                    'metrics': test_metrics
                })
                
                # Move to next window
                start_idx += step_size
            
            # Aggregate results
            aggregated = self._aggregate_forward_walking_results(results)
            
            self.logger.info(f"Forward-walking validation complete. "
                           f"Average win rate: {aggregated['avg_win_rate']:.2f}%")
            
            return aggregated
            
        except Exception as e:
            self.logger.error(f"Error in forward-walking validation: {e}")
            return {}
    
    def _generate_signals_for_period(self, data: pd.DataFrame) -> List[Dict]:
        """Generate Elliott Wave signals for a specific period"""
        try:
            # Import here to avoid circular imports
            from ..analysis.enhanced_wave_detector import EnhancedWaveDetector
            
            detector = EnhancedWaveDetector()
            analysis_result = detector.detect_elliott_waves(data, "BACKTEST_SYMBOL")
            
            waves = analysis_result.get('waves', [])
            signals = []
            
            for i, wave in enumerate(waves):
                # Generate buy/sell signals based on wave position
                signal_type = self._determine_signal_from_wave(wave, i, waves)
                
                if signal_type != SignalType.HOLD:
                    signals.append({
                        'timestamp': wave.get('end_time', data.index[-1]),
                        'signal_type': signal_type,
                        'price': wave.get('end_price', data['close'].iloc[-1]),
                        'confidence': wave.get('confidence', 0.5),
                        'wave_position': wave.get('position', 'unknown'),
                        'pattern': wave.get('pattern', 'impulse')
                    })
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return []
    
    def _determine_signal_from_wave(self, wave: Dict, index: int, all_waves: List[Dict]) -> SignalType:
        """Determine trading signal from Elliott Wave position"""
        try:
            wave_position = wave.get('position', '').lower()
            direction = wave.get('direction', '').lower()
            confidence = wave.get('confidence', 0.0)
            
            # Only trade high-confidence signals
            if confidence < 0.6:
                return SignalType.HOLD
            
            # Wave 3 and Wave 5 are typically strong buy signals (in impulse waves)
            if wave_position in ['wave_3', 'wave_5'] and direction == 'bullish':
                return SignalType.BUY
            
            # Wave A and Wave C are typically sell signals (in corrective waves)
            if wave_position in ['wave_a', 'wave_c'] and direction == 'bearish':
                return SignalType.SELL
            
            # Wave 2 and Wave 4 corrections might be buy opportunities
            if wave_position in ['wave_2', 'wave_4'] and direction == 'bearish':
                # Look for reversal signal after correction
                if index < len(all_waves) - 1:
                    next_wave = all_waves[index + 1]
                    if next_wave.get('direction', '').lower() == 'bullish':
                        return SignalType.BUY
            
            return SignalType.HOLD
            
        except Exception as e:
            self.logger.debug(f"Error determining signal: {e}")
            return SignalType.HOLD
    
    def _process_signal(self, signal: Dict, current_price: float, 
                       current_capital: float) -> Optional[Trade]:
        """Process a trading signal and create trade"""
        try:
            signal_type = SignalType(signal['signal_type'])
            confidence = signal.get('confidence', 0.5)
            
            # Position sizing based on confidence and capital
            risk_per_trade = 0.02  # 2% risk per trade
            position_size = (current_capital * risk_per_trade * confidence)
            
            # Account for commission and slippage
            adjusted_price = current_price
            if signal_type == SignalType.BUY:
                adjusted_price *= (1 + self.slippage + self.commission)
            elif signal_type == SignalType.SELL:
                adjusted_price *= (1 - self.slippage - self.commission)
            
            trade = Trade(
                entry_time=signal['timestamp'],
                entry_price=adjusted_price,
                signal_type=signal_type,
                confidence=confidence,
                size=position_size / adjusted_price,  # Number of shares/units
                wave_position=WavePosition(signal.get('wave_position', 'unknown')),
                elliott_pattern=signal.get('pattern', 'unknown')
            )
            
            return trade
            
        except Exception as e:
            self.logger.debug(f"Error processing signal: {e}")
            return None
    
    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate comprehensive backtesting metrics"""
        try:
            if not self.trades:
                return BacktestMetrics()
            
            # Basic trade statistics
            completed_trades = [t for t in self.trades if t.exit_time is not None]
            winning_trades = [t for t in completed_trades if t.pnl > 0]
            losing_trades = [t for t in completed_trades if t.pnl < 0]
            
            total_trades = len(completed_trades)
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            
            # Calculate metrics
            metrics = BacktestMetrics()
            metrics.total_trades = total_trades
            metrics.winning_trades = win_count
            metrics.losing_trades = loss_count
            metrics.win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
            
            # P&L calculations
            metrics.total_pnl = sum(t.pnl for t in completed_trades)
            metrics.total_pnl_percent = (metrics.total_pnl / self.initial_capital * 100)
            
            if winning_trades:
                metrics.average_win = np.mean([t.pnl for t in winning_trades])
            if losing_trades:
                metrics.average_loss = np.mean([t.pnl for t in losing_trades])
            
            # Profit factor
            gross_profit = sum(t.pnl for t in winning_trades)
            gross_loss = abs(sum(t.pnl for t in losing_trades))
            metrics.profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
            
            # Drawdown calculations
            if self.equity_curve:
                equity_values = [eq[1] for eq in self.equity_curve]
                metrics.max_drawdown = self._calculate_max_drawdown(equity_values)
                metrics.max_drawdown_percent = (metrics.max_drawdown / self.initial_capital * 100)
            
            # Risk-adjusted returns
            if completed_trades:
                returns = [t.pnl_percent for t in completed_trades]
                metrics.sharpe_ratio = self._calculate_sharpe_ratio(returns)
                metrics.sortino_ratio = self._calculate_sortino_ratio(returns)
                metrics.volatility = np.std(returns) if returns else 0
            
            # Time-based metrics
            if completed_trades:
                durations = [t.trade_duration for t in completed_trades if t.trade_duration]
                if durations:
                    metrics.average_trade_duration = np.mean(durations)
            
            # Elliott Wave specific metrics
            metrics.wave_accuracy = self._calculate_wave_accuracy(completed_trades)
            metrics.pattern_performance = self._calculate_pattern_performance(completed_trades)
            
            # Overall confidence score
            metrics.confidence_score = self._calculate_overall_confidence(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return BacktestMetrics()
    
    def _calculate_max_drawdown(self, equity_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        try:
            max_dd = 0.0
            peak = equity_values[0]
            
            for value in equity_values:
                if value > peak:
                    peak = value
                else:
                    drawdown = peak - value
                    max_dd = max(max_dd, drawdown)
            
            return max_dd
            
        except Exception as e:
            self.logger.debug(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        try:
            if not returns or np.std(returns) == 0:
                return 0.0
            
            excess_returns = np.mean(returns) - risk_free_rate / 252  # Daily risk-free rate
            return excess_returns / np.std(returns) * np.sqrt(252)  # Annualized
            
        except Exception as e:
            self.logger.debug(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        try:
            if not returns:
                return 0.0
            
            excess_returns = np.mean(returns) - risk_free_rate / 252
            downside_returns = [r for r in returns if r < 0]
            
            if not downside_returns:
                return float('inf')
            
            downside_deviation = np.std(downside_returns)
            return excess_returns / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0.0
            
        except Exception as e:
            self.logger.debug(f"Error calculating Sortino ratio: {e}")
            return 0.0
    
    def _calculate_wave_accuracy(self, trades: List[Trade]) -> Dict[str, float]:
        """Calculate accuracy by Elliott Wave position"""
        try:
            wave_stats = {}
            
            for trade in trades:
                if trade.wave_position and trade.wave_position != WavePosition.WAVE_1:  # Skip unknown
                    wave_name = trade.wave_position.value
                    
                    if wave_name not in wave_stats:
                        wave_stats[wave_name] = {'wins': 0, 'total': 0}
                    
                    wave_stats[wave_name]['total'] += 1
                    if trade.pnl > 0:
                        wave_stats[wave_name]['wins'] += 1
            
            # Calculate accuracy percentages
            wave_accuracy = {}
            for wave, stats in wave_stats.items():
                accuracy = (stats['wins'] / stats['total'] * 100) if stats['total'] > 0 else 0
                wave_accuracy[wave] = accuracy
            
            return wave_accuracy
            
        except Exception as e:
            self.logger.debug(f"Error calculating wave accuracy: {e}")
            return {}
    
    def _calculate_pattern_performance(self, trades: List[Trade]) -> Dict[str, float]:
        """Calculate performance by Elliott Wave pattern"""
        try:
            pattern_stats = {}
            
            for trade in trades:
                pattern = trade.elliott_pattern
                if pattern and pattern != 'unknown':
                    if pattern not in pattern_stats:
                        pattern_stats[pattern] = []
                    pattern_stats[pattern].append(trade.pnl_percent)
            
            # Calculate average performance
            pattern_performance = {}
            for pattern, pnls in pattern_stats.items():
                pattern_performance[pattern] = np.mean(pnls) if pnls else 0.0
            
            return pattern_performance
            
        except Exception as e:
            self.logger.debug(f"Error calculating pattern performance: {e}")
            return {}
    
    def _calculate_overall_confidence(self, metrics: BacktestMetrics) -> float:
        """Calculate overall confidence score for the strategy"""
        try:
            confidence_factors = []
            
            # Win rate factor (0-1)
            if metrics.win_rate >= 60:
                confidence_factors.append(1.0)
            elif metrics.win_rate >= 50:
                confidence_factors.append(0.8)
            elif metrics.win_rate >= 40:
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.3)
            
            # Profit factor (0-1)
            if metrics.profit_factor >= 2.0:
                confidence_factors.append(1.0)
            elif metrics.profit_factor >= 1.5:
                confidence_factors.append(0.8)
            elif metrics.profit_factor >= 1.0:
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.2)
            
            # Sharpe ratio factor (0-1)
            if metrics.sharpe_ratio >= 1.5:
                confidence_factors.append(1.0)
            elif metrics.sharpe_ratio >= 1.0:
                confidence_factors.append(0.8)
            elif metrics.sharpe_ratio >= 0.5:
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.3)
            
            # Sample size factor
            if metrics.total_trades >= 100:
                confidence_factors.append(1.0)
            elif metrics.total_trades >= 50:
                confidence_factors.append(0.8)
            elif metrics.total_trades >= 20:
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.4)
            
            return np.mean(confidence_factors)
            
        except Exception as e:
            self.logger.debug(f"Error calculating overall confidence: {e}")
            return 0.5
    
    def generate_report(self, metrics: BacktestMetrics) -> str:
        """Generate comprehensive backtest report"""
        try:
            report = []
            report.append("=" * 60)
            report.append("ELLIOTT WAVE BACKTESTING REPORT")
            report.append("=" * 60)
            report.append("")
            
            # Basic Statistics
            report.append("BASIC STATISTICS:")
            report.append(f"Total Trades: {metrics.total_trades}")
            report.append(f"Winning Trades: {metrics.winning_trades}")
            report.append(f"Losing Trades: {metrics.losing_trades}")
            report.append(f"Win Rate: {metrics.win_rate:.2f}%")
            report.append("")
            
            # Performance Metrics
            report.append("PERFORMANCE METRICS:")
            report.append(f"Total P&L: ${metrics.total_pnl:,.2f}")
            report.append(f"Total Return: {metrics.total_pnl_percent:.2f}%")
            report.append(f"Average Win: ${metrics.average_win:,.2f}")
            report.append(f"Average Loss: ${metrics.average_loss:,.2f}")
            report.append(f"Profit Factor: {metrics.profit_factor:.2f}")
            report.append("")
            
            # Risk Metrics
            report.append("RISK METRICS:")
            report.append(f"Max Drawdown: ${metrics.max_drawdown:,.2f} ({metrics.max_drawdown_percent:.2f}%)")
            report.append(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
            report.append(f"Sortino Ratio: {metrics.sortino_ratio:.2f}")
            report.append(f"Volatility: {metrics.volatility:.2f}%")
            report.append("")
            
            # Elliott Wave Specific
            if metrics.wave_accuracy:
                report.append("ELLIOTT WAVE ACCURACY:")
                for wave, accuracy in metrics.wave_accuracy.items():
                    report.append(f"{wave}: {accuracy:.1f}%")
                report.append("")
            
            if metrics.pattern_performance:
                report.append("PATTERN PERFORMANCE:")
                for pattern, performance in metrics.pattern_performance.items():
                    report.append(f"{pattern}: {performance:.2f}%")
                report.append("")
            
            # Confidence Score
            report.append(f"OVERALL CONFIDENCE SCORE: {metrics.confidence_score:.2f}/1.00")
            report.append("")
            
            # Additional Information
            if metrics.average_trade_duration:
                report.append(f"Average Trade Duration: {metrics.average_trade_duration}")
            
            report.append("=" * 60)
            
            return "\n".join(report)
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return "Error generating backtest report"
    
    def _filter_data_by_date(self, data: pd.DataFrame, 
                           start_date: Optional[datetime], 
                           end_date: Optional[datetime]) -> pd.DataFrame:
        """Filter market data by date range"""
        try:
            filtered_data = data.copy()
            
            if start_date:
                filtered_data = filtered_data[filtered_data.index >= start_date]
            if end_date:
                filtered_data = filtered_data[filtered_data.index <= end_date]
            
            return filtered_data
            
        except Exception as e:
            self.logger.debug(f"Error filtering data by date: {e}")
            return data
    
    def _filter_signals_by_date(self, signals: List[Dict],
                              start_date: Optional[datetime],
                              end_date: Optional[datetime]) -> List[Dict]:
        """Filter signals by date range"""
        try:
            filtered_signals = []
            
            for signal in signals:
                signal_time = signal.get('timestamp')
                if signal_time:
                    if start_date and signal_time < start_date:
                        continue
                    if end_date and signal_time > end_date:
                        continue
                    filtered_signals.append(signal)
            
            return filtered_signals
            
        except Exception as e:
            self.logger.debug(f"Error filtering signals by date: {e}")
            return signals
    
    def _get_market_data_at_time(self, data: pd.DataFrame, 
                               timestamp: datetime) -> Optional[pd.Series]:
        """Get market data row closest to specified timestamp"""
        try:
            # Find closest timestamp
            if isinstance(data.index, pd.DatetimeIndex):
                closest_idx = data.index.get_indexer([timestamp], method='nearest')[0]
                return data.iloc[closest_idx]
            else:
                # If index is not datetime, use last available data point
                return data.iloc[-1]
                
        except Exception as e:
            self.logger.debug(f"Error getting market data at time: {e}")
            return None
    
    def _aggregate_forward_walking_results(self, results: List[Dict]) -> Dict:
        """Aggregate results from forward-walking validation"""
        try:
            if not results:
                return {}
            
            # Extract metrics from all periods
            all_metrics = [r['metrics'] for r in results]
            
            aggregated = {
                'num_periods': len(results),
                'avg_win_rate': np.mean([m.win_rate for m in all_metrics]),
                'avg_profit_factor': np.mean([m.profit_factor for m in all_metrics if m.profit_factor != float('inf')]),
                'avg_sharpe_ratio': np.mean([m.sharpe_ratio for m in all_metrics]),
                'avg_max_drawdown': np.mean([m.max_drawdown_percent for m in all_metrics]),
                'consistency_score': np.std([m.total_pnl_percent for m in all_metrics]),  # Lower is better
                'avg_confidence': np.mean([m.confidence_score for m in all_metrics]),
                'period_results': results
            }
            
            return aggregated
            
        except Exception as e:
            self.logger.error(f"Error aggregating forward-walking results: {e}")
            return {}
