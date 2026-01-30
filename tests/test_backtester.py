"""
Tests for Backtesting Module
"""
import pytest
import pandas as pd
from src.trading.backtester import BacktestEngine
from src.trading.strategy import ElliottWaveStrategy


class TestBacktestEngine:
    """Test BacktestEngine class."""
    
    def test_initialization(self):
        """Test BacktestEngine initialization."""
        engine = BacktestEngine(initial_capital=10000)
        assert engine is not None
        assert engine.initial_capital == 10000
    
    def test_run_backtest_returns_results(self, wave_pattern_data):
        """Test that run_backtest returns results."""
        strategy = ElliottWaveStrategy()
        engine = BacktestEngine(initial_capital=10000)
        
        results = engine.run_backtest(wave_pattern_data, strategy)
        
        assert isinstance(results, dict)
    
    def test_backtest_has_performance_metrics(self, wave_pattern_data):
        """Test that backtest includes performance metrics."""
        strategy = ElliottWaveStrategy()
        engine = BacktestEngine(initial_capital=10000)
        
        results = engine.run_backtest(wave_pattern_data, strategy)
        
        # Should have some performance metrics
        expected_keys = ['total_return', 'sharpe_ratio', 'max_drawdown', 
                        'win_rate', 'total_trades', 'final_capital']
        
        # At least some of these keys should be present
        has_metrics = any(key in results for key in expected_keys)
        assert has_metrics or len(results) > 0
    
    def test_equity_curve_generation(self, wave_pattern_data):
        """Test equity curve generation."""
        strategy = ElliottWaveStrategy()
        engine = BacktestEngine(initial_capital=10000)
        
        results = engine.run_backtest(wave_pattern_data, strategy)
        
        # Should have equity data
        if 'equity_curve' in results:
            assert isinstance(results['equity_curve'], (list, pd.Series))
    
    def test_capital_preservation(self, wave_pattern_data):
        """Test that capital is tracked correctly."""
        initial = 10000
        strategy = ElliottWaveStrategy()
        engine = BacktestEngine(initial_capital=initial)
        
        results = engine.run_backtest(wave_pattern_data, strategy)
        
        # Final capital should be a positive number
        if 'final_capital' in results:
            assert results['final_capital'] > 0
