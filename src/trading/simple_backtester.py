"""
Simple wrapper for backtesting functionality
"""

from .backtester import BacktestEngine

class Backtester:
    """Simple wrapper for the complex BacktestEngine for web app compatibility."""
    
    def __init__(self):
        self.engine = BacktestEngine()
    
    def run_backtest(self, symbol, data, wave_detector, start_capital=10000):
        """Simple interface for web app backtesting."""
        try:
            # Simple backtest results for web interface
            return {
                'symbol': symbol,
                'total_trades': 0,
                'profitable_trades': 0,
                'win_rate': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'start_capital': start_capital,
                'end_capital': start_capital,
                'trades': [],
                'message': 'Backtesting feature is under development'
            }
        except Exception as e:
            return {
                'symbol': symbol,
                'error': str(e),
                'total_trades': 0,
                'win_rate': 0.0,
                'total_return': 0.0
            }
