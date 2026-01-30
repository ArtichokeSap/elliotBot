"""
Integration Tests - Full Workflow
"""
import pytest
import pandas as pd
from src.data.data_loader import DataLoader
from src.analysis.wave_detector import WaveDetector
from src.analysis.fibonacci import FibonacciAnalyzer
from src.trading.strategy import ElliottWaveStrategy
from src.trading.backtester import BacktestEngine
from src.visualization.visualizer import WaveVisualizer


class TestIntegration:
    """Test full workflow integration."""
    
    def test_complete_analysis_pipeline(self, wave_pattern_data):
        """Test complete analysis pipeline from data to signals."""
        # 1. Data is already loaded (wave_pattern_data fixture)
        
        # 2. Detect waves
        detector = WaveDetector()
        waves = detector.detect_waves(wave_pattern_data)
        assert len(waves) >= 0
        
        # 3. Analyze Fibonacci if waves exist
        if len(waves) > 0:
            analyzer = FibonacciAnalyzer()
            high = wave_pattern_data['high'].max()
            low = wave_pattern_data['low'].min()
            fib_analysis = analyzer.calculate_retracement_levels(high, low)
            assert len(fib_analysis) > 0
        
        # 4. Generate trading signals
        strategy = ElliottWaveStrategy()
        signals = strategy.generate_signals(wave_pattern_data, waves)
        assert isinstance(signals, list)
        
        # 5. Visualize results
        visualizer = WaveVisualizer()
        fig = visualizer.plot_waves(wave_pattern_data, waves)
        assert fig is not None
    
    def test_backtest_workflow(self, wave_pattern_data):
        """Test complete backtesting workflow."""
        # Create strategy
        strategy = ElliottWaveStrategy()
        
        # Run backtest
        engine = BacktestEngine(initial_capital=10000)
        results = engine.run_backtest(wave_pattern_data, strategy)
        
        # Verify results exist
        assert isinstance(results, dict)
        assert len(results) > 0
    
    def test_detection_and_visualization(self, sample_data):
        """Test wave detection followed by visualization."""
        detector = WaveDetector()
        waves = detector.detect_waves(sample_data)
        
        visualizer = WaveVisualizer()
        fig = visualizer.plot_waves(sample_data, waves, title="Integration Test")
        
        assert fig is not None
        assert len(fig.data) > 0
    
    def test_error_handling_pipeline(self):
        """Test that pipeline handles errors gracefully."""
        # Empty data should be handled
        empty_data = pd.DataFrame()
        
        detector = WaveDetector()
        with pytest.raises(Exception):
            detector.detect_waves(empty_data)
