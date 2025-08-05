#!/usr/bin/env python3
"""
Comprehensive Elliott Wave Analysis Example

This example demonstrates the complete Elliott Wave analysis system:
1. Data fetching from Binance
2. Wave detection and validation
3. Future wave projections
4. Historical pattern matching
5. Comprehensive visualization

Usage:
    python examples/comprehensive_elliott_analysis.py
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.binance_data_fetcher import BinanceDataFetcher
from analysis.wave_detector import WaveDetector
from analysis.wave_validator import WaveValidator
from analysis.wave_projector import WaveProjector
from analysis.pattern_memory import PatternMemory, PatternCategory
from analysis.fibonacci import FibonacciAnalyzer
from visualization.visualizer import WaveVisualizer
from utils.logger import setup_logger

# Setup logging
setup_logger()
logger = logging.getLogger(__name__)


def main():
    """Main function demonstrating comprehensive Elliott Wave analysis."""
    
    print("🚀 Starting Comprehensive Elliott Wave Analysis")
    print("=" * 60)
    
    try:
        # 1. Initialize all components
        print("\n1. Initializing Components...")
        data_fetcher = BinanceDataFetcher()
        detector = WaveDetector()
        validator = WaveValidator()
        projector = WaveProjector()
        pattern_memory = PatternMemory()
        fibonacci_analyzer = FibonacciAnalyzer()
        visualizer = WaveVisualizer()
        
        print("✅ All components initialized")
        
        # 2. Fetch market data
        print("\n2. Fetching Market Data...")
        symbol = "BTCUSDT"
        timeframe = "1h"
        
        # Get historical data (last 7 days)
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)
        
        data = data_fetcher.get_historical_ohlcv(
            symbol=symbol,
            interval=timeframe,
            start_time=int(start_time.timestamp() * 1000),
            end_time=int(end_time.timestamp() * 1000),
            limit=1000
        )
        
        if data.empty:
            print("❌ No data received")
            return
        
        print(f"✅ Fetched {len(data)} data points for {symbol}")
        print(f"   Date range: {data.index[0]} to {data.index[-1]}")
        print(f"   Price range: ${data['low'].min():.2f} - ${data['high'].max():.2f}")
        
        # 3. Detect Elliott Waves
        print("\n3. Detecting Elliott Waves...")
        waves = detector.detect_waves(data)
        
        if not waves:
            print("❌ No waves detected")
            return
        
        print(f"✅ Detected {len(waves)} waves")
        for i, wave in enumerate(waves):
            print(f"   Wave {i+1}: {wave.wave_type.value} - {wave.price_change_pct:.1%} change")
        
        # 4. Validate wave pattern
        print("\n4. Validating Wave Pattern...")
        validation_result = validator.validate_wave_pattern(waves, data)
        
        print(f"✅ Validation complete")
        print(f"   Overall score: {validation_result.overall_score:.1%}")
        print(f"   Passed critical rules: {validation_result.passed_critical_rules}")
        print(f"   Warnings: {len(validation_result.warnings)}")
        
        if validation_result.warnings:
            for warning in validation_result.warnings[:3]:  # Show first 3 warnings
                print(f"   ⚠️  {warning}")
        
        # 5. Perform Fibonacci analysis
        print("\n5. Performing Fibonacci Analysis...")
        fibonacci_analysis = fibonacci_analyzer.analyze_waves(waves, data)
        
        print(f"✅ Fibonacci analysis complete")
        if fibonacci_analysis.levels:
            print(f"   Key levels: {len(fibonacci_analysis.levels)}")
            for level in fibonacci_analysis.levels[:3]:  # Show first 3 levels
                print(f"   📊 {level.level_type.value}: ${level.price:.2f}")
        
        # 6. Generate future projections
        print("\n6. Generating Future Projections...")
        current_wave = max(waves, key=lambda w: w.end_point.timestamp)
        
        projection_scenarios = projector.generate_comprehensive_projections(
            current_wave=current_wave,
            all_waves=waves,
            data=data,
            include_alternatives=True,
            max_scenarios=3
        )
        
        print(f"✅ Generated {len(projection_scenarios)} projection scenarios")
        for i, scenario in enumerate(projection_scenarios):
            print(f"   Scenario {i+1}: {scenario.scenario_name}")
            print(f"      Confidence: {scenario.overall_confidence:.1%}")
            print(f"      Primary: {scenario.primary_projection.description}")
            print(f"      Alternatives: {len(scenario.alternative_projections)}")
        
        # 7. Find similar historical patterns
        print("\n7. Finding Similar Historical Patterns...")
        pattern_matches = pattern_memory.find_similar_patterns(
            current_waves=waves,
            current_validation=validation_result,
            current_fibonacci=fibonacci_analysis,
            symbol=symbol,
            timeframe=timeframe,
            min_similarity=0.5,
            max_results=5
        )
        
        print(f"✅ Found {len(pattern_matches)} similar patterns")
        for i, match in enumerate(pattern_matches):
            print(f"   Match {i+1}: {match.similarity_score:.1%} similarity")
            print(f"      Quality: {match.match_quality}")
            print(f"      Historical: {match.historical_pattern.pattern_id}")
            if match.outcome_prediction:
                print(f"      Outcome: {match.outcome_prediction}")
        
        # 8. Store current pattern in memory
        print("\n8. Storing Pattern in Memory...")
        pattern_id = pattern_memory.store_pattern(
            symbol=symbol,
            timeframe=timeframe,
            waves=waves,
            validation_result=validation_result,
            fibonacci_analysis=fibonacci_analysis,
            market_condition="bullish",
            tags=["example", "comprehensive"],
            notes="Comprehensive analysis example pattern"
        )
        
        print(f"✅ Stored pattern: {pattern_id}")
        
        # 9. Get pattern memory statistics
        print("\n9. Pattern Memory Statistics...")
        stats = pattern_memory.get_pattern_statistics()
        
        print(f"✅ Memory statistics:")
        print(f"   Total patterns: {stats.total_patterns}")
        print(f"   Average confidence: {stats.average_confidence:.1%}")
        print(f"   Categories: {list(stats.patterns_by_category.keys())}")
        
        # 10. Create comprehensive visualization
        print("\n10. Creating Comprehensive Visualization...")
        
        # Create comprehensive analysis chart
        fig = visualizer.plot_comprehensive_analysis(
            data=data,
            waves=waves,
            projection_scenarios=projection_scenarios,
            pattern_matches=pattern_matches,
            validation_result=validation_result,
            fibonacci_analysis=fibonacci_analysis,
            title=f"Comprehensive Elliott Wave Analysis - {symbol}"
        )
        
        # Save the chart
        output_path = f"comprehensive_analysis_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
        fig.write_html(output_path)
        
        print(f"✅ Comprehensive analysis chart saved to: {output_path}")
        
        # 11. Create specialized dashboards
        print("\n11. Creating Specialized Dashboards...")
        
        # Projection dashboard
        projection_fig = visualizer.create_projection_dashboard(
            data=data,
            waves=waves,
            projection_scenarios=projection_scenarios,
            validation_result=validation_result,
            fibonacci_analysis=fibonacci_analysis,
            title=f"Projection Dashboard - {symbol}"
        )
        
        projection_path = f"projection_dashboard_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
        projection_fig.write_html(projection_path)
        print(f"✅ Projection dashboard saved to: {projection_path}")
        
        # Pattern memory dashboard
        memory_fig = visualizer.create_pattern_memory_dashboard(
            data=data,
            waves=waves,
            pattern_matches=pattern_matches,
            pattern_memory_stats={
                'total_patterns': stats.total_patterns,
                'average_confidence': stats.average_confidence,
                'patterns_by_category': len(stats.patterns_by_category),
                'recent_patterns': len(stats.recent_patterns)
            },
            validation_result=validation_result,
            fibonacci_analysis=fibonacci_analysis,
            title=f"Pattern Memory Dashboard - {symbol}"
        )
        
        memory_path = f"pattern_memory_dashboard_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
        memory_fig.write_html(memory_path)
        print(f"✅ Pattern memory dashboard saved to: {memory_path}")
        
        # 12. Summary and recommendations
        print("\n12. Analysis Summary and Recommendations...")
        print("=" * 60)
        
        # Current market position
        current_price = data['close'].iloc[-1]
        print(f"📊 Current {symbol} Price: ${current_price:.2f}")
        
        # Wave analysis summary
        print(f"🌊 Wave Analysis:")
        print(f"   Pattern: {validation_result.pattern_type}")
        print(f"   Confidence: {validation_result.overall_score:.1%}")
        print(f"   Current wave: {current_wave.wave_type.value}")
        
        # Projection summary
        if projection_scenarios:
            best_scenario = projection_scenarios[0]
            print(f"🔮 Best Projection:")
            print(f"   Scenario: {best_scenario.scenario_name}")
            print(f"   Confidence: {best_scenario.overall_confidence:.1%}")
            print(f"   Description: {best_scenario.primary_projection.description}")
            
            if best_scenario.primary_projection.fibonacci_targets:
                targets = best_scenario.primary_projection.fibonacci_targets
                print(f"   Targets: {', '.join([f'${t:.2f}' for t in targets[:3]])}")
        
        # Historical pattern insights
        if pattern_matches:
            best_match = pattern_matches[0]
            print(f"📚 Historical Pattern Insights:")
            print(f"   Best match: {best_match.similarity_score:.1%} similarity")
            print(f"   Quality: {best_match.match_quality}")
            if best_match.outcome_prediction:
                print(f"   Historical outcome: {best_match.outcome_prediction}")
            if best_match.predicted_targets:
                print(f"   Historical targets: {', '.join([f'${t:.2f}' for t in best_match.predicted_targets])}")
        
        # Risk assessment
        print(f"⚠️  Risk Assessment:")
        if validation_result.overall_score < 0.7:
            print(f"   Low confidence pattern - exercise caution")
        if pattern_matches and best_match.similarity_score < 0.8:
            print(f"   Limited historical similarity - consider alternatives")
        if projection_scenarios and best_scenario.overall_confidence < 0.6:
            print(f"   Low projection confidence - monitor closely")
        
        # Recommendations
        print(f"💡 Recommendations:")
        if validation_result.passed_critical_rules:
            print(f"   ✅ Pattern passes critical Elliott Wave rules")
        else:
            print(f"   ❌ Pattern violates critical rules - reconsider analysis")
        
        if best_scenario.primary_projection.invalidation_levels:
            invalidation = best_scenario.primary_projection.invalidation_levels[0]
            print(f"   🛑 Key invalidation level: ${invalidation:.2f}")
        
        print(f"   📈 Monitor price action for pattern confirmation")
        print(f"   🔄 Update analysis as new data becomes available")
        
        print("\n" + "=" * 60)
        print("🎉 Comprehensive Elliott Wave Analysis Complete!")
        print(f"📁 Check the generated HTML files for detailed visualizations")
        
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main() 