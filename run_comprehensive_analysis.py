#!/usr/bin/env python3
"""
Advanced AI-Powered Elliott Wave Analysis Engine

This is the main runner script for the comprehensive Elliott Wave analysis system.
It demonstrates all features including data fetching, wave detection, validation,
projections, pattern memory, and advanced visualization.

Usage:
    python run_comprehensive_analysis.py --symbol BTCUSDT --timeframe 1h
    python run_comprehensive_analysis.py --symbol ETHUSDT --timeframe 4h --days 14
    python run_comprehensive_analysis.py --symbol BTCUSDT --timeframe 1d --export-json
"""

import argparse
import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.binance_data_fetcher import BinanceDataFetcher
from src.analysis.wave_detector import WaveDetector
from src.analysis.wave_validator import WaveValidator
from src.analysis.wave_projector import WaveProjector
from src.analysis.pattern_memory import PatternMemory, PatternCategory
from src.analysis.fibonacci import FibonacciAnalyzer
from src.visualization.visualizer import WaveVisualizer
from src.utils.logger import setup_logger
from src.utils.config import get_config


def setup_argparse():
    """Setup command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Advanced AI-Powered Elliott Wave Analysis Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --symbol BTCUSDT --timeframe 1h
  %(prog)s --symbol ETHUSDT --timeframe 4h --days 14
  %(prog)s --symbol BTCUSDT --timeframe 1d --export-json --output-dir ./analysis
  %(prog)s --symbol BTCUSDT --timeframe 1h --validate-only
  %(prog)s --symbol BTCUSDT --timeframe 1h --projections-only
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTCUSDT',
        help='Trading symbol (default: BTCUSDT)'
    )
    
    parser.add_argument(
        '--timeframe',
        type=str,
        default='1h',
        choices=['1m', '5m', '15m', '1h', '4h', '1d'],
        help='Timeframe for analysis (default: 1h)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days of historical data to fetch (default: 7)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Output directory for charts and data (default: ./output)'
    )
    
    parser.add_argument(
        '--export-json',
        action='store_true',
        help='Export analysis results to JSON'
    )
    
    parser.add_argument(
        '--export-csv',
        action='store_true',
        help='Export analysis results to CSV'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Run only wave detection and validation'
    )
    
    parser.add_argument(
        '--projections-only',
        action='store_true',
        help='Run only wave detection and projections'
    )
    
    parser.add_argument(
        '--memory-only',
        action='store_true',
        help='Run only pattern memory analysis'
    )
    
    parser.add_argument(
        '--no-charts',
        action='store_true',
        help='Skip chart generation'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def create_output_directory(output_dir):
    """Create output directory if it doesn't exist."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir


def run_comprehensive_analysis(args):
    """Run the comprehensive Elliott Wave analysis."""
    
    print("🚀 Advanced AI-Powered Elliott Wave Analysis Engine")
    print("=" * 70)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Days: {args.days}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 70)
    
    # Create output directory
    output_dir = create_output_directory(args.output_dir)
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger(level=log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = get_config(args.config)
        
        # Initialize components
        print("\n1. 🔧 Initializing Analysis Components...")
        data_fetcher = BinanceDataFetcher()
        detector = WaveDetector()
        validator = WaveValidator()
        projector = WaveProjector()
        pattern_memory = PatternMemory()
        fibonacci_analyzer = FibonacciAnalyzer()
        visualizer = WaveVisualizer()
        
        print("✅ All components initialized successfully")
        
        # Fetch market data
        print(f"\n2. 📊 Fetching Market Data for {args.symbol}...")
        end_time = datetime.now()
        start_time = end_time - timedelta(days=args.days)
        
        data = data_fetcher.get_historical_ohlcv(
            symbol=args.symbol,
            interval=args.timeframe,
            start_time=int(start_time.timestamp() * 1000),
            end_time=int(end_time.timestamp() * 1000),
            limit=1000
        )
        
        if data.empty:
            print("❌ No data received from API")
            return False
        
        print(f"✅ Successfully fetched {len(data)} data points")
        print(f"   Date range: {data.index[0]} to {data.index[-1]}")
        print(f"   Price range: ${data['low'].min():.2f} - ${data['high'].max():.2f}")
        print(f"   Volume: {data['volume'].sum():,.0f}")
        
        # Detect Elliott Waves
        print(f"\n3. 🌊 Detecting Elliott Waves...")
        waves = detector.detect_waves(data)
        
        if not waves:
            print("❌ No Elliott Waves detected in the data")
            return False
        
        print(f"✅ Detected {len(waves)} Elliott Waves")
        for i, wave in enumerate(waves):
            print(f"   Wave {i+1}: {wave.wave_type.value} - {wave.price_change_pct:.1%} change over {wave.duration} periods")
        
        # Validate wave pattern
        print(f"\n4. ✅ Validating Wave Pattern...")
        validation_result = validator.validate_wave_pattern(waves, data)
        
        print(f"✅ Validation completed")
        print(f"   Pattern type: {validation_result.pattern_type}")
        print(f"   Overall confidence: {validation_result.overall_score:.1%}")
        print(f"   Critical rules passed: {validation_result.passed_critical_rules}")
        print(f"   Warnings: {len(validation_result.warnings)}")
        
        if validation_result.warnings:
            print("   ⚠️  Warnings:")
            for warning in validation_result.warnings[:5]:  # Show first 5 warnings
                print(f"      - {warning}")
        
        # Perform Fibonacci analysis
        print(f"\n5. 📐 Performing Fibonacci Analysis...")
        fibonacci_analysis = fibonacci_analyzer.analyze_waves(waves, data)
        
        print(f"✅ Fibonacci analysis completed")
        if fibonacci_analysis.levels:
            print(f"   Key Fibonacci levels: {len(fibonacci_analysis.levels)}")
            for level in fibonacci_analysis.levels[:5]:  # Show first 5 levels
                print(f"      {level.level_type.value}: ${level.price:.2f}")
        
        # Early exit if validate-only mode
        if args.validate_only:
            print(f"\n✅ Validation-only analysis completed")
            return True
        
        # Generate future projections
        print(f"\n6. 🔮 Generating Future Wave Projections...")
        current_wave = max(waves, key=lambda w: w.end_point.timestamp)
        
        projection_scenarios = projector.generate_comprehensive_projections(
            current_wave=current_wave,
            all_waves=waves,
            data=data,
            include_alternatives=True,
            max_scenarios=5
        )
        
        print(f"✅ Generated {len(projection_scenarios)} projection scenarios")
        for i, scenario in enumerate(projection_scenarios):
            print(f"   Scenario {i+1}: {scenario.scenario_name}")
            print(f"      Confidence: {scenario.overall_confidence:.1%}")
            print(f"      Primary projection: {scenario.primary_projection.description}")
            print(f"      Alternative projections: {len(scenario.alternative_projections)}")
            
            if scenario.primary_projection.fibonacci_targets:
                targets = scenario.primary_projection.fibonacci_targets[:3]
                print(f"      Key targets: {', '.join([f'${t:.2f}' for t in targets])}")
        
        # Early exit if projections-only mode
        if args.projections_only:
            print(f"\n✅ Projections-only analysis completed")
            return True
        
        # Find similar historical patterns
        print(f"\n7. 📚 Finding Similar Historical Patterns...")
        pattern_matches = pattern_memory.find_similar_patterns(
            current_waves=waves,
            current_validation=validation_result,
            current_fibonacci=fibonacci_analysis,
            symbol=args.symbol,
            timeframe=args.timeframe,
            min_similarity=0.5,
            max_results=10
        )
        
        print(f"✅ Found {len(pattern_matches)} similar historical patterns")
        for i, match in enumerate(pattern_matches[:5]):  # Show first 5 matches
            print(f"   Match {i+1}: {match.similarity_score:.1%} similarity ({match.match_quality})")
            print(f"      Historical pattern: {match.historical_pattern.pattern_id}")
            print(f"      Category: {match.historical_pattern.pattern_category.value}")
            if match.outcome_prediction:
                print(f"      Historical outcome: {match.outcome_prediction}")
        
        # Store current pattern in memory
        print(f"\n8. 💾 Storing Current Pattern in Memory...")
        pattern_id = pattern_memory.store_pattern(
            symbol=args.symbol,
            timeframe=args.timeframe,
            waves=waves,
            validation_result=validation_result,
            fibonacci_analysis=fibonacci_analysis,
            market_condition="analyzed",
            tags=["automated", "comprehensive"],
            notes=f"Automated analysis run on {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        
        print(f"✅ Stored pattern with ID: {pattern_id}")
        
        # Get pattern memory statistics
        stats = pattern_memory.get_pattern_statistics()
        print(f"   Memory database now contains {stats.total_patterns} patterns")
        print(f"   Average confidence: {stats.average_confidence:.1%}")
        
        # Early exit if memory-only mode
        if args.memory_only:
            print(f"\n✅ Memory-only analysis completed")
            return True
        
        # Generate visualizations
        if not args.no_charts:
            print(f"\n9. 📈 Generating Visualizations...")
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            
            # Comprehensive analysis chart
            comprehensive_fig = visualizer.plot_comprehensive_analysis(
                data=data,
                waves=waves,
                projection_scenarios=projection_scenarios,
                pattern_matches=pattern_matches,
                validation_result=validation_result,
                fibonacci_analysis=fibonacci_analysis,
                title=f"Comprehensive Elliott Wave Analysis - {args.symbol}"
            )
            
            comprehensive_path = os.path.join(output_dir, f"comprehensive_analysis_{args.symbol}_{timestamp}.html")
            comprehensive_fig.write_html(comprehensive_path)
            print(f"✅ Comprehensive analysis chart: {comprehensive_path}")
            
            # Projection dashboard
            projection_fig = visualizer.create_projection_dashboard(
                data=data,
                waves=waves,
                projection_scenarios=projection_scenarios,
                validation_result=validation_result,
                fibonacci_analysis=fibonacci_analysis,
                title=f"Projection Dashboard - {args.symbol}"
            )
            
            projection_path = os.path.join(output_dir, f"projection_dashboard_{args.symbol}_{timestamp}.html")
            projection_fig.write_html(projection_path)
            print(f"✅ Projection dashboard: {projection_path}")
            
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
                title=f"Pattern Memory Dashboard - {args.symbol}"
            )
            
            memory_path = os.path.join(output_dir, f"pattern_memory_dashboard_{args.symbol}_{timestamp}.html")
            memory_fig.write_html(memory_path)
            print(f"✅ Pattern memory dashboard: {memory_path}")
        
        # Export results
        if args.export_json or args.export_csv:
            print(f"\n10. 📤 Exporting Analysis Results...")
            
            # Prepare export data
            export_data = {
                'analysis_info': {
                    'symbol': args.symbol,
                    'timeframe': args.timeframe,
                    'analysis_date': datetime.now().isoformat(),
                    'data_points': len(data),
                    'waves_detected': len(waves),
                    'validation_score': validation_result.overall_score,
                    'pattern_type': validation_result.pattern_type
                },
                'waves': [
                    {
                        'wave_type': wave.wave_type.value,
                        'start_price': wave.start_point.price,
                        'end_price': wave.end_point.price,
                        'price_change': wave.price_change,
                        'price_change_pct': wave.price_change_pct,
                        'duration': wave.duration,
                        'confidence': wave.confidence
                    }
                    for wave in waves
                ],
                'validation': {
                    'overall_score': validation_result.overall_score,
                    'passed_critical_rules': validation_result.passed_critical_rules,
                    'warnings': validation_result.warnings,
                    'recommendations': validation_result.recommendations
                },
                'projections': [
                    {
                        'scenario_name': scenario.scenario_name,
                        'confidence': scenario.overall_confidence,
                        'primary_projection': {
                            'description': scenario.primary_projection.description,
                            'fibonacci_targets': scenario.primary_projection.fibonacci_targets,
                            'invalidation_levels': scenario.primary_projection.invalidation_levels
                        }
                    }
                    for scenario in projection_scenarios
                ],
                'pattern_matches': [
                    {
                        'similarity_score': match.similarity_score,
                        'match_quality': match.match_quality,
                        'historical_pattern_id': match.historical_pattern.pattern_id,
                        'outcome_prediction': match.outcome_prediction
                    }
                    for match in pattern_matches
                ]
            }
            
            # Export JSON
            if args.export_json:
                json_path = os.path.join(output_dir, f"analysis_results_{args.symbol}_{timestamp}.json")
                with open(json_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                print(f"✅ JSON export: {json_path}")
            
            # Export CSV
            if args.export_csv:
                # Export waves to CSV
                waves_df = pd.DataFrame(export_data['waves'])
                waves_csv_path = os.path.join(output_dir, f"waves_{args.symbol}_{timestamp}.csv")
                waves_df.to_csv(waves_csv_path, index=False)
                print(f"✅ Waves CSV export: {waves_csv_path}")
                
                # Export projections to CSV
                projections_data = []
                for scenario in export_data['projections']:
                    projections_data.append({
                        'scenario_name': scenario['scenario_name'],
                        'confidence': scenario['confidence'],
                        'description': scenario['primary_projection']['description'],
                        'targets': ', '.join([f'${t:.2f}' for t in scenario['primary_projection']['fibonacci_targets']])
                    })
                
                projections_df = pd.DataFrame(projections_data)
                projections_csv_path = os.path.join(output_dir, f"projections_{args.symbol}_{timestamp}.csv")
                projections_df.to_csv(projections_csv_path, index=False)
                print(f"✅ Projections CSV export: {projections_csv_path}")
        
        # Final summary
        print(f"\n🎉 Analysis Complete!")
        print("=" * 70)
        
        current_price = data['close'].iloc[-1]
        print(f"📊 Current {args.symbol} Price: ${current_price:.2f}")
        print(f"🌊 Elliott Wave Pattern: {validation_result.pattern_type}")
        print(f"✅ Validation Confidence: {validation_result.overall_score:.1%}")
        
        if projection_scenarios:
            best_scenario = projection_scenarios[0]
            print(f"🔮 Best Projection: {best_scenario.scenario_name}")
            print(f"📈 Projection Confidence: {best_scenario.overall_confidence:.1%}")
            
            if best_scenario.primary_projection.fibonacci_targets:
                targets = best_scenario.primary_projection.fibonacci_targets[:3]
                print(f"🎯 Key Targets: {', '.join([f'${t:.2f}' for t in targets])}")
        
        if pattern_matches:
            best_match = pattern_matches[0]
            print(f"📚 Best Historical Match: {best_match.similarity_score:.1%} similarity")
            if best_match.outcome_prediction:
                print(f"📖 Historical Outcome: {best_match.outcome_prediction}")
        
        print(f"💾 Pattern stored in memory: {pattern_id}")
        print(f"📁 Results saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        print(f"❌ Analysis failed: {e}")
        return False


def main():
    """Main function."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    success = run_comprehensive_analysis(args)
    
    if success:
        print(f"\n✅ Analysis completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Analysis failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 