"""
Comprehensive Testing and Example Script for Enhanced Elliott Wave ML System
Demonstrates usage of the enhanced S/R detection and ML training framework
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_enhanced_sr_detector():
    """Test the Enhanced Support & Resistance Detector"""
    try:
        from src.analysis.enhanced_sr_detector import EnhancedSRDetector
        from src.data.data_loader import DataLoader
        
        logger.info("🔍 Testing Enhanced S/R Detector...")
        
        # Initialize components
        data_loader = DataLoader()
        sr_detector = EnhancedSRDetector()
        
        # Load test data
        symbol = 'AAPL'
        market_data = data_loader.get_yahoo_data(symbol, period='1y', interval='1d')
        
        if market_data.empty:
            logger.error(f"No data available for {symbol}")
            return False
        
        # Run S/R detection
        sr_results = sr_detector.detect_sr_levels(market_data)
        
        # Display results
        logger.info(f"📊 S/R Detection Results for {symbol}:")
        logger.info(f"   Support levels found: {len(sr_results['support_levels'])}")
        logger.info(f"   Resistance levels found: {len(sr_results['resistance_levels'])}")
        
        # Show top 3 support and resistance levels
        for i, level in enumerate(sr_results['support_levels'][:3], 1):
            logger.info(f"   Support {i}: ${level.price:.2f} - {level.strength} touches, "
                       f"conviction: {level.conviction:.2f}, method: {level.formation_method}")
        
        for i, level in enumerate(sr_results['resistance_levels'][:3], 1):
            logger.info(f"   Resistance {i}: ${level.price:.2f} - {level.strength} touches, "
                       f"conviction: {level.conviction:.2f}, method: {level.formation_method}")
        
        # Test output formatting
        formatted_output = sr_detector.format_sr_output(sr_results)
        logger.info(f"📋 Formatted Output:")
        logger.info(f"   Support levels: {len(formatted_output['support_levels'])}")
        logger.info(f"   Resistance levels: {len(formatted_output['resistance_levels'])}")
        
        # Test strong levels only
        strong_levels = sr_detector.get_strong_levels_only(sr_results)
        logger.info(f"💪 Strong Levels (3+ touches):")
        logger.info(f"   Strong support: {len(strong_levels['support_levels'])}")
        logger.info(f"   Strong resistance: {len(strong_levels['resistance_levels'])}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing Enhanced S/R Detector: {e}")
        return False

def test_ml_dataset_generation():
    """Test the ML Dataset Generation Pipeline"""
    try:
        from src.ml.dataset_generator import DatasetGenerator
        
        logger.info("📊 Testing ML Dataset Generation...")
        
        # Initialize dataset generator
        generator = DatasetGenerator(output_dir="test_datasets")
        
        # Test symbols (small set for testing)
        symbols = ['AAPL', 'MSFT']
        start_date = '2023-01-01'
        end_date = '2023-06-30'
        
        # Generate dataset
        dataset_path = generator.generate_comprehensive_dataset(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframes=['1d']
        )
        
        if dataset_path and Path(dataset_path).exists():
            logger.info(f"✅ Dataset generated successfully: {dataset_path}")
            
            # Load and analyze the dataset
            results = generator.load_dataset(dataset_path)
            logger.info(f"📈 Dataset Analysis:")
            logger.info(f"   Total samples: {len(results)}")
            
            if results:
                hit_rate = sum(1 for r in results if r.hit) / len(results)
                avg_confluence = sum(r.confluence_score for r in results) / len(results)
                
                logger.info(f"   Hit rate: {hit_rate:.3f}")
                logger.info(f"   Average confluence score: {avg_confluence:.1f}")
                
                # Analyze by confidence level
                high_confidence = [r for r in results if r.confidence_level == 'HIGH']
                if high_confidence:
                    high_hit_rate = sum(1 for r in high_confidence if r.hit) / len(high_confidence)
                    logger.info(f"   High confidence samples: {len(high_confidence)}")
                    logger.info(f"   High confidence hit rate: {high_hit_rate:.3f}")
            
            return True
        else:
            logger.error("Dataset generation failed")
            return False
        
    except Exception as e:
        logger.error(f"Error testing ML dataset generation: {e}")
        return False

def test_ml_training_framework():
    """Test the ML Training Framework"""
    try:
        from src.ml.training_framework import MLTrainingFramework, TrainingData
        
        logger.info("🤖 Testing ML Training Framework...")
        
        # Initialize training framework
        trainer = MLTrainingFramework(data_dir="test_ml_data", models_dir="test_models")
        
        # Create synthetic training data for testing
        training_data = []
        
        for i in range(100):  # 100 synthetic samples
            sample = TrainingData(
                symbol='TEST',
                timestamp=datetime.now().isoformat(),
                timeframe='1d',
                wave_count=f"{(i % 3) + 1}-{(i % 3) + 2}",
                projected_wave=str((i % 5) + 1),
                target_zone=[100 + i, 102 + i],
                confluence_score=(i % 10) + 1,
                confluence_methods=[f"method_{j}" for j in range((i % 3) + 1)],
                current_price=100 + i * 0.5,
                trend_direction='up' if i % 2 == 0 else 'down',
                rsi=30 + (i % 40),
                macd_signal='bullish' if i % 2 == 0 else 'bearish',
                volume_ratio=0.8 + (i % 5) * 0.1,
                volatility=0.01 + (i % 10) * 0.002,
                wave_structure_quality=0.3 + (i % 7) * 0.1,
                fibonacci_alignment=0.4 + (i % 6) * 0.1,
                rule_compliance=0.5 + (i % 5) * 0.1,
                pattern_clarity=0.6 + (i % 4) * 0.1,
                similar_pattern_success_rate=0.5 + (i % 3) * 0.15,
                timeframe_success_rate=0.6 + (i % 4) * 0.1,
                symbol_success_rate=0.55 + (i % 3) * 0.15,
                hit=i % 3 == 0,  # 1/3 hit rate
                hit_accuracy=0.8 if i % 3 == 0 else 0.0,
                time_to_hit=i % 20 + 1 if i % 3 == 0 else None,
                max_adverse_move=0.01 + (i % 10) * 0.005
            )
            training_data.append(sample)
        
        logger.info(f"📊 Created {len(training_data)} synthetic training samples")
        
        # Prepare datasets
        datasets = trainer.prepare_datasets(training_data)
        
        if not datasets:
            logger.error("Failed to prepare datasets")
            return False
        
        logger.info(f"✅ Prepared {len(datasets)} dataset types")
        
        # Train models (if sklearn is available)
        try:
            import sklearn
            
            # Train wave confidence classifier
            wave_performance = trainer.train_wave_confidence_classifier(datasets)
            logger.info(f"🎯 Wave Confidence Model - Accuracy: {wave_performance.accuracy:.3f}")
            
            # Train scenario ranking model
            scenario_performance = trainer.train_scenario_ranking_model(datasets)
            logger.info(f"📊 Scenario Ranking Model - Score: {scenario_performance.accuracy:.3f}")
            
            # Train zone confidence model
            zone_performance = trainer.train_zone_confidence_model(datasets)
            logger.info(f"🎯 Zone Confidence Model - Score: {zone_performance.accuracy:.3f}")
            
            # Test predictions
            test_features = {
                'confluence_score': 7,
                'current_price': 150.0,
                'rsi': 65,
                'volume_ratio': 1.2,
                'volatility': 0.025,
                'wave_structure_quality': 0.8,
                'fibonacci_alignment': 0.75,
                'rule_compliance': 0.9,
                'pattern_clarity': 0.7,
                'similar_pattern_success_rate': 0.65,
                'timeframe_success_rate': 0.7,
                'symbol_success_rate': 0.6
            }
            
            confidence = trainer.predict_wave_confidence(test_features)
            logger.info(f"🔮 Prediction Test - Wave Confidence: {confidence:.3f}")
            
            return True
            
        except ImportError:
            logger.warning("Scikit-learn not available. Skipping model training tests.")
            return True
        
    except Exception as e:
        logger.error(f"Error testing ML training framework: {e}")
        return False

def test_enhanced_confluence_integration():
    """Test integration of enhanced S/R with confluence analyzer"""
    try:
        from src.analysis.technical_confluence import TechnicalConfluenceAnalyzer
        from src.analysis.enhanced_wave_detector import EnhancedWaveDetector
        from src.data.data_loader import DataLoader
        
        logger.info("🧩 Testing Enhanced Confluence Integration...")
        
        # Initialize components
        data_loader = DataLoader()
        wave_detector = EnhancedWaveDetector()
        confluence_analyzer = TechnicalConfluenceAnalyzer()
        
        # Load test data
        symbol = 'BTC-USD'
        market_data = data_loader.get_yahoo_data(symbol, period='6mo', interval='1d')
        
        if market_data.empty:
            logger.warning(f"No data for {symbol}, trying AAPL instead")
            symbol = 'AAPL'
            market_data = data_loader.get_yahoo_data(symbol, period='6mo', interval='1d')
        
        if market_data.empty:
            logger.error("No market data available")
            return False
        
        # Run Elliott Wave analysis
        elliott_result = wave_detector.detect_elliott_waves(market_data, symbol)
        
        if not elliott_result or elliott_result.get('validation_score', 0) < 0.1:
            logger.warning("No valid Elliott Wave patterns detected")
            return False
        
        logger.info(f"📈 Elliott Wave Analysis:")
        logger.info(f"   Wave structure: {elliott_result.get('wave_structure', 'unknown')}")
        logger.info(f"   Validation score: {elliott_result.get('validation_score', 0):.3f}")
        logger.info(f"   Waves detected: {len(elliott_result.get('waves', []))}")
        
        # Run enhanced confluence analysis
        target_zones = confluence_analyzer.analyze_target_zones(
            market_data, elliott_result, '1d'
        )
        
        logger.info(f"🎯 Confluence Analysis Results:")
        logger.info(f"   Target zones found: {len(target_zones)}")
        
        for i, zone in enumerate(target_zones[:3], 1):
            logger.info(f"   Target {i}: ${zone.price_level:.2f}")
            logger.info(f"     Wave: {zone.wave_target}")
            logger.info(f"     Confluence score: {zone.confluence_score}")
            logger.info(f"     Confidence: {zone.confidence_level}")
            logger.info(f"     Confluences: {zone.confluences[:3]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing enhanced confluence integration: {e}")
        return False

def demonstrate_full_pipeline():
    """Demonstrate the complete enhanced Elliott Wave ML pipeline"""
    try:
        logger.info("🚀 Demonstrating Full Enhanced Elliott Wave ML Pipeline")
        logger.info("=" * 70)
        
        # Test 1: Enhanced S/R Detection
        logger.info("\n📍 Step 1: Enhanced Support & Resistance Detection")
        sr_success = test_enhanced_sr_detector()
        
        # Test 2: Enhanced Confluence Integration
        logger.info("\n📍 Step 2: Enhanced Confluence Analysis Integration")
        confluence_success = test_enhanced_confluence_integration()
        
        # Test 3: ML Dataset Generation
        logger.info("\n📍 Step 3: ML Dataset Generation Pipeline")
        dataset_success = test_ml_dataset_generation()
        
        # Test 4: ML Training Framework
        logger.info("\n📍 Step 4: ML Training Framework")
        ml_success = test_ml_training_framework()
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("🎯 PIPELINE TEST SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Enhanced S/R Detection: {'✅ PASS' if sr_success else '❌ FAIL'}")
        logger.info(f"Confluence Integration: {'✅ PASS' if confluence_success else '❌ FAIL'}")
        logger.info(f"ML Dataset Generation: {'✅ PASS' if dataset_success else '❌ FAIL'}")
        logger.info(f"ML Training Framework: {'✅ PASS' if ml_success else '❌ FAIL'}")
        
        overall_success = all([sr_success, confluence_success, dataset_success, ml_success])
        logger.info(f"Overall Pipeline: {'✅ SUCCESS' if overall_success else '❌ FAILED'}")
        
        if overall_success:
            logger.info("\n🎉 Enhanced Elliott Wave ML System is ready for production!")
            logger.info("\n📚 Next Steps:")
            logger.info("   1. Generate comprehensive datasets with more symbols and timeframes")
            logger.info("   2. Train models on larger datasets for better accuracy")
            logger.info("   3. Implement real-time prediction API endpoints")
            logger.info("   4. Set up automated model retraining pipeline")
            logger.info("   5. Add performance monitoring and model drift detection")
        
        return overall_success
        
    except Exception as e:
        logger.error(f"Error in full pipeline demonstration: {e}")
        return False

def run_example_analysis():
    """Run an example analysis showing all new features"""
    try:
        logger.info("\n🔬 EXAMPLE: Comprehensive Elliott Wave Analysis with ML")
        logger.info("=" * 70)
        
        from src.data.data_loader import DataLoader
        from src.analysis.enhanced_wave_detector import EnhancedWaveDetector
        from src.analysis.technical_confluence import TechnicalConfluenceAnalyzer
        from src.analysis.enhanced_sr_detector import EnhancedSRDetector
        
        # Initialize all components
        data_loader = DataLoader()
        wave_detector = EnhancedWaveDetector()
        confluence_analyzer = TechnicalConfluenceAnalyzer()
        sr_detector = EnhancedSRDetector()
        
        # Analyze TSLA as example
        symbol = 'TSLA'
        logger.info(f"📈 Analyzing {symbol}...")
        
        # Load data
        market_data = data_loader.get_yahoo_data(symbol, period='1y', interval='1d')
        current_price = market_data['close'].iloc[-1]
        
        logger.info(f"Current price: ${current_price:.2f}")
        
        # 1. Enhanced S/R Analysis
        logger.info("\n🔍 1. Enhanced Support & Resistance Analysis:")
        sr_results = sr_detector.detect_sr_levels(market_data)
        
        strong_support = [lvl for lvl in sr_results['support_levels'] if lvl.strength >= 3]
        strong_resistance = [lvl for lvl in sr_results['resistance_levels'] if lvl.strength >= 3]
        
        logger.info(f"   Strong Support Levels: {len(strong_support)}")
        for lvl in strong_support[:3]:
            logger.info(f"     ${lvl.price:.2f} ({lvl.strength} touches, {lvl.formation_method})")
        
        logger.info(f"   Strong Resistance Levels: {len(strong_resistance)}")
        for lvl in strong_resistance[:3]:
            logger.info(f"     ${lvl.price:.2f} ({lvl.strength} touches, {lvl.formation_method})")
        
        # 2. Elliott Wave Analysis
        logger.info("\n🌊 2. Elliott Wave Analysis:")
        elliott_result = wave_detector.detect_elliott_waves(market_data, symbol)
        
        if elliott_result and elliott_result.get('validation_score', 0) > 0.3:
            logger.info(f"   Wave Structure: {elliott_result.get('wave_structure', 'unknown')}")
            logger.info(f"   Validation Score: {elliott_result.get('validation_score', 0):.3f}")
            logger.info(f"   Direction: {elliott_result.get('direction', 'neutral')}")
            logger.info(f"   Waves Detected: {len(elliott_result.get('waves', []))}")
            
            # 3. Confluence Analysis
            logger.info("\n🎯 3. Technical Confluence Analysis:")
            target_zones = confluence_analyzer.analyze_target_zones(
                market_data, elliott_result, '1d'
            )
            
            logger.info(f"   Target Zones Found: {len(target_zones)}")
            
            for i, zone in enumerate(target_zones[:5], 1):
                price_change = ((zone.price_level - current_price) / current_price) * 100
                logger.info(f"   Target {i}: ${zone.price_level:.2f} ({price_change:+.1f}%)")
                logger.info(f"     Wave: {zone.wave_target}")
                logger.info(f"     Confidence: {zone.confidence_level}")
                logger.info(f"     Confluence Score: {zone.confluence_score}")
                logger.info(f"     Risk/Reward: {zone.risk_reward_ratio:.2f}")
                logger.info(f"     Key Confluences: {', '.join(zone.confluences[:3])}")
                
                # Check S/R confluence
                sr_confluence_count = 0
                for sr_level in sr_results['support_levels'] + sr_results['resistance_levels']:
                    if abs(zone.price_level - sr_level.price) / zone.price_level <= 0.01:
                        sr_confluence_count += 1
                
                if sr_confluence_count > 0:
                    logger.info(f"     ✅ S/R Confluence: {sr_confluence_count} level(s)")
                
                logger.info("")
        else:
            logger.warning("   No valid Elliott Wave patterns detected")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in example analysis: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Enhanced Elliott Wave System - Comprehensive Testing")
    print("=" * 60)
    
    # Run full pipeline demonstration
    success = demonstrate_full_pipeline()
    
    if success:
        # Run example analysis
        print("\n" + "=" * 60)
        run_example_analysis()
    
    print("\n" + "=" * 60)
    print("Testing complete!")
