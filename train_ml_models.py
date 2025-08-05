"""
Example ML Training Script for Elliott Wave System
Demonstrates comprehensive training pipeline for wave confidence, scenario ranking, and zone scoring
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_full_training_pipeline():
    """Run the complete ML training pipeline"""
    try:
        from src.ml.training_framework import MLTrainingFramework
        from src.ml.dataset_generator import DatasetGenerator
        
        logger.info("🚀 Starting Full ML Training Pipeline for Elliott Wave System")
        logger.info("=" * 80)
        
        # Configuration
        symbols = [
            # Major stocks
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META',
            # Crypto (if available)
            'BTC-USD', 'ETH-USD',
            # Forex
            'EURUSD=X', 'GBPUSD=X', 'USDJPY=X'
        ]
        
        start_date = '2022-01-01'
        end_date = '2023-12-31'
        timeframes = ['1d']  # Start with daily timeframe
        
        # Step 1: Generate Comprehensive Dataset
        logger.info("📊 Step 1: Generating Comprehensive Training Dataset")
        logger.info("-" * 60)
        
        dataset_generator = DatasetGenerator(output_dir="ml_datasets")
        
        dataset_path = dataset_generator.generate_comprehensive_dataset(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframes=timeframes
        )
        
        if not dataset_path or not Path(dataset_path).exists():
            logger.error("❌ Dataset generation failed")
            return False
        
        logger.info(f"✅ Dataset generated: {dataset_path}")
        
        # Load and analyze dataset
        training_data = dataset_generator.load_dataset(dataset_path)
        
        if not training_data:
            logger.error("❌ Failed to load training data")
            return False
        
        logger.info(f"📈 Dataset Analysis:")
        logger.info(f"   Total samples: {len(training_data)}")
        
        # Analyze dataset quality
        hit_rate = sum(1 for sample in training_data if sample.hit) / len(training_data)
        avg_confluence = sum(sample.confluence_score for sample in training_data) / len(training_data)
        
        confidence_distribution = {}
        for sample in training_data:
            confidence_distribution[sample.confidence_level] = confidence_distribution.get(sample.confidence_level, 0) + 1
        
        logger.info(f"   Overall hit rate: {hit_rate:.3f}")
        logger.info(f"   Average confluence score: {avg_confluence:.1f}")
        logger.info(f"   Confidence distribution: {confidence_distribution}")
        
        # Analyze by confidence level
        for confidence_level in ['HIGH', 'MEDIUM', 'LOW']:
            confidence_samples = [s for s in training_data if s.confidence_level == confidence_level]
            if confidence_samples:
                conf_hit_rate = sum(1 for s in confidence_samples if s.hit) / len(confidence_samples)
                logger.info(f"   {confidence_level} confidence hit rate: {conf_hit_rate:.3f} ({len(confidence_samples)} samples)")
        
        # Step 2: Train ML Models
        logger.info("\n🤖 Step 2: Training Machine Learning Models")
        logger.info("-" * 60)
        
        ml_trainer = MLTrainingFramework(data_dir="ml_training_data", models_dir="trained_models")
        
        # Prepare datasets for different tasks
        datasets = ml_trainer.prepare_datasets(training_data)
        
        if not datasets:
            logger.error("❌ Failed to prepare ML datasets")
            return False
        
        logger.info(f"✅ Prepared {len(datasets)} ML dataset types")
        
        # Check if sklearn is available
        try:
            import sklearn
            sklearn_available = True
        except ImportError:
            logger.warning("⚠️ Scikit-learn not available. ML training will be limited.")
            sklearn_available = False
        
        model_results = {}
        
        if sklearn_available:
            # Train Wave Confidence Classifier
            logger.info("\n🎯 Training Wave Confidence Classifier...")
            wave_performance = ml_trainer.train_wave_confidence_classifier(datasets)
            model_results['wave_confidence'] = wave_performance
            
            logger.info(f"   Accuracy: {wave_performance.accuracy:.3f}")
            logger.info(f"   Precision: {wave_performance.precision:.3f}")
            logger.info(f"   Recall: {wave_performance.recall:.3f}")
            logger.info(f"   Cross-validation mean: {np.mean(wave_performance.cross_val_scores):.3f}")
            
            # Show top features
            if wave_performance.feature_importance:
                top_features = sorted(wave_performance.feature_importance.items(), 
                                    key=lambda x: x[1], reverse=True)[:5]
                logger.info("   Top 5 Features:")
                for feature, importance in top_features:
                    logger.info(f"     {feature}: {importance:.3f}")
            
            # Train Scenario Ranking Model
            logger.info("\n📊 Training Scenario Ranking Model...")
            scenario_performance = ml_trainer.train_scenario_ranking_model(datasets)
            model_results['scenario_ranking'] = scenario_performance
            
            logger.info(f"   Regression score: {scenario_performance.accuracy:.3f}")
            logger.info(f"   Cross-validation mean: {np.mean(scenario_performance.cross_val_scores):.3f}")
            
            # Train Zone Confidence Model
            logger.info("\n🎯 Training Zone Confidence Model...")
            zone_performance = ml_trainer.train_zone_confidence_model(datasets)
            model_results['zone_confidence'] = zone_performance
            
            logger.info(f"   Regression score: {zone_performance.accuracy:.3f}")
            logger.info(f"   Cross-validation mean: {np.mean(zone_performance.cross_val_scores):.3f}")
        
        # Step 3: Model Validation and Testing
        logger.info("\n🔍 Step 3: Model Validation and Testing")
        logger.info("-" * 60)
        
        if sklearn_available and model_results:
            # Test predictions on sample data
            test_scenarios = [
                {
                    'name': 'High Confluence Bullish Wave 5',
                    'features': {
                        'confluence_score': 8,
                        'current_price': 150.0,
                        'rsi': 45,
                        'volume_ratio': 1.5,
                        'volatility': 0.02,
                        'wave_structure_quality': 0.85,
                        'fibonacci_alignment': 0.9,
                        'rule_compliance': 0.95,
                        'pattern_clarity': 0.8,
                        'similar_pattern_success_rate': 0.75,
                        'timeframe_success_rate': 0.7,
                        'symbol_success_rate': 0.65
                    }
                },
                {
                    'name': 'Medium Confluence Corrective Wave C',
                    'features': {
                        'confluence_score': 4,
                        'current_price': 200.0,
                        'rsi': 25,
                        'volume_ratio': 0.8,
                        'volatility': 0.035,
                        'wave_structure_quality': 0.6,
                        'fibonacci_alignment': 0.65,
                        'rule_compliance': 0.7,
                        'pattern_clarity': 0.65,
                        'similar_pattern_success_rate': 0.55,
                        'timeframe_success_rate': 0.6,
                        'symbol_success_rate': 0.58
                    }
                },
                {
                    'name': 'Low Confluence Uncertain Pattern',
                    'features': {
                        'confluence_score': 2,
                        'current_price': 75.0,
                        'rsi': 55,
                        'volume_ratio': 1.0,
                        'volatility': 0.045,
                        'wave_structure_quality': 0.4,
                        'fibonacci_alignment': 0.45,
                        'rule_compliance': 0.5,
                        'pattern_clarity': 0.4,
                        'similar_pattern_success_rate': 0.45,
                        'timeframe_success_rate': 0.5,
                        'symbol_success_rate': 0.48
                    }
                }
            ]
            
            logger.info("🔮 Testing Model Predictions:")
            
            for scenario in test_scenarios:
                logger.info(f"\n   Scenario: {scenario['name']}")
                
                # Test wave confidence
                wave_conf = ml_trainer.predict_wave_confidence(scenario['features'])
                logger.info(f"     Wave Confidence: {wave_conf:.3f}")
                
                # Test zone confidence
                zone_conf = ml_trainer.predict_zone_confidence(scenario['features'])
                logger.info(f"     Zone Confidence: {zone_conf:.3f}")
                
                # Provide interpretation
                if wave_conf > 0.7:
                    conf_level = "HIGH"
                elif wave_conf > 0.5:
                    conf_level = "MEDIUM"
                else:
                    conf_level = "LOW"
                
                logger.info(f"     Interpretation: {conf_level} confidence prediction")
        
        # Step 4: Generate Training Report
        logger.info("\n📋 Step 4: Generating Training Report")
        logger.info("-" * 60)
        
        report = {
            'training_info': {
                'dataset_path': dataset_path,
                'training_date': datetime.now().isoformat(),
                'symbols': symbols,
                'date_range': {'start': start_date, 'end': end_date},
                'timeframes': timeframes
            },
            'dataset_stats': {
                'total_samples': len(training_data),
                'overall_hit_rate': hit_rate,
                'average_confluence_score': avg_confluence,
                'confidence_distribution': confidence_distribution
            },
            'model_performance': {}
        }
        
        if sklearn_available and model_results:
            for model_name, performance in model_results.items():
                report['model_performance'][model_name] = {
                    'accuracy': performance.accuracy,
                    'precision': performance.precision,
                    'recall': performance.recall,
                    'cross_val_mean': np.mean(performance.cross_val_scores),
                    'cross_val_std': np.std(performance.cross_val_scores),
                    'training_samples': performance.training_samples
                }
        
        # Save training report
        report_path = Path("ml_training_reports") / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📊 Training report saved: {report_path}")
        
        # Step 5: Summary and Recommendations
        logger.info("\n🎯 Step 5: Training Summary and Recommendations")
        logger.info("-" * 60)
        
        logger.info("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"   📊 Dataset: {len(training_data)} samples with {hit_rate:.1%} hit rate")
        
        if sklearn_available and model_results:
            logger.info("   🤖 Models trained:")
            for model_name, performance in model_results.items():
                logger.info(f"     {model_name}: {performance.accuracy:.3f} accuracy")
        
        logger.info("\n💡 RECOMMENDATIONS:")
        
        if hit_rate < 0.4:
            logger.info("   📈 Consider adjusting target zone tolerance or confluence thresholds")
        elif hit_rate > 0.8:
            logger.info("   🎯 Excellent hit rate - consider tightening targets for better precision")
        
        if avg_confluence < 3:
            logger.info("   🔧 Consider expanding confluence methods or adjusting scoring")
        
        if sklearn_available:
            logger.info("   🚀 Ready for production deployment!")
            logger.info("   🔄 Set up automated retraining pipeline with new data")
            logger.info("   📊 Monitor model performance and drift over time")
        else:
            logger.info("   ⚠️ Install scikit-learn and optionally XGBoost for full ML capabilities")
        
        logger.info("\n📚 NEXT STEPS:")
        logger.info("   1. Integrate trained models into the web application")
        logger.info("   2. Add real-time prediction API endpoints")
        logger.info("   3. Implement model performance monitoring")
        logger.info("   4. Set up automated model updates")
        logger.info("   5. Expand to more symbols and timeframes")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in training pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

def quick_training_demo():
    """Quick training demonstration with minimal data"""
    try:
        from src.ml.training_framework import MLTrainingFramework, TrainingData
        
        logger.info("🚀 Quick ML Training Demo")
        logger.info("=" * 50)
        
        # Create synthetic training data for demo
        logger.info("📊 Creating synthetic training data...")
        
        training_data = []
        np.random.seed(42)  # For reproducible results
        
        for i in range(200):  # 200 synthetic samples
            # Create realistic-looking synthetic data
            confluence_score = np.random.randint(1, 11)
            wave_quality = 0.3 + np.random.random() * 0.7
            
            # Higher confluence and quality should correlate with higher hit rate
            hit_probability = 0.1 + (confluence_score / 10) * 0.4 + wave_quality * 0.3
            hit = np.random.random() < hit_probability
            
            sample = TrainingData(
                symbol=np.random.choice(['AAPL', 'MSFT', 'GOOGL', 'TSLA']),
                timestamp=datetime.now().isoformat(),
                timeframe='1d',
                wave_count=np.random.choice(['1-2-3', 'A-B-C', '3-4-5']),
                projected_wave=np.random.choice(['4', '5', 'C']),
                target_zone=[100 + i, 102 + i],
                confluence_score=confluence_score,
                confluence_methods=[f"method_{j}" for j in range(np.random.randint(1, 4))],
                current_price=100 + i * 0.5,
                trend_direction=np.random.choice(['up', 'down']),
                rsi=20 + np.random.random() * 60,
                macd_signal=np.random.choice(['bullish', 'bearish']),
                volume_ratio=0.5 + np.random.random() * 1.5,
                volatility=0.01 + np.random.random() * 0.04,
                wave_structure_quality=wave_quality,
                fibonacci_alignment=0.3 + np.random.random() * 0.7,
                rule_compliance=0.4 + np.random.random() * 0.6,
                pattern_clarity=0.3 + np.random.random() * 0.7,
                similar_pattern_success_rate=0.4 + np.random.random() * 0.4,
                timeframe_success_rate=0.5 + np.random.random() * 0.3,
                symbol_success_rate=0.45 + np.random.random() * 0.3,
                hit=hit,
                hit_accuracy=0.7 + np.random.random() * 0.3 if hit else 0.0,
                time_to_hit=np.random.randint(1, 21) if hit else None,
                max_adverse_move=0.005 + np.random.random() * 0.03
            )
            training_data.append(sample)
        
        logger.info(f"✅ Created {len(training_data)} synthetic samples")
        
        # Analyze synthetic data
        hit_rate = sum(1 for s in training_data if s.hit) / len(training_data)
        avg_confluence = sum(s.confluence_score for s in training_data) / len(training_data)
        
        logger.info(f"📈 Synthetic Data Analysis:")
        logger.info(f"   Hit rate: {hit_rate:.3f}")
        logger.info(f"   Average confluence: {avg_confluence:.1f}")
        
        # Train models
        try:
            import sklearn
            
            logger.info("\n🤖 Training ML Models...")
            
            trainer = MLTrainingFramework(data_dir="demo_ml_data", models_dir="demo_models")
            
            # Prepare datasets
            datasets = trainer.prepare_datasets(training_data)
            
            # Train models
            wave_perf = trainer.train_wave_confidence_classifier(datasets)
            scenario_perf = trainer.train_scenario_ranking_model(datasets)
            zone_perf = trainer.train_zone_confidence_model(datasets)
            
            logger.info("✅ Model Training Results:")
            logger.info(f"   Wave Confidence: {wave_perf.accuracy:.3f} accuracy")
            logger.info(f"   Scenario Ranking: {scenario_perf.accuracy:.3f} score")
            logger.info(f"   Zone Confidence: {zone_perf.accuracy:.3f} score")
            
            # Test prediction
            test_features = {
                'confluence_score': 7,
                'current_price': 150.0,
                'rsi': 45,
                'volume_ratio': 1.2,
                'volatility': 0.025,
                'wave_structure_quality': 0.8,
                'fibonacci_alignment': 0.75,
                'rule_compliance': 0.85,
                'pattern_clarity': 0.7,
                'similar_pattern_success_rate': 0.65,
                'timeframe_success_rate': 0.7,
                'symbol_success_rate': 0.6
            }
            
            confidence = trainer.predict_wave_confidence(test_features)
            logger.info(f"🔮 Test Prediction: {confidence:.3f} confidence")
            
            logger.info("\n✅ Quick training demo completed successfully!")
            return True
            
        except ImportError:
            logger.warning("⚠️ Scikit-learn not available. Install with: pip install scikit-learn")
            return False
        
    except Exception as e:
        logger.error(f"❌ Error in quick training demo: {e}")
        return False

if __name__ == "__main__":
    print("🤖 Elliott Wave ML Training System")
    print("=" * 50)
    
    # Ask user which mode to run
    print("\nSelect training mode:")
    print("1. Full Training Pipeline (comprehensive, takes longer)")
    print("2. Quick Demo (synthetic data, fast)")
    
    try:
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == "1":
            logger.info("Starting full training pipeline...")
            success = run_full_training_pipeline()
        elif choice == "2":
            logger.info("Starting quick demo...")
            success = quick_training_demo()
        else:
            logger.info("Running quick demo by default...")
            success = quick_training_demo()
        
        if success:
            print("\n🎉 Training completed successfully!")
        else:
            print("\n❌ Training failed. Check logs for details.")
            
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("Training session ended.")
