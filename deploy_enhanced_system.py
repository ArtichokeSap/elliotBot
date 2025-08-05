"""
Production Deployment Script for Enhanced Elliott Wave ML System
Integrates enhanced S&R detection and ML models into the existing web application
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedElliottWaveSystem:
    """Enhanced Elliott Wave System with ML and advanced S&R detection"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the enhanced system"""
        self.config = self._load_config(config_path)
        self.ml_trainer = None
        self.sr_detector = None
        self.confluence_analyzer = None
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load system configuration"""
        default_config = {
            'ml_models': {
                'enabled': True,
                'models_dir': 'trained_models',
                'retrain_interval_days': 30,
                'min_confidence_threshold': 0.5
            },
            'enhanced_sr': {
                'enabled': True,
                'min_conviction': 0.6,
                'max_levels_per_type': 10,
                'zone_tolerance': 0.005
            },
            'data_sources': {
                'yahoo_finance': True,
                'alpha_vantage': False,
                'polygon': False
            },
            'analysis': {
                'default_timeframes': ['1d', '4h', '1h'],
                'lookback_periods': {'1d': 252, '4h': 504, '1h': 720},
                'confluence_threshold': 3
            },
            'api': {
                'rate_limit': 100,  # requests per minute
                'cache_ttl': 300,   # seconds
                'enable_batch_analysis': True
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                # Merge with defaults
                default_config.update(user_config)
                logger.info(f"✅ Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"⚠️ Error loading config {config_path}: {e}. Using defaults.")
        
        return default_config
    
    def initialize_components(self) -> bool:
        """Initialize all system components"""
        try:
            logger.info("🚀 Initializing Enhanced Elliott Wave System Components")
            
            # Initialize Enhanced S&R Detector
            if self.config['enhanced_sr']['enabled']:
                try:
                    from src.analysis.enhanced_sr_detector import EnhancedSRDetector
                    self.sr_detector = EnhancedSRDetector(
                        min_conviction=self.config['enhanced_sr']['min_conviction'],
                        max_levels_per_type=self.config['enhanced_sr']['max_levels_per_type']
                    )
                    logger.info("✅ Enhanced S&R Detector initialized")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize Enhanced S&R Detector: {e}")
                    return False
            
            # Initialize ML Training Framework
            if self.config['ml_models']['enabled']:
                try:
                    from src.ml.training_framework import MLTrainingFramework
                    self.ml_trainer = MLTrainingFramework(
                        models_dir=self.config['ml_models']['models_dir']
                    )
                    
                    # Load trained models if available
                    if self.ml_trainer.load_trained_models():
                        logger.info("✅ ML models loaded successfully")
                    else:
                        logger.warning("⚠️ No trained models found. Run training first.")
                        
                except Exception as e:
                    logger.error(f"❌ Failed to initialize ML framework: {e}")
                    # Continue without ML for now
                    self.config['ml_models']['enabled'] = False
            
            # Initialize Technical Confluence Analyzer
            try:
                from src.analysis.technical_confluence import TechnicalConfluenceAnalyzer
                self.confluence_analyzer = TechnicalConfluenceAnalyzer()
                logger.info("✅ Technical Confluence Analyzer initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Confluence Analyzer: {e}")
                return False
            
            logger.info("🎉 All components initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error initializing system: {e}")
            return False
    
    def analyze_symbol(
        self, 
        symbol: str, 
        timeframe: str = '1d',
        include_ml_predictions: bool = True,
        include_enhanced_sr: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive Elliott Wave analysis with enhanced features
        
        Args:
            symbol: Trading symbol to analyze
            timeframe: Analysis timeframe
            include_ml_predictions: Whether to include ML model predictions
            include_enhanced_sr: Whether to include enhanced S&R analysis
            
        Returns:
            Complete analysis results with enhanced features
        """
        try:
            logger.info(f"📊 Analyzing {symbol} on {timeframe}")
            
            # Get market data
            from src.data.data_fetcher import DataFetcher
            data_fetcher = DataFetcher()
            
            lookback = self.config['analysis']['lookback_periods'].get(timeframe, 252)
            df = data_fetcher.get_historical_data(symbol, timeframe, lookback)
            
            if df is None or df.empty:
                return {'error': f'No data available for {symbol}'}
            
            # Base Elliott Wave analysis
            base_analysis = self.confluence_analyzer.analyze_confluence(df, symbol)
            
            # Enhanced S&R Detection
            enhanced_sr_results = {}
            if include_enhanced_sr and self.sr_detector:
                try:
                    sr_analysis = self.sr_detector.detect_support_resistance(df)
                    enhanced_sr_results = {
                        'support_levels': [level.price for level in sr_analysis.support_levels],
                        'resistance_levels': [level.price for level in sr_analysis.resistance_levels],
                        'support_zones': [
                            {'min': zone.min_price, 'max': zone.max_price, 'conviction': zone.conviction}
                            for zone in sr_analysis.support_zones
                        ],
                        'resistance_zones': [
                            {'min': zone.min_price, 'max': zone.max_price, 'conviction': zone.conviction}
                            for zone in sr_analysis.resistance_zones
                        ],
                        'detection_summary': sr_analysis.detection_summary
                    }
                    logger.info(f"✅ Enhanced S&R: {len(enhanced_sr_results['support_levels'])} support, {len(enhanced_sr_results['resistance_levels'])} resistance")
                except Exception as e:
                    logger.error(f"❌ Enhanced S&R detection failed: {e}")
                    enhanced_sr_results = {'error': 'Enhanced S&R detection failed'}
            
            # ML Predictions
            ml_predictions = {}
            if include_ml_predictions and self.ml_trainer and self.config['ml_models']['enabled']:
                try:
                    # Extract features for ML prediction
                    current_price = float(df['close'].iloc[-1])
                    
                    # Calculate technical indicators for ML features
                    rsi = self._calculate_rsi(df['close'])
                    volume_ratio = float(df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1])
                    volatility = float(df['close'].pct_change().rolling(20).std().iloc[-1])
                    
                    ml_features = {
                        'confluence_score': base_analysis.get('confluence_score', 0),
                        'current_price': current_price,
                        'rsi': rsi,
                        'volume_ratio': volume_ratio,
                        'volatility': volatility,
                        'wave_structure_quality': base_analysis.get('structure_quality', 0.5),
                        'fibonacci_alignment': base_analysis.get('fibonacci_score', 0.5),
                        'rule_compliance': base_analysis.get('rule_compliance', 0.5),
                        'pattern_clarity': base_analysis.get('pattern_clarity', 0.5),
                        'similar_pattern_success_rate': 0.6,  # Default, could be historical
                        'timeframe_success_rate': 0.65,       # Default, could be calculated
                        'symbol_success_rate': 0.6            # Default, could be symbol-specific
                    }
                    
                    # Get ML predictions
                    wave_confidence = self.ml_trainer.predict_wave_confidence(ml_features)
                    zone_confidence = self.ml_trainer.predict_zone_confidence(ml_features)
                    
                    ml_predictions = {
                        'wave_confidence': float(wave_confidence),
                        'zone_confidence': float(zone_confidence),
                        'confidence_level': self._get_confidence_level(wave_confidence),
                        'recommendation': self._generate_ml_recommendation(wave_confidence, zone_confidence),
                        'model_features_used': list(ml_features.keys())
                    }
                    
                    logger.info(f"✅ ML Predictions: Wave {wave_confidence:.3f}, Zone {zone_confidence:.3f}")
                    
                except Exception as e:
                    logger.error(f"❌ ML prediction failed: {e}")
                    ml_predictions = {'error': 'ML prediction failed'}
            
            # Combine all results
            comprehensive_analysis = {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'base_analysis': base_analysis,
                'enhanced_sr': enhanced_sr_results,
                'ml_predictions': ml_predictions,
                'system_info': {
                    'enhanced_sr_enabled': include_enhanced_sr and bool(self.sr_detector),
                    'ml_enabled': include_ml_predictions and self.config['ml_models']['enabled'],
                    'version': '2.0-enhanced'
                }
            }
            
            # Generate trading signals based on comprehensive analysis
            comprehensive_analysis['trading_signals'] = self._generate_trading_signals(comprehensive_analysis)
            
            return comprehensive_analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {e}")
            return {'error': f'Analysis failed for {symbol}: {str(e)}'}
    
    def batch_analyze(self, symbols: List[str], timeframe: str = '1d') -> Dict[str, Any]:
        """Analyze multiple symbols efficiently"""
        logger.info(f"📊 Batch analyzing {len(symbols)} symbols")
        
        results = {}
        successful = 0
        
        for symbol in symbols:
            try:
                analysis = self.analyze_symbol(symbol, timeframe)
                results[symbol] = analysis
                
                if 'error' not in analysis:
                    successful += 1
                    
            except Exception as e:
                logger.error(f"❌ Failed to analyze {symbol}: {e}")
                results[symbol] = {'error': str(e)}
        
        logger.info(f"✅ Batch analysis complete: {successful}/{len(symbols)} successful")
        
        return {
            'batch_results': results,
            'summary': {
                'total_symbols': len(symbols),
                'successful': successful,
                'failed': len(symbols) - successful,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _calculate_rsi(self, prices, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1])
        except:
            return 50.0  # Default neutral RSI
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Convert numeric confidence to level"""
        if confidence >= 0.7:
            return 'HIGH'
        elif confidence >= 0.5:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _generate_ml_recommendation(self, wave_conf: float, zone_conf: float) -> str:
        """Generate trading recommendation based on ML predictions"""
        if wave_conf >= 0.7 and zone_conf >= 0.7:
            return 'STRONG_SIGNAL'
        elif wave_conf >= 0.6 and zone_conf >= 0.6:
            return 'MODERATE_SIGNAL'
        elif wave_conf >= 0.5 or zone_conf >= 0.5:
            return 'WEAK_SIGNAL'
        else:
            return 'NO_CLEAR_SIGNAL'
    
    def _generate_trading_signals(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive trading signals"""
        signals = {
            'primary_signal': 'NEUTRAL',
            'confidence': 'MEDIUM',
            'supporting_factors': [],
            'risk_factors': [],
            'entry_zones': [],
            'target_zones': [],
            'stop_loss_zones': []
        }
        
        try:
            base = analysis.get('base_analysis', {})
            ml_pred = analysis.get('ml_predictions', {})
            sr_analysis = analysis.get('enhanced_sr', {})
            
            # Determine primary signal
            confluence_score = base.get('confluence_score', 0)
            ml_confidence = ml_pred.get('wave_confidence', 0.5)
            
            if confluence_score >= 6 and ml_confidence >= 0.7:
                signals['primary_signal'] = 'STRONG_BUY'
                signals['confidence'] = 'HIGH'
            elif confluence_score >= 4 and ml_confidence >= 0.6:
                signals['primary_signal'] = 'BUY'
                signals['confidence'] = 'MEDIUM'
            elif confluence_score >= 2 and ml_confidence >= 0.5:
                signals['primary_signal'] = 'WEAK_BUY'
                signals['confidence'] = 'LOW'
            
            # Add supporting factors
            if confluence_score >= 5:
                signals['supporting_factors'].append('High technical confluence')
            if ml_confidence >= 0.7:
                signals['supporting_factors'].append('High ML confidence')
            if len(sr_analysis.get('support_levels', [])) >= 2:
                signals['supporting_factors'].append('Strong support levels identified')
            
            # Add target zones from base analysis
            if 'targets' in base:
                signals['target_zones'] = base['targets']
            
            # Use enhanced S&R for entry and stop levels
            if sr_analysis:
                signals['entry_zones'] = sr_analysis.get('support_levels', [])[:3]
                signals['stop_loss_zones'] = sr_analysis.get('support_levels', [])[-2:] if len(sr_analysis.get('support_levels', [])) > 1 else []
            
        except Exception as e:
            logger.error(f"❌ Error generating trading signals: {e}")
        
        return signals

def create_web_api_endpoints():
    """Create Flask API endpoints for the enhanced system"""
    
    try:
        from flask import Flask, jsonify, request
        from flask_cors import CORS
        
        app = Flask(__name__)
        CORS(app)
        
        # Initialize the enhanced system
        enhanced_system = EnhancedElliottWaveSystem()
        
        if not enhanced_system.initialize_components():
            logger.error("❌ Failed to initialize enhanced system")
            return None
        
        @app.route('/api/v2/analyze', methods=['POST'])
        def analyze_enhanced():
            """Enhanced analysis endpoint"""
            try:
                data = request.get_json()
                symbol = data.get('symbol')
                timeframe = data.get('timeframe', '1d')
                include_ml = data.get('include_ml', True)
                include_sr = data.get('include_enhanced_sr', True)
                
                if not symbol:
                    return jsonify({'error': 'Symbol is required'}), 400
                
                result = enhanced_system.analyze_symbol(
                    symbol=symbol,
                    timeframe=timeframe,
                    include_ml_predictions=include_ml,
                    include_enhanced_sr=include_sr
                )
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"❌ API error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @app.route('/api/v2/batch_analyze', methods=['POST'])
        def batch_analyze_enhanced():
            """Enhanced batch analysis endpoint"""
            try:
                data = request.get_json()
                symbols = data.get('symbols', [])
                timeframe = data.get('timeframe', '1d')
                
                if not symbols:
                    return jsonify({'error': 'Symbols list is required'}), 400
                
                result = enhanced_system.batch_analyze(symbols, timeframe)
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"❌ Batch API error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @app.route('/api/v2/status', methods=['GET'])
        def system_status():
            """System status endpoint"""
            status = {
                'system': 'Enhanced Elliott Wave Analysis v2.0',
                'components': {
                    'enhanced_sr_detector': bool(enhanced_system.sr_detector),
                    'ml_framework': bool(enhanced_system.ml_trainer),
                    'confluence_analyzer': bool(enhanced_system.confluence_analyzer)
                },
                'configuration': enhanced_system.config,
                'timestamp': datetime.now().isoformat()
            }
            return jsonify(status)
        
        logger.info("✅ Enhanced API endpoints created")
        return app
        
    except Exception as e:
        logger.error(f"❌ Error creating API endpoints: {e}")
        return None

def main():
    """Main deployment function"""
    print("🚀 Enhanced Elliott Wave System - Production Deployment")
    print("=" * 60)
    
    try:
        # Initialize enhanced system
        logger.info("🔧 Initializing Enhanced Elliott Wave System...")
        
        enhanced_system = EnhancedElliottWaveSystem()
        
        if not enhanced_system.initialize_components():
            print("❌ Failed to initialize system components")
            return False
        
        # Test with a sample analysis
        logger.info("🧪 Running system validation test...")
        
        test_result = enhanced_system.analyze_symbol('AAPL', '1d')
        
        if 'error' in test_result:
            logger.error(f"❌ System test failed: {test_result['error']}")
            return False
        
        logger.info("✅ System validation passed!")
        
        # Print system status
        print("\n📊 SYSTEM STATUS:")
        print(f"   Enhanced S&R Detection: {'✅ Active' if enhanced_system.sr_detector else '❌ Disabled'}")
        print(f"   ML Framework: {'✅ Active' if enhanced_system.ml_trainer and enhanced_system.config['ml_models']['enabled'] else '❌ Disabled'}")
        print(f"   Technical Confluence: {'✅ Active' if enhanced_system.confluence_analyzer else '❌ Disabled'}")
        
        # Display test results summary
        if 'enhanced_sr' in test_result and test_result['enhanced_sr']:
            sr_info = test_result['enhanced_sr']
            print(f"   S&R Levels: {len(sr_info.get('support_levels', []))} support, {len(sr_info.get('resistance_levels', []))} resistance")
        
        if 'ml_predictions' in test_result and test_result['ml_predictions']:
            ml_info = test_result['ml_predictions']
            print(f"   ML Confidence: {ml_info.get('confidence_level', 'N/A')}")
        
        # Ask user about web API deployment
        print("\n🌐 WEB API DEPLOYMENT:")
        deploy_api = input("Deploy enhanced web API? (y/n): ").strip().lower() == 'y'
        
        if deploy_api:
            app = create_web_api_endpoints()
            if app:
                print("✅ Enhanced API endpoints ready!")
                print("   Available endpoints:")
                print("   POST /api/v2/analyze - Enhanced single symbol analysis")
                print("   POST /api/v2/batch_analyze - Enhanced batch analysis")
                print("   GET  /api/v2/status - System status")
                
                print("\n🚀 Starting enhanced web server...")
                app.run(host='0.0.0.0', port=5000, debug=False)
            else:
                print("❌ Failed to create API endpoints")
        else:
            print("✅ System ready for programmatic use!")
            
            # Show usage example
            print("\n💡 USAGE EXAMPLE:")
            print("```python")
            print("from deploy_enhanced_system import EnhancedElliottWaveSystem")
            print("")
            print("system = EnhancedElliottWaveSystem()")
            print("system.initialize_components()")
            print("")
            print("# Analyze single symbol")
            print("result = system.analyze_symbol('AAPL', '1d')")
            print("")
            print("# Batch analyze")
            print("batch_result = system.batch_analyze(['AAPL', 'MSFT', 'GOOGL'])")
            print("```")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️ Deployment interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ Deployment error: {e}")
        print(f"❌ Deployment failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Enhanced Elliott Wave System deployed successfully!")
    else:
        print("\n❌ Deployment failed. Check logs for details.")
