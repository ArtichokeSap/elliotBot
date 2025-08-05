"""
Elliott Wave Bot - Web Application (Fixed Version)
Interactive web interface for Elliott Wave analysis with enhanced Fibonacci levels and future predictions
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import sys
import os
import types
import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_loader import DataLoader
from src.analysis.wave_detector import WaveDetector
from src.analysis.fibonacci import FibonacciAnalyzer
from src.analysis.elliott_wave_validator import ElliottWaveValidator
from src.analysis.comprehensive_elliott_validator import ComprehensiveElliottValidator
from src.analysis.enhanced_wave_detector import EnhancedWaveDetector
from src.analysis.ml_wave_accuracy import MLWaveAccuracy
from src.analysis.auto_tuning import WaveAutoTuner
from src.analysis.backtesting_engine import BacktestingEngine
from src.visualization.visualizer import WaveVisualizer
from src.visualization.tradingview_style import TradingViewStyleVisualizer
from src.visualization.comprehensive_visualizer import ComprehensiveWaveVisualizer

# NEW: Import Technical Analysis System
from src.analysis.technical_confluence import TechnicalConfluenceAnalyzer
from src.data.enhanced_data_fetcher import EnhancedDataFetcher

# Charts disabled - data-only mode
# import plotly.graph_objects as go
# import plotly.utils

# Initialize Technical Analysis System
enhanced_data_fetcher = EnhancedDataFetcher()
technical_confluence_analyzer = TechnicalConfluenceAnalyzer()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.secret_key = 'elliott_wave_secret_key_2025'

# Initialize components with comprehensive Elliott Wave analysis
data_loader = DataLoader()

# Add get_data method if it doesn't exist (for backward compatibility)
if not hasattr(data_loader, 'get_data'):
    def get_data_method(symbol: str, source: str = 'auto', **kwargs):
        """Unified method to get market data from various sources."""
        try:
            # Auto-detect source based on symbol format
            if source == 'auto':
                if '/' in symbol or symbol.upper().endswith('USDT'):
                    source = 'binance'
                else:
                    source = 'yahoo'
            
            # Route to appropriate data source
            if source == 'yahoo':
                return data_loader.get_yahoo_data(symbol, **kwargs)
            elif source == 'binance':
                return data_loader.get_binance_data(symbol, **kwargs)
            elif source == 'csv':
                file_path = kwargs.get('file_path')
                if not file_path:
                    raise ValueError("file_path required for CSV source")
                return data_loader.load_csv_data(file_path, symbol)
            else:
                raise ValueError(f"Unsupported data source: {source}")
                
        except Exception as e:
            logger.error(f"Error loading data for {symbol} from {source}: {e}")
            # Return empty DataFrame with proper structure as fallback
            import pandas as pd
            df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            df.index.name = 'timestamp'
            return df
    
    # Bind the method to the data_loader instance
    import types
    data_loader.get_data = types.MethodType(get_data_method, data_loader)
    logger.info("Added get_data method to DataLoader instance")

wave_detector = WaveDetector()
enhanced_detector = EnhancedWaveDetector(min_wave_size=0.02, lookback_periods=5)
comprehensive_validator = ComprehensiveElliottValidator()

# Initialize ML components
try:
    ml_accuracy = MLWaveAccuracy()
    auto_tuner = WaveAutoTuner()
    backtesting_engine = BacktestingEngine()
    ml_features_available = True
    logger.info("ML features initialized successfully")
except Exception as e:
    logger.warning(f"ML features not available: {e}")
    ml_accuracy = None
    auto_tuner = None
    backtesting_engine = None
    ml_features_available = False

# Adjust wave detector for better web performance
wave_detector.zigzag_threshold = 0.03  # More sensitive (was 0.05)
wave_detector.min_wave_length = 3      # Shorter minimum waves
wave_detector.confidence_threshold = 0.5  # Lower confidence threshold

fibonacci_analyzer = FibonacciAnalyzer()
elliott_validator = ElliottWaveValidator()
visualizer = WaveVisualizer()
tv_visualizer = TradingViewStyleVisualizer()
comprehensive_visualizer = ComprehensiveWaveVisualizer()

# Trading pairs configuration
TRADING_PAIRS = {
    'forex': {
        'EURUSD': 'EURUSD=X',
        'GBPUSD': 'GBPUSD=X', 
        'USDJPY': 'USDJPY=X',
        'AUDUSD': 'AUDUSD=X',
        'USDCAD': 'USDCAD=X',
        'USDCHF': 'USDCHF=X',
        'NZDUSD': 'NZDUSD=X',
        'EURGBP': 'EURGBP=X',
        'EURJPY': 'EURJPY=X',
        'GBPJPY': 'GBPJPY=X'
    },
    'crypto': {
        'BTC-USD': 'BTC-USD',
        'ETH-USD': 'ETH-USD',
        'ADA-USD': 'ADA-USD',
        'DOT-USD': 'DOT-USD',
        'LINK-USD': 'LINK-USD',
        'XRP-USD': 'XRP-USD',
        'LTC-USD': 'LTC-USD',
        'BCH-USD': 'BCH-USD',
        'BNB-USD': 'BNB-USD',
        'SOL-USD': 'SOL-USD'
    },
    'stocks': {
        'AAPL': 'AAPL',
        'MSFT': 'MSFT',
        'GOOGL': 'GOOGL',
        'AMZN': 'AMZN',
        'TSLA': 'TSLA',
        'NVDA': 'NVDA',
        'META': 'META',
        'NFLX': 'NFLX',
        'AMD': 'AMD',
        'INTC': 'INTC'
    },
    'commodities': {
        'Gold': 'GC=F',
        'Silver': 'SI=F',
        'Crude Oil': 'CL=F',
        'Natural Gas': 'NG=F',
        'Copper': 'HG=F',
        'Platinum': 'PL=F',
        'Palladium': 'PA=F'
    }
}

# Timeframe configuration
TIMEFRAMES = {
    '1m': {'interval': '1m', 'period': '1d', 'label': '1 Minute'},
    '5m': {'interval': '5m', 'period': '5d', 'label': '5 Minutes'},
    '15m': {'interval': '15m', 'period': '5d', 'label': '15 Minutes'},
    '30m': {'interval': '30m', 'period': '1mo', 'label': '30 Minutes'},
    '1h': {'interval': '1h', 'period': '1mo', 'label': '1 Hour'},
    '4h': {'interval': '1h', 'period': '3mo', 'label': '4 Hours'},
    '1d': {'interval': '1d', 'period': '1y', 'label': '1 Day'},
    '1wk': {'interval': '1wk', 'period': '2y', 'label': '1 Week'},
    '1mo': {'interval': '1mo', 'period': '5y', 'label': '1 Month'}
}

def create_ascii_wave_table(wave_data, market_summary):
    """Create ASCII-style table for Elliott Wave analysis results."""
    if not wave_data:
        return "No Elliott Waves detected."
    
    # Header
    lines = []
    lines.append("=" * 80)
    lines.append(f"ELLIOTT WAVE ANALYSIS - {market_summary['symbol']}")
    lines.append("=" * 80)
    lines.append(f"Current Price: ${market_summary['current_price']:.4f} | Change 24h: {market_summary['change_24h']:+.2f}%")
    lines.append(f"Timeframe: {market_summary['timeframe']} | Data Points: {market_summary['data_points']}")
    lines.append("-" * 80)
    
    # Table header
    lines.append("| Wave | Direction |     Start Price |       End Price | Change % | Confidence |")
    lines.append("|------|-----------|-----------------|-----------------|----------|------------|")
    
    # Wave rows
    for wave in wave_data:
        wave_type = wave.get('type', 'Unknown')[:6]  # Limit to 6 chars
        direction = wave.get('direction', 'N/A')[:9]  # Limit to 9 chars
        start_price = wave.get('start_price', 0)
        end_price = wave.get('end_price', 0)
        change_pct = wave.get('price_change', 0)
        confidence = wave.get('confidence', 0) * 100
        
        lines.append(f"| {wave_type:<4} | {direction:<9} | ${start_price:>13.4f} | ${end_price:>13.4f} | {change_pct:>6.2f}% | {confidence:>8.1f}% |")
    
    lines.append("-" * 80)
    lines.append(f"Analysis completed at: {market_summary.get('last_update', 'Unknown')}")
    lines.append("=" * 80)
    
    return "\n".join(lines)

@app.route('/')
def index():
    """Main page with Elliott Wave analysis interface."""
    return render_template('index.html', 
                         trading_pairs=TRADING_PAIRS, 
                         timeframes=TIMEFRAMES)

@app.route('/api/analyze', methods=['POST'])
def analyze_pair():
    """API endpoint to analyze a specific trading pair and timeframe."""
    
    # Initialize response variables
    validation_results = []
    fibonacci_levels = []
    future_predictions = []
    chart_path = None
    analysis_result = None  # Initialize here to ensure scope
    
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        timeframe = data.get('timeframe', '1d')
        
        logger.info(f"🚀 API Request: Analyzing {symbol} on {timeframe} timeframe")
        print(f"🚀 API Request: Analyzing {symbol} on {timeframe} timeframe")
        
        # Get timeframe configuration
        tf_config = TIMEFRAMES.get(timeframe, TIMEFRAMES['1d'])
        
        # Load market data
        try:
            market_data = data_loader.get_yahoo_data(
                symbol, 
                period=tf_config['period'],
                interval=tf_config['interval']
            )
            
            if market_data.empty:
                return jsonify({
                    'success': False,
                    'error': f'No data available for {symbol}'
                })
            
            logger.info(f"Loaded {len(market_data)} data points for {symbol}")
            
            # Debug: Check data quality
            if len(market_data) < 10:
                logger.warning(f"Limited data for {symbol}: only {len(market_data)} points")
                
        except Exception as e:
            logger.error(f"Data loading error for {symbol}: {e}")
            return jsonify({
                'success': False,
                'error': f'Failed to load data for {symbol}: {str(e)}'
            })
        
        # NEW: Technical Confluence Analysis System
        waves = []
        validation_results = []
        fibonacci_levels = []
        future_predictions = []
        chart_path = None
        target_zones = []
        
        try:
            logger.info("🚀 Starting Enhanced Elliott Wave + Technical Confluence Analysis...")
            print("🚀 Starting Enhanced Elliott Wave + Technical Confluence Analysis...")
            
            # Step 1: Elliott Wave Detection
            analysis_result = enhanced_detector.detect_elliott_waves(market_data, symbol)
            print(f"📊 Elliott Wave Analysis result keys: {list(analysis_result.keys())}")
            print(f"🎯 Elliott Wave Validation score: {analysis_result.get('validation_score', 0):.1%}")
            
            # Step 2: Technical Confluence Analysis
            if analysis_result and analysis_result.get('validation_score', 0) > 0.0:
                logger.info("🧩 Running Technical Confluence Analysis...")
                print("🧩 Running Technical Confluence Analysis...")
                
                # Use the new technical confluence analyzer
                target_zones = technical_confluence_analyzer.analyze_target_zones(
                    market_data, 
                    analysis_result, 
                    timeframe
                )
                
                print(f"🎯 Found {len(target_zones)} target zones")
                logger.info(f"Found {len(target_zones)} confluence-based target zones")
                
                # Convert Elliott Wave results to web format
                for wave_data in analysis_result['waves']:
                    waves.append({
                        'type': wave_data['wave'],
                        'direction': 'UP' if wave_data['direction'] == 'bullish' else 'DOWN',
                        'start_time': str(wave_data['start_time']),
                        'end_time': str(wave_data['end_time']),
                        'start_price': float(wave_data['start_price']),
                        'end_price': float(wave_data['end_price']),
                        'price_change': float(((wave_data['end_price'] - wave_data['start_price']) / wave_data['start_price']) * 100),
                        'confidence': float(wave_data['confidence']),
                        'wave_type': wave_data['wave_type'],
                        'length': float(wave_data['length']),
                        'duration': int(wave_data['duration'])
                    })
                
                # Convert target zones to validation results format
                validation_results = [{
                    'type': analysis_result['wave_structure'].upper(),
                    'score': round(analysis_result['validation_score'] * 100, 1),
                    'status': 'HIGH_CONFIDENCE' if analysis_result['validation_score'] >= 0.8 else 'MODERATE_CONFIDENCE' if analysis_result['validation_score'] >= 0.6 else 'LOW_CONFIDENCE',
                    'rule_compliance': analysis_result['rule_compliance'],
                    'fibonacci_levels': analysis_result['fibonacci_levels'],
                    'recommendations': analysis_result['recommendations'],
                    'issues': analysis_result['issues'],
                    'target_zones_count': len(target_zones),
                    'high_confidence_targets': len([tz for tz in target_zones if tz.confidence_level == "HIGH"]),
                    'confluence_analysis': True
                }]
                
                # Extract Technical Confluence Fibonacci levels
                if target_zones:
                    for target_zone in target_zones[:10]:  # Top 10 targets
                        fibonacci_levels.append({
                            'level': f'{target_zone.wave_target} Target',
                            'value': float(target_zone.price_level),
                            'type': 'confluence_target',
                            'confluence_score': target_zone.confluence_score,
                            'confidence_level': target_zone.confidence_level,
                            'confluences': target_zone.confluences[:3],  # Top 3 confluences for predictions
                            'all_confluences': target_zone.confluences,  # All confluences for detailed analysis[:5],  # Top 5 confluences for table
                            'all_confluences': target_zone.confluences,  # All confluences for detailed view
                            'probability': target_zone.probability,
                            'risk_reward': target_zone.risk_reward_ratio
                        })
                
                # Create Enhanced Future Predictions from Target Zones
                current_price = float(market_data['close'].iloc[-1])
                high_confidence_targets = [tz for tz in target_zones if tz.confidence_level == "HIGH"]
                medium_confidence_targets = [tz for tz in target_zones if tz.confidence_level == "MEDIUM"]
                
                if high_confidence_targets:
                    best_target = high_confidence_targets[0]
                    price_change_pct = ((best_target.price_level - current_price) / current_price) * 100
                    
                    future_predictions.append({
                        'pattern': f'{best_target.elliott_basis} - High Confidence Target',
                        'probability': f"High ({best_target.probability:.0%})",
                        'targets': [{
                            'level': f'{best_target.wave_target} (Best)',
                            'price': best_target.price_level,
                            'ratio': f'{price_change_pct:+.1f}%'
                        }],
                        'expected_move': f"Target: {best_target.wave_target} ({price_change_pct:+.1f}%)",
                        'timeframe': timeframe,
                        'confidence': best_target.probability,
                        'confluence_score': best_target.confluence_score,
                        'confluences': best_target.confluences,
                        'validation_score': round(analysis_result['validation_score'] * 100, 1)
                    })
                    
                    # Add additional targets if available
                    for i, target in enumerate(high_confidence_targets[1:3], 1):  # Next 2 targets
                        price_change = ((target.price_level - current_price) / current_price) * 100
                        future_predictions.append({
                            'pattern': f'{target.elliott_basis} - Alternative Target {i+1}',
                            'probability': f"High ({target.probability:.0%})",
                            'targets': [{
                                'level': f'{target.wave_target}',
                                'price': target.price_level,
                                'ratio': f'{price_change:+.1f}%'
                            }],
                            'expected_move': f"Alternative: {target.wave_target} ({price_change:+.1f}%)",
                            'timeframe': timeframe,
                            'confidence': target.probability,
                            'confluence_score': target.confluence_score,
                            'confluences': target.confluences[:2]
                        })
                
                elif medium_confidence_targets:
                    best_target = medium_confidence_targets[0]
                    price_change_pct = ((best_target.price_level - current_price) / current_price) * 100
                    
                    future_predictions.append({
                        'pattern': f'{best_target.elliott_basis} - Medium Confidence',
                        'probability': f"Medium ({best_target.probability:.0%})",
                        'targets': [{
                            'level': f'{best_target.wave_target}',
                            'price': best_target.price_level,
                            'ratio': f'{price_change_pct:+.1f}%'
                        }],
                        'expected_move': f"Target: {best_target.wave_target} ({price_change_pct:+.1f}%)",
                        'timeframe': timeframe,
                        'confidence': best_target.probability,
                        'confluence_score': best_target.confluence_score,
                        'confluences': best_target.confluences[:3],  # Top 3 confluences for medium targets
                        'all_confluences': best_target.confluences   # All confluences
                    })
                
                logger.info(f"✅ Technical Confluence Analysis Complete: {len(target_zones)} targets, {len(high_confidence_targets)} high confidence")
                print(f"✅ Technical Confluence Analysis Complete: {len(target_zones)} targets, {len(high_confidence_targets)} high confidence")
                
            else:
                logger.warning("⚠️ Elliott Wave analysis yielded low confidence, using basic trend analysis...")
                print("⚠️ Elliott Wave analysis yielded low confidence, using basic trend analysis...")
                
                # Basic fallback analysis
                if len(market_data) >= 10:
                    price_changes = market_data['close'].pct_change().dropna()
                    trend = 'UP' if price_changes.tail(5).mean() > 0 else 'DOWN'
                    
                    waves.append({
                        'type': 'TREND',
                        'direction': trend,
                        'start_time': str(market_data.index[0]),
                        'end_time': str(market_data.index[-1]),
                        'start_price': float(market_data['close'].iloc[0]),
                        'end_price': float(market_data['close'].iloc[-1]),
                        'price_change': float(((market_data['close'].iloc[-1] - market_data['close'].iloc[0]) / market_data['close'].iloc[0]) * 100),
                        'confidence': 0.5,
                        'wave_type': 'trend',
                        'length': float(abs(market_data['close'].iloc[-1] - market_data['close'].iloc[0])),
                        'duration': len(market_data)
                    })
                    
                    validation_results = [{
                        'type': 'BASIC_TREND_ANALYSIS',
                        'score': 50.0,
                        'status': 'LOW_CONFIDENCE',
                        'rule_compliance': {},
                        'fibonacci_levels': {},
                        'recommendations': ['Consider using different timeframe for Elliott Wave analysis'],
                        'issues': ['Insufficient Elliott Wave pattern clarity'],
                        'target_zones_count': 0,
                        'high_confidence_targets': 0,
                        'confluence_analysis': False
                    }]
                
        except Exception as e:
            logger.error(f"Enhanced analysis error: {e}")
            print(f"❌ Enhanced analysis error: {e}")
            traceback.print_exc()
            
            # Create minimal fallback response
            waves = []
            validation_results = []
            fibonacci_levels = []
            future_predictions = []
            target_zones = []
                
        except Exception as e:
            logger.error(f"Wave detection error: {e}")
            print(f"❌ Wave detection error: {e}")
            import traceback
            traceback.print_exc()
            
            # Create minimal fallback response
            waves = []
            validation_results = []
            fibonacci_levels = []
            future_predictions = []
        
        # Enhanced Fibonacci levels and future predictions - replaced with Technical Confluence Analysis above
        # All processing completed in the Technical Confluence Analysis section
        
        except Exception as e:
            logger.error(f"Enhanced Technical Analysis error: {e}")
            print(f"❌ Enhanced Technical Analysis error: {e}")
            traceback.print_exc()
            
            # Create minimal fallback response
            waves = []
            validation_results = []
            fibonacci_levels = []
            future_predictions = []
            target_zones = []
        
        # Market summary (moved here before ASCII table creation)
        current_price = market_data['close'].iloc[-1]
        high_52w = market_data['high'].max()
        low_52w = market_data['low'].min()
        
        market_summary = {
            'symbol': symbol,
            'current_price': round(current_price, 4),
            'high_52w': round(high_52w, 4),
            'low_52w': round(low_52w, 4),
            'change_24h': round(((current_price - market_data['close'].iloc[-2]) / market_data['close'].iloc[-2]) * 100, 2) if len(market_data) > 1 else 0,
            'data_points': len(market_data),
            'timeframe': tf_config['label'],
            'last_update': market_data.index[-1].strftime('%Y-%m-%d %H:%M:%S UTC')
        }
        
        # Chart rendering is DISABLED - data-only mode
        
        # Prepare wave data for response (only if not already processed by comprehensive analysis)
        wave_data = []
        if analysis_result and analysis_result.get('validation_score', 0) > 0.0:
            # Use waves already converted in comprehensive analysis section
            wave_data = [
                {
                    'type': wave.get('type', wave.get('wave', 'Unknown')),
                    'direction': wave.get('direction', 'UNKNOWN'),
                    'start_date': wave.get('start_time', ''),
                    'end_date': wave.get('end_time', ''),
                    'start_price': wave.get('start_price', 0),
                    'end_price': wave.get('end_price', 0),
                    'price_change': wave.get('price_change', 0),
                    'confidence': wave.get('confidence', 0)
                }
                for wave in waves
            ]
        else:
            # Fallback for old format waves (if any)
            for wave in waves:
                if hasattr(wave, 'wave_type'):  # Old format
                    wave_data.append({
                        'type': wave.wave_type.value,
                        'direction': 'UP' if wave.direction.value == 1 else 'DOWN',
                        'start_date': wave.start_point.timestamp.strftime('%Y-%m-%d %H:%M'),
                        'end_date': wave.end_point.timestamp.strftime('%Y-%m-%d %H:%M'),
                        'start_price': round(wave.start_point.price, 4),
                        'end_price': round(wave.end_point.price, 4),
                        'price_change': round(((wave.end_point.price - wave.start_point.price) / wave.start_point.price) * 100, 2),
                        'confidence': round(wave.confidence, 3)
                    })
        
        # Create ASCII table now that wave_data is properly constructed
        ascii_table = create_ascii_wave_table(wave_data, market_summary)
        
        # Debug analysis result contents
        if analysis_result:
            logger.info(f"🔍 Analysis result validation_score: {analysis_result.get('validation_score', 'MISSING')}")
            logger.info(f"🔍 Analysis result wave_structure: {analysis_result.get('wave_structure', 'MISSING')}")
            logger.info(f"🔍 Analysis result direction: {analysis_result.get('direction', 'MISSING')}")
            print(f"🔍 DEBUG: validation_score = {analysis_result.get('validation_score', 'MISSING')}")
        
        return jsonify({
            'success': True,
            'validation_score': analysis_result.get('validation_score', 0) if analysis_result else 0,
            'wave_structure': analysis_result.get('wave_structure', 'unknown') if analysis_result else 'unknown',
            'direction': analysis_result.get('direction', 'neutral') if analysis_result else 'neutral',
            'waves': wave_data,  # Original wave data
            'ascii_table': ascii_table,  # ASCII-style table
            'fibonacci_levels': fibonacci_levels,
            'future_predictions': future_predictions,
            'validation_results': validation_results,
            'chart': None,  # Charts disabled - data-only mode
            'market_summary': market_summary,
            'analysis_timestamp': datetime.now().isoformat(),
            'comprehensive_analysis': bool(analysis_result and analysis_result.get('validation_score', 0) > 0),
            'data_mode': True,  # Indicates chart is disabled, data-only mode
            'wave_count': len(wave_data),
            # NEW: Technical Confluence Analysis Results
            'target_zones': [
                {
                    'price_level': float(tz.price_level),
                    'wave_target': tz.wave_target,
                    'elliott_basis': tz.elliott_basis,
                    'confluence_score': tz.confluence_score,
                    'confidence_level': tz.confidence_level,
                    'probability': tz.probability,
                    'risk_reward_ratio': tz.risk_reward_ratio,
                    'confluences': tz.confluences[:5],  # Top 5 confluences for table display
                    'all_confluences': tz.confluences,  # All confluences for detailed analysis
                    'confluence_methods': {
                        'fibonacci': [c for c in tz.confluences if 'Fibonacci' in c],
                        'support_resistance': [c for c in tz.confluences if any(sr in c for sr in ['Support', 'Resistance', 'Level'])],
                        'momentum': [c for c in tz.confluences if any(m in c for m in ['RSI', 'MACD', 'Momentum', 'Stochastic'])],
                        'pattern': [c for c in tz.confluences if any(p in c for p in ['Pattern', 'Triangle', 'Channel', 'Wedge'])],
                        'volume': [c for c in tz.confluences if 'Volume' in c],
                        'harmonic': [c for c in tz.confluences if any(h in c for h in ['Harmonic', 'Gartley', 'Butterfly', 'Bat'])]
                    },
                    'price_change_pct': ((tz.price_level - current_price) / current_price) * 100
                }
                for tz in target_zones[:10]  # Top 10 target zones
            ] if target_zones else [],
            'confluence_summary': {
                'total_targets': len(target_zones),
                'high_confidence': len([tz for tz in target_zones if tz.confidence_level == "HIGH"]),
                'medium_confidence': len([tz for tz in target_zones if tz.confidence_level == "MEDIUM"]),
                'low_confidence': len([tz for tz in target_zones if tz.confidence_level == "LOW"]),
                'best_target': {
                    'price': float(target_zones[0].price_level),
                    'wave': target_zones[0].wave_target,
                    'confidence': target_zones[0].confidence_level,
                    'probability': target_zones[0].probability,
                    'confluences': len(target_zones[0].confluences)
                } if target_zones else None
            },
            'analysis_mode': 'technical_confluence'  # Indicates new analysis mode
        })
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({
            'success': False,
            'error': f'Analysis failed: {str(e)}'
        })

def create_simple_trend_analysis(data):
    """Create simple trend patterns when Elliott Waves are not detected."""
    from src.analysis.wave_detector import WavePoint, Wave, WaveType, Direction
    
    simple_waves = []
    
    try:
        # Find basic high and low points
        high_idx = data['high'].idxmax()
        low_idx = data['low'].idxmin()
        
        # Create start and end points
        start_idx = data.index[0]
        end_idx = data.index[-1]
        
        # Determine if it's more of an uptrend or downtrend
        start_price = data.loc[start_idx, 'close']
        end_price = data.loc[end_idx, 'close']
        
        if end_price > start_price:
            # Uptrend - create simple wave pattern
            if low_idx < high_idx:  # Low first, then high
                # Wave 1: Start to Low
                wave1 = Wave(
                    wave_type=WaveType.WAVE_1,
                    direction=Direction.DOWN,
                    start_point=WavePoint(start_idx, start_price),
                    end_point=WavePoint(low_idx, data.loc[low_idx, 'low']),
                    confidence=0.6
                )
                simple_waves.append(wave1)
                
                # Wave 3: Low to High  
                wave3 = Wave(
                    wave_type=WaveType.WAVE_3,
                    direction=Direction.UP,
                    start_point=WavePoint(low_idx, data.loc[low_idx, 'low']),
                    end_point=WavePoint(high_idx, data.loc[high_idx, 'high']),
                    confidence=0.7
                )
                simple_waves.append(wave3)
                
                # Wave 5: High to End
                wave5 = Wave(
                    wave_type=WaveType.WAVE_5,
                    direction=Direction.UP if end_price > data.loc[high_idx, 'high'] else Direction.DOWN,
                    start_point=WavePoint(high_idx, data.loc[high_idx, 'high']),
                    end_point=WavePoint(end_idx, end_price),
                    confidence=0.5
                )
                simple_waves.append(wave5)
        else:
            # Downtrend - create simple wave pattern
            if high_idx < low_idx:  # High first, then low
                # Wave A: Start to High
                wave_a = Wave(
                    wave_type=WaveType.WAVE_A,
                    direction=Direction.UP,
                    start_point=WavePoint(start_idx, start_price),
                    end_point=WavePoint(high_idx, data.loc[high_idx, 'high']),
                    confidence=0.6
                )
                simple_waves.append(wave_a)
                
                # Wave C: High to Low
                wave_c = Wave(
                    wave_type=WaveType.WAVE_C,
                    direction=Direction.DOWN,
                    start_point=WavePoint(high_idx, data.loc[high_idx, 'high']),
                    end_point=WavePoint(low_idx, data.loc[low_idx, 'low']),
                    confidence=0.7
                )
                simple_waves.append(wave_c)
                
    except Exception as e:
        logger.error(f"Error creating simple trend analysis: {e}")
    
    return simple_waves

def create_professional_web_chart(data, waves, symbol, timeframe, validation_results=None, analysis_result=None):
    """Chart creation disabled - data-only mode."""
    logger.info("Chart creation disabled - returning None (data-only mode)")
    return None

def create_simple_web_chart(data, waves, symbol, timeframe):
    """Chart creation disabled - data-only mode."""
    logger.info("Chart creation disabled - returning None (data-only mode)")
    return None

def create_web_chart(data, waves, symbol, timeframe):
    """Chart creation disabled - data-only mode."""
    logger.info("Chart creation disabled - returning None (data-only mode)")
    return None
    
    fig = go.Figure()
    
    # Add candlestick chart with TradingView-style colors
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name=symbol,
        increasing_line_color='#00b050',  # TradingView green
        decreasing_line_color='#ff3b3b',  # TradingView red  
        increasing_fillcolor='#e6ffe6',   # Light green fill
        decreasing_fillcolor='#ffe6e6',   # Light red fill
        line=dict(width=1),
        showlegend=False
    ))
    
    # TradingView-style Elliott Wave colors (clean and professional)
    wave_colors = {
        'WAVE_1': {'color': '#2962FF', 'width': 2, 'dash': 'solid'},  # Blue
        'WAVE_2': {'color': '#FF6D00', 'width': 2, 'dash': 'dot'},    # Orange
        'WAVE_3': {'color': '#00C851', 'width': 3, 'dash': 'solid'},  # Green (strongest)
        'WAVE_4': {'color': '#9C27B0', 'width': 2, 'dash': 'dot'},    # Purple
        'WAVE_5': {'color': '#FF1744', 'width': 2, 'dash': 'solid'},  # Red
        'WAVE_A': {'color': '#FF6D00', 'width': 2, 'dash': 'solid'},  # Orange
        'WAVE_B': {'color': '#607D8B', 'width': 2, 'dash': 'dot'},    # Blue Grey
        'WAVE_C': {'color': '#E91E63', 'width': 3, 'dash': 'solid'}   # Pink
    }
    
    # Add Elliott Wave lines with TradingView styling
    for i, wave in enumerate(waves):
        wave_style = wave_colors.get(wave.wave_type.value, {'color': '#FFC107', 'width': 2, 'dash': 'solid'})
        wave_label = wave.wave_type.value.split('_')[-1]
        color = wave_style['color']
        
        # Add wave line
        fig.add_trace(go.Scatter(
            x=[wave.start_point.timestamp, wave.end_point.timestamp],
            y=[wave.start_point.price, wave.end_point.price],
            mode='lines+markers+text',
            line=dict(
                color=color, 
                width=wave_style['width'],
                dash=wave_style['dash']
            ),
            marker=dict(
                size=8, 
                color=color,
                line=dict(color='white', width=2)  # White outline for better visibility
            ),
            text=['', wave_label],
            textposition='top center',
            textfont=dict(size=14, color=color, family="Inter, Arial, sans-serif", style="bold"),
            name=f'Wave {wave_label}',
            showlegend=False,
            hovertemplate=f'<b>Wave {wave_label}</b><br>' +
                         'Price: %{y:.4f}<br>' +
                         f'Confidence: {wave.get("confidence", 0):.1%}<extra></extra>'
        ))
        
        # Add Fibonacci retracements for corrective waves (2, 4, B)
        if wave.get('wave_type', '') in ['WAVE_2', 'WAVE_4', 'WAVE_B']:
            add_fibonacci_levels(fig, wave, color)
    
    # Add future pattern prediction with timeframe
    if len(waves) >= 2:
        add_future_pattern_prediction(fig, data, waves, timeframe)
    
    # TradingView-style layout configuration (WHITE THEME)
    fig.update_layout(
        # Remove title for cleaner look
        title=None,
        
        # TradingView WHITE theme
        plot_bgcolor='white',
        paper_bgcolor='white',
        
        # Chart dimensions
        height=600,
        margin=dict(l=80, r=80, t=20, b=60),
        
        # Remove legend completely
        showlegend=False,
        
        # X-axis styling (TradingView white style)
        xaxis=dict(
            title=None,
            gridcolor='#e6e6e6',
            gridwidth=1,
            showgrid=False,  # Clean look without grid
            zeroline=False,
            showline=False,
            linecolor='#e6e6e6',
            tickfont=dict(color='#333333', size=12, family='Inter, Arial, sans-serif'),
            rangeslider=dict(visible=False),
            showspikes=False,
            fixedrange=True
        ),
        
        # Y-axis styling (TradingView white style) 
        yaxis=dict(
            title=None,
            gridcolor='#e6e6e6',
            gridwidth=1,
            showgrid=False,  # Clean look without grid
            zeroline=False,
            showline=False,
            linecolor='#e6e6e6',
            tickfont=dict(color='#333333', size=12, family='Inter, Arial, sans-serif'),
            side='right',  # TradingView has price scale on right
            showspikes=False,
            tickformat='.4f',
            fixedrange=True
        ),
        
        # Font styling
        font=dict(
            family="Inter, Arial, sans-serif",
            size=12,
            color='#333333'
        ),
        
        # Interaction settings
        dragmode='pan',
        hovermode='x unified',
        
        # Disable various UI elements for cleaner look
        selectdirection='any'
    )
    
    # Update hover label styling to match TradingView white theme
    fig.update_traces(
        hoverlabel=dict(
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#cccccc',
            font=dict(color='#333333', size=12)
        )
    )
    
    return fig

# Chart functions disabled - data-only mode

def add_fibonacci_levels(fig, wave, color):
    """Chart functions disabled - data-only mode."""
    pass

def add_future_pattern_prediction(fig, data, waves, timeframe):
    """Chart functions disabled - data-only mode."""
    pass

def analyze_wave_structure(waves, data):
    """Chart functions disabled - data-only mode."""
    return {"pattern": "Data-only mode", "confidence": 0.0, "direction": "neutral", "expected_move": 0}

def get_timeframe_multiplier(timeframe):
    """Chart functions disabled - data-only mode."""
    return 0.20

def calculate_dynamic_target(current_price, waves, data, volatility, recent_trend, timeframe):
    """Chart functions disabled - data-only mode."""
    return None

def get_volatility_factor(timeframe):
    """Get volatility impact factor based on timeframe."""
    factors = {
        '1m': 0.02,   '5m': 0.03,   '15m': 0.04,  '30m': 0.05,
        '1h': 0.06,   '4h': 0.08,   '1d': 0.10,   '1wk': 0.12,  '1mo': 0.15
    }
    return factors.get(timeframe, 0.08)

def get_trend_factor(timeframe):
    """Get trend impact factor based on timeframe."""
    factors = {
        '1m': 0.1,    '5m': 0.15,   '15m': 0.2,   '30m': 0.25,
        '1h': 0.3,    '4h': 0.4,    '1d': 0.5,    '1wk': 0.6,   '1mo': 0.7
    }
    return factors.get(timeframe, 0.3)

def get_max_move_for_timeframe(timeframe):
    """Get maximum reasonable price move for timeframe."""
    max_moves = {
        '1m': 0.02,   '5m': 0.03,   '15m': 0.05,  '30m': 0.08,
        '1h': 0.12,   '4h': 0.20,   '1d': 0.35,   '1wk': 0.50,  '1mo': 0.80
    }
    return max_moves.get(timeframe, 0.25)

def calculate_fibonacci_target(waves, current_price):
    """Calculate Fibonacci-based price target from wave relationships."""
    try:
        if len(waves) < 2:
            return None
        
        # Find impulse waves for Fibonacci relationships
        impulse_waves = [w for w in waves if w.wave_type.value in ['WAVE_1', 'WAVE_3', 'WAVE_5']]
        
        if len(impulse_waves) >= 2:
            # Use Wave 1 and Wave 3 relationship to project Wave 5
            wave_1 = impulse_waves[0]
            wave_3 = impulse_waves[1] if len(impulse_waves) > 1 else impulse_waves[0]
            
            wave_1_size = abs(wave_1.end_point.price - wave_1.start_point.price)
            wave_3_size = abs(wave_3.end_point.price - wave_3.start_point.price)
            
            # Common Fibonacci relationships
            if wave_3_size > wave_1_size * 1.5:  # Strong Wave 3
                # Wave 5 often equals Wave 1 in strong Wave 3 scenarios
                fib_target = current_price + (wave_1_size * 1.0)
            else:
                # Wave 5 often equals 1.618 * Wave 1
                fib_target = current_price + (wave_1_size * 1.618)
            
            return fib_target
        
        return None
        
    except Exception as e:
        logger.error(f"Error calculating Fibonacci target: {e}")
        return None

@app.route('/api/pairs')
def get_trading_pairs():
    """API endpoint to get available trading pairs."""
    return jsonify(TRADING_PAIRS)

@app.route('/api/timeframes')
def get_timeframes():
    """API endpoint to get available timeframes."""
    return jsonify(TIMEFRAMES)

@app.route('/network-test')
def network_test():
    """Network test page to diagnose connectivity issues."""
    return render_template('network_test.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'ml_features': ml_features_available
    })

@app.route('/api/ml/accuracy', methods=['POST'])
def get_wave_accuracy():
    """Get ML-based wave accuracy prediction."""
    if not ml_features_available or not ml_accuracy:
        logger.warning("ML features not available for accuracy prediction")
        return jsonify({'error': 'ML features not available', 'accuracy': 0.5})
    
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'UNKNOWN')
        logger.info(f"Processing ML accuracy request for symbol: {symbol}")
        
        # Get market data
        logger.info(f"Loading market data for {symbol}")
        logger.info(f"DataLoader object type: {type(data_loader)}")
        logger.info(f"DataLoader has get_data: {hasattr(data_loader, 'get_data')}")
        logger.info(f"DataLoader methods: {[m for m in dir(data_loader) if not m.startswith('_')]}")
        
        # Use get_yahoo_data directly instead of get_data for compatibility
        market_data = data_loader.get_yahoo_data(symbol, period='2y')
        
        if market_data is None or market_data.empty:
            logger.warning(f"No market data available for {symbol}")
            return jsonify({'error': 'No market data available', 'accuracy': 0.5})
        
        logger.info(f"Market data loaded: {len(market_data)} rows")
        
        # Get wave accuracy prediction
        logger.info("Calling ML accuracy prediction")
        accuracy_result = ml_accuracy.predict_wave_accuracy(market_data, symbol)
        
        logger.info(f"ML accuracy prediction completed: {accuracy_result}")
        
        return jsonify({
            'symbol': symbol,
            'accuracy_score': accuracy_result.get('accuracy_score', 0.5),
            'confidence_level': accuracy_result.get('confidence_level', 'Low'),
            'pattern_match_score': accuracy_result.get('pattern_match_score', 0.0),
            'features': accuracy_result.get('features', {}),
            'similar_patterns': accuracy_result.get('similar_patterns', [])
        })
        
    except Exception as e:
        logger.error(f"Error in wave accuracy prediction: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e), 'accuracy': 0.5})

@app.route('/api/ml/auto-tune', methods=['POST'])
def auto_tune_parameters():
    """Auto-tune Elliott Wave detection parameters."""
    if not ml_features_available or not auto_tuner:
        return jsonify({'error': 'Auto-tuning not available'})
    
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'BTC-USD')
        timeframe = data.get('timeframe', '1h')
        
        # Get market data
        market_data = data_loader.get_yahoo_data(symbol, period='2y')
        if market_data is None or market_data.empty:
            return jsonify({'error': 'No market data available'})
        
        # Run optimization
        tuning_result = auto_tuner.optimize_parameters(market_data, symbol, timeframe)
        
        # Apply optimized parameters to detector
        auto_tuner.auto_configure_detector(enhanced_detector, symbol, timeframe)
        
        return jsonify({
            'symbol': symbol,
            'timeframe': timeframe,
            'optimal_threshold': tuning_result.optimal_threshold,
            'optimal_min_wave_length': tuning_result.optimal_min_wave_length,
            'optimal_lookback_periods': tuning_result.optimal_lookback_periods,
            'confidence_score': tuning_result.confidence_score,
            'validation_accuracy': tuning_result.validation_accuracy,
            'multi_timeframe_confirmed': tuning_result.multi_timeframe_confirmed,
            'status': 'Parameters optimized and applied'
        })
        
    except Exception as e:
        logger.error(f"Error in auto-tuning: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/ml/backtest', methods=['POST'])
def run_backtest():
    """Run comprehensive backtesting."""
    if not ml_features_available or not backtesting_engine:
        return jsonify({'error': 'Backtesting engine not available'})
    
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'BTC-USD')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        # Get market data
        market_data = data_loader.get_yahoo_data(symbol, period='2y')
        if market_data is None or market_data.empty:
            return jsonify({'error': 'No market data available'})
        
        # Generate Elliott Wave signals
        analysis_result = enhanced_detector.detect_elliott_waves(market_data, symbol)
        waves = analysis_result.get('waves', [])
        
        # Convert waves to trading signals
        signals = []
        for i, wave in enumerate(waves):
            signal = {
                'timestamp': datetime.now() - timedelta(days=len(waves)-i),
                'signal_type': 'buy' if i % 2 == 0 else 'sell',  # Simplified
                'price': wave.get('end_price', 100),
                'confidence': wave.get('confidence', 0.5),
                'wave_position': wave.get('position', 'unknown'),
                'pattern': wave.get('pattern', 'impulse')
            }
            signals.append(signal)
        
        # Run backtest
        metrics = backtesting_engine.run_backtest(
            market_data, signals,
            datetime.fromisoformat(start_date) if start_date else None,
            datetime.fromisoformat(end_date) if end_date else None
        )
        
        # Generate report
        report = backtesting_engine.generate_report(metrics)
        
        return jsonify({
            'symbol': symbol,
            'total_trades': metrics.total_trades,
            'win_rate': metrics.win_rate,
            'total_return': metrics.total_pnl_percent,
            'max_drawdown': metrics.max_drawdown_percent,
            'sharpe_ratio': metrics.sharpe_ratio,
            'profit_factor': metrics.profit_factor,
            'confidence_score': metrics.confidence_score,
            'wave_accuracy': metrics.wave_accuracy,
            'pattern_performance': metrics.pattern_performance,
            'detailed_report': report
        })
        
    except Exception as e:
        logger.error(f"Error in backtesting: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/ml/forward-test', methods=['POST'])
def run_forward_test():
    """Run forward-walking validation."""
    if not ml_features_available or not backtesting_engine:
        return jsonify({'error': 'Forward testing not available'})
    
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'BTC-USD')
        
        # Get market data
        market_data = data_loader.get_yahoo_data(symbol, period='2y')
        if market_data is None or market_data.empty:
            return jsonify({'error': 'No market data available'})
        
        # Run forward-walking validation
        results = backtesting_engine.run_forward_walking_validation(market_data)
        
        return jsonify({
            'symbol': symbol,
            'num_periods': results.get('num_periods', 0),
            'avg_win_rate': results.get('avg_win_rate', 0),
            'avg_profit_factor': results.get('avg_profit_factor', 0),
            'avg_sharpe_ratio': results.get('avg_sharpe_ratio', 0),
            'avg_max_drawdown': results.get('avg_max_drawdown', 0),
            'consistency_score': results.get('consistency_score', 0),
            'avg_confidence': results.get('avg_confidence', 0),
            'status': 'Forward-walking validation complete'
        })
        
    except Exception as e:
        logger.error(f"Error in forward testing: {e}")
        return jsonify({'error': str(e)})

# NEW: Technical Confluence Analysis API Endpoints

@app.route('/api/technical/analyze', methods=['POST'])
def technical_confluence_analysis():
    """Enhanced Elliott Wave + Technical Confluence Analysis API endpoint."""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'BTC/USDT')
        timeframe = data.get('timeframe', '1h')
        exchange = data.get('exchange', 'binance')
        limit = data.get('limit', 500)
        
        logger.info(f"🚀 Technical Confluence Analysis: {symbol} on {exchange} ({timeframe})")
        
        # Fetch market data using enhanced data fetcher
        market_data = enhanced_data_fetcher.fetch_ohlcv_data(symbol, timeframe, exchange, limit)
        
        if market_data.empty:
            return jsonify({
                'success': False,
                'error': f'No data available for {symbol} on {exchange}'
            })
        
        # Elliott Wave Analysis
        elliott_analysis = enhanced_detector.detect_elliott_waves(market_data, symbol)
        
        if not elliott_analysis or elliott_analysis.get('validation_score', 0) < 0.1:
            return jsonify({
                'success': False,
                'error': 'No valid Elliott Wave structures detected',
                'suggestion': 'Try a different timeframe or symbol'
            })
        
        # Technical Confluence Analysis
        target_zones = technical_confluence_analyzer.analyze_target_zones(
            market_data, elliott_analysis, timeframe
        )
        
        # Market Summary
        market_summary = enhanced_data_fetcher.get_market_summary(symbol, exchange)
        
        # Format response
        response = technical_confluence_analyzer.format_analysis_results(target_zones, market_summary or {})
        response.update({
            'elliott_wave_analysis': {
                'wave_structure': elliott_analysis.get('wave_structure', 'unknown'),
                'validation_score': elliott_analysis.get('validation_score', 0),
                'direction': elliott_analysis.get('direction', 'neutral'),
                'waves_detected': len(elliott_analysis.get('waves', [])),
                'rule_compliance': elliott_analysis.get('rule_compliance', {}),
                'recommendations': elliott_analysis.get('recommendations', [])
            },
            'market_data': {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'data_points': len(market_data),
                'current_price': float(market_data['close'].iloc[-1]),
                'last_update': market_data.index[-1].isoformat()
            }
        })
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Technical confluence analysis error: {e}")
        return jsonify({
            'success': False,
            'error': f'Analysis failed: {str(e)}'
        })

@app.route('/api/technical/multi-timeframe', methods=['POST'])
def multi_timeframe_analysis():
    """Multi-timeframe technical confluence analysis."""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'BTC/USDT')
        exchange = data.get('exchange', 'binance')
        timeframes = data.get('timeframes', ['1h', '4h', '1d'])
        
        results = {}
        
        for tf in timeframes:
            try:
                # Fetch data for this timeframe
                market_data = enhanced_data_fetcher.fetch_ohlcv_data(symbol, tf, exchange, 200)
                
                if not market_data.empty:
                    # Elliott Wave Analysis
                    elliott_analysis = enhanced_detector.detect_elliott_waves(market_data, symbol)
                    
                    if elliott_analysis and elliott_analysis.get('validation_score', 0) >= 0.1:
                        # Technical Confluence Analysis
                        target_zones = technical_confluence_analyzer.analyze_target_zones(
                            market_data, elliott_analysis, tf
                        )
                        
                        results[tf] = {
                            'validation_score': elliott_analysis.get('validation_score', 0),
                            'wave_structure': elliott_analysis.get('wave_structure', 'unknown'),
                            'direction': elliott_analysis.get('direction', 'neutral'),
                            'target_zones_count': len(target_zones),
                            'high_confidence_targets': len([tz for tz in target_zones if tz.confidence_level == "HIGH"]),
                            'best_target': {
                                'price': float(target_zones[0].price_level),
                                'wave': target_zones[0].wave_target,
                                'confidence': target_zones[0].confidence_level,
                                'probability': target_zones[0].probability
                            } if target_zones else None
                        }
                    else:
                        results[tf] = {
                            'validation_score': 0,
                            'wave_structure': 'none',
                            'direction': 'neutral',
                            'target_zones_count': 0,
                            'high_confidence_targets': 0,
                            'best_target': None
                        }
            except Exception as tf_error:
                logger.error(f"Error analyzing {tf}: {tf_error}")
                results[tf] = {'error': str(tf_error)}
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'exchange': exchange,
            'timeframes': results,
            'analysis_timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Multi-timeframe analysis error: {e}")
        return jsonify({
            'success': False,
            'error': f'Multi-timeframe analysis failed: {str(e)}'
        })

@app.route('/api/technical/confluence-details', methods=['GET'])
def confluence_details():
    """Get detailed confluence analysis for a specific price level."""
    try:
        symbol = request.args.get('symbol', 'BTC/USDT')
        price_level = float(request.args.get('price', 50000))
        exchange = request.args.get('exchange', 'binance')
        timeframe = request.args.get('timeframe', '1h')
        
        # Fetch market data
        market_data = enhanced_data_fetcher.fetch_ohlcv_data(symbol, timeframe, exchange, 500)
        
        if market_data.empty:
            return jsonify({
                'success': False,
                'error': f'No data available for {symbol}'
            })
        
        # Analyze confluence at specific price level
        confluence_details = technical_confluence_analyzer.analyze_confluence_at_price(
            market_data, price_level
        )
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'price_level': price_level,
            'confluence_analysis': confluence_details,
            'analysis_timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Confluence details error: {e}")
        return jsonify({
            'success': False,
            'error': f'Confluence analysis failed: {str(e)}'
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = True  # Force debug mode for better error logging
    
    print(f"\n🚀 Elliott Wave Bot Web Interface")
    print(f"📊 Enhanced with ML Features: {ml_features_available}")
    print(f"🌐 Running on http://localhost:{port}")
    print(f"📈 Data-Only Mode: Charts disabled, JSON + ASCII output enabled")
    if ml_features_available:
        print(f"🤖 ML Features: Wave Accuracy, Auto-Tuning, Backtesting")
    print(f"⚡ Debug Mode: {debug_mode}")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
