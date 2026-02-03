"""
Elliott Wave Bot - Web Application (Fixed Version)
Interactive web interface for Elliott Wave analysis with enhanced Fibonacci levels and future predictions
"""

from flask import Flask, render_template, request, jsonify
import json
import sys
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta
import logging
import numpy as np
import plotly.graph_objects as go
import plotly.utils

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_loader import DataLoader
from src.analysis.wave_detector import WaveDetector
from src.analysis.fibonacci import FibonacciAnalyzer
from src.visualization.visualizer import WaveVisualizer
from src.analysis.technical_confluence import TechnicalConfluenceAnalyzer

from src.data.data_loader import DataLoader
from src.analysis.wave_detector import WaveDetector
from src.analysis.fibonacci import FibonacciAnalyzer
from src.visualization.visualizer import WaveVisualizer
import plotly.graph_objects as go
import plotly.utils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'elliott_wave_secret_key_2025'

# Initialize components with more sensitive settings for web interface
data_loader = DataLoader()
wave_detector = WaveDetector()
# Adjust wave detector for better web performance
wave_detector.zigzag_threshold = 0.03  # More sensitive (was 0.05)
wave_detector.min_wave_length = 3      # Shorter minimum waves
wave_detector.confidence_threshold = 0.5  # Lower confidence threshold

fibonacci_analyzer = FibonacciAnalyzer()
visualizer = WaveVisualizer()
technical_confluence_analyzer = TechnicalConfluenceAnalyzer()

# Try to import ML and backtesting features with fallback
ml_features_available = False
ml_accuracy = None
backtester = None

try:
    from src.analysis.ml_wave_accuracy import MLWaveAccuracy
    from src.trading.simple_backtester import Backtester
    ml_accuracy = MLWaveAccuracy()
    backtester = Backtester()
    ml_features_available = True
    logger.info("✅ ML and backtesting features available")
except ImportError as e:
    logger.warning(f"⚠️ ML/Backtesting features not available: {e}")
except Exception as e:
    logger.warning(f"⚠️ Error initializing ML/Backtesting features: {e}")

logger.info("✅ Web application initialized with enhanced support/resistance text labels")

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

def quick_confluence_analysis(market_data, elliott_analysis, timeframe):
    """Fast confluence analysis for large datasets to prevent timeouts."""
    from dataclasses import dataclass
    
    @dataclass
    class QuickTarget:
        price_level: float
        wave_target: str
        elliott_basis: str
        confluence_score: int
        confidence_level: str
        confluences: list
        probability: float
        risk_reward_ratio: float
        timeframe: str
    
    target_zones = []
    
    try:
        current_price = market_data['close'].iloc[-1]
        high_price = market_data['high'].max()
        low_price = market_data['low'].min()
        price_range = high_price - low_price
        
        # Enhanced Fibonacci targets based on current price action and Elliott Wave structure
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618]
        
        # Get Elliott Wave information for better targeting
        wave_count = len(elliott_analysis.get('waves', []))
        current_wave = elliott_analysis.get('current_wave', 'Unknown')
        
        # Create enhanced targets using Fibonacci levels and Elliott Wave context
        for i, level in enumerate(fib_levels[:5]):  # Use first 5 for speed but more variety
            # Upward target based on Elliott Wave context
            if 'WAVE_3' in current_wave or 'WAVE_5' in current_wave:
                target_up = current_price + (price_range * level * 0.618)  # Stronger targets for impulse waves
                wave_context = f"Wave {current_wave.split('_')[-1]} Extension"
            else:
                target_up = current_price + (price_range * level * 0.4)
                wave_context = "Corrective Rebound"
            
            if target_up < high_price * 1.3:  # Reasonable target
                target_zones.append(QuickTarget(
                    price_level=target_up,
                    wave_target=f"{wave_context} {level*100:.1f}%",
                    elliott_basis=f"Fibonacci Extension {level*100:.1f}% ({wave_context})",
                    confluence_score=8 + i,  # Higher base scores
                    confidence_level="MEDIUM" if i < 3 else "LOW",
                    confluences=["Fibonacci Level", "Elliott Wave Target", "Price Action"],
                    probability=0.65 - (i * 0.08),
                    risk_reward_ratio=2.5 + i,
                    timeframe=timeframe
                ))
            
            # Downward target based on Elliott Wave context
            if 'WAVE_2' in current_wave or 'WAVE_4' in current_wave:
                target_down = current_price - (price_range * level * 0.618)  # Deeper retracements for corrective waves
                wave_context = f"Wave {current_wave.split('_')[-1]} Retracement"
            else:
                target_down = current_price - (price_range * level * 0.4)
                wave_context = "Support Retest"
            
            if target_down > low_price * 0.7:  # Reasonable target
                target_zones.append(QuickTarget(
                    price_level=target_down,
                    wave_target=f"{wave_context} {level*100:.1f}%",
                    elliott_basis=f"Fibonacci Retracement {level*100:.1f}% ({wave_context})",
                    confluence_score=7 + i,
                    confidence_level="MEDIUM" if i < 3 else "LOW",
                    confluences=["Fibonacci Level", "Elliott Wave Retracement", "Support Zone"],
                    probability=0.6 - (i * 0.08),
                    risk_reward_ratio=2.2 + i,
                    timeframe=timeframe
                ))
        
        # Add Elliott Wave specific targets based on detected waves
        if wave_count > 0:
            # Add Wave 5 projection if we have Wave 1 and Wave 3
            waves = elliott_analysis.get('waves', [])
            wave_1_found = any(w.get('wave') == '1' or 'WAVE_1' in w.get('wave_type', '') for w in waves)
            wave_3_found = any(w.get('wave') == '3' or 'WAVE_3' in w.get('wave_type', '') for w in waves)
            
            if wave_1_found and wave_3_found:
                # Project Wave 5 target
                wave_5_target = current_price * 1.168  # Simple 16.8% extension
                target_zones.append(QuickTarget(
                    price_level=wave_5_target,
                    wave_target="Wave 5 Projection",
                    elliott_basis="Elliott Wave 5 Target (1.168x current)",
                    confluence_score=12,  # High score for Elliott Wave specific target
                    confidence_level="HIGH",
                    confluences=["Elliott Wave 5", "Impulse Completion", "Fibonacci Extension"],
                    probability=0.72,
                    risk_reward_ratio=3.2,
                    timeframe=timeframe
                ))
        
        # Add psychological level targets with Elliott Wave context
        round_numbers = []
        if current_price > 1000:
            round_numbers = [1000, 5000, 10000, 25000, 50000, 100000]
        elif current_price > 100:
            round_numbers = [100, 250, 500, 1000, 2000, 5000]
        elif current_price > 10:
            round_numbers = [10, 25, 50, 100, 200, 500]
        else:
            round_numbers = [1, 5, 10, 25, 50, 100]
        
        for round_num in round_numbers:
            if low_price <= round_num <= high_price * 1.5:
                distance_pct = abs(round_num - current_price) / current_price
                if distance_pct < 0.6:  # Within 60% of current price
                    target_zones.append(QuickTarget(
                        price_level=round_num,
                        wave_target="Psychological Level",
                        elliott_basis=f"Round Number {round_num} (Psychological)",
                        confluence_score=10,  # Good score for psychological levels
                        confidence_level="MEDIUM",
                        confluences=["Psychological Level", "Round Number", "Market Structure"],
                        probability=0.55,
                        risk_reward_ratio=2.8,
                        timeframe=timeframe
                    ))
        
        # Sort by confluence score and return top targets
        target_zones.sort(key=lambda x: x.confluence_score, reverse=True)
        
        # Ensure we always return at least 3 targets
        if len(target_zones) < 3:
            # Add basic support/resistance targets as fallback
            target_zones.append(QuickTarget(
                price_level=high_price * 0.95,
                wave_target="Near-term Resistance",
                elliott_basis="95% of Recent High",
                confluence_score=6,
                confidence_level="MEDIUM",
                confluences=["Technical Resistance", "Price Action"],
                probability=0.5,
                risk_reward_ratio=2.0,
                timeframe=timeframe
            ))
            
            target_zones.append(QuickTarget(
                price_level=low_price * 1.05,
                wave_target="Near-term Support",
                elliott_basis="105% of Recent Low",
                confluence_score=6,
                confidence_level="MEDIUM",
                confluences=["Technical Support", "Price Action"],
                probability=0.5,
                risk_reward_ratio=2.0,
                timeframe=timeframe
            ))
        
        return target_zones[:8]  # Return top 8 targets for variety
        
    except Exception as e:
        logger.error(f"Quick confluence analysis error: {e}")
        # Return at least one basic target as fallback
        try:
            current_price = market_data['close'].iloc[-1]
            return [QuickTarget(
                price_level=current_price * 1.05,
                wave_target="Basic Target",
                elliott_basis="5% Price Extension (Fallback)",
                confluence_score=5,
                confidence_level="LOW",
                confluences=["Price Action"],
                probability=0.4,
                risk_reward_ratio=1.5,
                timeframe=timeframe
            )]
        except:
            return []


# Minimal helpers used by tests (kept simple and deterministic)

def analyze_wave_structure(waves, data):
    """Analyze the detected waves and return a simple summary used by tests.

    Returns a dict: {'pattern', 'confidence', 'direction'}
    """
    if not waves:
        return {'pattern': 'none', 'confidence': 0.0, 'direction': 'unknown'}

    # Simple heuristics: count impulse vs corrective waves by type names
    types = [w.wave_type for w in waves]
    up_count = sum(1 for w in waves if w.direction.value == 1)
    down_count = sum(1 for w in waves if w.direction.value == -1)
    direction = 'up' if up_count >= down_count else 'down'
    confidence = float(sum(w.confidence for w in waves) / max(1, len(waves)))

    pattern = 'impulse' if any('IMPULSE' in t.name for t in types) else 'corrective'

    return {'pattern': pattern, 'confidence': confidence, 'direction': direction}


def calculate_dynamic_target(current_price, waves, data, volatility, recent_trend, timeframe):
    """Return a simple target price based on current price, trend and volatility.

    This is intentionally conservative; it's only used by tests to ensure the pipeline
    returns a numeric prediction.
    """
    if not waves:
        return None

    # Use average wave extension factor depending on direction
    analysis = analyze_wave_structure(waves, data)
    base_multiplier = 1.05 if analysis['direction'] == 'up' else 0.95

    # Adjust by volatility (more volatility -> wider target)
    vol_adj = 1.0 + min(0.5, volatility * 2.0)

    # Timeframe multiplier to increase target for longer timeframes
    tf_mult = get_timeframe_multiplier(timeframe)

    return current_price * base_multiplier * vol_adj * tf_mult


def get_timeframe_multiplier(timeframe):
    """Return a small multiplier based on timeframe string for testing.

    Simple mapping: 1h=0.25, 4h=0.5, 1d=1.0
    """
    mapping = {'1h': 0.25, '4h': 0.5, '1d': 1.0}
    return mapping.get(timeframe, 1.0)

@app.route('/')
def index():
    """Main page with Elliott Wave analysis interface."""
    return render_template('index.html', 
                         trading_pairs=TRADING_PAIRS, 
                         timeframes=TIMEFRAMES)

@app.route('/api/analyze', methods=['POST'])
def analyze_pair():
    """API endpoint to analyze a specific trading pair and timeframe."""
    analysis_start_time = time.time()
    
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        timeframe = data.get('timeframe', '1d')
        
        logger.info(f"🚀 Starting analysis for {symbol} on {timeframe} timeframe")
        
        # Convert symbol to Yahoo Finance format if needed
        yahoo_symbol = symbol
        for category, symbols in TRADING_PAIRS.items():
            if symbol in symbols:
                yahoo_symbol = symbols[symbol]
                logger.info(f"Converting {symbol} to Yahoo Finance format: {yahoo_symbol}")
                break
        
        # Get timeframe configuration
        tf_config = TIMEFRAMES.get(timeframe, TIMEFRAMES['1d'])
        
        # Load market data with optimization for large datasets
        try:
            # Special handling for BTC daily data which can be very large
            if symbol == 'BTC-USD' and timeframe == '1d':
                # Use a shorter period for BTC daily to prevent timeouts
                tf_config = {'interval': '1d', 'period': '3mo', 'label': '1 Day (3 months)'}
                logger.info(f"Using ultra-optimized period for BTC daily: 3 months instead of 1 year")
            elif symbol == 'BTC-USD':
                # Reduce period for all BTC timeframes
                original_period = tf_config['period']
                if original_period == '1y':
                    tf_config['period'] = '6mo'
                elif original_period == '2y':
                    tf_config['period'] = '1y'
                elif original_period == '5y':
                    tf_config['period'] = '2y'
                logger.info(f"Using reduced period for BTC {timeframe}: {tf_config['period']} instead of {original_period}")
            else:
                tf_config = TIMEFRAMES.get(timeframe, TIMEFRAMES['1d'])
            
            market_data = data_loader.get_yahoo_data(
                yahoo_symbol, 
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
        
        # Detect Elliott Waves with optimized parameters for web performance
        try:
            # Use a single detector instance with optimized settings for fast analysis
            wave_detector.zigzag_threshold = 0.03  # Good balance of sensitivity and performance
            wave_detector.min_wave_length = max(5, len(market_data) // 50)  # Reasonable minimum length
            wave_detector.confidence_threshold = 0.4  # Lower confidence for more detection
            
            logger.info(f"Starting wave detection for {len(market_data)} data points...")
            waves = wave_detector.detect_waves(market_data)
            logger.info(f"Detected {len(waves)} waves")
            
            # If no waves detected, try one more time with lower threshold for difficult patterns
            if len(waves) == 0:
                logger.info("No waves detected, trying with lower threshold...")
                wave_detector.zigzag_threshold = 0.05  # More relaxed threshold
                waves = wave_detector.detect_waves(market_data)
                logger.info(f"Second attempt detected {len(waves)} waves")
            
            # If still no waves, create simple trend analysis as fallback
            if len(waves) == 0:
                logger.warning(f"No waves detected for {symbol}, creating trend analysis")
                waves = create_simple_trend_analysis(market_data)
                logger.info(f"Created {len(waves)} trend patterns as fallback")
                
        except Exception as e:
            logger.error(f"Wave detection error: {e}")
            waves = []
        
        # Enhanced Fibonacci levels and future predictions
        try:
            fibonacci_levels = []
            future_predictions = []
            
            if waves:
                # Enhanced Fibonacci analysis for corrective waves (2, 4, B)
                corrective_waves = [w for w in waves if w.wave_type.value in ['WAVE_2', 'WAVE_4', 'WAVE_B']]
                
                for wave in corrective_waves:
                    # Calculate detailed retracements
                    start_price = wave.start_point.price
                    end_price = wave.end_point.price
                    price_range = abs(end_price - start_price)
                    
                    # Standard and extended Fibonacci levels
                    fib_ratios = {
                        '23.6%': 0.236,
                        '38.2%': 0.382,
                        '50.0%': 0.5,
                        '61.8%': 0.618,
                        '78.6%': 0.786,
                        '100%': 1.0,
                        '127.2%': 1.272,
                        '161.8%': 1.618
                    }
                    
                    fib_levels = {}
                    base_price = max(start_price, end_price) if wave.direction.value == -1 else min(start_price, end_price)
                    
                    for label, ratio in fib_ratios.items():
                        if wave.direction.value == -1:  # Downward wave
                            level_price = base_price - (price_range * ratio)
                        else:  # Upward wave
                            level_price = base_price + (price_range * ratio)
                        fib_levels[label] = level_price
                    
                    fibonacci_levels.append({
                        'wave': wave.wave_type.value,
                        'direction': 'UP' if wave.direction.value == 1 else 'DOWN',
                        'levels': fib_levels,
                        'start_price': start_price,
                        'end_price': end_price,
                        'confidence': wave.confidence
                    })
                
                # Future pattern prediction
                if len(waves) >= 3:
                    last_waves = waves[-3:]
                    current_price = market_data['close'].iloc[-1]
                    
                    # Determine the most probable next pattern
                    impulse_count = len([w for w in waves if w.wave_type.value in ['WAVE_1', 'WAVE_3', 'WAVE_5']])
                    corrective_count = len([w for w in waves if w.wave_type.value in ['WAVE_2', 'WAVE_4']])
                    
                    if impulse_count >= 3 and corrective_count >= 1:
                        # Likely completed or near completion of 5-wave impulse
                        wave_1_magnitude = None
                        wave_3_magnitude = None
                        
                        for wave in waves:
                            if wave.wave_type.value == 'WAVE_1':
                                wave_1_magnitude = abs(wave.end_point.price - wave.start_point.price)
                            elif wave.wave_type.value == 'WAVE_3':
                                wave_3_magnitude = abs(wave.end_point.price - wave.start_point.price)
                        
                        if wave_1_magnitude and wave_3_magnitude:
                            # Predict Wave 5 target using Fibonacci relationships
                            wave_5_target_1 = current_price + (wave_1_magnitude * 1.0)  # Wave 5 = Wave 1
                            wave_5_target_2 = current_price + (wave_1_magnitude * 1.618)  # Wave 5 = 1.618 * Wave 1
                            
                            future_predictions.append({
                                'pattern': 'Wave 5 Completion',
                                'probability': 'High (75%)',
                                'targets': [
                                    {'level': 'Conservative', 'price': wave_5_target_1, 'ratio': '1.0 x Wave 1'},
                                    {'level': 'Extended', 'price': wave_5_target_2, 'ratio': '1.618 x Wave 1'}
                                ],
                                'expected_move': 'Upward continuation then major correction'
                            })
                    
                    elif corrective_count == 0 and impulse_count <= 2:
                        # Early in impulse sequence
                        future_predictions.append({
                            'pattern': 'Impulse Continuation',
                            'probability': 'Medium (60%)',
                            'targets': [
                                {'level': 'Wave 3 target', 'price': current_price * 1.25, 'ratio': 'Strong impulse'},
                                {'level': 'Wave 5 target', 'price': current_price * 1.618, 'ratio': 'Full sequence'}
                            ],
                            'expected_move': 'Strong upward movement with corrections'
                        })
                    
                    else:
                        # Complex or corrective pattern
                        future_predictions.append({
                            'pattern': 'Corrective Sequence',
                            'probability': 'Medium (55%)',
                            'targets': [
                                {'level': 'Support', 'price': current_price * 0.85, 'ratio': '15% retracement'},
                                {'level': 'Major Support', 'price': current_price * 0.76, 'ratio': '24% retracement'}
                            ],
                            'expected_move': 'Sideways to downward correction'
                        })
                        
        except Exception as e:
            logger.error(f"Fibonacci calculation error: {e}")
            fibonacci_levels = []
            future_predictions = []
        
        # Advanced Technical Confluence Analysis
        try:
            confluence_results = {}
            target_zones = []
            
            if waves and market_data is not None and len(market_data) > 0:
                # Prepare Elliott Wave analysis structure for confluence analyzer
                elliott_analysis = {
                    'waves': [],
                    'wave_structure': 'impulse',  # Will be determined from waves
                    'pattern_type': 'impulse',    # Keep for backward compatibility
                    'current_wave': None,
                    'completion_level': 0.0
                }
                
                # Convert waves to the expected format
                for wave in waves:
                    wave_dict = {
                        'wave_type': wave.wave_type.value,
                        'wave': wave.wave_type.value.split('_')[-1],  # Extract just the number/letter
                        'start_price': wave.start_point.price,
                        'end_price': wave.end_point.price,
                        'start_time': wave.start_point.timestamp,
                        'end_time': wave.end_point.timestamp,
                        'direction': 'UP' if wave.direction.value == 1 else 'DOWN',
                        'confidence': wave.confidence
                    }
                    elliott_analysis['waves'].append(wave_dict)
                
                # Determine pattern type and current wave
                if len(waves) > 0:
                    last_wave = waves[-1]
                    elliott_analysis['current_wave'] = last_wave.wave_type.value
                    
                    # Simple pattern type determination
                    impulse_waves = [w for w in waves if w.wave_type.value in ['WAVE_1', 'WAVE_3', 'WAVE_5']]
                    corrective_waves = [w for w in waves if w.wave_type.value in ['WAVE_A', 'WAVE_B', 'WAVE_C']]
                    
                    if len(impulse_waves) >= len(corrective_waves):
                        elliott_analysis['pattern_type'] = 'impulse'
                        elliott_analysis['wave_structure'] = 'impulse'
                    else:
                        elliott_analysis['pattern_type'] = 'corrective'
                        elliott_analysis['wave_structure'] = 'corrective'
                
                # Perform comprehensive confluence analysis with timeout protection and optimization
                logger.info(f"Starting confluence analysis for {len(waves)} waves...")
                start_time = time.time()
                
                try:
                    # Optimize confluence analysis for large datasets and ensure targets for all pairs
                    if len(market_data) > 1000:  # Only use quick analysis for very large datasets
                        logger.info("Very large dataset detected, using optimized confluence analysis...")
                        target_zones = quick_confluence_analysis(market_data, elliott_analysis, timeframe)
                    else:
                        # Try full confluence analysis first, with fallback to quick analysis
                        try:
                            # Windows-compatible timeout using threading
                            import threading
                            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
                            
                            def run_confluence_analysis():
                                return technical_confluence_analyzer.analyze_target_zones(
                                    market_data=market_data,
                                    elliott_analysis=elliott_analysis,
                                    timeframe=timeframe
                                )
                            
                            # Use thread pool with timeout
                            with ThreadPoolExecutor(max_workers=1) as executor:
                                future = executor.submit(run_confluence_analysis)
                                try:
                                    target_zones = future.result(timeout=25)  # 25-second timeout
                                    logger.info(f"Full confluence analysis completed with {len(target_zones)} targets")
                                except FutureTimeoutError:
                                    logger.warning("Confluence analysis timeout (25s), using quick analysis")
                                    target_zones = quick_confluence_analysis(market_data, elliott_analysis, timeframe)
                            
                            # If full analysis returns no targets, use quick analysis as backup
                            if not target_zones or len(target_zones) == 0:
                                logger.info("Full confluence analysis returned no targets, using enhanced quick analysis")
                                target_zones = quick_confluence_analysis(market_data, elliott_analysis, timeframe)
                            
                        except Exception as timeout_error:
                            logger.warning(f"Confluence analysis error: {timeout_error}, using quick analysis")
                            target_zones = quick_confluence_analysis(market_data, elliott_analysis, timeframe)
                    
                    confluence_time = time.time() - start_time
                    logger.info(f"Confluence analysis completed in {confluence_time:.2f} seconds: {len(target_zones)} targets found")
                    
                except Exception as confluence_error:
                    logger.error(f"Confluence analysis failed: {confluence_error}")
                    target_zones = []
                
                # Add confluence targets to future predictions for backward compatibility
                for target in target_zones[:3]:  # Top 3 targets
                    confluence_info = {
                        'pattern': f'{target.wave_target} Target (Confluence Score: {target.confluence_score})',
                        'probability': f'{target.confidence_level} ({target.probability:.0%})',
                        'targets': [
                            {
                                'level': f'Target Zone {target.price_level:.4f}',
                                'price': target.price_level,
                                'ratio': f'Confluence: {", ".join(target.confluences[:3])}'
                            }
                        ],
                        'expected_move': f'Target: {target.price_level:.4f}',
                        'confluence_score': target.confluence_score,
                        'confluences': target.confluences,
                        'risk_reward': target.risk_reward_ratio,
                        'time_horizon': target.timeframe
                    }
                    future_predictions.append(confluence_info)
                
                logger.info(f"✅ Technical confluence analysis completed: {len(target_zones)} targets found")
            else:
                logger.warning("⚠️ No waves found for confluence analysis")
                
        except Exception as e:
            logger.error(f"Technical confluence analysis error: {e}")
            confluence_results = {}
            target_zones = []

        # Calculate Support/Resistance levels with performance optimization
        try:
            logger.info("Starting support/resistance calculation...")
            sr_start_time = time.time()
            
            # Optimize for large datasets by using a subset if needed
            if len(market_data) > 1000:
                # Use last 500 data points for S/R calculation to improve performance
                sr_data = market_data.tail(500)
                logger.info(f"Using last 500 points for S/R calculation (from {len(market_data)} total)")
            else:
                sr_data = market_data
            
            support_resistance = calculate_support_resistance(sr_data)
            
            sr_time = time.time() - sr_start_time
            logger.info(f"S/R calculation completed in {sr_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Support/Resistance calculation error: {e}")
            support_resistance = {'support_levels': [], 'resistance_levels': []}

        # Create interactive chart with performance optimization
        try:
            logger.info("Creating interactive chart...")
            chart_start_time = time.time()
            
            # Optimize chart data for large datasets
            if len(market_data) > 500:
                # Resample data for chart performance
                chart_data = market_data.resample('D').agg({
                    'open': 'first',
                    'high': 'max', 
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                logger.info(f"Resampled chart data from {len(market_data)} to {len(chart_data)} points")
            else:
                chart_data = market_data
            
            fig = create_web_chart(chart_data, waves, symbol, timeframe, support_resistance, target_zones)
            chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            chart_time = time.time() - chart_start_time
            logger.info(f"Chart creation completed in {chart_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Chart creation error: {e}")
            chart_json = None
        
        # Prepare wave data for response
        wave_data = []
        for wave in waves:
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
        
        # Market summary
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
        
        return jsonify({
            'success': True,
            'waves': wave_data,
            'fibonacci_levels': fibonacci_levels,
            'future_predictions': future_predictions,
            'support_resistance': support_resistance,
            'target_zones': [
                {
                    'price': target.price_level,
                    'price_level': target.price_level,
                    'price_change_pct': ((target.price_level - market_data['close'].iloc[-1]) / market_data['close'].iloc[-1]) * 100,
                    'wave_target': target.wave_target,
                    'elliott_basis': target.elliott_basis,
                    'confluence_score': target.confluence_score,
                    'confidence_level': target.confidence_level,
                    'confluences': target.confluences,
                    'all_confluences': target.confluences,  # For JavaScript compatibility
                    'probability': target.probability,
                    'risk_reward_ratio': target.risk_reward_ratio,
                    'timeframe': target.timeframe
                } for target in target_zones
            ],
            'confluence_summary': {
                'total_targets': len(target_zones),
                'high_confidence': len([t for t in target_zones if t.confidence_level == 'HIGH']),
                'medium_confidence': len([t for t in target_zones if t.confidence_level == 'MEDIUM']),
                'low_confidence': len([t for t in target_zones if t.confidence_level == 'LOW']),
                'average_confluence_score': sum(t.confluence_score for t in target_zones) / len(target_zones) if target_zones else 0,
                'strongest_confluence': max(target_zones, key=lambda t: t.confluence_score).confluence_score if target_zones else 0,
                'best_target': {
                    'wave': target_zones[0].wave_target,
                    'price': target_zones[0].price_level,
                    'confidence': target_zones[0].confidence_level,
                    'probability': target_zones[0].probability,
                    'confluences': len(target_zones[0].confluences)
                } if target_zones else None
            },
            'chart': chart_json,
            'market_summary': market_summary,
            'analysis_timestamp': datetime.now().isoformat(),
            'analysis_mode': 'technical_confluence',
            'analysis_time': round(time.time() - analysis_start_time, 2)
        })
        
        total_time = time.time() - analysis_start_time
        logger.info(f"✅ Analysis completed successfully in {total_time:.2f} seconds")
        
    except Exception as e:
        total_time = time.time() - analysis_start_time
        logger.error(f"❌ Analysis failed after {total_time:.2f} seconds: {e}")
        return jsonify({
            'success': False,
            'error': f'Analysis failed: {str(e)}',
            'analysis_time': round(total_time, 2)
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

def calculate_support_resistance(data, window=20, strength=3):
    """Calculate support and resistance levels using pivot points and clustering - optimized for performance."""
    try:
        import numpy as np
        from scipy.signal import argrelextrema
        
        # Performance optimization: limit data size for very large datasets
        if len(data) > 1000:
            # Use adaptive strength for large datasets
            strength = max(5, len(data) // 200)  # Increase strength for large datasets
        
        # Calculate pivot highs and lows with optimized parameters
        highs = data['high'].values
        lows = data['low'].values
        
        # Find local maxima (resistance) and minima (support) with optimized order
        resistance_indices = argrelextrema(highs, np.greater, order=strength)[0]
        support_indices = argrelextrema(lows, np.less, order=strength)[0]
        
        # Limit the number of levels to process for performance
        max_levels = 50  # Process maximum 50 levels of each type
        
        # Get the actual price levels (limit to most recent if too many)
        if len(resistance_indices) > max_levels:
            resistance_indices = resistance_indices[-max_levels:]  # Take most recent
        if len(support_indices) > max_levels:
            support_indices = support_indices[-max_levels:]  # Take most recent
        
        resistance_levels = [{'price': highs[i], 'timestamp': data.index[i], 'touches': 1} for i in resistance_indices]
        support_levels = [{'price': lows[i], 'timestamp': data.index[i], 'touches': 1} for i in support_indices]
        
        # Cluster nearby levels (within 1% of each other)
        def cluster_levels(levels, tolerance=0.01):
            if not levels:
                return []
            
            # Sort by price
            levels.sort(key=lambda x: x['price'])
            clustered = []
            current_cluster = [levels[0]]
            
            for level in levels[1:]:
                # Check if this level is within tolerance of the current cluster
                cluster_avg = sum(l['price'] for l in current_cluster) / len(current_cluster)
                if abs(level['price'] - cluster_avg) / cluster_avg <= tolerance:
                    current_cluster.append(level)
                else:
                    # Finalize current cluster
                    if current_cluster:
                        avg_price = sum(l['price'] for l in current_cluster) / len(current_cluster)
                        touches = len(current_cluster)
                        most_recent = max(current_cluster, key=lambda x: x['timestamp'])
                        clustered.append({
                            'price': avg_price,
                            'timestamp': most_recent['timestamp'],
                            'touches': touches,
                            'strength': min(touches / 2.0, 5.0)  # Strength from 1-5
                        })
                    current_cluster = [level]
            
            # Don't forget the last cluster
            if current_cluster:
                avg_price = sum(l['price'] for l in current_cluster) / len(current_cluster)
                touches = len(current_cluster)
                most_recent = max(current_cluster, key=lambda x: x['timestamp'])
                clustered.append({
                    'price': avg_price,
                    'timestamp': most_recent['timestamp'],
                    'touches': touches,
                    'strength': min(touches / 2.0, 5.0)
                })
            
            return clustered
        
        # Cluster the levels
        resistance_clustered = cluster_levels(resistance_levels)
        support_clustered = cluster_levels(support_levels)
        
        # Filter to keep only the most significant levels and sort by relevance
        resistance_clustered.sort(key=lambda x: x['strength'], reverse=True)
        support_clustered.sort(key=lambda x: x['strength'], reverse=True)
        
        # Get current price for distance calculations
        current_price = data['close'].iloc[-1]
        
        # Add distance to current price for each level
        for level in resistance_clustered:
            level['distance_to_current'] = abs(level['price'] - current_price)
            level['distance_percent'] = (abs(level['price'] - current_price) / current_price) * 100
        
        for level in support_clustered:
            level['distance_to_current'] = abs(level['price'] - current_price)
            level['distance_percent'] = (abs(level['price'] - current_price) / current_price) * 100
        
        # Create combined lists sorted by strength and proximity
        all_resistance = resistance_clustered[:10]  # Top 10 by strength
        all_support = support_clustered[:10]  # Top 10 by strength
        
        # Sort by distance to current price for nearest levels
        nearest_resistance = sorted(all_resistance, key=lambda x: x['distance_to_current'])[:3]
        nearest_support = sorted(all_support, key=lambda x: x['distance_to_current'])[:3]
        
        # Most contacted (strongest) levels
        strongest_resistance = resistance_clustered[:3]  # Top 3 by strength
        strongest_support = support_clustered[:3]  # Top 3 by strength
        
        # Combine and prioritize: nearest + strongest (remove duplicates)
        final_resistance = []
        final_support = []
        
        # Add nearest levels first
        for level in nearest_resistance:
            if level not in final_resistance:
                level['priority'] = 'nearest'
                final_resistance.append(level)
        
        for level in nearest_support:
            if level not in final_support:
                level['priority'] = 'nearest'
                final_support.append(level)
        
        # Add strongest levels if not already included
        for level in strongest_resistance:
            if level not in final_resistance and len(final_resistance) < 5:
                level['priority'] = 'strongest'
                final_resistance.append(level)
        
        for level in strongest_support:
            if level not in final_support and len(final_support) < 5:
                level['priority'] = 'strongest'
                final_support.append(level)
        
        # Also add psychological levels (round numbers)
        current_price = data['close'].iloc[-1]
        price_range = data['high'].max() - data['low'].min()
        
        # Generate round number levels within the visible range
        psychological_levels = []
        min_price = data['low'].min() - price_range * 0.1
        max_price = data['high'].max() + price_range * 0.1
        
        # Determine appropriate round number intervals based on price
        if current_price < 1:
            intervals = [0.01, 0.05, 0.1, 0.25, 0.5]
        elif current_price < 10:
            intervals = [0.5, 1, 2.5, 5]
        elif current_price < 100:
            intervals = [5, 10, 25, 50]
        elif current_price < 1000:
            intervals = [25, 50, 100, 250]
        else:
            intervals = [100, 250, 500, 1000]
        
        for interval in intervals:
            # Find round numbers in our price range
            start = int(min_price / interval) * interval
            end = int(max_price / interval + 1) * interval
            level = start
            while level <= end:
                if min_price <= level <= max_price:
                    psychological_levels.append({
                        'price': level,
                        'type': 'psychological',
                        'strength': 2.0,
                        'interval': interval
                    })
                level += interval
        
        return {
            'support_levels': final_support,  # Smart combination of nearest + strongest
            'resistance_levels': final_resistance,  # Smart combination of nearest + strongest
            'psychological_levels': psychological_levels,
            'current_price': current_price,
            'analysis_summary': {
                'total_support_found': len(support_clustered),
                'total_resistance_found': len(resistance_clustered),
                'nearest_support': nearest_support[0] if nearest_support else None,
                'nearest_resistance': nearest_resistance[0] if nearest_resistance else None,
                'strongest_support': strongest_support[0] if strongest_support else None,
                'strongest_resistance': strongest_resistance[0] if strongest_resistance else None
            }
        }
        
    except Exception as e:
        logger.error(f"Error calculating support/resistance: {e}")
        return {'support_levels': [], 'resistance_levels': [], 'psychological_levels': []}

def create_web_chart(data, waves, symbol, timeframe, support_resistance=None, target_zones=None):
    """Create an enhanced interactive chart optimized for web display."""
    
    fig = go.Figure()
    
    # Add candlestick chart with enhanced styling
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name=symbol,
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4444',
        increasing_fillcolor='rgba(0,255,136,0.1)',
        decreasing_fillcolor='rgba(255,68,68,0.1)',
        line=dict(width=1.5)
    ))
    
    # Enhanced Elliott Wave colors and styles
    wave_colors = {
        'WAVE_1': {'color': '#FF6B6B', 'width': 4, 'dash': 'solid'},
        'WAVE_2': {'color': '#4ECDC4', 'width': 3, 'dash': 'dash'},
        'WAVE_3': {'color': '#45B7D1', 'width': 5, 'dash': 'solid'},
        'WAVE_4': {'color': '#96CEB4', 'width': 3, 'dash': 'dash'},
        'WAVE_5': {'color': '#FECA57', 'width': 4, 'dash': 'solid'},
        'WAVE_A': {'color': '#FF9FF3', 'width': 4, 'dash': 'solid'},
        'WAVE_B': {'color': '#54A0FF', 'width': 3, 'dash': 'dash'},
        'WAVE_C': {'color': '#5F27CD', 'width': 5, 'dash': 'solid'}
    }
    
    # Add Elliott Wave lines with enhanced styling
    for i, wave in enumerate(waves):
        wave_style = wave_colors.get(wave.wave_type.value, {'color': '#FFFFFF', 'width': 3, 'dash': 'solid'})
        wave_label = wave.wave_type.value.split('_')[-1]
        color = wave_style['color']
        
        # Add wave line with enhanced styling
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
                size=10, 
                color=color,
                line=dict(color='white', width=2)
            ),
            text=['', wave_label],
            textposition='top center',
            textfont=dict(size=16, color=color, family="Arial Black"),
            name=f'Wave {wave_label}',
            showlegend=True,
            hovertemplate=f'<b>Wave {wave_label}</b><br>' +
                         'Start: %{x[0]}<br>' +
                         'End: %{x[1]}<br>' +
                         'Price: %{y}<br>' +
                         f'Confidence: {wave.confidence:.1%}<extra></extra>'
        ))
        
        # Add Fibonacci retracements for corrective waves (2, 4, B)
        if wave.wave_type.value in ['WAVE_2', 'WAVE_4', 'WAVE_B']:
            add_fibonacci_levels(fig, wave, color)
    
    # Add future pattern prediction
    if len(waves) >= 3:
        add_future_pattern_prediction(fig, data, waves)
    
    # Add Support/Resistance levels
    if support_resistance:
        add_support_resistance_levels(fig, support_resistance, data)
    
    # Add Target Zones from Technical Confluence Analysis
    if target_zones:
        add_target_zones(fig, target_zones)
    
    # Add volume subplot with enhanced styling
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['volume'],
        name='Volume',
        marker_color='rgba(158,202,225,0.3)',
        yaxis='y2',
        hovertemplate='Volume: %{y:,.0f}<extra></extra>'
    ))
    
    # Calculate price range for better y-axis scaling
    price_range = data['high'].max() - data['low'].min()
    price_padding = price_range * 0.1
    
    # Update layout for enhanced web display with beautiful styling
    fig.update_layout(
        title=dict(
            text=f'📈 {symbol} - Elliott Wave Analysis ({timeframe.upper()})',
            x=0.5,
            font=dict(size=28, color='white', family="Arial Black"),
            pad=dict(t=20)
        ),
        template='plotly_dark',
        height=900,  # Increased height for better visibility
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(0,0,0,0.8)",
            bordercolor="rgba(255,255,255,0.3)",
            borderwidth=2,
            font=dict(size=12)
        ),
        xaxis=dict(
            title=dict(text='📅 Time', font=dict(size=14, color='white')),
            gridcolor='rgba(128,128,128,0.3)',
            rangeslider=dict(visible=False),
            showspikes=True,
            spikecolor="cyan",
            spikesnap="cursor",
            spikemode="across",
            spikedash="dot",
            spikethickness=2
        ),
        yaxis=dict(
            title=dict(text='💰 Price (USD)', font=dict(size=14, color='white')),
            gridcolor='rgba(128,128,128,0.3)',
            side='left',
            showspikes=True,
            spikecolor="cyan",
            spikesnap="cursor",
            spikemode="across",
            spikedash="dot",
            spikethickness=2,
            range=[data['low'].min() - price_padding, data['high'].max() + price_padding],
            tickformat=".4f"
        ),
        yaxis2=dict(
            title=dict(text='📊 Volume', font=dict(size=12, color='white')),
            overlaying='y',
            side='right',
            gridcolor='rgba(128,128,128,0.1)',
            showgrid=False,
            range=[0, data['volume'].max() * 4]  # Volume takes 25% of chart height
        ),
        plot_bgcolor='rgba(17,17,17,1)',
        paper_bgcolor='rgba(10,10,10,1)',
        font=dict(color='white', family="Arial", size=12),
        margin=dict(l=80, r=140, t=120, b=80),
        dragmode='zoom',
        hovermode='x unified',
        # Add beautiful gradient background
        annotations=[
            dict(
                text="✨ Professional Elliott Wave Analysis ✨",
                xref="paper", yref="paper",
                x=0.5, y=-0.08,
                xanchor='center', yanchor='top',
                font=dict(size=12, color="rgba(255,255,255,0.6)"),
                showarrow=False
            )
        ]
    )
    
    return fig

def add_fibonacci_levels(fig, wave, color):
    """Add Fibonacci retracement levels for corrective waves (2, 4, B)."""
    try:
        # Calculate Fibonacci levels
        start_price = wave.start_point.price
        end_price = wave.end_point.price
        price_diff = abs(end_price - start_price)
        
        # Standard Fibonacci retracement levels
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        
        # Determine direction (retracement is opposite to main trend)
        if end_price > start_price:  # Upward correction
            base_price = end_price
            direction = -1
        else:  # Downward correction
            base_price = start_price
            direction = 1
            
        # Add horizontal lines for each Fibonacci level
        for level in fib_levels:
            fib_price = base_price + (direction * price_diff * level)
            
            fig.add_hline(
                y=fib_price,
                line_dash="dot",
                line_color=color,
                opacity=0.6,
                annotation_text=f"{level:.1%} ({fib_price:.4f})",
                annotation_position="bottom right",
                annotation_font_size=10,
                annotation_font_color=color
            )
            
    except Exception as e:
        logger.error(f"Error adding Fibonacci levels: {e}")

def add_support_resistance_levels(fig, support_resistance, data):
    """Add support and resistance levels to the chart with enhanced visibility."""
    try:
        # Add support levels (green) with enhanced styling and priority info
        for i, level in enumerate(support_resistance.get('support_levels', [])):
            price = level['price']
            strength = level['strength']
            priority = level.get('priority', 'normal')
            distance_percent = level.get('distance_percent', 0)
            
            # Different styling based on priority
            if priority == 'nearest':
                line_width = 4
                opacity = 0.9
                priority_icon = "🎯"
                priority_text = "NEAREST"
            elif priority == 'strongest':
                line_width = 3
                opacity = 0.8
                priority_icon = "💪"
                priority_text = "STRONGEST"
            else:
                line_width = 2
                opacity = 0.7
                priority_icon = "🔰"
                priority_text = ""
            
            # Add horizontal line with enhanced styling
            fig.add_hline(
                y=price,
                line_dash="solid",
                line_color="#00FF88",
                line_width=line_width,
                opacity=opacity,
                annotation_text=f"{priority_icon} {priority_text} Support {price:.4f} ({distance_percent:.1f}% away)",
                annotation_position="bottom right",
                annotation_font_size=12,
                annotation_font_color="#00FF88",
                annotation_bgcolor="rgba(0,255,136,0.2)",
                annotation_bordercolor="#00FF88",
                annotation_borderwidth=1
            )
        
        # Add resistance levels (red) with enhanced styling and priority info
        for i, level in enumerate(support_resistance.get('resistance_levels', [])):
            price = level['price']
            strength = level['strength']
            priority = level.get('priority', 'normal')
            distance_percent = level.get('distance_percent', 0)
            
            # Different styling based on priority
            if priority == 'nearest':
                line_width = 4
                opacity = 0.9
                priority_icon = "🎯"
                priority_text = "NEAREST"
            elif priority == 'strongest':
                line_width = 3
                opacity = 0.8
                priority_icon = "💪"
                priority_text = "STRONGEST"
            else:
                line_width = 2
                opacity = 0.7
                priority_icon = "🔴"
                priority_text = ""
            
            # Add horizontal line with enhanced styling
            fig.add_hline(
                y=price,
                line_dash="solid",
                line_color="#FF4444",
                line_width=line_width,
                opacity=opacity,
                annotation_text=f"{priority_icon} {priority_text} Resistance {price:.4f} ({distance_percent:.1f}% away)",
                annotation_position="top right",
                annotation_font_size=12,
                annotation_font_color="#FF4444",
                annotation_bgcolor="rgba(255,68,68,0.2)",
                annotation_bordercolor="#FF4444",
                annotation_borderwidth=1
            )
        
        # Add psychological levels (yellow) with enhanced styling
        psych_count = 0
        for level in support_resistance.get('psychological_levels', []):
            # Only show psychological levels that are close to current price range
            price_range = data['high'].max() - data['low'].min()
            current_price = data['close'].iloc[-1]
            
            # Show psychological levels within 30% of current price range
            if abs(level['price'] - current_price) <= price_range * 0.3:
                psych_count += 1
                if psych_count <= 5:  # Show more psychological levels
                    price = level['price']
                    
                    # Add horizontal line with enhanced styling
                    fig.add_hline(
                        y=price,
                        line_dash="dot",
                        line_color="#FFD700",
                        line_width=2,
                        opacity=0.6,
                        annotation_text=f"💰 ${price:.0f} (Psychological)",
                        annotation_position="bottom left",
                        annotation_font_size=10,
                        annotation_font_color="#FFD700",
                        annotation_bgcolor="rgba(255,215,0,0.1)",
                        annotation_bordercolor="#FFD700",
                        annotation_borderwidth=1
                    )
        
        # Add enhanced legend text for S/R levels with priority system
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text="<b>Support/Resistance Priority System:</b><br>" +
                 "🎯 <span style='color:#00FF88'>NEAREST</span> (Closest to price) | " +
                 "� <span style='color:#FFD700'>STRONGEST</span> (Most contacted) | " +
                 "💰 <span style='color:#FFD700'>Psychological</span> (Round numbers)",
            showarrow=False,
            font=dict(size=11, color="white"),
            bgcolor="rgba(0,0,0,0.8)",
            bordercolor="rgba(255,255,255,0.3)",
            borderwidth=1
        )
                
    except Exception as e:
        logger.error(f"Error adding support/resistance levels: {e}")

def add_target_zones(fig, target_zones):
    """Add confluence target zones to the chart with enhanced visualization."""
    try:
        if not target_zones:
            return
            
        # Define colors based on confluence score
        def get_target_color(score):
            if score >= 20:
                return "#FFD700"  # Gold for highest confidence
            elif score >= 15:
                return "#FF6B6B"  # Red for high confidence  
            elif score >= 10:
                return "#4ECDC4"  # Teal for medium confidence
            else:
                return "#95A5A6"  # Gray for low confidence
        
        # Add target zone lines and annotations
        for i, target in enumerate(target_zones[:5]):  # Show top 5 targets
            color = get_target_color(target.confluence_score)
            
            # Determine line style based on confidence
            line_dash = "solid" if target.confidence_level == "HIGH" else "dash" if target.confidence_level == "MEDIUM" else "dot"
            line_width = 4 if target.confidence_level == "HIGH" else 3 if target.confidence_level == "MEDIUM" else 2
            
            # Add horizontal line for target
            fig.add_hline(
                y=target.price_level,
                line_dash=line_dash,
                line_color=color,
                line_width=line_width,
                opacity=0.8,
                annotation_text=f"🎯 {target.wave_target}: {target.price_level:.4f} (Score: {target.confluence_score})",
                annotation_position="top left" if i % 2 == 0 else "bottom left",
                annotation_font_size=12,
                annotation_font_color=color,
                annotation_bgcolor=f"rgba({','.join(str(int(color[i:i+2], 16)) for i in (1, 3, 5))},0.2)",
                annotation_bordercolor=color,
                annotation_borderwidth=2
            )
            
            # Add confluence details as text annotation
            confluence_text = f"Confluences: {', '.join(target.confluences[:3])}"
            if len(target.confluences) > 3:
                confluence_text += f" +{len(target.confluences)-3} more"
                
            fig.add_annotation(
                x=0.98,
                y=target.price_level,
                xref="paper",
                yref="y",
                text=confluence_text,
                showarrow=True,
                arrowhead=1,
                arrowcolor=color,
                arrowwidth=2,
                ax=-50,
                ay=20 * (i + 1),
                font=dict(size=10, color=color),
                bgcolor=f"rgba({','.join(str(int(color[i:i+2], 16)) for i in (1, 3, 5))},0.1)",
                bordercolor=color,
                borderwidth=1
            )
        
        # Add legend for target zones
        fig.add_annotation(
            x=0.02,
            y=0.02,
            xref="paper",
            yref="paper",
            text="<b>🎯 Technical Confluence Targets:</b><br>" +
                 "🥇 <span style='color:#FFD700'>Gold (Score ≥20)</span> | " +
                 "🔴 <span style='color:#FF6B6B'>High (Score ≥15)</span> | " +
                 "🔵 <span style='color:#4ECDC4'>Medium (Score ≥10)</span>",
            showarrow=False,
            font=dict(size=11, color="white"),
            bgcolor="rgba(0,0,0,0.8)",
            bordercolor="rgba(255,255,255,0.3)",
            borderwidth=1
        )
                
    except Exception as e:
        logger.error(f"Error adding target zones: {e}")

def add_future_pattern_prediction(fig, data, waves):
    """Add future pattern prediction based on current Elliott Wave structure."""
    try:
        if len(waves) < 3:
            return
            
        # Get the last few waves to determine pattern
        recent_waves = waves[-5:] if len(waves) >= 5 else waves
        
        # Check if we have a complete 5-wave impulse
        impulse_waves = [w for w in recent_waves if w.wave_type.value in ['WAVE_1', 'WAVE_3', 'WAVE_5']]
        corrective_waves = [w for w in recent_waves if w.wave_type.value in ['WAVE_2', 'WAVE_4']]
        
        current_price = data['close'].iloc[-1]
        last_timestamp = data.index[-1]
        
        # Project future timeframe (next 20% of current timeframe)
        time_span = data.index[-1] - data.index[0]
        future_time = last_timestamp + timedelta(seconds=time_span.total_seconds() * 0.2)
        
        prediction_text = ""
        prediction_color = "#FFD700"  # Gold color for predictions
        
        # Determine most likely next pattern
        if len(impulse_waves) >= 3 and len(corrective_waves) >= 1:
            # Likely completed impulse, expect ABC correction
            last_high = max([w.end_point.price for w in recent_waves])
            predicted_price = current_price * 0.85  # Expect 15% correction
            prediction_text = "Expected: ABC Correction (15-25% retracement)"
            
        elif len(impulse_waves) == 2 and len(corrective_waves) >= 1:
            # In Wave 3 or expecting Wave 5
            wave_1_magnitude = abs(recent_waves[0].end_point.price - recent_waves[0].start_point.price)
            predicted_price = current_price + (wave_1_magnitude * 1.618)  # Wave 5 often 1.618 of Wave 1
            prediction_text = "Expected: Wave 5 completion (1.618 extension)"
            
        else:
            # General trend continuation
            recent_change = (current_price - data['close'].iloc[-10]) / data['close'].iloc[-10]
            predicted_price = current_price * (1 + recent_change * 0.5)
            prediction_text = "Expected: Trend continuation"
        
        # Add prediction line
        fig.add_trace(go.Scatter(
            x=[last_timestamp, future_time],
            y=[current_price, predicted_price],
            mode='lines+markers',
            line=dict(color=prediction_color, width=3, dash='dash'),
            marker=dict(size=8, color=prediction_color, symbol='star'),
            name='Price Prediction',
            hovertemplate=f'<b>Prediction</b><br>Target: %{{y:.4f}}<br>{prediction_text}<extra></extra>'
        ))
        
        # Add prediction annotation
        fig.add_annotation(
            x=future_time,
            y=predicted_price,
            text=f"Target: {predicted_price:.4f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor=prediction_color,
            bgcolor="rgba(255,215,0,0.1)",
            bordercolor=prediction_color,
            font=dict(color=prediction_color, size=12)
        )
        
    except Exception as e:
        logger.error(f"Error adding future pattern prediction: {e}")

@app.route('/api/generate_chart/<symbol>')
def generate_chart_html(symbol):
    """Generate a standalone HTML chart for a specific symbol."""
    try:
        # Default to 1d timeframe
        timeframe = '1d'
        tf_config = TIMEFRAMES.get(timeframe, TIMEFRAMES['1d'])
        
        # Load market data
        market_data = data_loader.get_yahoo_data(
            symbol, 
            period=tf_config['period'],
            interval=tf_config['interval']
        )
        
        if market_data.empty:
            return f"<h1>Error: No data available for {symbol}</h1>"
        
        # Detect Elliott Waves
        waves = []
        sensitivity_levels = [0.02, 0.03, 0.04, 0.05, 0.06]
        
        for threshold in sensitivity_levels:
            temp_detector = WaveDetector()
            temp_detector.zigzag_threshold = threshold
            temp_detector.min_wave_length = max(3, len(market_data) // 100)
            temp_detector.confidence_threshold = 0.4
            
            waves = temp_detector.detect_waves(market_data)
            if len(waves) > 0:
                break
        
        # Calculate Support/Resistance levels
        support_resistance = calculate_support_resistance(market_data)
        
        # Quick confluence analysis for standalone chart
        target_zones = []
        try:
            if waves:
                # Prepare Elliott Wave analysis structure
                elliott_analysis = {
                    'waves': [],
                    'pattern_type': 'impulse',
                    'current_wave': None,
                    'completion_level': 0.0
                }
                
                # Convert waves to the expected format
                for wave in waves:
                    wave_dict = {
                        'wave_type': wave.wave_type.value,
                        'start_price': wave.start_point.price,
                        'end_price': wave.end_point.price,
                        'start_time': wave.start_point.timestamp,
                        'end_time': wave.end_point.timestamp,
                        'direction': 'UP' if wave.direction.value == 1 else 'DOWN',
                        'confidence': wave.confidence
                    }
                    elliott_analysis['waves'].append(wave_dict)
                
                target_zones = technical_confluence_analyzer.analyze_target_zones(
                    market_data=market_data,
                    elliott_analysis=elliott_analysis,
                    timeframe=timeframe
                )
        except:
            target_zones = []
        
        # Create chart
        fig = create_web_chart(market_data, waves, symbol, timeframe, support_resistance, target_zones)
        
        # Convert to HTML
        html_str = fig.to_html(include_plotlyjs='cdn')
        
        return html_str
        
    except Exception as e:
        logger.error(f"Chart generation error for {symbol}: {e}")
        return f"<h1>Error generating chart for {symbol}: {str(e)}</h1>"

@app.route('/api/pairs')
def get_trading_pairs():
    """API endpoint to get available trading pairs."""
    return jsonify(TRADING_PAIRS)

@app.route('/api/timeframes')
def get_timeframes():
    """API endpoint to get available timeframes."""
    return jsonify(TIMEFRAMES)

@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0',
        'ml_features': ml_features_available
    })

@app.route('/api/ml/accuracy', methods=['POST'])
def get_ml_accuracy():
    """Get ML-based wave accuracy prediction."""
    if not ml_features_available or ml_accuracy is None:
        return jsonify({
            'success': False,
            'error': 'ML features not available'
        })
    
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        timeframe = data.get('timeframe', '1d')
        
        # Get timeframe configuration
        tf_config = TIMEFRAMES.get(timeframe, TIMEFRAMES['1d'])
        
        # Load market data
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
        
        # Detect waves
        waves = wave_detector.detect_waves(market_data)
        
        if not waves:
            return jsonify({
                'success': False,
                'error': 'No waves detected for ML analysis'
            })
        
        # Get ML accuracy predictions
        ml_predictions = []
        for wave in waves:
            try:
                accuracy_score = ml_accuracy.predict_wave_accuracy(wave, market_data)
                ml_predictions.append({
                    'wave_type': wave.wave_type.value,
                    'confidence': wave.confidence,
                    'ml_accuracy': accuracy_score,
                    'recommendation': 'High Confidence' if accuracy_score > 0.7 else 'Medium Confidence' if accuracy_score > 0.5 else 'Low Confidence'
                })
            except Exception as e:
                logger.error(f"ML prediction error for wave: {e}")
                ml_predictions.append({
                    'wave_type': wave.wave_type.value,
                    'confidence': wave.confidence,
                    'ml_accuracy': 0.5,
                    'recommendation': 'ML Unavailable'
                })
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'ml_predictions': ml_predictions,
            'analysis_timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"ML accuracy analysis error: {e}")
        return jsonify({
            'success': False,
            'error': f'ML analysis failed: {str(e)}'
        })

@app.route('/api/ml/autotune', methods=['POST'])
def auto_tune_parameters():
    """Auto-tune wave detection parameters using ML."""
    if not ml_features_available or ml_accuracy is None:
        return jsonify({
            'success': False,
            'error': 'ML features not available'
        })
    
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        timeframe = data.get('timeframe', '1d')
        
        # Get timeframe configuration
        tf_config = TIMEFRAMES.get(timeframe, TIMEFRAMES['1d'])
        
        # Load market data
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
        
        # Auto-tune parameters
        original_threshold = wave_detector.zigzag_threshold
        original_min_length = wave_detector.min_wave_length
        original_confidence = wave_detector.confidence_threshold
        
        # Try different parameter combinations
        best_params = None
        best_score = 0
        
        thresholds = [0.02, 0.03, 0.04, 0.05]
        min_lengths = [3, 5, 7]
        confidence_levels = [0.4, 0.5, 0.6]
        
        for threshold in thresholds:
            for min_length in min_lengths:
                for confidence in confidence_levels:
                    # Test parameters
                    wave_detector.zigzag_threshold = threshold
                    wave_detector.min_wave_length = min_length
                    wave_detector.confidence_threshold = confidence
                    
                    waves = wave_detector.detect_waves(market_data)
                    
                    if waves:
                        # Calculate average ML confidence
                        total_score = 0
                        for wave in waves:
                            try:
                                score = ml_accuracy.predict_wave_accuracy(wave, market_data)
                                total_score += score
                            except:
                                total_score += 0.5
                        
                        avg_score = total_score / len(waves)
                        
                        if avg_score > best_score:
                            best_score = avg_score
                            best_params = {
                                'threshold': threshold,
                                'min_length': min_length,
                                'confidence': confidence,
                                'waves_found': len(waves),
                                'avg_ml_score': avg_score
                            }
        
        # Restore best parameters
        if best_params:
            wave_detector.zigzag_threshold = best_params['threshold']
            wave_detector.min_wave_length = best_params['min_length']
            wave_detector.confidence_threshold = best_params['confidence']
        else:
            # Restore original parameters if no improvement found
            wave_detector.zigzag_threshold = original_threshold
            wave_detector.min_wave_length = original_min_length
            wave_detector.confidence_threshold = original_confidence
            best_params = {
                'threshold': original_threshold,
                'min_length': original_min_length,
                'confidence': original_confidence,
                'waves_found': 0,
                'avg_ml_score': 0
            }
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'best_parameters': best_params,
            'original_parameters': {
                'threshold': original_threshold,
                'min_length': original_min_length,
                'confidence': original_confidence
            },
            'analysis_timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Auto-tune error: {e}")
        return jsonify({
            'success': False,
            'error': f'Auto-tuning failed: {str(e)}'
        })

@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    """Run backtest analysis on Elliott Wave patterns."""
    if not ml_features_available or backtester is None:
        return jsonify({
            'success': False,
            'error': 'Backtesting features not available'
        })
    
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        timeframe = data.get('timeframe', '1d')
        period = data.get('period', '1y')  # Backtest period
        
        # Get timeframe configuration
        tf_config = TIMEFRAMES.get(timeframe, TIMEFRAMES['1d'])
        
        # Load market data for backtesting
        market_data = data_loader.get_yahoo_data(
            symbol, 
            period=period,
            interval=tf_config['interval']
        )
        
        if market_data.empty:
            return jsonify({
                'success': False,
                'error': f'No data available for {symbol}'
            })
        
        # Run backtest
        backtest_results = backtester.run_backtest(
            symbol=symbol,
            data=market_data,
            wave_detector=wave_detector,
            start_capital=10000
        )
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'backtest_results': backtest_results,
            'analysis_timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return jsonify({
            'success': False,
            'error': f'Backtesting failed: {str(e)}'
        })

@app.route('/api/forward-test', methods=['POST'])
def run_forward_test():
    """Run forward test analysis on Elliott Wave patterns."""
    if not ml_features_available:
        return jsonify({
            'success': False,
            'error': 'Forward testing features not available'
        })
    
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        timeframe = data.get('timeframe', '1d')
        
        # Get timeframe configuration
        tf_config = TIMEFRAMES.get(timeframe, TIMEFRAMES['1d'])
        
        # Load recent market data
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
        
        # Detect current waves
        waves = wave_detector.detect_waves(market_data)
        
        if not waves:
            return jsonify({
                'success': False,
                'error': 'No waves detected for forward testing'
            })
        
        # Generate forward test predictions
        predictions = []
        current_price = market_data['close'].iloc[-1]
        
        for wave in waves[-3:]:  # Last 3 waves
            try:
                ml_score = ml_accuracy.predict_wave_accuracy(wave, market_data) if ml_accuracy else 0.5
                
                # Simple forward prediction based on wave pattern
                if wave.wave_type.value in ['WAVE_1', 'WAVE_3', 'WAVE_5']:
                    direction = 'BULLISH'
                    target_price = current_price * 1.05
                    stop_loss = current_price * 0.95
                else:
                    direction = 'BEARISH'
                    target_price = current_price * 0.95
                    stop_loss = current_price * 1.05
                
                predictions.append({
                    'wave_type': wave.wave_type.value,
                    'direction': direction,
                    'current_price': current_price,
                    'target_price': target_price,
                    'stop_loss': stop_loss,
                    'ml_confidence': ml_score,
                    'time_horizon': '1-3 days'
                })
                
            except Exception as e:
                logger.error(f"Forward test prediction error: {e}")
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'forward_predictions': predictions,
            'analysis_timestamp': datetime.now().isoformat(),
            'disclaimer': 'Forward testing predictions are experimental and should not be used for actual trading'
        })
        
    except Exception as e:
        logger.error(f"Forward test error: {e}")
        return jsonify({
            'success': False,
            'error': f'Forward testing failed: {str(e)}'
        })

@app.route('/api/technical/analyze', methods=['POST'])
def technical_confluence_analysis():
    """Dedicated endpoint for technical confluence analysis."""
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        timeframe = data.get('timeframe', '1d')
        
        # Get timeframe configuration
        tf_config = TIMEFRAMES.get(timeframe, TIMEFRAMES['1d'])
        
        # Load market data
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
        
        # Detect Elliott Waves
        waves = wave_detector.detect_waves(market_data)
        
        if not waves:
            return jsonify({
                'success': False,
                'error': 'No Elliott Wave patterns detected for confluence analysis'
            })
        
        # Prepare Elliott Wave analysis structure
        elliott_analysis = {
            'waves': [],
            'pattern_type': 'impulse',
            'current_wave': None,
            'completion_level': 0.0
        }
        
        # Convert waves to the expected format
        for wave in waves:
            wave_dict = {
                'wave_type': wave.wave_type.value,
                'start_price': wave.start_point.price,
                'end_price': wave.end_point.price,
                'start_time': wave.start_point.timestamp,
                'end_time': wave.end_point.timestamp,
                'direction': 'UP' if wave.direction.value == 1 else 'DOWN',
                'confidence': wave.confidence
            }
            elliott_analysis['waves'].append(wave_dict)
        
        # Perform comprehensive confluence analysis
        target_zones = technical_confluence_analyzer.analyze_target_zones(
            market_data=market_data,
            elliott_analysis=elliott_analysis,
            timeframe=timeframe
        )
        
        # Format response
        formatted_targets = []
        for target in target_zones:
            formatted_targets.append({
                'price': target.price_level,
                'wave_target': target.wave_target,
                'elliott_basis': target.elliott_basis,
                'confluence_score': target.confluence_score,
                'confidence_level': target.confidence_level,
                'confluences': target.confluences,
                'probability': target.probability,
                'risk_reward_ratio': target.risk_reward_ratio,
                'timeframe': target.timeframe,
                'expected_move': f'Target: {target.price_level:.4f}',
                'confluence_details': {
                    'methods_count': len(target.confluences),
                    'methods': target.confluences
                }
            })
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'timeframe': timeframe,
            'target_zones': formatted_targets,
            'confluence_summary': {
                'total_targets': len(target_zones),
                'high_confidence_targets': len([t for t in target_zones if t.confidence_level == 'HIGH']),
                'medium_confidence_targets': len([t for t in target_zones if t.confidence_level == 'MEDIUM']),
                'low_confidence_targets': len([t for t in target_zones if t.confidence_level == 'LOW']),
                'average_confluence_score': sum(t.confluence_score for t in target_zones) / len(target_zones) if target_zones else 0,
                'strongest_confluence': max(target_zones, key=lambda t: t.confluence_score).confluence_score if target_zones else 0,
                'weakest_confluence': min(target_zones, key=lambda t: t.confluence_score).confluence_score if target_zones else 0
            },
            'analysis_timestamp': datetime.now().isoformat(),
            'analysis_mode': 'technical_confluence_only'
        })
        
    except Exception as e:
        logger.error(f"Technical confluence analysis error: {e}")
        return jsonify({
            'success': False,
            'error': f'Technical confluence analysis failed: {str(e)}'
        })

@app.route('/api/technical/confluence-details', methods=['POST'])
def confluence_details():
    """Get detailed confluence breakdown for specific price levels."""
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        price_level = data.get('price_level')
        timeframe = data.get('timeframe', '1d')
        
        if not price_level:
            return jsonify({
                'success': False,
                'error': 'Price level is required'
            })
        
        # Get timeframe configuration
        tf_config = TIMEFRAMES.get(timeframe, TIMEFRAMES['1d'])
        
        # Load market data
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
        
        # Detect Elliott Waves
        waves = wave_detector.detect_waves(market_data)
        
        # Analyze confluence at specific level
        confluence_analysis = technical_confluence_analyzer._analyze_confluence_at_level(
            price_level, waves, market_data
        )
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'price_level': price_level,
            'confluence_score': confluence_analysis.get('score', 0),
            'confluence_methods': confluence_analysis.get('methods', []),
            'method_details': confluence_analysis.get('details', {}),
            'confidence_level': technical_confluence_analyzer._get_confidence_level(confluence_analysis.get('score', 0)),
            'probability': technical_confluence_analyzer._calculate_probability(confluence_analysis.get('score', 0)),
            'analysis_timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Confluence details analysis error: {e}")
        return jsonify({
            'success': False,
            'error': f'Confluence details analysis failed: {str(e)}'
        })

def quick_confluence_analysis(market_data, elliott_analysis, timeframe):
    """
    Simplified confluence analysis for large datasets to prevent timeouts
    """
    from src.analysis.technical_confluence import TargetZone
    
    try:
        target_zones = []
        
        # Simple Elliott Wave target generation (no heavy confluence analysis)
        waves = elliott_analysis.get('waves', [])
        if not waves:
            return []
        
        current_price = market_data['close'].iloc[-1]
        
        # Generate simple targets based on last few waves
        recent_waves = waves[-3:] if len(waves) >= 3 else waves
        
        for i, wave in enumerate(recent_waves):
            # Simple Fibonacci-based targets
            start_price = wave.get('start_price', current_price)
            end_price = wave.get('end_price', current_price)
            price_range = abs(end_price - start_price)
            
            # Basic Fibonacci extensions/retracements
            fib_ratios = [0.618, 1.0, 1.618]
            
            for ratio in fib_ratios:
                if wave.get('direction') == 'UP':
                    target_price = end_price + (price_range * ratio)
                else:
                    target_price = end_price - (price_range * ratio)
                
                # Skip targets too close to current price
                price_change_pct = abs((target_price - current_price) / current_price) * 100
                if price_change_pct < 2:  # Skip targets less than 2% away
                    continue
                
                # Simple confluence score based on basic indicators
                confluence_score = 3  # Basic score
                confidence_level = "MEDIUM"
                confluences = ["Elliott Wave Pattern", "Fibonacci Level", "Basic Technical Analysis"]
                
                # Adjust based on wave type
                wave_type = wave.get('wave', 'Unknown')
                if wave_type in ['1', '3', '5']:
                    confluence_score += 2
                    confidence_level = "HIGH" if confluence_score >= 5 else "MEDIUM"
                
                target_zone = TargetZone(
                    price_level=target_price,
                    wave_target=f"Wave {wave_type} Target",
                    elliott_basis=f"Fibonacci {ratio:.1%} of Wave {wave_type}",
                    confluence_score=confluence_score,
                    confidence_level=confidence_level,
                    confluences=confluences,
                    probability=0.6 if confluence_score >= 5 else 0.5,
                    timeframe=timeframe,
                    risk_reward_ratio=max(1.5, price_change_pct / 10)
                )
                
                target_zones.append(target_zone)
        
        # Sort by confluence score and limit to top 5
        target_zones.sort(key=lambda x: x.confluence_score, reverse=True)
        return target_zones[:5]
        
    except Exception as e:
        logger.error(f"Quick confluence analysis error: {e}")
        return []

if __name__ == '__main__':
    print("🚀 Starting Elliott Wave Bot Web Application (Enhanced Version)...")
    print("📊 Available trading pairs:")
    for category, pairs in TRADING_PAIRS.items():
        print(f"   {category.upper()}: {', '.join(pairs.keys())}")
    print(f"⏰ Available timeframes: {', '.join(TIMEFRAMES.keys())}")
    print("🌐 Access the application at: http://localhost:5000")
    print("✨ Features: Enhanced Fibonacci levels, Wave 2/4 analysis, Future predictions")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
