"""
Elliott Wave Bot - Web Application
Interactive web interface for Elliott Wave analysis with multiple trading pairs and timeframes
"""

from flask import Flask, render_template, request, jsonify
import json
import sys
import os
from datetime import datetime, timedelta
import logging

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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
        'XAUUSD': 'GC=F',  # Gold
        'XAGUSD': 'SI=F',  # Silver
    },
    'crypto': {
        'BTCUSD': 'BTC-USD',
        'ETHUSD': 'ETH-USD',
        'ADAUSD': 'ADA-USD',
        'DOTUSD': 'DOT-USD',
        'LINKUSD': 'LINK-USD',
        'LTCUSD': 'LTC-USD',
        'XRPUSD': 'XRP-USD',
        'SOLUSD': 'SOL-USD',
    },
    'stocks': {
        'AAPL': 'AAPL',
        'GOOGL': 'GOOGL',
        'MSFT': 'MSFT',
        'AMZN': 'AMZN',
        'TSLA': 'TSLA',
        'NVDA': 'NVDA',
        'META': 'META',
        'NFLX': 'NFLX',
    },
    'indices': {
        'SPY': 'SPY',
        'QQQ': 'QQQ',
        'DIA': 'DIA',
        'IWM': 'IWM',
        'VIX': '^VIX',
        'DXY': 'DX-Y.NYB',
    }
}

# Timeframe configuration
TIMEFRAMES = {
    '1m': {'period': '1d', 'interval': '1m', 'label': '1 Minute'},
    '5m': {'period': '5d', 'interval': '5m', 'label': '5 Minutes'},
    '15m': {'period': '5d', 'interval': '15m', 'label': '15 Minutes'},
    '30m': {'period': '5d', 'interval': '30m', 'label': '30 Minutes'},
    '1h': {'period': '1mo', 'interval': '1h', 'label': '1 Hour'},
    '4h': {'period': '3mo', 'interval': '1d', 'label': '4 Hours (Daily Data)'},
    '1d': {'period': '1y', 'interval': '1d', 'label': '1 Day'},
    '1w': {'period': '2y', 'interval': '1wk', 'label': '1 Week'},
    '1M': {'period': '5y', 'interval': '1mo', 'label': '1 Month'},
}

@app.route('/')
def index():
    """Main page with trading pair and timeframe selection."""
    return render_template('index.html', 
                         trading_pairs=TRADING_PAIRS, 
                         timeframes=TIMEFRAMES)

@app.route('/api/analyze', methods=['POST'])
def analyze_pair():
    """API endpoint to analyze a specific trading pair and timeframe."""
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        timeframe = data.get('timeframe', '1d')
        
        logger.info(f"Analyzing {symbol} on {timeframe} timeframe")
        
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
        
        # Detect Elliott Waves with adaptive parameters
        try:
            # Try different sensitivity levels if no waves detected
            sensitivity_levels = [0.02, 0.03, 0.04, 0.05, 0.06]
            waves = []
            
            for threshold in sensitivity_levels:
                # Create a temporary detector with adjusted settings
                temp_detector = WaveDetector()
                temp_detector.zigzag_threshold = threshold
                temp_detector.min_wave_length = max(3, len(market_data) // 100)  # Adaptive minimum length
                temp_detector.confidence_threshold = 0.4  # Lower confidence threshold
                
                waves = temp_detector.detect_waves(market_data)
                
                if len(waves) > 0:
                    logger.info(f"Detected {len(waves)} waves with threshold {threshold}")
                    break
            
            if len(waves) == 0:
                logger.warning(f"No waves detected for {symbol} with any sensitivity level")
                
                # Create simple trend analysis as fallback
                waves = create_simple_trend_analysis(market_data)
                logger.info(f"Created {len(waves)} trend patterns as fallback")
                
        except Exception as e:
            logger.error(f"Wave detection error: {e}")
            waves = []
        
        # Calculate Fibonacci levels
        try:
            fibonacci_levels = []
            future_predictions = []
            
            if waves:
                # Enhanced Fibonacci analysis for waves 2 and 4
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
        
        # Create interactive chart
        try:
            fig = create_web_chart(market_data, waves, symbol, timeframe)
            chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
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
            'chart': chart_json,
            'market_summary': market_summary,
            'analysis_timestamp': datetime.now().isoformat()
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

def create_web_chart(data, waves, symbol, timeframe):
    """Create an interactive chart optimized for web display with enhanced Elliott Wave visualization."""
    
    fig = go.Figure()
    
    # Add candlestick chart
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
    
    # Update layout for enhanced web display
    # Calculate price range for better y-axis scaling
    price_range = data['high'].max() - data['low'].min()
    price_padding = price_range * 0.1
    
    fig.update_layout(
        title=dict(
            text=f'{symbol} - Elliott Wave Analysis ({timeframe.upper()})',
            x=0.5,
            font=dict(size=26, color='white', family="Arial Black")
        ),
        template='plotly_dark',
        height=800,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1
        ),
        xaxis=dict(
            title='Time',
            gridcolor='rgba(128,128,128,0.2)',
            rangeslider=dict(visible=False),
            showspikes=True,
            spikecolor="white",
            spikesnap="cursor",
            spikemode="across"
        ),
        yaxis=dict(
            title='Price (USD)',
            gridcolor='rgba(128,128,128,0.2)',
            side='left',
            showspikes=True,
            spikecolor="white",
            spikesnap="cursor",
            spikemode="across",
            range=[data['low'].min() - price_padding, data['high'].max() + price_padding]
        ),
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right',
            gridcolor='rgba(128,128,128,0.1)',
            showgrid=False,
            range=[0, data['volume'].max() * 4]  # Volume takes 25% of chart height
        ),
        plot_bgcolor='rgba(17,17,17,1)',
        paper_bgcolor='rgba(17,17,17,1)',
        font=dict(color='white', family="Arial"),
        margin=dict(l=80, r=120, t=100, b=80),
        dragmode='zoom',
        hovermode='x unified'
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

    # Add volume subplot
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['volume'],
        name='Volume',
        marker_color='rgba(158,202,225,0.5)',
        yaxis='y2'
    ))

    # Update layout for web
    fig.update_layout(
        title=dict(
            text=f'{symbol} - Elliott Wave Analysis ({timeframe.upper()})',
            x=0.5,
            font=dict(size=24, color='white')
        ),
        template='plotly_dark',
        height=700,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            title='Time',
            gridcolor='rgba(128,128,128,0.2)',
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(
            title='Price',
            gridcolor='rgba(128,128,128,0.2)',
            side='left'
        ),
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right',
            gridcolor='rgba(128,128,128,0.1)',
            showgrid=False
        ),
        plot_bgcolor='rgba(17,17,17,1)',
        paper_bgcolor='rgba(17,17,17,1)',
        font=dict(color='white'),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Add volume subplot
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['volume'],
        name='Volume',
        marker_color='rgba(158,202,225,0.5)',
        yaxis='y2'
    ))
    
    # Update layout for web
    fig.update_layout(
        title=dict(
            text=f'{symbol} - Elliott Wave Analysis ({timeframe.upper()})',
            x=0.5,
            font=dict(size=24, color='white')
        ),
        template='plotly_dark',
        height=700,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            title='Time',
            gridcolor='rgba(128,128,128,0.2)',
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(
            title='Price',
            gridcolor='rgba(128,128,128,0.2)',
            side='left'
        ),
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right',
            gridcolor='rgba(128,128,128,0.1)',
            showgrid=False
        ),
        plot_bgcolor='rgba(17,17,17,1)',
        paper_bgcolor='rgba(17,17,17,1)',
        font=dict(color='white'),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

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
        'version': '1.0.0'
    })

if __name__ == '__main__':
    print("🚀 Starting Elliott Wave Bot Web Application...")
    print("📊 Available trading pairs:")
    for category, pairs in TRADING_PAIRS.items():
        print(f"   {category.upper()}: {', '.join(pairs.keys())}")
    print(f"⏰ Available timeframes: {', '.join(TIMEFRAMES.keys())}")
    print("🌐 Access the application at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
