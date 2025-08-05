"""
Elliott Wave Bot - Offline Version with Built-in Data
No external network dependencies - runs completely offline
"""

from flask import Flask, render_template, request, jsonify
import json
import sys
import os
from datetime import datetime, timedelta
import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.utils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'elliott_wave_offline_2025'

# Built-in sample data - no network required
SAMPLE_DATA = {
    'AAPL': {
        'name': 'Apple Inc.',
        'data': None  # Will be generated
    },
    'BTC-USD': {
        'name': 'Bitcoin USD',
        'data': None  # Will be generated
    },
    'EURUSD': {
        'name': 'EUR/USD',
        'data': None  # Will be generated
    },
    'TSLA': {
        'name': 'Tesla Inc.',
        'data': None  # Will be generated
    }
}

def generate_realistic_price_data(symbol, days=365):
    """Generate realistic price data for demonstration"""
    np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
    
    # Base parameters for different assets
    if symbol == 'AAPL':
        base_price = 150.0
        volatility = 0.025
        trend = 0.0003
    elif symbol == 'BTC-USD':
        base_price = 45000.0
        volatility = 0.06
        trend = 0.0008
    elif symbol == 'EURUSD':
        base_price = 1.10
        volatility = 0.008
        trend = 0.0001
    elif symbol == 'TSLA':
        base_price = 200.0
        volatility = 0.04
        trend = 0.0005
    else:
        base_price = 100.0
        volatility = 0.02
        trend = 0.0002
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate price series with realistic patterns
    prices = []
    current_price = base_price
    
    for i in range(len(dates)):
        # Add Elliott Wave-like patterns
        cycle_pos = (i % 100) / 100.0  # 100-day cycle
        
        # Create wave-like movement
        if cycle_pos < 0.2:  # Wave 1
            wave_trend = trend * 2
        elif cycle_pos < 0.3:  # Wave 2 (correction)
            wave_trend = -trend * 1.5
        elif cycle_pos < 0.5:  # Wave 3 (strongest)
            wave_trend = trend * 3
        elif cycle_pos < 0.6:  # Wave 4 (correction)
            wave_trend = -trend * 0.8
        elif cycle_pos < 0.8:  # Wave 5
            wave_trend = trend * 1.5
        else:  # ABC correction
            wave_trend = -trend * 1.2
        
        # Add random volatility
        random_change = np.random.normal(wave_trend, volatility)
        current_price *= (1 + random_change)
        
        # Ensure price doesn't go negative
        current_price = max(current_price, base_price * 0.1)
        prices.append(current_price)
    
    # Create OHLC data
    high_prices = [p * (1 + abs(np.random.normal(0, volatility * 0.3))) for p in prices]
    low_prices = [p * (1 - abs(np.random.normal(0, volatility * 0.3))) for p in prices]
    open_prices = [prices[0]] + prices[:-1]
    
    # Add volume
    base_volume = 1000000 if 'USD' in symbol else 50000000
    volumes = [int(base_volume * (1 + np.random.normal(0, 0.5))) for _ in prices]
    
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    return df

def detect_elliott_waves_simple(data, symbol):
    """Simple Elliott Wave detection without external dependencies"""
    try:
        # Find peaks and troughs
        closes = data['close'].values
        highs = data['high'].values
        lows = data['low'].values
        
        # Simple peak/trough detection
        window = max(5, len(data) // 50)
        peaks = []
        troughs = []
        
        for i in range(window, len(data) - window):
            # Check for peak
            if all(highs[i] >= highs[i-j] for j in range(1, window+1)) and \
               all(highs[i] >= highs[i+j] for j in range(1, window+1)):
                peaks.append((i, highs[i], data.index[i]))
            
            # Check for trough
            if all(lows[i] <= lows[i-j] for j in range(1, window+1)) and \
               all(lows[i] <= lows[i+j] for j in range(1, window+1)):
                troughs.append((i, lows[i], data.index[i]))
        
        # Combine and sort turning points
        turning_points = []
        for idx, price, time in peaks:
            turning_points.append({'idx': idx, 'price': price, 'time': time, 'type': 'peak'})
        for idx, price, time in troughs:
            turning_points.append({'idx': idx, 'price': price, 'time': time, 'type': 'trough'})
        
        turning_points.sort(key=lambda x: x['idx'])
        
        # Generate waves from turning points
        waves = []
        wave_labels = ['1', '2', '3', '4', '5', 'A', 'B', 'C']
        
        for i in range(len(turning_points) - 1):
            if i >= len(wave_labels):
                break
                
            start = turning_points[i]
            end = turning_points[i + 1]
            
            direction = 'bullish' if end['price'] > start['price'] else 'bearish'
            price_change = ((end['price'] - start['price']) / start['price']) * 100
            
            # Calculate confidence based on price move significance
            confidence = min(0.95, abs(price_change) / 10.0 + 0.3)
            
            waves.append({
                'wave': wave_labels[i],
                'wave_type': 'impulse' if i < 5 else 'corrective',
                'direction': direction,
                'start_time': start['time'],
                'end_time': end['time'],
                'start_price': start['price'],
                'end_price': end['price'],
                'length': abs(end['price'] - start['price']),
                'duration': (end['time'] - start['time']).days,
                'confidence': confidence
            })
        
        # Determine overall structure
        if len(waves) >= 5:
            wave_structure = 'impulse'
            validation_score = 0.75
        elif len(waves) >= 3:
            wave_structure = 'corrective'
            validation_score = 0.65
        else:
            wave_structure = 'trend'
            validation_score = 0.55
        
        # Calculate Fibonacci levels
        fibonacci_levels = {}
        if len(waves) >= 2:
            wave1 = waves[0]
            wave2 = waves[1]
            
            fibonacci_levels = {
                'wave1_length': wave1['length'],
                'wave2_retracement': abs(wave2['length'] / wave1['length']),
                'fibonacci_236': wave1['start_price'] + (wave1['length'] * 0.236),
                'fibonacci_382': wave1['start_price'] + (wave1['length'] * 0.382),
                'fibonacci_618': wave1['start_price'] + (wave1['length'] * 0.618)
            }
        
        return {
            'validation_score': validation_score,
            'wave_structure': wave_structure,
            'direction': 'bullish' if data['close'].iloc[-1] > data['close'].iloc[0] else 'bearish',
            'waves': waves,
            'fibonacci_levels': fibonacci_levels,
            'rule_compliance': {
                'wave_count': {'score': min(1.0, len(waves) / 5.0)},
                'pattern_clarity': {'score': validation_score}
            },
            'recommendations': [
                f"Detected {len(waves)} waves in {wave_structure} structure",
                f"Overall direction: {direction}",
                f"Confidence level: {validation_score:.1%}"
            ],
            'issues': [] if validation_score > 0.6 else ['Low confidence pattern - needs more data']
        }
        
    except Exception as e:
        logger.error(f"Error in simple Elliott Wave detection: {e}")
        return {
            'validation_score': 0.3,
            'wave_structure': 'trend',
            'direction': 'neutral',
            'waves': [],
            'fibonacci_levels': {},
            'rule_compliance': {},
            'recommendations': ['Unable to detect clear Elliott Wave patterns'],
            'issues': [f'Analysis error: {str(e)}']
        }

# Initialize sample data
for symbol in SAMPLE_DATA.keys():
    SAMPLE_DATA[symbol]['data'] = generate_realistic_price_data(symbol)

# Trading pairs configuration
TRADING_PAIRS = {
    'stocks': {
        'AAPL': 'Apple Inc.',
        'TSLA': 'Tesla Inc.',
        'GOOGL': 'Google',
        'MSFT': 'Microsoft'
    },
    'crypto': {
        'BTC-USD': 'Bitcoin USD',
        'ETH-USD': 'Ethereum USD',
        'ADA-USD': 'Cardano USD'
    },
    'forex': {
        'EURUSD': 'EUR/USD',
        'GBPUSD': 'GBP/USD',
        'USDJPY': 'USD/JPY'
    }
}

# Timeframe configuration
TIMEFRAMES = {
    '1h': {'label': '1 Hour'},
    '4h': {'label': '4 Hours'},
    '1d': {'label': '1 Day'},
    '1wk': {'label': '1 Week'}
}

@app.route('/')
def index():
    """Main page with Elliott Wave analysis interface."""
    try:
        return render_template('index.html', 
                             trading_pairs=TRADING_PAIRS, 
                             timeframes=TIMEFRAMES)
    except Exception as e:
        # Fallback if template not found
        return f"""
        <html>
        <head><title>Elliott Wave Bot - Offline</title></head>
        <body>
        <h1>🌊 Elliott Wave Bot - Offline Mode</h1>
        <p>Web interface template not found. Using API-only mode.</p>
        <h2>Available Endpoints:</h2>
        <ul>
            <li><a href="/api/health">Health Check</a></li>
            <li><a href="/api/pairs">Trading Pairs</a></li>
            <li><a href="/api/timeframes">Timeframes</a></li>
        </ul>
        <h2>Sample Analysis:</h2>
        <p>POST to /api/analyze with: {{"symbol": "AAPL", "timeframe": "1d"}}</p>
        </body>
        </html>
        """

@app.route('/api/analyze', methods=['POST'])
def analyze_pair():
    """API endpoint to analyze a specific trading pair and timeframe."""
    
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'AAPL')
        timeframe = data.get('timeframe', '1d')
        
        logger.info(f"🚀 Offline Analysis: {symbol} on {timeframe}")
        
        # Get sample data
        if symbol in SAMPLE_DATA:
            market_data = SAMPLE_DATA[symbol]['data']
        else:
            # Generate data for unknown symbols
            market_data = generate_realistic_price_data(symbol)
        
        # Perform Elliott Wave analysis
        analysis_result = detect_elliott_waves_simple(market_data, symbol)
        
        # Convert waves to web format
        waves = []
        for wave_data in analysis_result['waves']:
            waves.append({
                'type': wave_data['wave'],
                'direction': 'UP' if wave_data['direction'] == 'bullish' else 'DOWN',
                'start_time': str(wave_data['start_time']),
                'end_time': str(wave_data['end_time']),
                'start_price': float(wave_data['start_price']),
                'end_price': float(wave_data['end_price']),
                'price_change': float(((wave_data['end_price'] - wave_data['start_price']) / wave_data['start_price']) * 100),
                'confidence': float(wave_data['confidence'])
            })
        
        # Create validation results
        validation_results = [{
            'type': analysis_result['wave_structure'].upper(),
            'score': round(analysis_result['validation_score'] * 100, 1),
            'status': 'HIGH_CONFIDENCE' if analysis_result['validation_score'] >= 0.7 else 'MODERATE_CONFIDENCE',
            'recommendations': analysis_result['recommendations'],
            'issues': analysis_result['issues']
        }]
        
        # Create Fibonacci levels
        fibonacci_levels = []
        if analysis_result['fibonacci_levels']:
            for level_name, value in analysis_result['fibonacci_levels'].items():
                if isinstance(value, (int, float)):
                    fibonacci_levels.append({
                        'level': level_name.replace('_', ' ').title(),
                        'value': float(value),
                        'type': 'retracement' if 'retracement' in level_name else 'level'
                    })
        
        # Create future predictions
        future_predictions = []
        if waves:
            current_price = market_data['close'].iloc[-1]
            
            # Simple prediction based on wave structure
            if analysis_result['wave_structure'] == 'impulse':
                expected_move = 0.08  # 8% continuation
                pattern_desc = "Impulse Pattern - Continuation Expected"
            else:
                expected_move = 0.05  # 5% move
                pattern_desc = "Corrective Pattern - Reversal Possible"
            
            predicted_price = current_price * (1 + expected_move)
            
            future_predictions.append({
                'pattern': pattern_desc,
                'probability': f"Medium ({analysis_result['validation_score']:.0%})",
                'targets': [{
                    'level': 'Target',
                    'price': predicted_price,
                    'ratio': f'{expected_move * 100:+.1f}%'
                }],
                'expected_move': f"{'Upward' if expected_move > 0 else 'Downward'} movement",
                'timeframe': timeframe,
                'confidence': analysis_result['validation_score']
            })
        
        # Create chart
        fig = create_offline_chart(market_data, waves, symbol, timeframe)
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Market summary
        current_price = market_data['close'].iloc[-1]
        
        market_summary = {
            'symbol': symbol,
            'current_price': round(current_price, 4),
            'high_52w': round(market_data['high'].max(), 4),
            'low_52w': round(market_data['low'].min(), 4),
            'change_24h': round(((current_price - market_data['close'].iloc[-2]) / market_data['close'].iloc[-2]) * 100, 2),
            'data_points': len(market_data),
            'timeframe': TIMEFRAMES.get(timeframe, {'label': timeframe})['label'],
            'last_update': market_data.index[-1].strftime('%Y-%m-%d %H:%M:%S UTC')
        }
        
        wave_data = [{
            'type': wave.get('type', 'Unknown'),
            'direction': wave.get('direction', 'UNKNOWN'),
            'start_date': wave.get('start_time', ''),
            'end_date': wave.get('end_time', ''),
            'start_price': wave.get('start_price', 0),
            'end_price': wave.get('end_price', 0),
            'price_change': wave.get('price_change', 0),
            'confidence': wave.get('confidence', 0)
        } for wave in waves]
        
        return jsonify({
            'success': True,
            'validation_score': analysis_result.get('validation_score', 0.6),
            'wave_structure': analysis_result.get('wave_structure', 'trend'),
            'direction': analysis_result.get('direction', 'neutral'),
            'waves': wave_data,
            'fibonacci_levels': fibonacci_levels,
            'future_predictions': future_predictions,
            'validation_results': validation_results,
            'chart': chart_json,
            'market_summary': market_summary,
            'analysis_timestamp': datetime.now().isoformat(),
            'offline_mode': True,
            'data_source': 'Generated sample data'
        })
        
    except Exception as e:
        logger.error(f"❌ Offline analysis error: {e}")
        return jsonify({
            'success': False,
            'error': f'Offline analysis failed: {str(e)}',
            'offline_mode': True
        })

def create_offline_chart(data, waves, symbol, timeframe):
    """Create chart for offline mode"""
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name=symbol,
        increasing_line_color='#26A69A',
        decreasing_line_color='#EF5350'
    ))
    
    # Add wave lines
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336', '#FF5722', '#607D8B', '#E91E63']
    
    for i, wave in enumerate(waves):
        if isinstance(wave, dict):
            color = colors[i % len(colors)]
            
            try:
                start_time = pd.to_datetime(wave['start_time'])
                end_time = pd.to_datetime(wave['end_time'])
                
                fig.add_trace(go.Scatter(
                    x=[start_time, end_time],
                    y=[wave['start_price'], wave['end_price']],
                    mode='lines+markers+text',
                    line=dict(color=color, width=2),
                    marker=dict(size=8, color=color),
                    text=['', wave['type']],
                    textposition='top center',
                    name=f"Wave {wave['type']}",
                    showlegend=False
                ))
            except:
                continue
    
    fig.update_layout(
        title=f"{symbol} - Elliott Wave Analysis (Offline Mode)",
        xaxis_title="Date",
        yaxis_title="Price",
        height=600,
        template="plotly_white",
        hovermode='x unified'
    )
    
    return fig

@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0-offline',
        'mode': 'offline',
        'network_required': False,
        'sample_data_loaded': len(SAMPLE_DATA)
    })

@app.route('/api/pairs')
def get_trading_pairs():
    """API endpoint to get available trading pairs."""
    return jsonify(TRADING_PAIRS)

@app.route('/api/timeframes')
def get_timeframes():
    """API endpoint to get available timeframes."""
    return jsonify(TIMEFRAMES)

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🌊 ELLIOTT WAVE BOT - OFFLINE MODE")
    print("=" * 60)
    print("📊 Professional Elliott Wave Analysis System")
    print("🔌 NO NETWORK REQUIRED - Fully Offline")
    print("📈 Built-in sample data for demonstration")
    print("🎯 Comprehensive pattern detection")
    print("")
    print("✅ Sample data loaded for:")
    for symbol, info in SAMPLE_DATA.items():
        print(f"   • {symbol}: {info['name']}")
    print("")
    print("📊 Available trading pairs:")
    for category, pairs in TRADING_PAIRS.items():
        print(f"   {category.upper()}: {', '.join(pairs.keys())}")
    print("")
    print("⏰ Available timeframes: " + ", ".join(TIMEFRAMES.keys()))
    print("\n🌐 Starting server at: http://localhost:5000")
    print("🔧 No network errors possible - fully self-contained")
    print("=" * 60)
    print("")
    
    try:
        app.run(
            debug=False,
            host='127.0.0.1',  # Localhost only for security
            port=5000,
            use_reloader=False,
            threaded=True
        )
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("🔧 Try running as administrator or change the port.")
        print("💡 Alternative: python -m http.server 8000")
