"""
Simple Elliott Wave Web App with Visible Confluence Targets
This creates a minimal but working confluence display
"""

from flask import Flask, render_template, request, jsonify
import json
import sys
import os
from datetime import datetime
import logging
import numpy as np
import plotly.graph_objects as go
import plotly.utils

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.data.data_loader import DataLoader
from src.analysis.wave_detector import WaveDetector
from simple_confluence import SimpleConfluenceAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'elliott_wave_simple_key'

# Initialize components
data_loader = DataLoader()
wave_detector = WaveDetector()
confluence_analyzer = SimpleConfluenceAnalyzer()

# Simple configuration
SYMBOLS = {
    'AAPL': 'AAPL',
    'TSLA': 'TSLA', 
    'MSFT': 'MSFT',
    'BTC-USD': 'BTC-USD',
    'ETH-USD': 'ETH-USD'
}

@app.route('/')
def index():
    """Simple home page"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Simple Elliott Wave Confluence</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: white; }
            .container { max-width: 1200px; margin: 0 auto; }
            .controls { background: #2a2a2a; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
            .targets { background: #2a2a2a; padding: 20px; border-radius: 10px; margin: 20px 0; }
            .target-item { 
                background: #3a3a3a; 
                padding: 15px; 
                margin: 10px 0; 
                border-radius: 8px; 
                border-left: 5px solid #00ff88;
            }
            .target-high { border-left-color: #ffaa00; }
            .target-medium { border-left-color: #0088ff; }
            .target-low { border-left-color: #888888; }
            button { 
                background: #00ff88; 
                color: black; 
                border: none; 
                padding: 10px 20px; 
                border-radius: 5px; 
                cursor: pointer; 
                font-weight: bold;
            }
            button:hover { background: #00dd77; }
            select { 
                background: #3a3a3a; 
                color: white; 
                border: 1px solid #555; 
                padding: 8px; 
                border-radius: 5px;
            }
            .chart-container { background: #2a2a2a; padding: 20px; border-radius: 10px; }
            .loading { text-align: center; color: #00ff88; font-size: 18px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Elliott Wave Confluence Targets</h1>
            
            <div class="controls">
                <label>Symbol: </label>
                <select id="symbol">
                    <option value="AAPL">AAPL</option>
                    <option value="TSLA">TSLA</option>
                    <option value="MSFT">MSFT</option>
                    <option value="BTC-USD">BTC-USD</option>
                    <option value="ETH-USD">ETH-USD</option>
                </select>
                <button onclick="analyzeSymbol()">🚀 Analyze Confluence</button>
            </div>
            
            <div id="loading" class="loading" style="display: none;">
                ⏳ Analyzing confluence targets...
            </div>
            
            <div class="chart-container">
                <div id="chart" style="height: 600px;"></div>
            </div>
            
            <div id="targets" class="targets" style="display: none;">
                <h2>🎯 Confluence Targets</h2>
                <div id="target-list"></div>
            </div>
        </div>

        <script>
        async function analyzeSymbol() {
            const symbol = document.getElementById('symbol').value;
            const loading = document.getElementById('loading');
            const targets = document.getElementById('targets');
            
            loading.style.display = 'block';
            targets.style.display = 'none';
            
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ symbol: symbol })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    // Display chart
                    document.getElementById('chart').innerHTML = data.chart_html;
                    
                    // Display targets
                    displayTargets(data.confluence_targets);
                    targets.style.display = 'block';
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                loading.style.display = 'none';
            }
        }
        
        function displayTargets(targetList) {
            const container = document.getElementById('target-list');
            container.innerHTML = '';
            
            targetList.forEach((target, index) => {
                const div = document.createElement('div');
                div.className = `target-item target-${target.confidence.toLowerCase()}`;
                
                const changePercent = ((target.price - target.current_price) / target.current_price * 100).toFixed(2);
                const changeColor = changePercent >= 0 ? '#00ff88' : '#ff4444';
                
                div.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>${target.name}</strong>
                            <div style="color: #aaa; font-size: 14px;">Methods: ${target.methods.join(', ')}</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 20px; font-weight: bold;">$${target.price.toFixed(4)}</div>
                            <div style="color: ${changeColor};">${changePercent >= 0 ? '+' : ''}${changePercent}%</div>
                            <div style="color: #aaa;">Score: ${target.score}</div>
                        </div>
                    </div>
                `;
                
                container.appendChild(div);
            });
        }
        
        // Auto-analyze AAPL on load
        window.onload = () => analyzeSymbol();
        </script>
    </body>
    </html>
    '''

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Simple analysis endpoint"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'AAPL')
        
        logger.info(f"🔍 Analyzing {symbol}")
        
        # Load data
        market_data = data_loader.get_yahoo_data(symbol, period='1y', interval='1d')
        
        if market_data.empty:
            return jsonify({'success': False, 'error': f'No data for {symbol}'})
        
        # Detect waves
        waves = wave_detector.detect_waves(market_data)
        logger.info(f"📊 Found {len(waves)} waves")
        
        # Generate confluence targets
        confluence_targets = confluence_analyzer.analyze_simple_targets(market_data, waves)
        logger.info(f"🎯 Generated {len(confluence_targets)} confluence targets")
        
        # Create chart
        chart_html = create_simple_chart_html(market_data, waves, confluence_targets, symbol)
        
        # Format targets for response
        targets_data = []
        current_price = market_data['close'].iloc[-1]
        
        for target in confluence_targets:
            targets_data.append({
                'name': target.name,
                'price': target.price,
                'methods': target.methods,
                'score': target.score,
                'confidence': target.confidence,
                'current_price': current_price
            })
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'confluence_targets': targets_data,
            'chart_html': chart_html,
            'summary': {
                'total_targets': len(targets_data),
                'high_confidence': len([t for t in confluence_targets if t.confidence == 'HIGH']),
                'current_price': current_price
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Analysis error: {e}")
        return jsonify({'success': False, 'error': str(e)})

def create_simple_chart_html(market_data, waves, confluence_targets, symbol="BTC-USD"):
    """Create a simple HTML chart with inline JavaScript"""
    
    # Prepare chart data
    dates = [d.strftime('%Y-%m-%d') for d in market_data.index]
    open_prices = market_data['open'].tolist()
    high_prices = market_data['high'].tolist()
    low_prices = market_data['low'].tolist()
    close_prices = market_data['close'].tolist()
    
    # Wave points
    wave_lines = []
    for i, wave in enumerate(waves):
        wave_lines.append(f"""
        data.push({{
            x: ['{wave.start_date.strftime('%Y-%m-%d')}', '{wave.end_date.strftime('%Y-%m-%d')}'],
            y: [{wave.start_price}, {wave.end_price}],
            mode: 'lines',
            name: 'Wave {i+1}',
            line: {{color: 'yellow', width: 2}}
        }});""")
    
    # Target lines
    target_lines = []
    for target in confluence_targets:
        color = 'red' if target.confidence == 'HIGH' else 'orange' if target.confidence == 'MEDIUM' else 'gray'
        target_lines.append(f"""
        data.push({{
            x: ['{dates[0]}', '{dates[-1]}'],
            y: [{target.price}, {target.price}],
            mode: 'lines',
            name: '{target.name}',
            line: {{color: '{color}', width: 2, dash: 'dash'}}
        }});""")
    
    html_chart = f"""
    <div id="chart" style="width:100%; height:600px;"></div>
    <script>
        var candlestick = {{
            x: {dates},
            open: {open_prices},
            high: {high_prices},
            low: {low_prices},
            close: {close_prices},
            type: 'candlestick',
            name: '{symbol}',
            increasing: {{line: {{color: '#00ff00'}}}},
            decreasing: {{line: {{color: '#ff0000'}}}}
        }};
        
        var data = [candlestick];
        
        {chr(10).join(wave_lines)}
        
        {chr(10).join(target_lines)}
        
        var layout = {{
            title: '{symbol} - Elliott Wave & Confluence Analysis',
            xaxis: {{title: 'Date', color: '#fff'}},
            yaxis: {{title: 'Price', color: '#fff'}},
            paper_bgcolor: '#1e1e1e',
            plot_bgcolor: '#2d2d2d',
            font: {{color: '#fff'}},
            showlegend: true
        }};
        
        Plotly.newPlot('chart', data, layout);
    </script>
    """
    
    return html_chart

def create_simple_chart(data, waves, targets, symbol):
    """Create a simple chart with confluence targets"""
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
        decreasing_line_color='#ff4444'
    ))
    
    # Add Elliott waves
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
    for i, wave in enumerate(waves[:5]):  # Limit to 5 waves for clarity
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=[wave.start_point.timestamp, wave.end_point.timestamp],
            y=[wave.start_point.price, wave.end_point.price],
            mode='lines+markers+text',
            line=dict(color=color, width=3),
            marker=dict(size=8, color=color),
            text=['', wave.wave_type.value.split('_')[-1]],
            textposition='top center',
            name=f'Wave {wave.wave_type.value.split("_")[-1]}',
            showlegend=True
        ))
    
    # Add confluence target lines
    for target in targets:
        # Color based on confidence
        if target.confidence == 'HIGH':
            color = '#ffaa00'
            width = 3
        elif target.confidence == 'MEDIUM':
            color = '#0088ff'
            width = 2
        else:
            color = '#888888'
            width = 1
        
        # Add horizontal line
        fig.add_hline(
            y=target.price,
            line_dash="solid",
            line_color=color,
            line_width=width,
            annotation_text=f"🎯 {target.name}: ${target.price:.2f}",
            annotation_position="top right"
        )
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} - Elliott Wave with Confluence Targets',
        template='plotly_dark',
        height=600,
        showlegend=True,
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified'
    )
    
    return fig

if __name__ == '__main__':
    print("🚀 Starting Simple Elliott Wave Confluence App...")
    print("🌐 Access at: http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)
