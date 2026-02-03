# Elliott Wave Bot - Web Application Guide

## 🌐 Web Interface Overview

Your Elliott Wave Bot now includes a professional web interface that allows users to:

- **Select Trading Pairs**: Choose from Forex, Crypto, Stocks, and Indices
- **Multiple Timeframes**: From 1-minute to 1-month charts
- **Interactive Analysis**: Real-time Elliott Wave pattern detection
- **Professional Charts**: Candlestick charts with wave annotations
- **Fibonacci Levels**: Automatic retracement calculations
- **Export Functionality**: Save charts as PNG images

## 🚀 Quick Start

### Method 1: Simple Launch
```bash
python run_web.py
```

### Method 2: Full Launch (with auto-browser)
```bash
python launch_web.py
```

### Method 3: Direct Flask
```bash
cd web
python app.py
```

Then open your browser to: **http://localhost:5000**

## 📊 Available Trading Pairs

### 💱 Forex
- **EURUSD, GBPUSD, USDJPY**
- **AUDUSD, USDCAD, USDCHF, NZDUSD**
- **XAUUSD (Gold), XAGUSD (Silver)**

### 🪙 Cryptocurrency  
- **BTCUSD, ETHUSD, ADAUSD**
- **DOTUSD, LINKUSD, LTCUSD**
- **XRPUSD, SOLUSD**

### 📈 Stocks
- **AAPL, GOOGL, MSFT, AMZN**
- **TSLA, NVDA, META, NFLX**

### 📊 Indices
- **SPY, QQQ, DIA, IWM**
- **VIX, DXY**

## ⏰ Timeframe Options

| Timeframe | Data Period | Best For |
|-----------|-------------|----------|
| **1m** | 1 day | Scalping |
| **5m** | 5 days | Intraday |
| **15m** | 5 days | Short-term |
| **30m** | 5 days | Day trading |
| **1h** | 1 month | Swing trading |
| **4h** | 3 months | Position trading |
| **1d** | 1 year | Long-term analysis |
| **1w** | 2 years | Macro trends |
| **1M** | 5 years | Investment analysis |

## 🎯 How to Use

### 1. Select Your Analysis
1. **Choose Category**: Forex, Crypto, Stocks, or Indices
2. **Pick Trading Pair**: Select from dropdown
3. **Select Timeframe**: Choose appropriate timeframe
4. **Click "Analyze Waves"**: Start the analysis

### 2. Interpret Results
- **Interactive Chart**: Zoom, pan, and hover for details
- **Wave Table**: Shows detected Elliott Waves with confidence scores
- **Fibonacci Levels**: Retracement levels for recent waves
- **Market Summary**: Key statistics and current prices

### 3. Export and Share
- **Export Chart**: Save as PNG image
- **Refresh Data**: Get latest market data
- **Copy Analysis**: Share results with others

## 📱 Web Interface Features

### 🎨 Professional Design
- **Dark Theme**: Easy on the eyes for long analysis sessions
- **Responsive Layout**: Works on desktop, tablet, and mobile
- **Smooth Animations**: Professional user experience
- **Bootstrap Framework**: Modern and clean interface

### 📊 Interactive Charts
- **Plotly.js Integration**: High-performance charting
- **Candlestick Visualization**: Professional OHLC display
- **Wave Annotations**: Clear Elliott Wave labels (1,2,3,4,5,A,B,C)
- **Volume Analysis**: Volume bars with transparency
- **Zoom & Pan**: Detailed chart exploration

### 🔍 Real-time Analysis
- **Live Data**: Fresh market data from Yahoo Finance
- **Instant Processing**: Fast Elliott Wave detection
- **Confidence Scoring**: Quality assessment for each wave
- **Error Handling**: Graceful handling of data issues

## 🛠️ Technical Architecture

### Backend (Flask)
```
web/
├── app.py              # Main Flask application
├── templates/
│   └── index.html      # Main web page template
└── static/
    ├── css/
    │   └── style.css   # Custom styling
    └── js/
        └── app.js      # Frontend JavaScript
```

### API Endpoints
- **GET /**: Main web interface
- **POST /api/analyze**: Elliott Wave analysis
- **GET /api/pairs**: Available trading pairs
- **GET /api/timeframes**: Available timeframes
- **GET /api/health**: Health check

### Data Flow
1. **User Selection** → Frontend JavaScript
2. **API Request** → Flask Backend
3. **Data Loading** → Yahoo Finance API
4. **Wave Detection** → Elliott Wave Algorithm
5. **Chart Generation** → Plotly.js
6. **Results Display** → Interactive Web Interface

## 🚨 Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Error: Address already in use
# Solution: Try different port
python -c "from web.app import app; app.run(port=5001)"
```

#### 2. Module Import Errors
```bash
# Error: No module named 'flask'
# Solution: Install dependencies
pip install -r requirements-minimal.txt
```

#### 3. Data Loading Failures
```bash
# Error: No data available for symbol
# Solution: 
# - Check internet connection
# - Try different symbol
# - Use different timeframe
```

#### 4. Chart Not Displaying
```bash
# Error: Chart container empty
# Solution:
# - Check browser console for JavaScript errors
# - Ensure Plotly.js is loaded
# - Try refreshing the page
```

### Browser Compatibility
- ✅ **Chrome 80+** (Recommended)
- ✅ **Firefox 75+**
- ✅ **Safari 13+**
- ✅ **Edge 80+**

### Performance Tips
- **Use 1d timeframe** for initial testing
- **Popular symbols** (AAPL, BTCUSD) load faster
- **Clear browser cache** if charts don't update
- **Close other browser tabs** for better performance

## 🔧 Customization

### Adding New Trading Pairs
Edit `web/app.py` and add to `TRADING_PAIRS`:
```python
'forex': {
    'NEWPAIR': 'NEWPAIR=X',  # Add new forex pair
    ...
}
```

### Modifying Timeframes
Edit `TIMEFRAMES` in `web/app.py`:
```python
'2h': {'period': '1mo', 'interval': '2h', 'label': '2 Hours'},
```

### Styling Changes
Edit `web/static/css/style.css` for custom colors and layout.

### Adding Features
- Edit `web/static/js/app.js` for new frontend features
- Edit `web/app.py` for new API endpoints

## 📈 Advanced Usage

### Keyboard Shortcuts
- **Ctrl+Enter**: Analyze current selection
- **Escape**: Return to welcome screen
- **F5**: Refresh page and data

### URL Parameters (Future)
```
http://localhost:5000/?symbol=AAPL&timeframe=1d
```

### API Usage (Advanced)
```javascript
// Direct API call
fetch('/api/analyze', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({symbol: 'AAPL', timeframe: '1d'})
})
```

## 🚀 Deployment Options

### Local Development
```bash
python run_web.py  # Development server
```

### Production Deployment
```bash
# Using Gunicorn (Linux/Mac)
pip install gunicorn
cd web
gunicorn -w 4 -b 0.0.0.0:8000 app:app

# Using Waitress (Windows)
pip install waitress
cd web
waitress-serve --port=8000 app:app
```

### Cloud Deployment
- **Heroku**: Push to Heroku with Procfile
- **AWS**: Deploy on EC2 with nginx
- **Docker**: Containerize for any platform

## 🎉 Success!

Your Elliott Wave Bot web interface is now ready for professional trading analysis! 

**Access it at: http://localhost:5000**

Enjoy analyzing Elliott Wave patterns across multiple markets and timeframes! 📊🚀
