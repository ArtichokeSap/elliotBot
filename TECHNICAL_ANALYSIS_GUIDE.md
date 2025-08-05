# Elliott Wave Technical Analysis System
## Complete Technical Analysis with High-Probability Target Zone Detection

🚀 **SYSTEM OVERVIEW**

This system combines Elliott Wave Theory with technical confluence analysis to identify high-probability target zones for any trading pair. It supports multiple exchanges and provides both command-line and web API interfaces.

---

## 🌟 **KEY FEATURES**

### Elliott Wave Detection
- ✅ Automated wave pattern recognition
- ✅ Fibonacci projection calculations  
- ✅ Wave validation scoring
- ✅ Support for both impulse and corrective patterns

### Technical Confluence Analysis
- 🎯 **6 Confluence Methods:**
  1. Fibonacci levels (retracements & extensions)
  2. Support/Resistance zones  
  3. Momentum indicators (RSI, MACD)
  4. Chart patterns recognition
  5. Volume analysis
  6. Harmonic pattern detection

### Multi-Exchange Support
- 📊 **Binance** - Full OHLCV data + market summaries
- 📊 **Bybit** - Complete API integration
- 🔄 Rate limiting and error handling
- 📈 Real-time and historical data

### Target Zone Scoring
- 🔥 **HIGH Confidence**: 5+ confluences (85%+ probability)
- ⚠️ **MEDIUM Confidence**: 3-4 confluences (65%+ probability)  
- 🔽 **LOW Confidence**: <3 confluences (45%+ probability)

---

## 🚀 **QUICK START**

### 1. Command Line Analysis
```bash
# Basic analysis
python analyze.py BTC/USDT

# Advanced options
python analyze.py ETH/USDT -t 4h -e bybit --detailed

# List available symbols
python analyze.py --list-symbols -e binance
```

### 2. Windows Quick Start
```cmd
# Double-click or run:
start_technical_analysis.bat

# Follow the interactive menu
```

### 3. Web API Server
```bash
# Start API server
python src/api/technical_analysis_api.py

# Access at http://localhost:5001
```

---

## 📊 **API ENDPOINTS**

### Health Check
```
GET /api/health
```

### Single Symbol Analysis
```
GET /api/analyze?symbol=BTC/USDT&timeframe=1h&exchange=binance
```

### Multi-Timeframe Analysis
```
GET /api/multi-timeframe?symbol=ETH/USDT&exchange=binance
```

### Confluence Details
```
GET /api/confluence-details?symbol=BTC/USDT&price=50000
```

---

## 🎯 **SAMPLE OUTPUT**

### Command Line Results
```
🎯 TARGET ZONES ANALYSIS
============================================================
🔥 HIGH CONFIDENCE TARGETS:
   1. $114089.59 (-0.15%)
      Wave: Wave C
      Basis: Fibonacci Extension 161.8% of Wave A
      Confluence Score: 24/10
      Probability: 85.0%
      Risk/Reward: 0.07

🏆 BEST TARGET RECOMMENDATION:
   Target: $114089.59
   Expected Move: -0.15%
   Confidence: HIGH
   Wave: Wave C
   Probability: 85.0%
```

### API JSON Response
```json
{
  "success": true,
  "symbol": "BTC/USDT",
  "target_zones": [
    {
      "price_level": 114089.59,
      "wave_target": "Wave C",
      "elliott_basis": "Fibonacci Extension 161.8% of Wave A",
      "confluence_score": 24,
      "confidence_level": "HIGH",
      "probability": 0.85,
      "risk_reward_ratio": 0.07,
      "confluences": [
        "fibonacci_extension_1618",
        "support_resistance_zone",
        "momentum_divergence",
        "volume_confluence"
      ]
    }
  ]
}
```

---

## 🔧 **CONFIGURATION**

### API Configuration
Copy `api_config_template.yaml` to `api_config.yaml` and customize:

```yaml
# Server settings
server:
  host: "127.0.0.1"
  port: 5000
  debug: true

# Exchange settings
exchanges:
  binance:
    enabled: true
    rate_limit: 1200
  bybit:
    enabled: true
    rate_limit: 600

# Analysis parameters
analysis:
  confluence:
    fibonacci:
      retracement_levels: [0.236, 0.382, 0.5, 0.618, 0.786]
      extension_levels: [1.0, 1.236, 1.382, 1.618, 2.0, 2.618]
```

---

## 📚 **TECHNICAL DETAILS**

### Elliott Wave Theory Implementation
- **Wave Detection**: Zig-zag pattern identification
- **Pattern Validation**: Fibonacci relationships and wave rules
- **Projection Methods**: Extensions, retracements, and time analysis

### Confluence Scoring Algorithm
```python
# Each confluence method contributes points:
fibonacci_levels: 0-5 points
support_resistance: 0-3 points  
momentum_indicators: 0-2 points
chart_patterns: 0-4 points
volume_analysis: 0-2 points
harmonic_patterns: 0-3 points

# Total possible: 19 points
# Normalized to 0-10 scale for display
```

### Risk Management Features
- Risk/Reward ratio calculation
- Probability-based position sizing
- Stop-loss level suggestions
- Maximum drawdown analysis

---

## 🛠️ **SYSTEM ARCHITECTURE**

### Core Components
```
src/
├── analysis/
│   ├── technical_confluence.py    # Main confluence analyzer
│   ├── enhanced_wave_detector.py  # Elliott Wave detection  
│   └── ml_wave_accuracy.py        # ML validation system
├── data/
│   └── enhanced_data_fetcher.py   # Multi-exchange data client
├── api/
│   └── technical_analysis_api.py  # Flask web API
└── utils/                         # Helper functions
```

### External Dependencies
- **NumPy/Pandas**: Data processing
- **Requests**: API communication  
- **Flask**: Web API framework
- **TA-Lib** (optional): Technical indicators

---

## 🎮 **USAGE EXAMPLES**

### Crypto Analysis
```bash
# Bitcoin analysis
python analyze.py BTC/USDT -t 1h

# Ethereum on Bybit
python analyze.py ETH/USDT -e bybit -t 4h

# Detailed analysis with JSON export
python analyze.py SOL/USDT --detailed
```

### Multi-Timeframe Strategy
```bash
# API call for multiple timeframes
curl "http://localhost:5001/api/multi-timeframe?symbol=BTC/USDT&exchange=binance"
```

### Production Integration
```python
import requests

# Get analysis via API
response = requests.get("http://localhost:5001/api/analyze", params={
    "symbol": "BTC/USDT",
    "timeframe": "1h", 
    "exchange": "binance"
})

analysis = response.json()
best_target = analysis["target_zones"][0]
```

---

## 🚨 **IMPORTANT NOTES**

### Data Quality
- Minimum 200 candles recommended for accurate analysis
- Higher timeframes (4h, 1d) generally more reliable
- Volume data essential for confluence analysis

### Risk Disclaimer  
- This is a technical analysis tool, not financial advice
- Always use proper risk management
- Backtest strategies before live trading
- Market conditions can invalidate technical patterns

### Performance Tips
- Use caching for production deployments
- Implement rate limiting for API endpoints
- Monitor exchange API usage limits
- Consider WebSocket connections for real-time data

---

## 🔬 **TESTING & VALIDATION**

### Test Suite
```bash
# Run all tests
python test_technical_analysis.py

# Expected output:
# ✅ Enhanced Data Fetcher: PASSED
# ✅ Technical Analysis Core: PASSED  
# ✅ API Integration: PASSED
```

### Manual Testing
```bash
# Test different symbols
python analyze.py ETH/USDT
python analyze.py BNBUSDT 
python analyze.py ADAUSDT

# Test different timeframes
python analyze.py BTC/USDT -t 15m
python analyze.py BTC/USDT -t 4h
python analyze.py BTC/USDT -t 1d
```

---

## 📈 **ROADMAP & EXTENSIONS**

### Planned Features
- [ ] WebSocket real-time updates
- [ ] Email/SMS alert system
- [ ] Advanced ML pattern recognition
- [ ] Portfolio-level analysis
- [ ] Custom indicator framework

### Integration Possibilities
- Trading bot automation
- Discord/Telegram notifications
- TradingView webhook integration
- Portfolio management systems
- Risk management platforms

---

## 🆘 **TROUBLESHOOTING**

### Common Issues

**"No Elliott Wave patterns detected"**
- Try different timeframe (4h often works better)
- Ensure sufficient data points (500+ recommended)
- Check if symbol has enough volatility

**"API connection timeout"**  
- Check internet connection
- Verify exchange API is accessible
- Try different exchange (binance/bybit)

**"TA-Lib not available"**
- System works without TA-Lib (uses manual calculations)
- For better performance: `pip install TA-Lib`

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python analyze.py BTC/USDT
```

---

## 📞 **SUPPORT**

### Getting Help
1. Check logs in `logs/` directory
2. Run test suite: `python test_technical_analysis.py`
3. Try different symbols/timeframes
4. Review configuration in `api_config.yaml`

### System Requirements
- Python 3.8+
- Internet connection for exchange APIs
- 4GB+ RAM recommended
- Windows/Linux/MacOS compatible

---

**🎯 Ready to identify high-probability trading opportunities with Elliott Wave + Technical Confluence Analysis!**
