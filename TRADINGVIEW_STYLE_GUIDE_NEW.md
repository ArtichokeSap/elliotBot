# TradingView-Style Elliott Wave Visualization Guide

## 🎨 Professional Elliott Wave Charts

This guide explains the new TradingView-style Elliott Wave visualization features that create professional charts matching the style shown in your TradingView example.

## 🚀 Key Features Implemented

### 1. Professional Wave Labeling
- **Primary Degree**: Numbers 1, 2, 3, 4, 5 and letters A, B, C
- **Intermediate Degree**: Numbers in parentheses (1), (2), (3), (4), (5) and (A), (B), (C)
- **Minor Degree**: Roman numerals in parentheses (i), (ii), (iii), (iv), (v) and (a), (b), (c)

### 2. TradingView Color Scheme
- **Impulse Waves**: Professional blue (#2962FF) for up, red (#F23645) for down
- **Corrective Waves**: Green (#089981) for up, orange (#FF6D00) for down
- **Candlesticks**: Teal (#26A69A) for up, red (#EF5350) for down
- **Background**: Clean white with light gray grid lines

### 3. Professional Styling
- Clean candlestick charts with proper borders
- Volume bars with color-coded transparency
- Elliott Wave trend lines connecting all points
- Professional annotations positioned at wave endpoints
- TradingView-style layout and typography

### 4. Enhanced Features
- **Invalidation Levels**: Shows key Elliott Wave rule violations
- **Fibonacci Integration**: Displays retracement and extension levels
- **Interactive Hover**: Detailed wave information on hover
- **Multiple Degrees**: Support for different wave degree classifications
- **Professional Legend**: Clean legend positioning and styling

## 📊 Usage Examples

### Basic Usage
```python
from src.visualization.tradingview_style import create_tradingview_chart

# Create professional chart
fig = create_tradingview_chart(
    data=ohlcv_data,
    waves=detected_waves,
    symbol="AAPL",
    degree="primary",
    save_path="aapl_professional.html"
)
```

### Advanced Usage with Fibonacci
```python
from src.visualization.visualizer import WaveVisualizer

visualizer = WaveVisualizer()

# Create TradingView-style chart with all features
fig = visualizer.create_tradingview_chart(
    data=data,
    waves=waves,
    fibonacci_analysis=fib_analysis,
    title="AAPL - Elliott Wave Analysis",
    degree="primary",
    save_path="aapl_tradingview.html"
)
```

## 🌊 Wave Degree System

### Primary Degree (Default)
- Best for daily charts and main trends
- Uses numbers: 1, 2, 3, 4, 5, A, B, C
- Matches the style in your TradingView example

### Intermediate Degree
- For shorter timeframes or sub-waves
- Uses parentheses: (1), (2), (3), (4), (5), (A), (B), (C)

### Minor Degree
- For very short-term analysis
- Uses Roman numerals: (i), (ii), (iii), (iv), (v), (a), (b), (c)

## 🎯 Generated Files

The system now creates several professional charts:

### 1. Main TradingView-Style Charts
- `{symbol}_tradingview_elliott_waves.html` - Primary degree analysis
- `{symbol}_intermediate_waves.html` - Intermediate degree analysis

### 2. Key Features in Each Chart
- **Professional candlestick chart** with TradingView colors
- **Elliott Wave annotations** with proper labeling
- **Volume analysis** with color-coded bars
- **Interactive hover information** for each wave
- **Clean legend** and professional layout
- **Invalidation levels** (when applicable)
- **Fibonacci levels** (when available)

## 🔧 Configuration Options

### Wave Degree Selection
```python
# Primary degree (1, 2, 3, 4, 5)
degree="primary"

# Intermediate degree ((1), (2), (3), (4), (5))
degree="intermediate"

# Minor degree ((i), (ii), (iii), (iv), (v))
degree="minor"
```

### Color Customization
The TradingView-style visualizer uses professional colors:
- Impulse waves: Blue/Red for direction
- Corrective waves: Green/Orange for direction
- Candlesticks: Teal/Red with proper transparency
- Background: Clean white with light grid

## 📈 Professional Features

### 1. Elliott Wave Rules
- **Wave 2 Invalidation**: Cannot retrace below start of wave 1
- **Wave 4 Overlap**: Cannot overlap with wave 1 territory
- **Wave 3 Extension**: Cannot be the shortest impulse wave

### 2. Fibonacci Integration
- **Retracement Levels**: 23.6%, 38.2%, 50%, 61.8%, 78.6%
- **Extension Levels**: 127.2%, 161.8%, 261.8%
- **Color Coding**: Strong (red), medium (orange), weak (yellow)

### 3. Interactive Elements
- **Hover Information**: Wave type, confidence, price change
- **Zoom and Pan**: Full interactivity with Plotly
- **Legend Control**: Toggle wave visibility
- **Time Navigation**: Scroll through different periods

## 🎨 Visual Improvements

### Compared to Basic Charts
1. **Professional color scheme** matching TradingView
2. **Better wave labeling** with proper Elliott Wave notation
3. **Cleaner layout** with improved spacing and typography
4. **Enhanced annotations** positioned at optimal locations
5. **Volume integration** with color-coded transparency
6. **Interactive features** for better analysis

### TradingView Style Elements
- Clean white background
- Professional blue/red color scheme
- Proper candlestick styling
- Elliott Wave trend lines
- Clean typography and spacing
- Interactive hover information

## 🚀 Quick Start

1. **Run the Demo**:
   ```bash
   python demo_tradingview.py
   ```

2. **Multi-Symbol Analysis**:
   ```bash
   python examples/tradingview_elliott_analysis.py
   ```

3. **Open Generated Charts**:
   - Open any `.html` file in your browser
   - Charts are fully interactive with zoom, pan, and hover

## 💡 Tips for Best Results

1. **Use Daily Data**: Works best with daily timeframes for primary analysis
2. **Choose Appropriate Degree**: Match wave degree to your analysis timeframe
3. **Check Invalidation Levels**: Pay attention to Elliott Wave rule violations
4. **Combine with Fibonacci**: Use Fibonacci levels for target identification
5. **Interactive Analysis**: Use hover and zoom features for detailed examination

## 🎯 Professional Output

The new TradingView-style charts provide:
- **Publication-ready quality** for reports and presentations
- **Interactive analysis tools** for detailed wave examination
- **Professional appearance** matching industry standards
- **Comprehensive wave information** with confidence levels
- **Fibonacci integration** for target analysis
- **Volume confirmation** for wave validation

## 📋 Examples of Generated Charts

### Recent Outputs
- `aapl_tradingview_elliott_waves.html` - Apple with primary degree waves
- `btc-usd_tradingview_elliott_waves.html` - Bitcoin with professional styling
- `tsla_tradingview_elliott_waves.html` - Tesla with intermediate degree
- `nvda_tradingview_elliott_waves.html` - NVIDIA with complete analysis

This implementation creates charts that match the professional appearance and functionality of TradingView Elliott Wave analysis, as demonstrated in your example image.
