# Advanced AI-Powered Elliott Wave Analysis Engine

A comprehensive, production-ready Elliott Wave analysis system that combines real-time market data, strict rule validation, advanced projections, historical pattern matching, and professional visualization.

## 🚀 Features

### Core Analysis
- **Real-time Data Fetching**: Binance API integration with fallback to CoinGecko
- **Advanced Wave Detection**: Multi-degree Elliott Wave identification with ZigZag algorithm
- **Strict Rule Validation**: Complete Elliott Wave Theory enforcement
- **Fibonacci Analysis**: Comprehensive retracement and extension calculations
- **Future Projections**: Multiple scenario generation with confidence scoring
- **Pattern Memory**: Historical pattern matching and learning system

### Visualization
- **TradingView-Style Charts**: Professional candlestick charts with wave overlays
- **Multi-Degree Visualization**: Hierarchical wave display with color coding
- **Projection Dashboards**: Interactive future scenario visualization
- **Pattern Memory Dashboards**: Historical pattern analysis and insights
- **Comprehensive Analysis**: All-in-one visualization combining all features

### Advanced Features
- **Time Symmetry Analysis**: Duration ratio validation
- **Complex Pattern Support**: Triangles, diagonals, WXY corrections
- **Risk/Reward Analysis**: Invalidation levels and target calculations
- **Confidence Scoring**: Multi-factor confidence assessment
- **Export Capabilities**: JSON, CSV, and HTML chart exports

## 📁 Project Structure

```
elliottBot/
├── src/
│   ├── data/
│   │   └── binance_data_fetcher.py      # Real-time data fetching
│   ├── analysis/
│   │   ├── wave_detector.py             # Elliott Wave detection
│   │   ├── wave_validator.py            # Rule validation
│   │   ├── wave_projector.py            # Future projections
│   │   ├── pattern_memory.py            # Historical pattern matching
│   │   └── fibonacci.py                 # Fibonacci analysis
│   ├── visualization/
│   │   ├── visualizer.py                # Main visualization engine
│   │   └── tradingview_style.py         # TradingView-style charts
│   └── utils/
│       ├── config.py                    # Configuration management
│       ├── logger.py                    # Logging system
│       └── helpers.py                   # Utility functions
├── examples/
│   ├── comprehensive_elliott_analysis.py # Complete system example
│   ├── basic_analysis.py                # Basic usage example
│   └── ...                              # Additional examples
├── output/                              # Generated charts and data
├── run_comprehensive_analysis.py        # Main runner script
└── requirements.txt                     # Dependencies
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd elliottBot

# Install dependencies
pip install -r requirements.txt

# Run comprehensive analysis
python run_comprehensive_analysis.py --symbol BTCUSDT --timeframe 1h
```

### Dependencies
```
pandas>=1.5.0
numpy>=1.21.0
plotly>=5.0.0
requests>=2.28.0
scipy>=1.9.0
```

## 🎯 Usage

### Command Line Interface

#### Basic Analysis
```bash
# Analyze BTC/USDT on 1-hour timeframe
python run_comprehensive_analysis.py --symbol BTCUSDT --timeframe 1h

# Analyze ETH/USDT on 4-hour timeframe with 14 days of data
python run_comprehensive_analysis.py --symbol ETHUSDT --timeframe 4h --days 14

# Analyze on daily timeframe with JSON export
python run_comprehensive_analysis.py --symbol BTCUSDT --timeframe 1d --export-json
```

#### Advanced Options
```bash
# Validate only (no projections)
python run_comprehensive_analysis.py --symbol BTCUSDT --timeframe 1h --validate-only

# Projections only (skip pattern memory)
python run_comprehensive_analysis.py --symbol BTCUSDT --timeframe 1h --projections-only

# Custom output directory
python run_comprehensive_analysis.py --symbol BTCUSDT --timeframe 1h --output-dir ./my_analysis

# Export to multiple formats
python run_comprehensive_analysis.py --symbol BTCUSDT --timeframe 1h --export-json --export-csv
```

### Programmatic Usage

#### Complete Analysis
```python
from src.data.binance_data_fetcher import BinanceDataFetcher
from src.analysis.wave_detector import WaveDetector
from src.analysis.wave_validator import WaveValidator
from src.analysis.wave_projector import WaveProjector
from src.analysis.pattern_memory import PatternMemory
from src.analysis.fibonacci import FibonacciAnalyzer
from src.visualization.visualizer import WaveVisualizer

# Initialize components
data_fetcher = BinanceDataFetcher()
detector = WaveDetector()
validator = WaveValidator()
projector = WaveProjector()
pattern_memory = PatternMemory()
fibonacci_analyzer = FibonacciAnalyzer()
visualizer = WaveVisualizer()

# Fetch data
data = data_fetcher.get_historical_ohlcv("BTCUSDT", "1h", limit=1000)

# Detect waves
waves = detector.detect_waves(data)

# Validate pattern
validation_result = validator.validate_wave_pattern(waves, data)

# Generate projections
current_wave = max(waves, key=lambda w: w.end_point.timestamp)
projection_scenarios = projector.generate_comprehensive_projections(
    current_wave=current_wave,
    all_waves=waves,
    data=data
)

# Find similar patterns
pattern_matches = pattern_memory.find_similar_patterns(
    current_waves=waves,
    current_validation=validation_result
)

# Create visualization
fig = visualizer.plot_comprehensive_analysis(
    data=data,
    waves=waves,
    projection_scenarios=projection_scenarios,
    pattern_matches=pattern_matches,
    validation_result=validation_result
)

fig.show()
```

#### Individual Components
```python
# Wave detection only
waves = detector.detect_waves(data)
print(f"Detected {len(waves)} waves")

# Validation only
validation_result = validator.validate_wave_pattern(waves, data)
print(f"Validation score: {validation_result.overall_score:.1%}")

# Fibonacci analysis only
fibonacci_analysis = fibonacci_analyzer.analyze_waves(waves, data)
for level in fibonacci_analysis.levels:
    print(f"{level.level_type.value}: ${level.price:.2f}")
```

## 📊 Output Examples

### Generated Charts
- **Comprehensive Analysis**: All features combined in one chart
- **Projection Dashboard**: 6-panel dashboard with projections and statistics
- **Pattern Memory Dashboard**: Historical pattern analysis and insights
- **Validation Dashboard**: Rule compliance and confidence breakdown

### Data Exports
- **JSON**: Complete analysis results in structured format
- **CSV**: Wave data and projections in tabular format
- **HTML**: Interactive charts for web viewing

### Sample Output
```
🚀 Advanced AI-Powered Elliott Wave Analysis Engine
==================================================
Symbol: BTCUSDT
Timeframe: 1h
Days: 7
Output Directory: ./output

✅ Successfully fetched 168 data points
   Date range: 2024-01-15 to 2024-01-22
   Price range: $41,250.00 - $43,500.00

✅ Detected 5 Elliott Waves
   Wave 1: IMPULSE_1 - +2.5% change over 12 periods
   Wave 2: IMPULSE_2 - -1.2% change over 8 periods
   Wave 3: IMPULSE_3 - +4.1% change over 15 periods
   Wave 4: IMPULSE_4 - -0.8% change over 6 periods
   Wave 5: IMPULSE_5 - +1.8% change over 10 periods

✅ Validation completed
   Pattern type: impulse
   Overall confidence: 87.5%
   Critical rules passed: True

✅ Generated 3 projection scenarios
   Scenario 1: Standard ABC Correction
      Confidence: 78.2%
      Primary projection: Wave A correction
      Key targets: $42,150.00, $41,800.00, $41,450.00

✅ Found 8 similar historical patterns
   Match 1: 89.2% similarity (excellent)
      Historical pattern: BTCUSDT_1h_20240115_IMPULSE_1-2-3-4-5
      Historical outcome: ABC correction followed by new impulse

🎉 Analysis Complete!
==================================================
📊 Current BTCUSDT Price: $43,250.00
🌊 Elliott Wave Pattern: impulse
✅ Validation Confidence: 87.5%
🔮 Best Projection: Standard ABC Correction
📈 Projection Confidence: 78.2%
🎯 Key Targets: $42,150.00, $41,800.00, $41,450.00
📚 Best Historical Match: 89.2% similarity
📖 Historical Outcome: ABC correction followed by new impulse
```

## 🔧 Configuration

### Configuration File
Create a `config.yaml` file to customize analysis parameters:

```yaml
# Wave detection settings
wave_detection:
  zigzag_threshold: 0.02
  min_wave_length: 5
  max_wave_length: 100

# Validation settings
validation:
  fibonacci_tolerance: 0.1
  time_symmetry_tolerance: 0.2
  overlap_tolerance: 0.01

# Projection settings
projection:
  historical_pattern_weight: 0.3
  rule_compliance_weight: 0.4
  market_context_weight: 0.3

# Pattern memory settings
pattern_memory:
  structural_weight: 0.3
  fibonacci_weight: 0.25
  temporal_weight: 0.2
  price_action_weight: 0.25

# Visualization settings
visualization:
  chart_height: 800
  chart_width: 1200
  theme: "plotly_dark"
  show_volume: true
```

### Environment Variables
```bash
# API configuration
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key

# Logging
LOG_LEVEL=INFO
LOG_FILE=elliott_analysis.log

# Output
OUTPUT_DIR=./analysis_results
```

## 📈 Analysis Features

### Elliott Wave Detection
- **Multi-degree Analysis**: Primary, Intermediate, Minor, Minute, Minuette
- **ZigZag Algorithm**: Robust swing point detection
- **Pattern Recognition**: Impulse, corrective, triangle, diagonal patterns
- **Time Symmetry**: Duration ratio validation

### Rule Validation
- **Impulse Wave Rules**:
  - Wave 2 must not retrace past Wave 1 start
  - Wave 3 cannot be the shortest among 1, 3, 5
  - Wave 4 must not overlap Wave 1 territory
  - Wave 1, 3, 5 must subdivide into 5 smaller waves

- **Fibonacci Guidelines**:
  - Wave 2 ≈ 50-61.8% of Wave 1
  - Wave 3 ≈ 161.8% of Wave 1
  - Wave 5 ≈ 61.8-100% of Wave 1 or 3

- **Corrective Wave Rules**:
  - Wave B should not exceed 100% of Wave A
  - Wave C should equal or extend Wave A
  - Corrective waves should subdivide into 3 waves

### Future Projections
- **Multiple Scenarios**: Primary and alternative projections
- **Confidence Scoring**: Based on historical patterns and rule compliance
- **Fibonacci Targets**: Retracement and extension levels
- **Invalidation Levels**: Risk management points
- **Time Projections**: Estimated completion times

### Pattern Memory
- **Historical Matching**: Find similar patterns from database
- **Similarity Scoring**: Multi-factor similarity calculation
- **Outcome Prediction**: Based on historical pattern results
- **Learning System**: Continuously improve with new data
- **Pattern Categories**: Impulse, corrective, triangle, diagonal, complex

## 🎨 Visualization Features

### Chart Types
- **Candlestick Charts**: Professional price action display
- **Wave Overlays**: Color-coded Elliott Wave annotations
- **Fibonacci Levels**: Retracement and extension lines
- **Projection Paths**: Future scenario visualization
- **Volume Analysis**: Volume profile integration

### Interactive Features
- **Hover Tooltips**: Detailed wave and projection information
- **Legend Controls**: Toggle different analysis components
- **Zoom and Pan**: Interactive chart navigation
- **Export Options**: PNG, PDF, HTML export

### Dashboard Panels
- **Main Chart**: Complete analysis overview
- **Confidence Levels**: Projection confidence distribution
- **Risk/Reward Analysis**: Risk/reward ratio histogram
- **Pattern Types**: Projection type distribution
- **Fibonacci Targets**: Price target distribution
- **Historical Outcomes**: Pattern memory insights

## 🔍 Advanced Analysis

### Complex Patterns
- **Triangles**: Contracting and expanding triangles
- **Diagonals**: Leading and ending diagonals
- **Complex Corrections**: WXY and WXYXZ patterns
- **Flats**: Regular, expanded, and running flats
- **Zigzags**: Simple and complex zigzag patterns

### Time Analysis
- **Duration Ratios**: Time symmetry validation
- **Fibonacci Time**: Time-based Fibonacci projections
- **Pattern Duration**: Historical pattern timing analysis
- **Completion Estimates**: Projected pattern completion times

### Risk Management
- **Invalidation Levels**: Pattern failure points
- **Stop Loss Levels**: Risk management targets
- **Position Sizing**: Risk/reward based sizing
- **Multiple Timeframes**: Cross-timeframe analysis

## 🚀 Performance

### Optimization Features
- **Efficient Algorithms**: Optimized wave detection algorithms
- **Caching**: Pattern memory caching for faster lookups
- **Parallel Processing**: Multi-threaded analysis where applicable
- **Memory Management**: Efficient data structure usage

### Scalability
- **Large Datasets**: Handle thousands of data points
- **Multiple Symbols**: Analyze multiple assets simultaneously
- **Real-time Updates**: Continuous analysis capabilities
- **Batch Processing**: Process multiple analyses efficiently

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd elliottBot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 src/
black src/
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints for all functions
- Add comprehensive docstrings
- Write unit tests for new features

### Testing
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_wave_detector.py

# Run with coverage
python -m pytest --cov=src
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Elliott Wave International for Elliott Wave Theory
- Binance for market data API
- Plotly for visualization capabilities
- The Elliott Wave community for insights and feedback

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Join our community discussions
- Check the documentation and examples

---

**Disclaimer**: This software is for educational and research purposes. Trading involves risk, and past performance does not guarantee future results. Always conduct your own analysis and consider consulting with financial professionals before making trading decisions. 