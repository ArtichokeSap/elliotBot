# Elliott Wave Bot - AI Agent Instructions

## Overview
Elliott Wave trading bot using technical analysis and ML for pattern recognition in financial markets. The system detects Elliott Wave patterns (1-2-3-4-5 impulse, A-B-C corrective), performs Fibonacci analysis, generates trading signals, and includes backtesting capabilities.

## Architecture

### Core Data Flow
1. **Data Loading** (`src/data/`) → **Wave Detection** (`src/analysis/`) → **Signal Generation** (`src/trading/`) → **Visualization** (`src/visualization/`)
2. All modules import from `src.*` structure; examples add parent to `sys.path` with `sys.path.append(os.path.join(os.path.dirname(__file__), '..'))` 

### Module Structure
- **`src/data/data_loader.py`**: OHLCV data from Yahoo Finance/Binance. Use `DataLoader().get_yahoo_data(symbol, period="1y")` for quick data fetching
- **`src/analysis/wave_detector.py`**: Core Elliott Wave detection. Returns `List[Wave]` with `WavePoint` dataclass containing `timestamp`, `price`, `wave_type`, `confidence`
- **`src/analysis/fibonacci.py`**: Fibonacci retracement/extension analysis integrated with wave patterns
- **`src/trading/strategy.py`**: Signal generation using `ElliottWaveStrategy`. Returns `TradingSignal` with entry/exit prices, stop-loss, take-profit
- **`src/trading/backtester.py`**: Historical strategy validation with equity curves and performance metrics
- **`src/visualization/visualizer.py`**: Plotly-based interactive charts using `WaveVisualizer().plot_waves(data, waves)`

### Key Data Structures
```python
# Wave detection returns this structure
@dataclass
class Wave:
    start_point: WavePoint  # timestamp, price, index, wave_type
    end_point: WavePoint
    wave_type: WaveType  # Enum: IMPULSE_1-5, CORRECTIVE_A-C
    direction: TrendDirection  # UP/DOWN/SIDEWAYS
    confidence: float
    fibonacci_ratios: Dict[str, float]
```

## Critical Patterns

### Configuration Management
- **Always use**: `from src.utils.config import get_config` then `config = get_config()`
- Config searches for `config.yaml` at project root (copied from `config_template.yaml`)
- Access nested values: `config.get('wave_detection.zigzag_threshold', 0.05)` with dot notation

### Import Pattern (CRITICAL)
All imports use `from ..module import Class` relative imports within `src/`. Examples must add parent to path:
```python
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.data_loader import DataLoader
```

### Logging Standard
```python
from src.utils.logger import get_logger
logger = get_logger(__name__)  # Auto-configured with file/console handlers
logger.info("message")  # Don't use print() in src/ modules
```

### Wave Detection Workflow
1. Load data with `DataLoader`
2. Call `detector.detect_waves(data)` → returns `List[Wave]`
3. Get current state: `detector.get_current_wave_count(data)` → dict with `current_wave`, `confidence`, `analysis`, `next_target`
4. Wave confidence threshold default is 0.7 (configurable in `config.yaml`)

### Visualization Pattern
```python
visualizer = WaveVisualizer()
fig = visualizer.plot_waves(data, waves, fibonacci_analysis=None, title="...")
fig.write_html("output.html")  # Standard output format
# OR fig.show() for browser display
```

## Development Workflows

### Quick Testing
```powershell
python tools/quick_test.py  # Fast system health check
python test_installation.py  # Verify all dependencies
python main.py  # Run full analysis on default symbol
```

### Running Examples
All examples in `examples/` are standalone scripts:
```powershell
python examples/basic_analysis.py  # Single symbol analysis
python examples/backtesting.py  # Strategy validation
python examples/signal_generation.py  # Trading signals
```

### Installation
- **Recommended**: `pip install -r requirements-minimal.txt` (core packages only)
- **Full**: `pip install -r requirements.txt` (includes optional packages)
- TA-Lib is optional; system has built-in indicators if it fails to install

## Project-Specific Conventions

### Wave Type Enums
Use `WaveType.IMPULSE_1` through `WaveType.IMPULSE_5` and `WaveType.CORRECTIVE_A/B/C`. Never use string literals for wave types.

### Fibonacci Integration
Fibonacci levels are calculated as part of wave analysis. Access via `wave.fibonacci_ratios` dict. Standard levels: 0.236, 0.382, 0.5, 0.618, 0.786 (retracement) and 1.272, 1.618, 2.618 (extension).

### DataFrame Expectations
OHLCV DataFrames must have columns: `['open', 'high', 'low', 'close', 'volume']` with DatetimeIndex. All dates are pandas Timestamps.

### Signal Confidence Scoring
All signals and waves include `confidence: float` (0.0-1.0). Default threshold 0.7 means only act on high-confidence patterns.

### Error Handling Pattern
```python
try:
    # operation
    logger.info("Success message")
except Exception as e:
    logger.error(f"Context: {e}")
    return fallback_value
```

### Security Patterns (CRITICAL)
**Always validate user inputs:**
```python
from main import validate_symbol, validate_period, sanitize_filename

# Validate before use
symbol = validate_symbol(user_input)  # Raises ValueError if invalid
period = validate_period(time_period)  # Whitelist check
filename = sanitize_filename(name)    # Path traversal protection
```

**Model loading security:**
- **NEVER use pickle.load()** - use `joblib.load()` instead
- Only load models from trusted sources
- Validate file existence before loading

**File operations:**
- Always sanitize filenames from user input
- Validate paths stay within project directory
- Use Path objects from pathlib for safety

## Integration Points

### External Dependencies
- **yfinance**: Primary data source, rate-limited to 1 req/sec (configurable)
- **scikit-learn**: ML models for wave classification (XGBoost optional)
- **plotly**: Interactive charts, not matplotlib (matplotlib only for optional static exports)

### Output Formats
- Charts: HTML files via `fig.write_html()` - browser-viewable, no server required
- Logs: `logs/elliott_bot.log` with rotating handler
- Data: Pickle/CSV supported via SQLAlchemy in `src/data/storage.py`

## Common Tasks

### Adding New Wave Type
1. Add enum to `WaveType` in `src/analysis/wave_detector.py`
2. Update `_classify_wave_type()` method with detection logic
3. Add visualization color mapping in `WaveVisualizer.colors` dict

### Creating New Strategy
1. Inherit from `ElliottWaveStrategy` in `src/trading/strategy.py`
2. Override `generate_signals()` method returning `List[TradingSignal]`
3. Test with `BacktestEngine` before live use

### Debugging Wave Detection
- Enable DEBUG logging: `config.yaml` → `general.log_level: "DEBUG"`
- Use `tools/optimize_detection.py` to tune `zigzag_threshold` and `min_wave_length`
- Check `detector.validate_wave_rules(wave)` for rule violations

## Testing
- Run `pytest` from project root
- Use `python validate_imports.py` to check circular dependencies
- Health check: `python tools/health_check.py`

## Documentation Structure
```
docs/
├── reference/          # Evergreen guides (USAGE_GUIDE.md, INSTALL.md, etc.)
├── changelog/          # One-off fix documentation (SECURITY_FIXES.md, etc.)
└── sessions/           # Chronological development session logs
    ├── _SESSION_TEMPLATE.md
    └── YYYY-MM-DD-title.md
```

---

## 🔴 SESSION CONTEXT SKILL (Generalized)

> **Skill Design**: See `docs/reference/SESSION_CONTEXT_SKILL_DESIGN.md` for full architecture.
> **Configuration**: See `.ai/session-context/config.yaml` for settings.

### File Locations
```
.ai/session-context/
├── config.yaml          # Skill configuration
├── summary.md           # Rolling summary (ALWAYS read first - ~500 tokens)
├── recent/              # Last 7 days of sessions
├── archive/             # Monthly compressed archives
└── _templates/          # Session and summary templates
```

### Trigger Phrases
**Logging**: "log session", "save session", "end session", "record session"
**Resumption**: "resume session", "continue session", "catch me up", "where were we"
**Quick**: "quick context" (summary only, no git checks)

---

### 🔴 SESSION LOGGING PROCEDURE

When triggered:

1. **Create session file**: `.ai/session-context/recent/YYYY-MM-DD-[brief-title].md`
2. **Use template**: `.ai/session-context/_templates/session.md`
3. **Fill in**: Summary, Goals, Changes, Decisions, Problems, Next Steps, AI Context
4. **Update summary.md**: Add entry to recent sessions table, update current state
5. **Check archival**: If any sessions in `recent/` are >7 days old, archive them
6. **Confirm with user**: Show file path, offer adjustments

**What NOT to Log**: Trivial Q&A, sessions user declines

---

### 🟢 SESSION RESUMPTION PROCEDURE

**Tiered Context Loading** (optimizes context window):

| Tier | Files | Tokens | When |
|------|-------|--------|------|
| 1 | `summary.md` | ~500 | Always |
| 2 | `recent/*.md` | ~1500 | Full resume |
| 3 | `archive/*.md` | Variable | Historical search |

**For "quick context"**: Read only `summary.md`, report to user.

**For full resume**:
1. Read `summary.md` (always)
2. Run `git status` and `git diff --stat`
3. Read recent sessions if needed for detail
4. Summarize: state, next steps, blockers
5. Offer options from next steps

---

### 📦 ARCHIVAL (Prevents Unbounded Growth)

**Automatic rules** (configured in `config.yaml`):
- Sessions >7 days old: Move from `recent/` to `archive/`
- Archive format: Monthly summaries (`archive/2026-01.md`)
- Max 12 monthly archives, then compress to yearly

**When archiving**:
1. Extract: date, title, key accomplishments, decisions
2. Append to monthly archive file
3. Delete original from `recent/`
4. Regenerate `summary.md` if needed