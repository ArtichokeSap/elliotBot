"""
Technical Confluence Analysis System
Combines Elliott Wave Theory with multiple technical analysis methods to identify high-probability target zones
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from enum import Enum

# Technical Analysis Libraries
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning("TA-Lib not available. Some technical indicators will use manual calculations.")

# Import Enhanced S/R Detector
try:
    from .enhanced_sr_detector import EnhancedSRDetector
    ENHANCED_SR_AVAILABLE = True
except ImportError:
    ENHANCED_SR_AVAILABLE = False
    logging.warning("Enhanced S/R Detector not available. Using basic S/R detection.")

@dataclass
class TargetZone:
    """Represents a projected target zone with confluence analysis"""
    price_level: float
    wave_target: str  # e.g., "Wave 5", "Wave C"
    elliott_basis: str  # e.g., "Fibonacci Extension 161.8%"
    confluence_score: int
    confidence_level: str  # "HIGH", "MEDIUM", "LOW"
    confluences: List[str]
    probability: float
    timeframe: str
    risk_reward_ratio: float

@dataclass
class FibonacciLevel:
    """Fibonacci retracement/extension level"""
    level: float  # price level
    ratio: float  # fibonacci ratio (0.618, 1.618, etc.)
    type: str    # "retracement" or "extension"
    swing_high: float
    swing_low: float
    
@dataclass
class SupportResistanceZone:
    """Support/Resistance zone"""
    level: float
    strength: int  # number of touches
    type: str     # "support", "resistance", "supply", "demand"
    volume_confirmation: bool

class ConfluenceType(Enum):
    """Types of technical confluence"""
    FIBONACCI = "fibonacci"
    SUPPORT_RESISTANCE = "support_resistance" 
    MOMENTUM = "momentum"
    CHART_PATTERNS = "chart_patterns"
    HARMONIC_PATTERNS = "harmonic_patterns"
    VOLUME = "volume"

class TechnicalConfluenceAnalyzer:
    """
    Advanced technical analysis system combining Elliott Wave with confluence methods
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize Enhanced S/R Detector
        if ENHANCED_SR_AVAILABLE:
            self.sr_detector = EnhancedSRDetector()
            self.logger.info("🔍 Enhanced S/R Detector initialized")
        else:
            self.sr_detector = None
            self.logger.warning("Using basic S/R detection")
        
        # Fibonacci ratios for extensions and retracements
        self.fib_retracements = [0.236, 0.382, 0.5, 0.618, 0.786]
        self.fib_extensions = [1.0, 1.272, 1.414, 1.618, 2.0, 2.618, 3.618, 4.236]
        
        # Confidence thresholds
        self.high_confidence_threshold = 5
        self.medium_confidence_threshold = 3
        
    def analyze_target_zones(self, market_data: pd.DataFrame, elliott_analysis: Dict, 
                           timeframe: str) -> List[TargetZone]:
        """
        Main analysis function that combines Elliott Wave projections with technical confluence
        """
        try:
            target_zones = []
            
            # 1. Get Elliott Wave projections
            elliott_targets = self._get_elliott_wave_targets(elliott_analysis, market_data)
            
            if not elliott_targets:
                self.logger.warning("No Elliott Wave targets found")
                return []
            
            # 2. For each Elliott target, analyze technical confluence
            for target in elliott_targets:
                confluence_analysis = self._analyze_confluence_at_level(
                    market_data, target['price'], target, timeframe
                )
                
                # 3. Create target zone with confluence score
                target_zone = TargetZone(
                    price_level=target['price'],
                    wave_target=target['wave'],
                    elliott_basis=target['basis'],
                    confluence_score=confluence_analysis['score'],
                    confidence_level=self._get_confidence_level(confluence_analysis['score']),
                    confluences=confluence_analysis['methods'],
                    probability=self._calculate_probability(confluence_analysis['score']),
                    timeframe=timeframe,
                    risk_reward_ratio=self._calculate_risk_reward(market_data['close'].iloc[-1], target['price'])
                )
                
                target_zones.append(target_zone)
            
            # 4. Sort by confluence score (highest first)
            target_zones.sort(key=lambda x: x.confluence_score, reverse=True)
            
            return target_zones
            
        except Exception as e:
            self.logger.error(f"Error in target zone analysis: {e}")
            return []
    
    def _get_elliott_wave_targets(self, elliott_analysis: Dict, market_data: pd.DataFrame) -> List[Dict]:
        """Extract projected target levels from Elliott Wave analysis"""
        targets = []
        
        try:
            current_price = market_data['close'].iloc[-1]
            
            # Get wave structure and current position
            wave_structure = elliott_analysis.get('wave_structure', 'unknown')
            waves = elliott_analysis.get('waves', [])
            
            if not waves:
                return targets
            
            # Project next wave targets based on current structure
            if wave_structure.upper() in ['IMPULSE', 'MOTIVE']:
                targets.extend(self._project_impulse_targets(waves, current_price))
            elif wave_structure.upper() in ['CORRECTIVE', 'ZIGZAG', 'FLAT']:
                targets.extend(self._project_corrective_targets(waves, current_price))
            elif wave_structure.upper() == 'TRIANGLE':
                targets.extend(self._project_triangle_targets(waves, current_price))
            
            return targets
            
        except Exception as e:
            self.logger.error(f"Error getting Elliott targets: {e}")
            return []
    
    def _project_impulse_targets(self, waves: List[Dict], current_price: float) -> List[Dict]:
        """Project targets for impulse wave structures"""
        targets = []
        
        try:
            if len(waves) < 3:
                return targets
            
            # Find wave positions
            wave_1 = next((w for w in waves if w.get('wave') == '1'), None)
            wave_3 = next((w for w in waves if w.get('wave') == '3'), None)
            
            if wave_1 and wave_3:
                # Calculate Wave 5 targets using Fibonacci extensions
                wave_1_length = abs(wave_1['end_price'] - wave_1['start_price'])
                wave_3_start = wave_3['start_price']
                
                # Common Wave 5 projections
                extensions = [0.618, 1.0, 1.618]
                for ext in extensions:
                    target_price = wave_3_start + (wave_1_length * ext)
                    targets.append({
                        'price': target_price,
                        'wave': 'Wave 5',
                        'basis': f'Fibonacci Extension {ext:.1%} of Wave 1',
                        'type': 'impulse_completion'
                    })
            
            return targets
            
        except Exception as e:
            self.logger.error(f"Error projecting impulse targets: {e}")
            return []
    
    def _project_corrective_targets(self, waves: List[Dict], current_price: float) -> List[Dict]:
        """Project targets for corrective wave structures"""
        targets = []
        
        try:
            if len(waves) < 2:
                return targets
            
            # Find A and B waves for C wave projection
            wave_a = next((w for w in waves if w.get('wave') == 'A'), None)
            wave_b = next((w for w in waves if w.get('wave') == 'B'), None)
            
            if wave_a and wave_b:
                # Calculate Wave C targets
                wave_a_length = abs(wave_a['end_price'] - wave_a['start_price'])
                wave_b_end = wave_b['end_price']
                
                # Common Wave C projections
                extensions = [1.0, 1.272, 1.618]
                for ext in extensions:
                    if wave_a['direction'] == 'bearish':
                        target_price = wave_b_end - (wave_a_length * ext)
                    else:
                        target_price = wave_b_end + (wave_a_length * ext)
                    
                    targets.append({
                        'price': target_price,
                        'wave': 'Wave C',
                        'basis': f'Fibonacci Extension {ext:.1%} of Wave A',
                        'type': 'corrective_completion'
                    })
            
            return targets
            
        except Exception as e:
            self.logger.error(f"Error projecting corrective targets: {e}")
            return []
    
    def _project_triangle_targets(self, waves: List[Dict], current_price: float) -> List[Dict]:
        """Project targets for triangle wave structures"""
        targets = []
        
        try:
            # Triangle breakout typically equals the widest part of the triangle
            if len(waves) >= 4:
                wave_a = next((w for w in waves if w.get('wave') == 'A'), None)
                wave_b = next((w for w in waves if w.get('wave') == 'B'), None)
                
                if wave_a and wave_b:
                    triangle_height = abs(wave_a['end_price'] - wave_a['start_price'])
                    
                    # Project breakout targets
                    targets.append({
                        'price': current_price + triangle_height,
                        'wave': 'Triangle Breakout Up',
                        'basis': 'Triangle Height Projection',
                        'type': 'triangle_breakout'
                    })
                    
                    targets.append({
                        'price': current_price - triangle_height,
                        'wave': 'Triangle Breakout Down',
                        'basis': 'Triangle Height Projection',
                        'type': 'triangle_breakout'
                    })
            
            return targets
            
        except Exception as e:
            self.logger.error(f"Error projecting triangle targets: {e}")
            return []
    
    def _analyze_confluence_at_level(self, market_data: pd.DataFrame, target_price: float, 
                                   elliott_target: Dict, timeframe: str) -> Dict:
        """Analyze technical confluence at a specific price level"""
        confluence_methods = []
        total_score = 0
        
        try:
            # 1. Fibonacci Analysis
            fib_confluence = self._check_fibonacci_confluence(market_data, target_price)
            if fib_confluence:
                confluence_methods.extend(fib_confluence)
                total_score += len(fib_confluence)
            
            # 2. Support/Resistance Analysis
            sr_confluence = self._check_support_resistance_confluence(market_data, target_price)
            if sr_confluence:
                confluence_methods.extend(sr_confluence)
                total_score += len(sr_confluence)
            
            # 3. Momentum Indicators
            momentum_confluence = self._check_momentum_confluence(market_data, target_price)
            if momentum_confluence:
                confluence_methods.extend(momentum_confluence)
                total_score += len(momentum_confluence)
            
            # 4. Chart Patterns
            pattern_confluence = self._check_chart_pattern_confluence(market_data, target_price)
            if pattern_confluence:
                confluence_methods.extend(pattern_confluence)
                total_score += len(pattern_confluence)
            
            # 5. Volume Analysis
            volume_confluence = self._check_volume_confluence(market_data, target_price)
            if volume_confluence:
                confluence_methods.extend(volume_confluence)
                total_score += len(volume_confluence)
            
            # 6. Harmonic Patterns
            harmonic_confluence = self._check_harmonic_confluence(market_data, target_price)
            if harmonic_confluence:
                confluence_methods.extend(harmonic_confluence)
                total_score += len(harmonic_confluence)
            
            return {
                'score': total_score,
                'methods': confluence_methods,
                'details': {
                    'fibonacci': fib_confluence or [],
                    'support_resistance': sr_confluence or [],
                    'momentum': momentum_confluence or [],
                    'patterns': pattern_confluence or [],
                    'volume': volume_confluence or [],
                    'harmonic': harmonic_confluence or []
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing confluence: {e}")
            return {'score': 0, 'methods': [], 'details': {}}
    
    def _check_fibonacci_confluence(self, market_data: pd.DataFrame, target_price: float) -> List[str]:
        """Check for Fibonacci level confluences"""
        confluences = []
        tolerance = 0.002  # 0.2% tolerance for Fibonacci levels
        
        try:
            # Find significant swing highs and lows
            swings = self._find_swing_points(market_data)
            
            for swing_high, swing_low in swings:
                swing_range = abs(swing_high - swing_low)
                
                # Check retracements
                for ratio in self.fib_retracements:
                    if swing_high > swing_low:  # Uptrend retracement
                        fib_level = swing_high - (swing_range * ratio)
                    else:  # Downtrend retracement
                        fib_level = swing_low + (swing_range * ratio)
                    
                    if abs(target_price - fib_level) / target_price <= tolerance:
                        confluences.append(f"Fibonacci {ratio:.1%} Retracement")
                
                # Check extensions
                for ratio in self.fib_extensions:
                    if swing_high > swing_low:  # Uptrend extension
                        fib_level = swing_low + (swing_range * ratio)
                    else:  # Downtrend extension
                        fib_level = swing_high - (swing_range * ratio)
                    
                    if abs(target_price - fib_level) / target_price <= tolerance:
                        confluences.append(f"Fibonacci {ratio:.1%} Extension")
            
            return list(set(confluences))  # Remove duplicates
            
        except Exception as e:
            self.logger.error(f"Error checking Fibonacci confluence: {e}")
            return []
    
    def _check_support_resistance_confluence(self, market_data: pd.DataFrame, target_price: float) -> List[str]:
        """Check for Support/Resistance confluences using Enhanced S/R Detector"""
        confluences = []
        tolerance = 0.003  # 0.3% tolerance for S/R levels
        
        try:
            if self.sr_detector:
                # Use Enhanced S/R Detector
                sr_results = self.sr_detector.detect_sr_levels(market_data)
                
                # Check support levels
                for level in sr_results.get('support_levels', []):
                    if abs(target_price - level.price) / target_price <= tolerance:
                        method_info = f" ({level.formation_method})" if level.formation_method != 'basic' else ""
                        confluences.append(
                            f"Support Level - {level.strength} touches, "
                            f"conviction: {level.conviction:.2f}{method_info}"
                        )
                
                # Check resistance levels
                for level in sr_results.get('resistance_levels', []):
                    if abs(target_price - level.price) / target_price <= tolerance:
                        method_info = f" ({level.formation_method})" if level.formation_method != 'basic' else ""
                        confluences.append(
                            f"Resistance Level - {level.strength} touches, "
                            f"conviction: {level.conviction:.2f}{method_info}"
                        )
                
                self.logger.debug(f"Enhanced S/R: Found {len(confluences)} confluences at {target_price}")
                
            else:
                # Fallback to basic S/R detection
                sr_levels = self._find_support_resistance_levels(market_data)
                
                for level in sr_levels:
                    if abs(target_price - level['price']) / target_price <= tolerance:
                        confluences.append(f"{level['type'].title()} Level ({level['strength']} touches)")
            
            # Check for supply/demand zones (basic implementation)
            supply_demand_zones = self._find_supply_demand_zones(market_data)
            
            for zone in supply_demand_zones:
                if zone['low'] <= target_price <= zone['high']:
                    confluences.append(f"{zone['type'].title()} Zone")
            
            return confluences
            
        except Exception as e:
            self.logger.error(f"Error checking S/R confluence: {e}")
            return []
    
    def _check_momentum_confluence(self, market_data: pd.DataFrame, target_price: float) -> List[str]:
        """Check for momentum indicator confluences"""
        confluences = []
        
        try:
            current_price = market_data['close'].iloc[-1]
            price_change_pct = (target_price - current_price) / current_price
            
            # Calculate technical indicators
            rsi = self._calculate_rsi(market_data['close'])
            macd_line, macd_signal, macd_hist = self._calculate_macd(market_data['close'])
            stoch_k, stoch_d = self._calculate_stochastic(market_data)
            
            current_rsi = rsi.iloc[-1]
            current_macd = macd_line.iloc[-1]
            current_signal = macd_signal.iloc[-1]
            current_stoch_k = stoch_k.iloc[-1]
            
            # RSI confluence
            if price_change_pct > 0:  # Bullish target
                if current_rsi < 70:  # Not overbought
                    confluences.append("RSI Not Overbought")
                if current_rsi < 30:  # Oversold bounce
                    confluences.append("RSI Oversold Bounce")
            else:  # Bearish target
                if current_rsi > 30:  # Not oversold
                    confluences.append("RSI Not Oversold")
                if current_rsi > 70:  # Overbought drop
                    confluences.append("RSI Overbought Drop")
            
            # MACD confluence
            if price_change_pct > 0 and current_macd > current_signal:
                confluences.append("MACD Bullish Crossover")
            elif price_change_pct < 0 and current_macd < current_signal:
                confluences.append("MACD Bearish Crossover")
            
            # Stochastic confluence
            if price_change_pct > 0 and current_stoch_k < 80:
                confluences.append("Stochastic Not Overbought")
            elif price_change_pct < 0 and current_stoch_k > 20:
                confluences.append("Stochastic Not Oversold")
            
            return confluences
            
        except Exception as e:
            self.logger.error(f"Error checking momentum confluence: {e}")
            return []
    
    def _check_chart_pattern_confluence(self, market_data: pd.DataFrame, target_price: float) -> List[str]:
        """Check for chart pattern confluences"""
        confluences = []
        
        try:
            # Simple pattern detection
            highs = market_data['high'].values
            lows = market_data['low'].values
            closes = market_data['close'].values
            
            # Double top/bottom detection
            if self._detect_double_top(highs, closes, target_price):
                confluences.append("Double Top Pattern")
            
            if self._detect_double_bottom(lows, closes, target_price):
                confluences.append("Double Bottom Pattern")
            
            # Head and shoulders detection
            if self._detect_head_shoulders(highs, closes, target_price):
                confluences.append("Head & Shoulders Pattern")
            
            # Triangle patterns
            triangle_type = self._detect_triangle_pattern(highs, lows)
            if triangle_type and self._triangle_target_matches(market_data, target_price):
                confluences.append(f"{triangle_type} Triangle Pattern")
            
            return confluences
            
        except Exception as e:
            self.logger.error(f"Error checking chart patterns: {e}")
            return []
    
    def _check_volume_confluence(self, market_data: pd.DataFrame, target_price: float) -> List[str]:
        """Check for volume-based confluences"""
        confluences = []
        
        try:
            volumes = market_data['volume'].values
            closes = market_data['close'].values
            current_price = closes[-1]
            
            # Volume profile analysis (simplified)
            volume_zones = self._calculate_volume_profile(market_data)
            
            for zone in volume_zones:
                if abs(target_price - zone['price']) / target_price <= 0.005:  # 0.5% tolerance
                    if zone['volume_pct'] > 0.1:  # High volume zone (>10% of total)
                        confluences.append(f"High Volume Node ({zone['volume_pct']:.1%})")
            
            # VWAP confluence
            vwap = self._calculate_vwap(market_data)
            if abs(target_price - vwap.iloc[-1]) / target_price <= 0.003:
                confluences.append("VWAP Level")
            
            # Volume spike analysis
            avg_volume = np.mean(volumes[-20:])  # 20-period average
            recent_volume = volumes[-1]
            
            if recent_volume > avg_volume * 1.5:  # 50% above average
                confluences.append("High Volume Confirmation")
            
            return confluences
            
        except Exception as e:
            self.logger.error(f"Error checking volume confluence: {e}")
            return []
    
    def _check_harmonic_confluence(self, market_data: pd.DataFrame, target_price: float) -> List[str]:
        """Check for harmonic pattern confluences"""
        confluences = []
        
        try:
            # Simplified harmonic pattern detection
            swings = self._find_swing_points(market_data, min_periods=5)
            
            if len(swings) >= 2:
                for i in range(len(swings) - 1):
                    swing_high, swing_low = swings[i]
                    
                    # Check for Gartley pattern ratios
                    if self._check_gartley_ratios(swing_high, swing_low, target_price):
                        confluences.append("Gartley Pattern PRZ")
                    
                    # Check for Bat pattern ratios
                    if self._check_bat_ratios(swing_high, swing_low, target_price):
                        confluences.append("Bat Pattern PRZ")
                    
                    # Check for Butterfly pattern ratios
                    if self._check_butterfly_ratios(swing_high, swing_low, target_price):
                        confluences.append("Butterfly Pattern PRZ")
            
            return confluences
            
        except Exception as e:
            self.logger.error(f"Error checking harmonic confluence: {e}")
            return []
    
    # Helper methods for technical calculations
    
    def _find_swing_points(self, market_data: pd.DataFrame, min_periods: int = 10) -> List[Tuple[float, float]]:
        """Find significant swing highs and lows"""
        try:
            highs = market_data['high'].values
            lows = market_data['low'].values
            
            swing_points = []
            
            # Simple peak/valley detection
            for i in range(min_periods, len(highs) - min_periods):
                # Check for swing high
                if all(highs[i] >= highs[j] for j in range(i - min_periods, i + min_periods + 1) if j != i):
                    # Find corresponding low before this high
                    low_before = min(lows[max(0, i - min_periods * 2):i])
                    swing_points.append((highs[i], low_before))
                
                # Check for swing low
                if all(lows[i] <= lows[j] for j in range(i - min_periods, i + min_periods + 1) if j != i):
                    # Find corresponding high before this low
                    high_before = max(highs[max(0, i - min_periods * 2):i])
                    swing_points.append((high_before, lows[i]))
            
            return swing_points[-5:]  # Return last 5 swing points
            
        except Exception as e:
            self.logger.error(f"Error finding swing points: {e}")
            return []
    
    def _find_support_resistance_levels(self, market_data: pd.DataFrame) -> List[Dict]:
        """Find horizontal support and resistance levels"""
        try:
            levels = []
            highs = market_data['high'].values
            lows = market_data['low'].values
            tolerance = 0.002  # 0.2% tolerance for level clustering
            
            # Find resistance levels (highs that were tested multiple times)
            unique_highs = []
            for high in highs:
                if not any(abs(high - uh) / high <= tolerance for uh in unique_highs):
                    touches = sum(1 for h in highs if abs(h - high) / high <= tolerance)
                    if touches >= 2:  # At least 2 touches
                        unique_highs.append(high)
                        levels.append({
                            'price': high,
                            'type': 'resistance',
                            'strength': touches
                        })
            
            # Find support levels (lows that were tested multiple times)
            unique_lows = []
            for low in lows:
                if not any(abs(low - ul) / low <= tolerance for ul in unique_lows):
                    touches = sum(1 for l in lows if abs(l - low) / low <= tolerance)
                    if touches >= 2:  # At least 2 touches
                        unique_lows.append(low)
                        levels.append({
                            'price': low,
                            'type': 'support',
                            'strength': touches
                        })
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Error finding S/R levels: {e}")
            return []
    
    def _find_supply_demand_zones(self, market_data: pd.DataFrame) -> List[Dict]:
        """Find supply and demand zones"""
        try:
            zones = []
            
            # Simplified supply/demand zone detection
            # Look for areas where price moved away quickly (high momentum)
            closes = market_data['close'].values
            volumes = market_data['volume'].values
            highs = market_data['high'].values
            lows = market_data['low'].values
            
            for i in range(20, len(closes) - 5):
                # Check for demand zone (strong move up from this area)
                move_up = closes[i + 5] / closes[i] - 1
                avg_volume = np.mean(volumes[i:i + 5])
                
                if move_up > 0.03 and avg_volume > np.mean(volumes[i - 10:i]):  # 3% move with high volume
                    zones.append({
                        'low': min(lows[i - 2:i + 3]),
                        'high': max(highs[i - 2:i + 3]),
                        'type': 'demand'
                    })
                
                # Check for supply zone (strong move down from this area)
                move_down = closes[i] / closes[i + 5] - 1
                
                if move_down > 0.03 and avg_volume > np.mean(volumes[i - 10:i]):  # 3% move with high volume
                    zones.append({
                        'low': min(lows[i - 2:i + 3]),
                        'high': max(highs[i - 2:i + 3]),
                        'type': 'supply'
                    })
            
            return zones[-10:]  # Return last 10 zones
            
        except Exception as e:
            self.logger.error(f"Error finding supply/demand zones: {e}")
            return []
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        try:
            if TALIB_AVAILABLE:
                return pd.Series(talib.RSI(prices.values, timeperiod=period), index=prices.index)
            else:
                # Manual RSI calculation
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return pd.Series(index=prices.index, data=50)  # Neutral RSI
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        try:
            if TALIB_AVAILABLE:
                macd, macd_signal, macd_hist = talib.MACD(prices.values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
                return (pd.Series(macd, index=prices.index), 
                       pd.Series(macd_signal, index=prices.index),
                       pd.Series(macd_hist, index=prices.index))
            else:
                # Manual MACD calculation
                ema_fast = prices.ewm(span=fast).mean()
                ema_slow = prices.ewm(span=slow).mean()
                macd_line = ema_fast - ema_slow
                macd_signal = macd_line.ewm(span=signal).mean()
                macd_hist = macd_line - macd_signal
                return macd_line, macd_signal, macd_hist
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            return pd.Series(index=prices.index, data=0), pd.Series(index=prices.index, data=0), pd.Series(index=prices.index, data=0)
    
    def _calculate_stochastic(self, market_data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        try:
            if TALIB_AVAILABLE:
                slowk, slowd = talib.STOCH(market_data['high'].values, market_data['low'].values, 
                                         market_data['close'].values, fastk_period=k_period, 
                                         slowk_period=d_period, slowd_period=d_period)
                return pd.Series(slowk, index=market_data.index), pd.Series(slowd, index=market_data.index)
            else:
                # Manual Stochastic calculation
                lowest_low = market_data['low'].rolling(window=k_period).min()
                highest_high = market_data['high'].rolling(window=k_period).max()
                k_percent = 100 * ((market_data['close'] - lowest_low) / (highest_high - lowest_low))
                d_percent = k_percent.rolling(window=d_period).mean()
                return k_percent, d_percent
        except Exception as e:
            self.logger.error(f"Error calculating Stochastic: {e}")
            return pd.Series(index=market_data.index, data=50), pd.Series(index=market_data.index, data=50)
    
    def _calculate_volume_profile(self, market_data: pd.DataFrame, bins: int = 20) -> List[Dict]:
        """Calculate simplified volume profile"""
        try:
            prices = market_data['close'].values
            volumes = market_data['volume'].values
            
            # Create price bins
            price_min, price_max = prices.min(), prices.max()
            price_bins = np.linspace(price_min, price_max, bins)
            
            volume_profile = []
            total_volume = volumes.sum()
            
            for i in range(len(price_bins) - 1):
                bin_low, bin_high = price_bins[i], price_bins[i + 1]
                bin_mask = (prices >= bin_low) & (prices < bin_high)
                bin_volume = volumes[bin_mask].sum()
                
                if bin_volume > 0:
                    volume_profile.append({
                        'price': (bin_low + bin_high) / 2,
                        'volume': bin_volume,
                        'volume_pct': bin_volume / total_volume
                    })
            
            return sorted(volume_profile, key=lambda x: x['volume'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error calculating volume profile: {e}")
            return []
    
    def _calculate_vwap(self, market_data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        try:
            typical_price = (market_data['high'] + market_data['low'] + market_data['close']) / 3
            volume = market_data['volume']
            return (typical_price * volume).cumsum() / volume.cumsum()
        except Exception as e:
            self.logger.error(f"Error calculating VWAP: {e}")
            return market_data['close']
    
    # Pattern detection helpers
    
    def _detect_double_top(self, highs: np.ndarray, closes: np.ndarray, target_price: float) -> bool:
        """Detect double top pattern"""
        try:
            if len(highs) < 20:
                return False
            
            # Find two recent peaks at similar levels
            peaks = []
            for i in range(10, len(highs) - 10):
                if all(highs[i] >= highs[j] for j in range(i - 5, i + 6) if j != i):
                    peaks.append((i, highs[i]))
            
            if len(peaks) >= 2:
                peak1, peak2 = peaks[-2], peaks[-1]
                if abs(peak1[1] - peak2[1]) / peak1[1] <= 0.02:  # Within 2%
                    neckline = min(closes[peak1[0]:peak2[0]])
                    target = neckline - (peak1[1] - neckline)
                    return abs(target - target_price) / target_price <= 0.03
            
            return False
        except:
            return False
    
    def _detect_double_bottom(self, lows: np.ndarray, closes: np.ndarray, target_price: float) -> bool:
        """Detect double bottom pattern"""
        try:
            if len(lows) < 20:
                return False
            
            # Find two recent troughs at similar levels
            troughs = []
            for i in range(10, len(lows) - 10):
                if all(lows[i] <= lows[j] for j in range(i - 5, i + 6) if j != i):
                    troughs.append((i, lows[i]))
            
            if len(troughs) >= 2:
                trough1, trough2 = troughs[-2], troughs[-1]
                if abs(trough1[1] - trough2[1]) / trough1[1] <= 0.02:  # Within 2%
                    neckline = max(closes[trough1[0]:trough2[0]])
                    target = neckline + (neckline - trough1[1])
                    return abs(target - target_price) / target_price <= 0.03
            
            return False
        except:
            return False
    
    def _detect_head_shoulders(self, highs: np.ndarray, closes: np.ndarray, target_price: float) -> bool:
        """Detect head and shoulders pattern"""
        try:
            # Simplified head and shoulders detection
            # Look for three peaks with middle one being highest
            if len(highs) < 30:
                return False
            
            peaks = []
            for i in range(10, len(highs) - 10):
                if all(highs[i] >= highs[j] for j in range(i - 5, i + 6) if j != i):
                    peaks.append((i, highs[i]))
            
            if len(peaks) >= 3:
                # Check last three peaks for head and shoulders
                left_shoulder, head, right_shoulder = peaks[-3], peaks[-2], peaks[-1]
                
                # Head should be higher than both shoulders
                if (head[1] > left_shoulder[1] and head[1] > right_shoulder[1] and
                    abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1] <= 0.05):
                    
                    # Calculate neckline and target
                    neckline = min(closes[left_shoulder[0]:right_shoulder[0]])
                    target = neckline - (head[1] - neckline)
                    return abs(target - target_price) / target_price <= 0.03
            
            return False
        except:
            return False
    
    def _detect_triangle_pattern(self, highs: np.ndarray, lows: np.ndarray) -> Optional[str]:
        """Detect triangle patterns"""
        try:
            if len(highs) < 20:
                return None
            
            # Find recent highs and lows
            recent_highs = highs[-20:]
            recent_lows = lows[-20:]
            
            # Calculate trends in highs and lows
            high_trend = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
            low_trend = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]
            
            # Determine triangle type
            if high_trend < -0.001 and low_trend > 0.001:
                return "Symmetrical"
            elif high_trend < -0.001 and abs(low_trend) <= 0.001:
                return "Descending"
            elif abs(high_trend) <= 0.001 and low_trend > 0.001:
                return "Ascending"
            
            return None
        except:
            return None
    
    def _triangle_target_matches(self, market_data: pd.DataFrame, target_price: float) -> bool:
        """Check if target matches triangle breakout projection"""
        try:
            current_price = market_data['close'].iloc[-1]
            highs = market_data['high'].values[-20:]
            lows = market_data['low'].values[-20:]
            
            triangle_height = max(highs) - min(lows)
            
            # Check both breakout directions
            upside_target = current_price + triangle_height
            downside_target = current_price - triangle_height
            
            return (abs(target_price - upside_target) / upside_target <= 0.05 or
                   abs(target_price - downside_target) / abs(downside_target) <= 0.05)
        except:
            return False
    
    # Harmonic pattern helpers
    
    def _check_gartley_ratios(self, swing_high: float, swing_low: float, target_price: float) -> bool:
        """Check if target aligns with Gartley pattern ratios"""
        try:
            # Gartley pattern: 0.618 retracement of XA, then 0.786 projection
            xa_range = abs(swing_high - swing_low)
            gartley_target = swing_low + (xa_range * 0.786)
            return abs(target_price - gartley_target) / target_price <= 0.02
        except:
            return False
    
    def _check_bat_ratios(self, swing_high: float, swing_low: float, target_price: float) -> bool:
        """Check if target aligns with Bat pattern ratios"""
        try:
            # Bat pattern: 0.382 or 0.5 retracement, then 0.886 projection
            xa_range = abs(swing_high - swing_low)
            bat_target = swing_low + (xa_range * 0.886)
            return abs(target_price - bat_target) / target_price <= 0.02
        except:
            return False
    
    def _check_butterfly_ratios(self, swing_high: float, swing_low: float, target_price: float) -> bool:
        """Check if target aligns with Butterfly pattern ratios"""
        try:
            # Butterfly pattern: 0.786 retracement, then 1.27 or 1.618 projection
            xa_range = abs(swing_high - swing_low)
            butterfly_target1 = swing_low + (xa_range * 1.27)
            butterfly_target2 = swing_low + (xa_range * 1.618)
            return (abs(target_price - butterfly_target1) / target_price <= 0.02 or
                   abs(target_price - butterfly_target2) / target_price <= 0.02)
        except:
            return False
    
    # Scoring and confidence calculation
    
    def _get_confidence_level(self, score: int) -> str:
        """Convert confluence score to confidence level"""
        if score >= self.high_confidence_threshold:
            return "HIGH"
        elif score >= self.medium_confidence_threshold:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _calculate_probability(self, score: int) -> float:
        """Calculate probability based on confluence score"""
        # Non-linear scaling: more confluences = exponentially higher probability
        base_prob = 0.3  # 30% base probability
        max_prob = 0.85  # 85% maximum probability
        
        if score == 0:
            return base_prob
        
        # Exponential scaling
        scaled_score = min(score / 8.0, 1.0)  # Normalize to 0-1 with max at 8 confluences
        probability = base_prob + (max_prob - base_prob) * (scaled_score ** 0.7)
        
        return round(probability, 3)
    
    def _calculate_risk_reward(self, current_price: float, target_price: float, stop_loss_pct: float = 0.02) -> float:
        """Calculate risk-reward ratio"""
        try:
            price_change = abs(target_price - current_price)
            stop_loss_distance = current_price * stop_loss_pct
            
            if stop_loss_distance == 0:
                return 0.0
            
            return round(price_change / stop_loss_distance, 2)
        except:
            return 0.0

    def format_analysis_results(self, target_zones: List[TargetZone], 
                              market_summary: Dict) -> Dict:
        """Format analysis results for output"""
        try:
            # Categorize by confidence
            high_confidence = [tz for tz in target_zones if tz.confidence_level == "HIGH"]
            medium_confidence = [tz for tz in target_zones if tz.confidence_level == "MEDIUM"]
            low_confidence = [tz for tz in target_zones if tz.confidence_level == "LOW"]
            
            return {
                'market_summary': market_summary,
                'analysis_timestamp': datetime.now().isoformat(),
                'total_targets': len(target_zones),
                'high_confidence_targets': len(high_confidence),
                'medium_confidence_targets': len(medium_confidence),
                'low_confidence_targets': len(low_confidence),
                'target_zones': {
                    'high_confidence': [self._format_target_zone(tz) for tz in high_confidence],
                    'medium_confidence': [self._format_target_zone(tz) for tz in medium_confidence],
                    'low_confidence': [self._format_target_zone(tz) for tz in low_confidence]
                },
                'summary': {
                    'best_target': self._format_target_zone(target_zones[0]) if target_zones else None,
                    'avg_confluence_score': round(np.mean([tz.confluence_score for tz in target_zones]), 1) if target_zones else 0,
                    'avg_probability': round(np.mean([tz.probability for tz in target_zones]), 3) if target_zones else 0
                }
            }
        except Exception as e:
            self.logger.error(f"Error formatting results: {e}")
            return {'error': str(e)}
    
    def _format_target_zone(self, target_zone: TargetZone) -> Dict:
        """Format a single target zone for output"""
        current_price = 100  # This should be passed from the calling function
        price_change_pct = ((target_zone.price_level - current_price) / current_price) * 100
        
        return {
            'price_level': round(target_zone.price_level, 6),
            'wave_target': target_zone.wave_target,
            'elliott_basis': target_zone.elliott_basis,
            'confluence_score': target_zone.confluence_score,
            'confidence_level': target_zone.confidence_level,
            'probability': f"{target_zone.probability:.1%}",
            'price_change_pct': f"{price_change_pct:+.2f}%",
            'risk_reward_ratio': f"{target_zone.risk_reward_ratio}:1",
            'confluences': target_zone.confluences,
            'timeframe': target_zone.timeframe
        }
