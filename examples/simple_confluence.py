"""
Simple Confluence Analysis System
Creates basic but visible confluence targets for Elliott Wave analysis
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any

class SimpleConfluenceTarget:
    def __init__(self, price: float, name: str, methods: List[str], score: int):
        self.price = price
        self.name = name
        self.methods = methods
        self.score = score
        self.confidence = "HIGH" if score >= 3 else "MEDIUM" if score >= 2 else "LOW"

class SimpleConfluenceAnalyzer:
    def __init__(self):
        self.fib_ratios = [0.618, 1.0, 1.272, 1.618, 2.0, 2.618]
    
    def analyze_simple_targets(self, market_data: pd.DataFrame, waves: List) -> List[SimpleConfluenceTarget]:
        """Generate simple but visible confluence targets"""
        targets = []
        current_price = market_data['close'].iloc[-1]
        
        if not waves:
            # Create basic percentage targets if no waves
            for pct in [5, 10, 15, 20]:
                up_target = current_price * (1 + pct/100)
                down_target = current_price * (1 - pct/100)
                
                targets.append(SimpleConfluenceTarget(
                    price=up_target,
                    name=f"Upside Target {pct}%",
                    methods=[f"{pct}% Extension", "Basic Target"],
                    score=2
                ))
                
                targets.append(SimpleConfluenceTarget(
                    price=down_target,
                    name=f"Downside Target {pct}%",
                    methods=[f"{pct}% Retracement", "Support Level"],
                    score=2
                ))
            
            return sorted(targets, key=lambda x: abs(x.price - current_price))[:5]
        
        # Analyze with waves
        try:
            # Find highest and lowest prices for Fibonacci analysis
            high_price = market_data['high'].max()
            low_price = market_data['low'].min()
            price_range = high_price - low_price
            
            # Create Fibonacci targets
            for ratio in self.fib_ratios:
                # Upward projections from low
                fib_up = low_price + (price_range * ratio)
                # Downward projections from high  
                fib_down = high_price - (price_range * ratio)
                
                if fib_up > current_price * 0.8 and fib_up < current_price * 1.5:
                    methods = [f"Fibonacci {ratio}", "Wave Extension"]
                    if self._near_round_number(fib_up):
                        methods.append("Psychological Level")
                    
                    targets.append(SimpleConfluenceTarget(
                        price=fib_up,
                        name=f"Fib {ratio} Upside",
                        methods=methods,
                        score=len(methods)
                    ))
                
                if fib_down > current_price * 0.5 and fib_down < current_price * 1.2:
                    methods = [f"Fibonacci {ratio}", "Retracement"]
                    if self._near_round_number(fib_down):
                        methods.append("Psychological Level")
                    
                    targets.append(SimpleConfluenceTarget(
                        price=fib_down,
                        name=f"Fib {ratio} Downside", 
                        methods=methods,
                        score=len(methods)
                    ))
            
            # Add support/resistance levels
            sr_levels = self._find_simple_sr_levels(market_data)
            for level in sr_levels:
                methods = ["Support/Resistance"]
                if self._near_round_number(level):
                    methods.append("Psychological Level")
                
                targets.append(SimpleConfluenceTarget(
                    price=level,
                    name="S/R Level",
                    methods=methods,
                    score=len(methods)
                ))
            
            # Sort by confluence score and distance from current price
            targets.sort(key=lambda x: (x.score, -abs(x.price - current_price)), reverse=True)
            return targets[:8]  # Return top 8 targets
            
        except Exception as e:
            print(f"Error in confluence analysis: {e}")
            return []
    
    def _find_simple_sr_levels(self, data: pd.DataFrame) -> List[float]:
        """Find basic support and resistance levels"""
        try:
            highs = data['high'].rolling(window=10).max()
            lows = data['low'].rolling(window=10).min()
            
            # Find recent highs and lows
            recent_highs = highs.tail(50).unique()
            recent_lows = lows.tail(50).unique()
            
            levels = []
            current_price = data['close'].iloc[-1]
            
            # Add significant levels within reasonable range
            for high in recent_highs:
                if not np.isnan(high) and current_price * 0.8 < high < current_price * 1.3:
                    levels.append(high)
            
            for low in recent_lows:
                if not np.isnan(low) and current_price * 0.7 < low < current_price * 1.2:
                    levels.append(low)
            
            return list(set(levels))[:5]  # Return unique levels
        except:
            return []
    
    def _near_round_number(self, price: float) -> bool:
        """Check if price is near a round number"""
        # Check for round numbers based on price level
        if price < 1:
            return abs(price - round(price, 1)) < 0.01
        elif price < 10:
            return abs(price - round(price)) < 0.1
        elif price < 100:
            return abs(price - round(price/5)*5) < 0.5
        else:
            return abs(price - round(price/10)*10) < 1.0
