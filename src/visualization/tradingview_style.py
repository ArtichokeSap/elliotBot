"""
TradingView-style Elliott Wave Visualization
Creates professional Elliott Wave charts matching TradingView appearance
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import List, Dict, Any, Optional, Tuple
import logging

from ..utils.logger import get_logger
from ..analysis.wave_detector import Wave, WaveType, TrendDirection
from ..analysis.fibonacci import FibonacciAnalysis, FibonacciLevel

logger = get_logger(__name__)


class TradingViewStyleVisualizer:
    """
    TradingView-style Elliott Wave visualizer with professional appearance
    """
    
    def __init__(self):
        """Initialize TradingView-style visualizer"""
        
        # TradingView-like color scheme
        self.colors = {
            # Wave colors - professional blue/red scheme
            'impulse_up': '#2962FF',      # TradingView blue
            'impulse_down': '#F23645',    # TradingView red
            'corrective_up': '#089981',   # TradingView green
            'corrective_down': '#FF6D00', # TradingView orange
            
            # Chart elements
            'background': '#FFFFFF',      # White background
            'grid': '#E1E3E7',           # Light gray grid
            'text': '#363A45',           # Dark gray text
            'border': '#D1D4DC',         # Border color
            
            # Fibonacci colors
            'fib_strong': '#FF5722',      # Strong level (red)
            'fib_medium': '#FF9800',      # Medium level (orange)
            'fib_weak': '#FFC107',        # Weak level (yellow)
            
            # Support/Resistance
            'support': '#4CAF50',         # Green
            'resistance': '#F44336',      # Red
            'invalidation': '#E91E63',    # Pink for invalidation levels
            
            # Candlesticks
            'candle_up': '#26A69A',       # Teal for up candles
            'candle_down': '#EF5350',     # Red for down candles
            'candle_up_border': '#26A69A',
            'candle_down_border': '#EF5350',
            
            # Volume
            'volume_up': 'rgba(38, 166, 154, 0.6)',    # Transparent teal
            'volume_down': 'rgba(239, 83, 80, 0.6)',   # Transparent red
        }
        
        # Wave labeling mapping - professional style
        self.wave_labels = {
            # Primary degree waves (Roman numerals in parentheses)
            WaveType.IMPULSE_1: "(i)",
            WaveType.IMPULSE_2: "(ii)", 
            WaveType.IMPULSE_3: "(iii)",
            WaveType.IMPULSE_4: "(iv)",
            WaveType.IMPULSE_5: "(v)",
            
            # Intermediate degree waves (numbers)
            # We'll use these for smaller timeframes
            WaveType.CORRECTIVE_A: "a",
            WaveType.CORRECTIVE_B: "b", 
            WaveType.CORRECTIVE_C: "c",
        }
        
        # Alternative labeling for different degrees
        self.alternative_labels = {
            'primary': {
                WaveType.IMPULSE_1: "1",
                WaveType.IMPULSE_2: "2", 
                WaveType.IMPULSE_3: "3",
                WaveType.IMPULSE_4: "4",
                WaveType.IMPULSE_5: "5",
                WaveType.CORRECTIVE_A: "A",
                WaveType.CORRECTIVE_B: "B", 
                WaveType.CORRECTIVE_C: "C",
            },
            'intermediate': {
                WaveType.IMPULSE_1: "(1)",
                WaveType.IMPULSE_2: "(2)", 
                WaveType.IMPULSE_3: "(3)",
                WaveType.IMPULSE_4: "(4)",
                WaveType.IMPULSE_5: "(5)",
                WaveType.CORRECTIVE_A: "(A)",
                WaveType.CORRECTIVE_B: "(B)", 
                WaveType.CORRECTIVE_C: "(C)",
            }
        }
        
        logger.info("TradingView-style visualizer initialized")
    
    def create_professional_chart(
        self, 
        data: pd.DataFrame, 
        waves: List[Wave],
        symbol: str = "Chart",
        fibonacci_analysis: Optional[FibonacciAnalysis] = None,
        degree: str = "primary",
        show_invalidation: bool = True,
        show_validation: bool = True,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create a professional TradingView-style Elliott Wave chart with rule validation
        
        Args:
            data: OHLCV DataFrame
            waves: List of detected waves
            symbol: Symbol name for title
            fibonacci_analysis: Optional Fibonacci analysis
            degree: Wave degree ('primary', 'intermediate', 'minor')
            show_invalidation: Whether to show invalidation levels
            show_validation: Whether to show rule validation indicators
            save_path: Optional path to save HTML
            
        Returns:
            Plotly figure
        """
        try:
            # Ensure proper column names
            data = self._normalize_columns(data)
            
            # Create subplot structure
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                row_heights=[0.8, 0.2],
                subplot_titles=(
                    f'{symbol} - Elliott Wave Analysis (Rule-Validated)',
                    'Volume'
                )
            )
            
            # 1. Add candlestick chart with TradingView styling
            self._add_candlestick_chart(fig, data, symbol)
            
            # 2. Add Elliott Wave analysis with validation
            if show_validation:
                self._add_validated_elliott_waves(fig, data, waves, degree, show_invalidation)
            else:
                self._add_elliott_waves(fig, data, waves, degree, show_invalidation)
            
            # 3. Add Fibonacci levels
            if fibonacci_analysis:
                self._add_fibonacci_levels(fig, fibonacci_analysis, data)
            
            # 4. Add volume chart
            self._add_volume_chart(fig, data)
            
            # 5. Apply TradingView styling
            self._apply_tradingview_styling(fig, symbol)
            
            # Save if path provided
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Professional chart saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating professional chart: {e}")
            raise
    
    def _normalize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to lowercase"""
        column_mapping = {
            'Open': 'open', 'High': 'high', 'Low': 'low', 
            'Close': 'close', 'Volume': 'volume',
            'Adj Close': 'adj_close'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in data.columns:
                data = data.rename(columns={old_col: new_col})
        
        return data
    
    def _add_candlestick_chart(self, fig: go.Figure, data: pd.DataFrame, symbol: str):
        """Add professional candlestick chart"""
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name=symbol,
                increasing_line_color=self.colors['candle_up_border'],
                decreasing_line_color=self.colors['candle_down_border'],
                increasing_fillcolor=self.colors['candle_up'],
                decreasing_fillcolor=self.colors['candle_down'],
                line=dict(width=1),
                opacity=0.9
            ),
            row=1, col=1
        )
    
    def _add_elliott_waves(self, fig: go.Figure, waves: List[Wave], data: pd.DataFrame, degree: str):
        """Add Elliott Wave lines and labels with professional styling"""
        
        # Get appropriate labels for the degree
        if degree in self.alternative_labels:
            labels = self.alternative_labels[degree]
        else:
            labels = self.wave_labels
        
        # Track all wave points for the main trend line
        wave_points = []
        
        for i, wave in enumerate(waves):
            try:
                # Get wave information
                start_time = wave.start_point.timestamp
                end_time = wave.end_point.timestamp
                start_price = wave.start_point.price
                end_price = wave.end_point.price
                wave_type = wave.wave_type
                
                # Get wave label
                wave_label = labels.get(wave_type, "?")
                
                # Determine color based on wave type and direction
                color = self._get_wave_color(wave)
                
                # Store points for trend line
                wave_points.extend([(start_time, start_price), (end_time, end_price)])
                
                # Add wave line with professional styling
                fig.add_trace(
                    go.Scatter(
                        x=[start_time, end_time],
                        y=[start_price, end_price],
                        mode='lines',
                        line=dict(
                            color=color, 
                            width=2,
                            dash='solid'
                        ),
                        name=f'Wave {wave_label}',
                        showlegend=False,
                        hovertemplate=(
                            f'<b>Wave {wave_label}</b><br>' +
                            'Start: %{x}<br>' +
                            'Price: $%{y:.2f}<br>' +
                            f'Confidence: {wave.confidence:.1%}<br>' +
                            f'Change: {wave.price_change_pct:.1%}' +
                            '<extra></extra>'
                        )
                    ),
                    row=1, col=1
                )
                
                # Add professional wave label
                self._add_wave_label(fig, wave, wave_label, color)
                
            except Exception as e:
                logger.warning(f"Error adding wave {i}: {e}")
                continue
        
                # Add main trend line connecting all waves
        if len(wave_points) > 1:
            self._add_trend_line(fig, wave_points)
    
    def _add_validated_elliott_waves(self, fig: go.Figure, data: pd.DataFrame, waves: List[Wave], degree: str, show_invalidation: bool):
        """Add Elliott Waves with validation indicators and rule compliance"""
        try:
            # Import validator
            from ..analysis.elliott_wave_validator import ElliottWaveValidator
            validator = ElliottWaveValidator()
            
            # Get appropriate labels for the degree
            if degree in self.alternative_labels:
                labels = self.alternative_labels[degree]
            else:
                labels = self.wave_labels
            
            # Group waves into structures for validation
            validated_structures = []
            
            # Look for impulse patterns (5 waves)
            for i in range(len(waves) - 4):
                impulse_candidate = waves[i:i+5]
                if all(w.wave_type.value in ['1', '2', '3', '4', '5'] for w in impulse_candidate):
                    structure = validator.validate_impulse_structure(impulse_candidate, data)
                    validated_structures.append((structure, i, i+5))
            
            # Look for corrective patterns (3 waves)  
            for i in range(len(waves) - 2):
                corrective_candidate = waves[i:i+3]
                if all(w.wave_type.value in ['A', 'B', 'C'] for w in corrective_candidate):
                    structure = validator.validate_corrective_structure(corrective_candidate, data)
                    validated_structures.append((structure, i, i+3))
            
            # Track all wave points for the main trend line
            wave_points = []
            validated_wave_indices = set()
            
            # Add validated structures first (these get special styling)
            for structure, start_idx, end_idx in validated_structures:
                if structure.validation_score > 0.6:  # High confidence structures
                    for j in range(start_idx, end_idx):
                        if j < len(waves):
                            validated_wave_indices.add(j)
                            wave = waves[j]
                            self._add_validated_wave(fig, wave, labels, structure.validation_score)
                            
                            # Store points for trend line
                            wave_points.extend([
                                (wave.start_point.timestamp, wave.start_point.price),
                                (wave.end_point.timestamp, wave.end_point.price)
                            ])
            
            # Add remaining waves with standard styling
            for i, wave in enumerate(waves):
                if i not in validated_wave_indices:
                    self._add_standard_wave(fig, wave, labels)
                    
                    # Store points for trend line
                    wave_points.extend([
                        (wave.start_point.timestamp, wave.start_point.price),
                        (wave.end_point.timestamp, wave.end_point.price)
                    ])
            
            # Add trend line
            if len(wave_points) > 1:
                self._add_trend_line(fig, wave_points)
            
            # Add validation indicators
            self._add_validation_indicators(fig, validated_structures, data)
            
            # Add invalidation levels if requested
            if show_invalidation:
                self._add_invalidation_levels(fig, waves, data)
                
        except Exception as e:
            logger.error(f"Error adding validated Elliott waves: {e}")
            # Fallback to standard wave display
            self._add_elliott_waves(fig, data, waves, degree, show_invalidation)
    
    def _add_validated_wave(self, fig: go.Figure, wave: Wave, labels: Dict, validation_score: float):
        """Add a validated wave with special styling indicating rule compliance"""
        
        wave_label = labels.get(wave.wave_type, "?")
        
        # Enhanced styling for validated waves
        if validation_score >= 0.8:
            color = self._get_wave_color(wave)
            width = 3
            opacity = 1.0
            icon = "✅"
        elif validation_score >= 0.6:
            color = self._get_wave_color(wave)
            width = 2.5
            opacity = 0.9
            icon = "⚠️"
        else:
            color = 'gray'
            width = 2
            opacity = 0.7
            icon = "❌"
        
        # Add wave line
        fig.add_trace(
            go.Scatter(
                x=[wave.start_point.timestamp, wave.end_point.timestamp],
                y=[wave.start_point.price, wave.end_point.price],
                mode='lines',
                line=dict(
                    color=color,
                    width=width
                ),
                opacity=opacity,
                name=f'Wave {wave_label} {icon}',
                showlegend=False,
                hovertemplate=(
                    f'<b>Wave {wave_label}</b> {icon}<br>' +
                    'Start: %{x}<br>' +
                    'Price: $%{y:.2f}<br>' +
                    f'Confidence: {wave.confidence:.1%}<br>' +
                    f'Validation Score: {validation_score:.1%}<br>' +
                    f'Change: {wave.price_change_pct:.1%}' +
                    '<extra></extra>'
                )
            ),
            row=1, col=1
        )
        
        # Add enhanced wave label
        self._add_validated_wave_label(fig, wave, wave_label, color, validation_score)
    
    def _add_standard_wave(self, fig: go.Figure, wave: Wave, labels: Dict):
        """Add a standard wave without validation indicators"""
        
        wave_label = labels.get(wave.wave_type, "?")
        color = self._get_wave_color(wave)
        
        # Add wave line
        fig.add_trace(
            go.Scatter(
                x=[wave.start_point.timestamp, wave.end_point.timestamp],
                y=[wave.start_point.price, wave.end_point.price],
                mode='lines',
                line=dict(
                    color=color,
                    width=1.5
                ),
                opacity=0.6,
                name=f'Wave {wave_label}',
                showlegend=False,
                hovertemplate=(
                    f'<b>Wave {wave_label}</b><br>' +
                    'Start: %{x}<br>' +
                    'Price: $%{y:.2f}<br>' +
                    f'Confidence: {wave.confidence:.1%}<br>' +
                    f'Change: {wave.price_change_pct:.1%}' +
                    '<extra></extra>'
                )
            ),
            row=1, col=1
        )
        
        # Add standard wave label
        self._add_wave_label(fig, wave, wave_label, color)
    
    def _add_validated_wave_label(self, fig: go.Figure, wave: Wave, label: str, color: str, validation_score: float):
        """Add wave label with validation indicator"""
        
        # Calculate label position (middle of wave)
        mid_time = wave.start_point.timestamp + (wave.end_point.timestamp - wave.start_point.timestamp) / 2
        mid_price = (wave.start_point.price + wave.end_point.price) / 2
        
        # Offset label above/below line based on wave direction
        price_offset = abs(wave.end_point.price - wave.start_point.price) * 0.02
        if wave.direction == TrendDirection.UP:
            label_price = mid_price + price_offset
        else:
            label_price = mid_price - price_offset
        
        # Add validation icon
        if validation_score >= 0.8:
            validation_icon = "✅"
        elif validation_score >= 0.6:
            validation_icon = "⚠️"
        else:
            validation_icon = "❌"
        
        # Add annotation
        fig.add_annotation(
            x=mid_time,
            y=label_price,
            text=f"<b>{label}</b> {validation_icon}",
            showarrow=False,
            font=dict(
                size=12,
                color=color,
                family="Arial Black"
            ),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=color,
            borderwidth=1,
            row=1, col=1
        )
    
    def _add_validation_indicators(self, fig: go.Figure, validated_structures, data: pd.DataFrame):
        """Add visual indicators for Elliott Wave rule compliance"""
        
        try:
            y_min = data['low'].min()
            y_max = data['high'].max()
            y_range = y_max - y_min
            
            for i, (structure, start_idx, end_idx) in enumerate(validated_structures[:3]):  # Show top 3
                if structure.validation_score > 0.6:
                    
                    # Position validation box
                    box_y = y_max - (i * 0.15 * y_range)
                    
                    # Add validation score annotation
                    score_text = f"Structure {i+1}: {structure.validation_score:.1%}"
                    if structure.validation_score >= 0.8:
                        box_color = "lightgreen"
                        score_text += " ✅"
                    elif structure.validation_score >= 0.6:
                        box_color = "lightyellow" 
                        score_text += " ⚠️"
                    else:
                        box_color = "lightcoral"
                        score_text += " ❌"
                    
                    fig.add_annotation(
                        x=data.index[-len(data)//4],  # Position on right side
                        y=box_y,
                        text=score_text,
                        showarrow=False,
                        font=dict(size=10, color="black"),
                        bgcolor=box_color,
                        bordercolor="gray",
                        borderwidth=1,
                        row=1, col=1
                    )
                    
        except Exception as e:
            logger.warning(f"Error adding validation indicators: {e}")
    
    def _add_elliott_waves(self, fig: go.Figure, data: pd.DataFrame, waves: List[Wave], degree: str, show_invalidation: bool):
        """Add Elliott Wave lines and labels with professional styling (standard method)"""
        
        # Get appropriate labels for the degree
        if degree in self.alternative_labels:
            labels = self.alternative_labels[degree]
        else:
            labels = self.wave_labels
        
        # Track all wave points for the main trend line
        wave_points = []
        
        for i, wave in enumerate(waves):
            try:
                # Get wave information
                start_time = wave.start_point.timestamp
                end_time = wave.end_point.timestamp
                start_price = wave.start_point.price
                end_price = wave.end_point.price
                wave_type = wave.wave_type
                
                # Get wave label
                wave_label = labels.get(wave_type, "?")
                
                # Determine color based on wave type and direction
                color = self._get_wave_color(wave)
                
                # Store points for trend line
                wave_points.extend([(start_time, start_price), (end_time, end_price)])
                
                # Add wave line with professional styling
                fig.add_trace(
                    go.Scatter(
                        x=[start_time, end_time],
                        y=[start_price, end_price],
                        mode='lines',
                        line=dict(
                            color=color, 
                            width=2,
                            dash='solid'
                        ),
                        name=f'Wave {wave_label}',
                        showlegend=False,
                        hovertemplate=(
                            f'<b>Wave {wave_label}</b><br>' +
                            'Start: %{x}<br>' +
                            'Price: $%{y:.2f}<br>' +
                            f'Confidence: {wave.confidence:.1%}<br>' +
                            f'Change: {wave.price_change_pct:.1%}' +
                            '<extra></extra>'
                        )
                    ),
                    row=1, col=1
                )
                
                # Add professional wave label
                self._add_wave_label(fig, wave, wave_label, color)
                
            except Exception as e:
                logger.warning(f"Error adding wave {i}: {e}")
                continue
        
        # Add main trend line connecting all waves
        if len(wave_points) > 1:
            self._add_trend_line(fig, wave_points)
    
    def _get_wave_color(self, wave: Wave) -> str:
        """Get appropriate color for wave based on type and direction"""
        if wave.wave_type in [WaveType.IMPULSE_1, WaveType.IMPULSE_3, WaveType.IMPULSE_5]:
            # Impulse waves
            return self.colors['impulse_up'] if wave.direction == TrendDirection.UP else self.colors['impulse_down']
        else:
            # Corrective waves  
            return self.colors['corrective_up'] if wave.direction == TrendDirection.UP else self.colors['corrective_down']
    
    def _add_wave_label(self, fig: go.Figure, wave: Wave, label: str, color: str):
        """Add professional wave label annotation"""
        
        # Calculate label position
        start_time = wave.start_point.timestamp
        end_time = wave.end_point.timestamp
        start_price = wave.start_point.price
        end_price = wave.end_point.price
        
        # Position label at the end point with offset
        if wave.direction == TrendDirection.UP:
            # For upward waves, place label above the end point
            label_y = end_price * 1.015
            ay = -20
        else:
            # For downward waves, place label below the end point
            label_y = end_price * 0.985
            ay = 20
        
        fig.add_annotation(
            x=end_time,
            y=label_y,
            text=f"<b>{label}</b>",
            showarrow=True,
            arrowhead=0,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=color,
            ax=0,
            ay=ay,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor=color,
            borderwidth=1,
            font=dict(
                size=14, 
                color=color, 
                family="Arial, sans-serif"
            ),
            row=1,
            col=1
        )
    
    def _add_trend_line(self, fig: go.Figure, wave_points: List[Tuple]):
        """Add main trend line connecting wave points"""
        
        # Remove duplicates while preserving order
        unique_points = []
        seen = set()
        for point in wave_points:
            if point not in seen:
                unique_points.append(point)
                seen.add(point)
        
        if len(unique_points) < 2:
            return
        
        wave_x = [point[0] for point in unique_points]
        wave_y = [point[1] for point in unique_points]
        
        fig.add_trace(
            go.Scatter(
                x=wave_x,
                y=wave_y,
                mode='lines',
                line=dict(
                    color='rgba(54, 58, 69, 0.4)', 
                    width=1,
                    dash='dot'
                ),
                name='Elliott Wave Structure',
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
    
    def _add_invalidation_levels(self, fig: go.Figure, waves: List[Wave], data: pd.DataFrame):
        """Add invalidation levels for Elliott Wave rules"""
        
        if len(waves) < 2:
            return
        
        try:
            # Add key invalidation levels based on Elliott Wave rules
            for i, wave in enumerate(waves):
                if wave.wave_type == WaveType.IMPULSE_2 and i > 0:
                    # Wave 2 cannot retrace below start of wave 1
                    prev_wave = waves[i-1]
                    invalidation_price = prev_wave.start_point.price
                    
                    self._add_horizontal_line(
                        fig, invalidation_price, 
                        "Invalidation Level", 
                        self.colors['invalidation'],
                        dash='dash'
                    )
                
                elif wave.wave_type == WaveType.IMPULSE_4 and i > 2:
                    # Wave 4 cannot overlap with wave 1 territory
                    wave_1 = waves[i-3]  # Assuming sequential wave numbering
                    invalidation_price = wave_1.end_point.price
                    
                    self._add_horizontal_line(
                        fig, invalidation_price,
                        "Wave 4 Overlap Level",
                        self.colors['invalidation'],
                        dash='dashdot'
                    )
        
        except Exception as e:
            logger.warning(f"Error adding invalidation levels: {e}")
    
    def _add_horizontal_line(self, fig: go.Figure, price: float, name: str, color: str, dash: str = 'solid'):
        """Add horizontal line across the chart"""
        
        # Get x-axis range from data
        x_range = fig.data[0].x  # Assuming first trace is the candlestick
        
        fig.add_trace(
            go.Scatter(
                x=[x_range[0], x_range[-1]],
                y=[price, price],
                mode='lines',
                line=dict(color=color, width=1, dash=dash),
                name=name,
                showlegend=True,
                hovertemplate=f'{name}: ${price:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    def _add_fibonacci_levels(self, fig: go.Figure, fib_analysis: FibonacciAnalysis, data: pd.DataFrame):
        """Add Fibonacci retracement and extension levels"""
        
        x_range = [data.index[0], data.index[-1]]
        
        for level in fib_analysis.key_levels[:5]:  # Show top 5 levels
            # Determine line style and color based on importance
            if level.ratio in [0.382, 0.618]:
                color = self.colors['fib_strong']
                width = 2
            elif level.ratio in [0.236, 0.786]:
                color = self.colors['fib_medium'] 
                width = 1.5
            else:
                color = self.colors['fib_weak']
                width = 1
            
            # Add Fibonacci line
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=[level.price, level.price],
                    mode='lines',
                    line=dict(
                        color=color,
                        width=width,
                        dash='solid' if level.level_type == 'retracement' else 'dash'
                    ),
                    name=f"Fib {level.ratio:.1%}",
                    showlegend=True,
                    hovertemplate=f"Fibonacci {level.ratio:.1%}<br>Price: ${level.price:.2f}<extra></extra>"
                ),
                row=1, col=1
            )
            
            # Add level label on the right
            fig.add_annotation(
                x=data.index[-1],
                y=level.price,
                text=f"{level.ratio:.1%}",
                showarrow=False,
                xanchor='left',
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor=color,
                font=dict(size=10, color=color),
                row=1, col=1
            )
    
    def _add_volume_chart(self, fig: go.Figure, data: pd.DataFrame):
        """Add volume chart with color coding"""
        
        # Color volume bars based on price movement
        colors = []
        for i in range(len(data)):
            if data['close'].iloc[i] >= data['open'].iloc[i]:
                colors.append(self.colors['volume_up'])
            else:
                colors.append(self.colors['volume_down'])
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['volume'],
                name="Volume",
                marker_color=colors,
                showlegend=False,
                hovertemplate='Volume: %{y:,.0f}<extra></extra>'
            ),
            row=2, col=1
        )
    
    def _apply_tradingview_styling(self, fig: go.Figure, symbol: str):
        """Apply TradingView-like styling to the chart"""
        
        fig.update_layout(
            # Overall layout
            title=dict(
                text=f"<b>{symbol} - Elliott Wave Analysis</b>",
                x=0.02,
                y=0.98,
                font=dict(size=18, color=self.colors['text'], family="Arial, sans-serif")
            ),
            
            # Background and appearance
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            
            # Dimensions
            height=700,
            width=1200,
            
            # Remove range slider
            xaxis_rangeslider_visible=False,
            
            # Hover mode
            hovermode='x unified',
            
            # Legend styling
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.98,
                xanchor="left", 
                x=1.01,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor=self.colors['border'],
                borderwidth=1,
                font=dict(size=10, color=self.colors['text'])
            ),
            
            # Margins
            margin=dict(l=50, r=120, t=50, b=50)
        )
        
        # Style price axis
        fig.update_yaxes(
            title_text="Price ($)",
            showgrid=True,
            gridcolor=self.colors['grid'],
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor=self.colors['border'],
            mirror=True,
            tickformat='$,.2f',
            tickfont=dict(size=10, color=self.colors['text']),
            row=1, col=1
        )
        
        # Style volume axis
        fig.update_yaxes(
            title_text="Volume",
            showgrid=True,
            gridcolor=self.colors['grid'],
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor=self.colors['border'],
            mirror=True,
            tickformat=',.0f',
            tickfont=dict(size=10, color=self.colors['text']),
            row=2, col=1
        )
        
        # Style time axis
        fig.update_xaxes(
            title_text="",
            showgrid=True,
            gridcolor=self.colors['grid'],
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor=self.colors['border'],
            mirror=True,
            tickfont=dict(size=10, color=self.colors['text'])
        )
        
        # Remove subplot titles for cleaner look
        fig.layout.annotations = tuple([
            ann for ann in fig.layout.annotations 
            if not (hasattr(ann, 'text') and 'Volume' in str(ann.text))
        ])


def create_tradingview_chart(
    data: pd.DataFrame, 
    waves: List[Wave], 
    symbol: str = "CHART",
    fibonacci_analysis: Optional[FibonacciAnalysis] = None,
    degree: str = "primary",
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Convenience function to create TradingView-style Elliott Wave chart
    
    Args:
        data: OHLCV DataFrame
        waves: List of Elliott Waves
        symbol: Chart symbol/title
        fibonacci_analysis: Optional Fibonacci analysis
        degree: Wave degree labeling ('primary', 'intermediate', 'minor')
        save_path: Optional path to save HTML file
        
    Returns:
        Plotly figure object
    """
    visualizer = TradingViewStyleVisualizer()
    return visualizer.create_professional_chart(
        data=data,
        waves=waves,
        symbol=symbol,
        fibonacci_analysis=fibonacci_analysis,
        degree=degree,
        save_path=save_path
    )
