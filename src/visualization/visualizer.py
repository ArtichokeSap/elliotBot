"""
Visualization module for Elliott Wave analysis using Plotly.
Enhanced with TradingView-style professional charts.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import List, Dict, Any, Optional, Tuple
import logging

from ..utils.logger import get_logger
from ..utils.config import get_config
from ..analysis.wave_detector import Wave, WaveType, TrendDirection
from ..analysis.fibonacci import FibonacciAnalysis, FibonacciLevel
from ..analysis.wave_validator import WaveValidator, WaveValidation, ValidationSeverity
from ..analysis.wave_projector import WaveProjector, ProjectionScenario, WaveProjection, ProjectionType, ProjectionConfidence
from ..analysis.pattern_memory import PatternMemory, PatternMatch, HistoricalPattern, PatternCategory
from .tradingview_style import TradingViewStyleVisualizer, create_tradingview_chart

logger = get_logger(__name__)


class WaveVisualizer:
    """
    Main visualization class for Elliott Wave analysis.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize visualizer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.theme = self.config.get('visualization.default_theme', 'plotly_dark')
        self.show_fibonacci = self.config.get('visualization.show_fibonacci', True)
        self.show_volume = self.config.get('visualization.show_volume', True)
        self.chart_height = self.config.get('visualization.chart_height', 800)
        self.chart_width = self.config.get('visualization.chart_width', 1200)
        
        # Enhanced color scheme with TradingView-like colors
        self.colors = {
            'impulse_up': '#2962FF',      # Professional blue for upward impulse waves
            'impulse_down': '#F23645',    # Professional red for downward impulse waves
            'corrective_up': '#089981',   # Professional green for upward corrective waves
            'corrective_down': '#FF6D00', # Professional orange for downward corrective waves
            'fibonacci': '#FF5722',       # Strong red for Fibonacci levels
            'support': '#4CAF50',         # Green for support levels
            'resistance': '#F44336',      # Red for resistance levels
            'volume': '#9E9E9E',          # Gray for volume
            'invalidation': '#E91E63',    # Pink for invalidation levels
            'background': '#FFFFFF',      # White background
            'grid': '#E1E3E7',           # Light gray grid
            'text': '#363A45'            # Dark text
        }
        
        logger.info("WaveVisualizer initialized")
    
    def create_tradingview_chart(
        self, 
        data: pd.DataFrame, 
        waves: List[Wave],
        fibonacci_analysis: Optional[FibonacciAnalysis] = None,
        title: str = "Elliott Wave Analysis",
        degree: str = "primary",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create a professional TradingView-style Elliott Wave chart.
        
        Args:
            data: OHLCV DataFrame
            waves: List of detected waves
            fibonacci_analysis: Optional Fibonacci analysis
            title: Chart title/symbol
            degree: Wave degree ('primary', 'intermediate', 'minor')
            save_path: Optional path to save the chart
            
        Returns:
            Plotly figure
        """
        try:
            return create_tradingview_chart(
                data=data,
                waves=waves,
                symbol=title,
                fibonacci_analysis=fibonacci_analysis,
                degree=degree,
                save_path=save_path
            )
        except Exception as e:
            logger.error(f"Error creating TradingView-style chart: {e}")
            # Fallback to standard chart
            return self.plot_waves(data, waves, fibonacci_analysis, title, save_path)
    
    def plot_waves(
        self, 
        data: pd.DataFrame, 
        waves: List[Wave],
        fibonacci_analysis: Optional[FibonacciAnalysis] = None,
        title: str = "Elliott Wave Analysis",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create main Elliott Wave chart with price data and wave annotations.
        
        Args:
            data: OHLCV DataFrame
            waves: List of detected waves
            fibonacci_analysis: Optional Fibonacci analysis
            title: Chart title
            save_path: Optional path to save the chart
            
        Returns:
            Plotly figure
        """
        try:
            # Create subplots
            if self.show_volume:
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02,
                    row_heights=[0.7, 0.3],
                    subplot_titles=(title, "Volume")
                )
            else:
                fig = go.Figure()
            
            # Add candlestick chart
            candlestick = go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name="Price",
                increasing_line_color='#00ff00',
                decreasing_line_color='#ff0000'
            )
            
            if self.show_volume:
                fig.add_trace(candlestick, row=1, col=1)
            else:
                fig.add_trace(candlestick)
            
            # Add wave annotations
            if waves:
                self._add_wave_annotations(fig, waves, data, row=1 if self.show_volume else None)
            
            # Add Fibonacci levels
            if fibonacci_analysis and self.show_fibonacci:
                self._add_fibonacci_levels(fig, fibonacci_analysis, data, row=1 if self.show_volume else None)
            
            # Add volume if enabled
            if self.show_volume:
                self._add_volume_chart(fig, data, row=2)
            
            # Update layout
            self._update_layout(fig, title)
            
            # Save if path provided
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Chart saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating wave chart: {e}")
            raise
    
    def _add_wave_annotations(self, fig: go.Figure, waves: List[Wave], data: pd.DataFrame, row: Optional[int] = None):
        """
        Add wave lines and labels to the chart.
        
        Args:
            fig: Plotly figure
            waves: List of waves
            data: OHLCV DataFrame
            row: Row number for subplot
        """
        for wave in waves:
            # Determine color based on wave type and direction
            color = self._get_wave_color(wave)
            
            # Draw wave line
            fig.add_trace(
                go.Scatter(
                    x=[wave.start_point.timestamp, wave.end_point.timestamp],
                    y=[wave.start_point.price, wave.end_point.price],
                    mode='lines',
                    line=dict(color=color, width=2),
                    name=f"Wave {wave.wave_type.value}",
                    showlegend=False,
                    hovertemplate=f"Wave {wave.wave_type.value}<br>" +
                                 f"Confidence: {wave.confidence:.2f}<br>" +
                                 f"Start: {wave.start_point.price:.2f}<br>" +
                                 f"End: {wave.end_point.price:.2f}<br>" +
                                 f"Change: {wave.price_change_pct:.2%}<extra></extra>"
                ),
                row=row, col=1
            )
            
            # Add wave label
            mid_timestamp = wave.start_point.timestamp + (wave.end_point.timestamp - wave.start_point.timestamp) / 2
            mid_price = (wave.start_point.price + wave.end_point.price) / 2
            
            fig.add_annotation(
                x=mid_timestamp,
                y=mid_price,
                text=f"<b>{wave.wave_type.value}</b>",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor=color,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor=color,
                font=dict(size=12, color='black'),
                row=row,
                col=1
            )
    
    def _get_wave_color(self, wave: Wave) -> str:
        """
        Get color for wave based on type and direction.
        
        Args:
            wave: Wave object
            
        Returns:
            Color string
        """
        if wave.wave_type in [WaveType.IMPULSE_1, WaveType.IMPULSE_3, WaveType.IMPULSE_5]:
            return self.colors['impulse_up'] if wave.direction == TrendDirection.UP else self.colors['impulse_down']
        else:
            return self.colors['corrective_up'] if wave.direction == TrendDirection.UP else self.colors['corrective_down']
    
    def _add_fibonacci_levels(self, fig: go.Figure, fib_analysis: FibonacciAnalysis, data: pd.DataFrame, row: Optional[int] = None):
        """
        Add Fibonacci retracement and extension levels to the chart.
        
        Args:
            fig: Plotly figure
            fib_analysis: Fibonacci analysis
            data: OHLCV DataFrame
            row: Row number for subplot
        """
        x_range = [data.index[0], data.index[-1]]
        
        # Add key Fibonacci levels
        for level in fib_analysis.key_levels:
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=[level.price, level.price],
                    mode='lines',
                    line=dict(
                        color=self.colors['fibonacci'],
                        width=1,
                        dash='dash' if level.level_type == 'extension' else 'solid'
                    ),
                    name=f"Fib {level.ratio:.3f}",
                    showlegend=False,
                    hovertemplate=f"Fibonacci {level.ratio:.1%}<br>" +
                                 f"Price: {level.price:.2f}<br>" +
                                 f"Type: {level.level_type}<extra></extra>"
                ),
                row=row, col=1
            )
            
            # Add level label
            fig.add_annotation(
                x=data.index[-1],
                y=level.price,
                text=f"{level.ratio:.1%}",
                showarrow=False,
                xanchor='left',
                bgcolor='rgba(255,215,0,0.8)',
                font=dict(size=10, color='black'),
                row=row,
                col=1
            )
    
    def _add_volume_chart(self, fig: go.Figure, data: pd.DataFrame, row: int):
        """
        Add volume chart to subplot.
        
        Args:
            fig: Plotly figure
            data: OHLCV DataFrame
            row: Row number
        """
        # Color volume bars based on price movement
        colors = ['green' if close >= open else 'red' 
                 for close, open in zip(data['close'], data['open'])]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['volume'],
                name="Volume",
                marker_color=colors,
                opacity=0.7,
                showlegend=False
            ),
            row=row, col=1
        )
    
    def _update_layout(self, fig: go.Figure, title: str):
        """
        Update chart layout with theme and styling.
        
        Args:
            fig: Plotly figure
            title: Chart title
        """
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16)
            ),
            template=self.theme,
            height=self.chart_height,
            width=self.chart_width,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Price", row=1, col=1)
        
        if self.show_volume:
            fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    def plot_fibonacci_analysis(
        self, 
        data: pd.DataFrame, 
        fib_analysis: FibonacciAnalysis,
        title: str = "Fibonacci Analysis"
    ) -> go.Figure:
        """
        Create dedicated Fibonacci analysis chart.
        
        Args:
            data: OHLCV DataFrame
            fib_analysis: Fibonacci analysis
            title: Chart title
            
        Returns:
            Plotly figure
        """
        try:
            fig = go.Figure()
            
            # Add price data
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='blue', width=2)
                )
            )
            
            # Add swing points
            fig.add_trace(
                go.Scatter(
                    x=[data.index[0], data.index[-1]],  # Approximate positions
                    y=[fib_analysis.swing_low, fib_analysis.swing_high],
                    mode='markers',
                    name='Swing Points',
                    marker=dict(size=10, color='red', symbol='diamond')
                )
            )
            
            # Add all Fibonacci levels
            x_range = [data.index[0], data.index[-1]]
            
            for level in fib_analysis.retracements + fib_analysis.extensions:
                line_style = 'solid' if level.is_key_level else 'dot'
                line_width = 2 if level.is_key_level else 1
                
                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=[level.price, level.price],
                        mode='lines',
                        line=dict(
                            color=self.colors['fibonacci'],
                            width=line_width,
                            dash=line_style
                        ),
                        name=f"Fib {level.ratio:.1%}",
                        showlegend=level.is_key_level
                    )
                )
            
            # Highlight current price
            fig.add_hline(
                y=fib_analysis.current_price,
                line_dash="dash",
                line_color="white",
                annotation_text=f"Current: {fib_analysis.current_price:.2f}"
            )
            
            # Update layout
            fig.update_layout(
                title=title,
                template=self.theme,
                height=600,
                width=self.chart_width,
                xaxis_title="Date",
                yaxis_title="Price",
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating Fibonacci chart: {e}")
            raise
    
    def plot_wave_progression(self, waves: List[Wave], title: str = "Wave Progression") -> go.Figure:
        """
        Create a chart showing wave progression over time.
        
        Args:
            waves: List of waves
            title: Chart title
            
        Returns:
            Plotly figure
        """
        try:
            fig = go.Figure()
            
            if not waves:
                fig.add_annotation(
                    text="No waves detected",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=20)
                )
                return fig
            
            # Create wave progression line
            x_points = []
            y_points = []
            wave_labels = []
            
            for i, wave in enumerate(waves):
                if i == 0:
                    x_points.append(wave.start_point.timestamp)
                    y_points.append(wave.start_point.price)
                    wave_labels.append(f"Start")
                
                x_points.append(wave.end_point.timestamp)
                y_points.append(wave.end_point.price)
                wave_labels.append(f"Wave {wave.wave_type.value}")
            
            # Add main progression line
            fig.add_trace(
                go.Scatter(
                    x=x_points,
                    y=y_points,
                    mode='lines+markers',
                    name='Wave Progression',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8, color='red')
                )
            )
            
            # Add wave labels
            for i, (x, y, label) in enumerate(zip(x_points[1:], y_points[1:], wave_labels[1:])):
                fig.add_annotation(
                    x=x, y=y,
                    text=label,
                    showarrow=True,
                    arrowhead=2,
                    bgcolor='rgba(255,255,255,0.8)',
                    font=dict(size=10)
                )
            
            # Update layout
            fig.update_layout(
                title=title,
                template=self.theme,
                height=500,
                width=self.chart_width,
                xaxis_title="Date",
                yaxis_title="Price",
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating wave progression chart: {e}")
            raise
    
    def create_dashboard(
        self, 
        data: pd.DataFrame, 
        waves: List[Wave],
        fibonacci_analysis: Optional[FibonacciAnalysis] = None,
        additional_indicators: Optional[pd.DataFrame] = None
    ) -> go.Figure:
        """
        Create comprehensive dashboard with multiple charts.
        
        Args:
            data: OHLCV DataFrame
            waves: List of detected waves
            fibonacci_analysis: Optional Fibonacci analysis
            additional_indicators: Optional technical indicators
            
        Returns:
            Plotly figure with dashboard layout
        """
        try:
            # Create subplot structure
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    "Elliott Wave Analysis", "Wave Progression",
                    "Fibonacci Levels", "RSI & MACD",
                    "Volume Analysis", "Wave Confidence"
                ),
                specs=[
                    [{"colspan": 2}, None],
                    [{"secondary_y": True}, {"secondary_y": True}],
                    [{"secondary_y": False}, {"secondary_y": False}]
                ],
                vertical_spacing=0.08,
                horizontal_spacing=0.05
            )
            
            # 1. Main Elliott Wave chart (top row, full width)
            candlestick = go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name="Price"
            )
            fig.add_trace(candlestick, row=1, col=1)
            
            if waves:
                self._add_wave_annotations(fig, waves, data, row=1)
            
            if fibonacci_analysis:
                self._add_fibonacci_levels(fig, fibonacci_analysis, data, row=1)
            
            # 2. Wave progression (row 2, col 1)
            if waves:
                wave_times = [w.end_point.timestamp for w in waves]
                wave_prices = [w.end_point.price for w in waves]
                wave_types = [w.wave_type.value for w in waves]
                
                fig.add_trace(
                    go.Scatter(
                        x=wave_times,
                        y=wave_prices,
                        mode='lines+markers',
                        name='Wave Progression',
                        text=wave_types,
                        textposition='top center'
                    ),
                    row=2, col=1
                )
            
            # 3. Technical indicators (row 2, col 2)
            if additional_indicators is not None:
                if 'rsi' in additional_indicators.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=additional_indicators.index,
                            y=additional_indicators['rsi'],
                            name='RSI',
                            line=dict(color='orange')
                        ),
                        row=2, col=2
                    )
                    
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=2)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=2)
            
            # 4. Volume analysis (row 3, col 1)
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['volume'],
                    name='Volume',
                    opacity=0.6
                ),
                row=3, col=1
            )
            
            # 5. Wave confidence (row 3, col 2)
            if waves:
                confidences = [w.confidence for w in waves]
                wave_labels = [w.wave_type.value for w in waves]
                
                fig.add_trace(
                    go.Bar(
                        x=wave_labels,
                        y=confidences,
                        name='Wave Confidence',
                        marker_color='lightblue'
                    ),
                    row=3, col=2
                )
            
            # Update layout
            fig.update_layout(
                title="Elliott Wave Trading Dashboard",
                template=self.theme,
                height=1000,
                width=self.chart_width,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            raise
    
    def save_chart(self, fig: go.Figure, filepath: str, format: str = 'html'):
        """
        Save chart to file.
        
        Args:
            fig: Plotly figure
            filepath: Output file path
            format: Output format ('html', 'png', 'pdf', 'svg')
        """
        try:
            if format.lower() == 'html':
                fig.write_html(filepath)
            elif format.lower() == 'png':
                fig.write_image(filepath)
            elif format.lower() == 'pdf':
                fig.write_image(filepath)
            elif format.lower() == 'svg':
                fig.write_image(filepath)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Chart saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving chart: {e}")
            raise

    def plot_multi_degree_waves(
        self, 
        data: pd.DataFrame, 
        waves: List[Wave],
        fibonacci_analysis: Optional[FibonacciAnalysis] = None,
        title: str = "Multi-Degree Elliott Wave Analysis",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create a chart showing waves at multiple degrees with different styling.
        
        Args:
            data: OHLCV DataFrame
            waves: List of waves with degree information
            fibonacci_analysis: Optional Fibonacci analysis
            title: Chart title
            save_path: Optional path to save the chart
            
        Returns:
            Plotly figure
        """
        try:
            # Create subplots
            if self.show_volume:
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02,
                    row_heights=[0.7, 0.3],
                    subplot_titles=(title, "Volume")
                )
            else:
                fig = go.Figure()
            
            # Add candlestick chart
            candlestick = go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name="Price",
                increasing_line_color='#00ff00',
                decreasing_line_color='#ff0000'
            )
            
            if self.show_volume:
                fig.add_trace(candlestick, row=1, col=1)
            else:
                fig.add_trace(candlestick)
            
            # Add wave annotations by degree
            if waves:
                self._add_multi_degree_wave_annotations(fig, waves, data, row=1 if self.show_volume else None)
            
            # Add Fibonacci levels
            if fibonacci_analysis:
                self._add_fibonacci_levels(fig, fibonacci_analysis, data, row=1 if self.show_volume else None)
            
            # Add volume chart
            if self.show_volume:
                self._add_volume_chart(fig, data, row=2)
            
            # Update layout
            self._update_layout(fig, title)
            
            # Save if path provided
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Multi-degree chart saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating multi-degree chart: {e}")
            raise

    def _add_multi_degree_wave_annotations(self, fig: go.Figure, waves: List[Wave], data: pd.DataFrame, row: Optional[int] = None):
        """
        Add wave annotations with degree-specific styling.
        
        Args:
            fig: Plotly figure
            waves: List of waves with degree information
            data: OHLCV DataFrame
            row: Row number for subplot
        """
        # Group waves by degree
        waves_by_degree = {}
        for wave in waves:
            degree = wave.degree
            if degree not in waves_by_degree:
                waves_by_degree[degree] = []
            waves_by_degree[degree].append(wave)
        
        # Add waves by degree with different styling
        for degree, degree_waves in waves_by_degree.items():
            line_width = self._get_line_width_by_degree(degree)
            opacity = self._get_opacity_by_degree(degree)
            
            for wave in degree_waves:
                color = self._get_wave_color(wave)
                
                # Draw wave line with degree-specific styling
                fig.add_trace(
                    go.Scatter(
                        x=[wave.start_point.timestamp, wave.end_point.timestamp],
                        y=[wave.start_point.price, wave.end_point.price],
                        mode='lines',
                        line=dict(
                            color=color, 
                            width=line_width,
                            opacity=opacity
                        ),
                        name=f"{degree.value} Wave {wave.wave_type.value}",
                        showlegend=True,
                        hovertemplate=f"<b>{degree.value} Wave {wave.wave_type.value}</b><br>" +
                                     f"Confidence: {wave.confidence:.2f}<br>" +
                                     f"Duration: {wave.duration} periods<br>" +
                                     f"Start: {wave.start_point.price:.2f}<br>" +
                                     f"End: {wave.end_point.price:.2f}<br>" +
                                     f"Change: {wave.price_change_pct:.2%}<extra></extra>"
                    ),
                    row=row, col=1
                )
                
                # Add wave label with degree prefix
                mid_timestamp = wave.start_point.timestamp + (wave.end_point.timestamp - wave.start_point.timestamp) / 2
                mid_price = (wave.start_point.price + wave.end_point.price) / 2
                
                # Different label styles based on degree
                label_text = self._get_degree_label(wave)
                font_size = self._get_font_size_by_degree(degree)
                
                fig.add_annotation(
                    x=mid_timestamp,
                    y=mid_price,
                    text=label_text,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor=color,
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor=color,
                    font=dict(size=font_size, color='black'),
                    row=row,
                    col=1
                )

    def _get_line_width_by_degree(self, degree) -> int:
        """Get line width based on wave degree."""
        width_map = {
            'PRIMARY': 3,
            'INTERMEDIATE': 2,
            'MINOR': 1,
            'MINUTE': 1,
            'MINUETTE': 1
        }
        return width_map.get(degree.value, 1)

    def _get_opacity_by_degree(self, degree) -> float:
        """Get opacity based on wave degree."""
        opacity_map = {
            'PRIMARY': 1.0,
            'INTERMEDIATE': 0.8,
            'MINOR': 0.6,
            'MINUTE': 0.4,
            'MINUETTE': 0.3
        }
        return opacity_map.get(degree.value, 0.5)

    def _get_font_size_by_degree(self, degree) -> int:
        """Get font size based on wave degree."""
        size_map = {
            'PRIMARY': 14,
            'INTERMEDIATE': 12,
            'MINOR': 10,
            'MINUTE': 8,
            'MINUETTE': 6
        }
        return size_map.get(degree.value, 10)

    def _get_degree_label(self, wave: Wave) -> str:
        """Get label text based on wave degree and type."""
        degree_prefix = {
            'PRIMARY': '',
            'INTERMEDIATE': '(',
            'MINOR': '((',
            'MINUTE': '((((',
            'MINUETTE': '((((('
        }
        degree_suffix = {
            'PRIMARY': '',
            'INTERMEDIATE': ')',
            'MINOR': '))',
            'MINUTE': '))))',
            'MINUETTE': ')))))'
        }
        
        prefix = degree_prefix.get(wave.degree.value, '')
        suffix = degree_suffix.get(wave.degree.value, '')
        return f"<b>{prefix}{wave.wave_type.value}{suffix}</b>"

    def add_time_symmetry_annotations(self, fig: go.Figure, time_symmetry_results: List[Dict[str, float]], data: pd.DataFrame, row: Optional[int] = None):
        """
        Add time symmetry information to the chart.
        
        Args:
            fig: Plotly figure
            time_symmetry_results: Results from validate_time_symmetry
            data: OHLCV DataFrame
            row: Row number for subplot
        """
        for result in time_symmetry_results:
            pattern = result['pattern']
            ratios = result['ratios']
            
            # Add text annotation for time symmetry info
            fig.add_annotation(
                x=data.index[-1],
                y=data['high'].max(),
                text=f"<b>Time Symmetry - {pattern.title()}</b><br>" +
                     f"1:3 Ratio: {ratios.get('1:3', 'N/A'):.2f}<br>" +
                     f"3:5 Ratio: {ratios.get('3:5', 'N/A'):.2f}<br>" +
                     f"2:4 Ratio: {ratios.get('2:4', 'N/A'):.2f}",
                showarrow=False,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='black',
                font=dict(size=10, color='black'),
                row=row,
                col=1
            )

    def add_future_wave_projections(
        self, 
        fig: go.Figure, 
        current_wave: Wave, 
        scenarios: List[Dict[str, Any]], 
        data: pd.DataFrame, 
        row: Optional[int] = None
    ):
        """
        Add future wave projections and scenarios to the chart.
        
        Args:
            fig: Plotly figure
            current_wave: Current wave
            scenarios: List of future wave scenarios
            data: OHLCV DataFrame
            row: Row number for subplot
        """
        if not scenarios:
            return
            
        # Get the most likely scenario (highest likelihood)
        primary_scenario = max(scenarios, key=lambda x: x['likelihood'])
        
        # Add projected wave path for primary scenario
        self._add_projected_wave_path(fig, current_wave, primary_scenario, data, row)
        
        # Add Fibonacci target levels for primary scenario
        self._add_fibonacci_targets(fig, primary_scenario, data, row)
        
        # Add scenario information annotation
        self._add_scenario_annotation(fig, scenarios, data, row)
        
        # Add alternative scenarios as dashed lines
        for scenario in scenarios[1:]:  # Skip primary scenario
            self._add_alternative_scenario(fig, current_wave, scenario, data, row)

    def _add_projected_wave_path(
        self, 
        fig: go.Figure, 
        current_wave: Wave, 
        scenario: Dict[str, Any], 
        data: pd.DataFrame, 
        row: Optional[int] = None
    ):
        """Add projected wave path for the primary scenario."""
        # Calculate projected end point based on Fibonacci targets
        primary_target = scenario['fibonacci_targets'][0] if scenario['fibonacci_targets'] else current_wave.end_point.price
        
        # Create projected wave line (dashed, different color)
        fig.add_trace(
            go.Scatter(
                x=[current_wave.end_point.timestamp, data.index[-1] + pd.Timedelta(hours=24)],
                y=[current_wave.end_point.price, primary_target],
                mode='lines',
                line=dict(
                    color='rgba(255, 165, 0, 0.7)',  # Orange with transparency
                    width=2,
                    dash='dash'
                ),
                name=f"Projected {scenario['next_wave']}",
                showlegend=True,
                hovertemplate=f"<b>Projected {scenario['next_wave']}</b><br>" +
                             f"Scenario: {scenario['name']}<br>" +
                             f"Likelihood: {scenario['likelihood']:.1%}<br>" +
                             f"Target: {primary_target:.2f}<br>" +
                             f"Direction: {scenario['direction']}<extra></extra>"
            ),
            row=row, col=1
        )

    def _add_fibonacci_targets(
        self, 
        fig: go.Figure, 
        scenario: Dict[str, Any], 
        data: pd.DataFrame, 
        row: Optional[int] = None
    ):
        """Add Fibonacci target levels for the scenario."""
        if not scenario.get('fibonacci_targets'):
            return
            
        x_range = [data.index[-1], data.index[-1] + pd.Timedelta(days=7)]
        
        for i, target in enumerate(scenario['fibonacci_targets']):
            color = ['#FF6B35', '#FF8C42', '#FFA07A'][i % 3]  # Different orange shades
            
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=[target, target],
                    mode='lines',
                    line=dict(
                        color=color,
                        width=1,
                        dash='dot'
                    ),
                    name=f"Fib Target {i+1}",
                    showlegend=False,
                    hovertemplate=f"Fibonacci Target {i+1}<br>" +
                                 f"Price: {target:.2f}<br>" +
                                 f"Scenario: {scenario['name']}<extra></extra>"
                ),
                row=row, col=1
            )

    def _add_scenario_annotation(
        self, 
        fig: go.Figure, 
        scenarios: List[Dict[str, Any]], 
        data: pd.DataFrame, 
        row: Optional[int] = None
    ):
        """Add scenario information as chart annotation."""
        # Sort scenarios by likelihood
        sorted_scenarios = sorted(scenarios, key=lambda x: x['likelihood'], reverse=True)
        
        # Create annotation text
        annotation_text = "<b>Future Wave Scenarios:</b><br>"
        for i, scenario in enumerate(sorted_scenarios[:3]):  # Show top 3 scenarios
            annotation_text += f"{scenario['scenario']}. {scenario['name']}<br>" + \
                             f"   Likelihood: {scenario['likelihood']:.1%}<br>" + \
                             f"   Next: {scenario['next_wave']}<br>"
        
        fig.add_annotation(
            x=data.index[-1] - pd.Timedelta(days=7),
            y=data['high'].max(),
            text=annotation_text,
            showarrow=False,
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='orange',
            borderwidth=2,
            font=dict(size=10, color='black'),
            row=row,
            col=1
        )

    def _add_alternative_scenario(
        self, 
        fig: go.Figure, 
        current_wave: Wave, 
        scenario: Dict[str, Any], 
        data: pd.DataFrame, 
        row: Optional[int] = None
    ):
        """Add alternative scenario as a lighter dashed line."""
        if not scenario.get('fibonacci_targets'):
            return
            
        # Use second target for alternative scenario
        alt_target = scenario['fibonacci_targets'][1] if len(scenario['fibonacci_targets']) > 1 else scenario['fibonacci_targets'][0]
        
        fig.add_trace(
            go.Scatter(
                x=[current_wave.end_point.timestamp, data.index[-1] + pd.Timedelta(hours=12)],
                y=[current_wave.end_point.price, alt_target],
                mode='lines',
                line=dict(
                    color='rgba(128, 128, 128, 0.5)',  # Gray with low opacity
                    width=1,
                    dash='dot'
                ),
                name=f"Alt {scenario['next_wave']}",
                showlegend=True,
                hovertemplate=f"<b>Alternative {scenario['next_wave']}</b><br>" +
                             f"Scenario: {scenario['name']}<br>" +
                             f"Likelihood: {scenario['likelihood']:.1%}<br>" +
                             f"Target: {alt_target:.2f}<extra></extra>"
            ),
            row=row, col=1
        )

    def plot_waves_with_projections(
        self, 
        data: pd.DataFrame, 
        waves: List[Wave],
        projection_scenarios: List[ProjectionScenario],
        validation_result: Optional[WaveValidation] = None,
        fibonacci_analysis: Optional[FibonacciAnalysis] = None,
        title: str = "Elliott Wave Analysis with Projections",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create a chart showing waves with projection scenarios.
        
        Args:
            data: OHLCV DataFrame
            waves: List of detected waves
            projection_scenarios: List of projection scenarios
            validation_result: Optional validation result
            fibonacci_analysis: Optional Fibonacci analysis
            title: Chart title
            save_path: Optional path to save the chart
            
        Returns:
            Plotly figure
        """
        try:
            # Create subplots
            if self.show_volume:
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02,
                    row_heights=[0.7, 0.3],
                    subplot_titles=(title, "Volume")
                )
            else:
                fig = go.Figure()
            
            # Add candlestick chart
            candlestick = go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name="Price",
                increasing_line_color='#00ff00',
                decreasing_line_color='#ff0000'
            )
            
            if self.show_volume:
                fig.add_trace(candlestick, row=1, col=1)
            else:
                fig.add_trace(candlestick)
            
            # Add wave annotations
            if waves:
                if validation_result:
                    self._add_validated_wave_annotations(fig, waves, validation_result, data, row=1 if self.show_volume else None)
                else:
                    self._add_wave_annotations(fig, waves, data, row=1 if self.show_volume else None)
            
            # Add projection scenarios
            if projection_scenarios:
                self._add_projection_scenarios(fig, projection_scenarios, data, row=1 if self.show_volume else None)
            
            # Add Fibonacci levels
            if fibonacci_analysis:
                self._add_fibonacci_levels(fig, fibonacci_analysis, data, row=1 if self.show_volume else None)
            
            # Add volume chart
            if self.show_volume:
                self._add_volume_chart(fig, data, row=2)
            
            # Update layout
            self._update_layout_with_projections(fig, title, projection_scenarios)
            
            # Save if path provided
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Projection chart saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating projection chart: {e}")
            raise

    def _add_projection_scenarios(
        self, 
        fig: go.Figure, 
        scenarios: List[ProjectionScenario], 
        data: pd.DataFrame, 
        row: Optional[int] = None
    ):
        """Add projection scenarios to the chart."""
        if not scenarios:
            return
        
        # Get primary scenario
        primary_scenario = scenarios[0]
        
        # Add primary projection
        self._add_projection_path(fig, primary_scenario.primary_projection, data, row, is_primary=True)
        
        # Add alternative projections
        for i, alt_projection in enumerate(primary_scenario.alternative_projections):
            self._add_projection_path(fig, alt_projection, data, row, is_primary=False, alt_index=i)
        
        # Add projection information
        self._add_projection_info(fig, primary_scenario, data, row)

    def _add_projection_path(
        self, 
        fig: go.Figure, 
        projection: WaveProjection, 
        data: pd.DataFrame, 
        row: Optional[int] = None,
        is_primary: bool = True,
        alt_index: int = 0
    ):
        """Add projection path to the chart."""
        current_price = data['close'].iloc[-1]
        current_time = data.index[-1]
        
        # Determine line style based on projection type and confidence
        line_style = self._get_projection_line_style(projection, is_primary)
        
        # Create projection path
        for i, target in enumerate(projection.fibonacci_targets):
            # Calculate time target
            time_target = projection.time_targets[i] if i < len(projection.time_targets) else current_time + pd.Timedelta(hours=24)
            
            # Create path line
            fig.add_trace(
                go.Scatter(
                    x=[current_time, time_target],
                    y=[current_price, target],
                    mode='lines',
                    line=dict(
                        color=line_style['color'],
                        width=line_style['width'],
                        dash=line_style['dash']
                    ),
                    name=f"{'Primary' if is_primary else f'Alt {alt_index+1}'} Target {i+1}",
                    showlegend=True,
                    hovertemplate=f"<b>{projection.description}</b><br>" +
                                 f"Target: {target:.2f}<br>" +
                                 f"Likelihood: {projection.likelihood:.1%}<br>" +
                                 f"Risk/Reward: {projection.risk_reward_ratios[i]:.1f}<br>" +
                                 f"Confidence: {projection.confidence.value}<extra></extra>"
                ),
                row=row, col=1
            )
            
            # Add target marker
            fig.add_trace(
                go.Scatter(
                    x=[time_target],
                    y=[target],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=line_style['color'],
                        symbol='diamond'
                    ),
                    name=f"Target {i+1}",
                    showlegend=False,
                    hovertemplate=f"Target {i+1}: {target:.2f}<extra></extra>"
                ),
                row=row, col=1
            )

    def _get_projection_line_style(self, projection: WaveProjection, is_primary: bool) -> Dict[str, Any]:
        """Get line style for projection based on type and confidence."""
        if is_primary:
            if projection.confidence == ProjectionConfidence.HIGH:
                return {'color': '#00FF00', 'width': 3, 'dash': 'solid'}  # Green, thick, solid
            elif projection.confidence == ProjectionConfidence.MEDIUM:
                return {'color': '#FFA500', 'width': 2, 'dash': 'solid'}  # Orange, medium, solid
            else:
                return {'color': '#FF0000', 'width': 2, 'dash': 'dash'}   # Red, medium, dashed
        else:
            if projection.confidence == ProjectionConfidence.HIGH:
                return {'color': '#00FF00', 'width': 2, 'dash': 'dot'}     # Green, medium, dotted
            elif projection.confidence == ProjectionConfidence.MEDIUM:
                return {'color': '#FFA500', 'width': 1, 'dash': 'dot'}     # Orange, thin, dotted
            else:
                return {'color': '#FF0000', 'width': 1, 'dash': 'dot'}     # Red, thin, dotted

    def _add_projection_info(
        self, 
        fig: go.Figure, 
        scenario: ProjectionScenario, 
        data: pd.DataFrame, 
        row: Optional[int] = None
    ):
        """Add projection information to the chart."""
        # Add scenario information
        fig.add_annotation(
            x=data.index[-1] - pd.Timedelta(days=3),
            y=data['high'].max(),
            text=f"<b>Projection Scenario</b><br>" +
                 f"Name: {scenario.scenario_name}<br>" +
                 f"Confidence: {scenario.overall_confidence:.1%}<br>" +
                 f"Primary: {scenario.primary_projection.description}<br>" +
                 f"Alternatives: {len(scenario.alternative_projections)}",
            showarrow=False,
            bgcolor='rgba(0, 255, 255, 0.9)',
            bordercolor='cyan',
            borderwidth=2,
            font=dict(size=10, color='black'),
            row=row,
            col=1
        )
        
        # Add invalidation levels
        self._add_invalidation_levels(fig, scenario.primary_projection, data, row)
        
        # Add recommendations
        if scenario.recommendations:
            self._add_recommendations(fig, scenario.recommendations, data, row)

    def _add_invalidation_levels(
        self, 
        fig: go.Figure, 
        projection: WaveProjection, 
        data: pd.DataFrame, 
        row: Optional[int] = None
    ):
        """Add invalidation levels to the chart."""
        x_range = [data.index[-1] - pd.Timedelta(days=7), data.index[-1] + pd.Timedelta(days=7)]
        
        for i, invalidation_level in enumerate(projection.invalidation_levels):
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=[invalidation_level, invalidation_level],
                    mode='lines',
                    line=dict(
                        color='red',
                        width=2,
                        dash='dash'
                    ),
                    name=f"Invalidation {i+1}",
                    showlegend=True,
                    hovertemplate=f"Invalidation Level {i+1}<br>" +
                                 f"Price: {invalidation_level:.2f}<br>" +
                                 f"Projection: {projection.description}<extra></extra>"
                ),
                row=row, col=1
            )

    def _add_recommendations(
        self, 
        fig: go.Figure, 
        recommendations: List[str], 
        data: pd.DataFrame, 
        row: Optional[int] = None
    ):
        """Add recommendations to the chart."""
        if not recommendations:
            return
        
        # Create recommendation text
        rec_text = "<b>Recommendations:</b><br>"
        for i, rec in enumerate(recommendations[:3]):  # Show first 3 recommendations
            rec_text += f"{i+1}. {rec}<br>"
        
        fig.add_annotation(
            x=data.index[-1] - pd.Timedelta(days=5),
            y=data['low'].min(),
            text=rec_text,
            showarrow=False,
            bgcolor='rgba(255, 255, 0, 0.9)',
            bordercolor='yellow',
            borderwidth=1,
            font=dict(size=9, color='black'),
            row=row,
            col=1
        )

    def _update_layout_with_projections(self, fig: go.Figure, title: str, scenarios: List[ProjectionScenario]):
        """Update layout with projection information."""
        if scenarios:
            # Add projection count to title
            projection_count = len(scenarios)
            alt_count = sum(len(s.alternative_projections) for s in scenarios)
            updated_title = f"{title} ({projection_count} scenarios, {alt_count} alternatives)"
        else:
            updated_title = title
        
        fig.update_layout(
            title=updated_title,
            height=self.chart_height,
            width=self.chart_width,
            template=self.theme,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

    def create_projection_dashboard(
        self, 
        data: pd.DataFrame, 
        waves: List[Wave],
        projection_scenarios: List[ProjectionScenario],
        validation_result: Optional[WaveValidation] = None,
        fibonacci_analysis: Optional[FibonacciAnalysis] = None,
        title: str = "Elliott Wave Projection Dashboard",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create a comprehensive projection dashboard.
        
        Args:
            data: OHLCV DataFrame
            waves: List of detected waves
            projection_scenarios: List of projection scenarios
            validation_result: Optional validation result
            fibonacci_analysis: Optional Fibonacci analysis
            title: Chart title
            save_path: Optional path to save the chart
            
        Returns:
            Plotly figure
        """
        try:
            # Create subplot structure for dashboard
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    "Elliott Wave Analysis with Projections",
                    "Projection Confidence Levels",
                    "Risk/Reward Analysis",
                    "Projection Types",
                    "Fibonacci Targets",
                    "Market Context"
                ),
                specs=[
                    [{"colspan": 2}, None],
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}]
                ],
                vertical_spacing=0.08,
                horizontal_spacing=0.05
            )
            
            # 1. Main chart with projections (top row, full width)
            candlestick = go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name="Price"
            )
            fig.add_trace(candlestick, row=1, col=1)
            
            if waves:
                if validation_result:
                    self._add_validated_wave_annotations(fig, waves, validation_result, data, row=1)
                else:
                    self._add_wave_annotations(fig, waves, data, row=1)
            
            if projection_scenarios:
                self._add_projection_scenarios(fig, projection_scenarios, data, row=1)
            
            if fibonacci_analysis:
                self._add_fibonacci_levels(fig, fibonacci_analysis, data, row=1)
            
            # 2. Projection confidence levels (row 2, col 1)
            self._add_confidence_chart(fig, projection_scenarios, row=2, col=1)
            
            # 3. Risk/reward analysis (row 2, col 2)
            self._add_risk_reward_chart(fig, projection_scenarios, row=2, col=2)
            
            # 4. Projection types (row 3, col 1)
            self._add_projection_types_chart(fig, projection_scenarios, row=3, col=1)
            
            # 5. Fibonacci targets (row 3, col 2)
            self._add_fibonacci_targets_chart(fig, projection_scenarios, row=3, col=2)
            
            # Update layout
            fig.update_layout(
                title=title,
                height=1000,
                width=1400,
                template=self.theme,
                showlegend=True
            )
            
            # Save if path provided
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Projection dashboard saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating projection dashboard: {e}")
            raise

    def _add_confidence_chart(self, fig: go.Figure, scenarios: List[ProjectionScenario], row: int, col: int):
        """Add confidence levels chart."""
        if not scenarios:
            return
        
        # Extract confidence levels
        confidences = []
        for scenario in scenarios:
            confidences.append(scenario.primary_projection.confidence.value)
            for alt in scenario.alternative_projections:
                confidences.append(alt.confidence.value)
        
        # Count by confidence level
        confidence_counts = {}
        for conf in confidences:
            confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
        
        # Create bar chart
        levels = list(confidence_counts.keys())
        counts = list(confidence_counts.values())
        colors = ['green' if level == 'high' else 'orange' if level == 'medium' else 'red' for level in levels]
        
        fig.add_trace(
            go.Bar(
                x=levels,
                y=counts,
                marker_color=colors,
                name="Confidence Levels",
                hovertemplate="%{y} projections<extra></extra>"
            ),
            row=row, col=col
        )
        
        fig.update_yaxes(title_text="Number of Projections", row=row, col=col)

    def _add_risk_reward_chart(self, fig: go.Figure, scenarios: List[ProjectionScenario], row: int, col: int):
        """Add risk/reward analysis chart."""
        if not scenarios:
            return
        
        # Extract risk/reward ratios
        ratios = []
        for scenario in scenarios:
            ratios.extend(scenario.primary_projection.risk_reward_ratios)
            for alt in scenario.alternative_projections:
                ratios.extend(alt.risk_reward_ratios)
        
        if ratios:
            fig.add_trace(
                go.Histogram(
                    x=ratios,
                    nbinsx=10,
                    name="Risk/Reward Distribution",
                    hovertemplate="Ratio: %{x}<br>Count: %{y}<extra></extra>"
                ),
                row=row, col=col
            )
        
        fig.update_xaxes(title_text="Risk/Reward Ratio", row=row, col=col)
        fig.update_yaxes(title_text="Frequency", row=row, col=col)

    def _add_projection_types_chart(self, fig: go.Figure, scenarios: List[ProjectionScenario], row: int, col: int):
        """Add projection types chart."""
        if not scenarios:
            return
        
        # Count projection types
        type_counts = {}
        for scenario in scenarios:
            proj_type = scenario.primary_projection.projection_type.value
            type_counts[proj_type] = type_counts.get(proj_type, 0) + 1
            for alt in scenario.alternative_projections:
                alt_type = alt.projection_type.value
                type_counts[alt_type] = type_counts.get(alt_type, 0) + 1
        
        # Create pie chart
        types = list(type_counts.keys())
        counts = list(type_counts.values())
        
        fig.add_trace(
            go.Pie(
                labels=types,
                values=counts,
                name="Projection Types",
                hovertemplate="Type: %{label}<br>Count: %{value}<extra></extra>"
            ),
            row=row, col=col
        )

    def _add_fibonacci_targets_chart(self, fig: go.Figure, scenarios: List[ProjectionScenario], row: int, col: int):
        """Add Fibonacci targets chart."""
        if not scenarios:
            return
        
        # Extract all Fibonacci targets
        all_targets = []
        for scenario in scenarios:
            all_targets.extend(scenario.primary_projection.fibonacci_targets)
            for alt in scenario.alternative_projections:
                all_targets.extend(alt.fibonacci_targets)
        
        if all_targets:
            # Create histogram of targets
            fig.add_trace(
                go.Histogram(
                    x=all_targets,
                    nbinsx=15,
                    name="Fibonacci Targets",
                    hovertemplate="Target: %{x}<br>Frequency: %{y}<extra></extra>"
                ),
                row=row, col=col
            )
        
        fig.update_xaxes(title_text="Price Targets", row=row, col=col)
        fig.update_yaxes(title_text="Frequency", row=row, col=col)

    def plot_advanced_projections(
        self, 
        data: pd.DataFrame, 
        waves: List[Wave],
        current_wave: Wave,
        scenarios: List[Dict[str, Any]],
        fibonacci_analysis: Optional[FibonacciAnalysis] = None,
        title: str = "Advanced Elliott Wave Projections",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create a chart with advanced projection features.
        
        Args:
            data: OHLCV DataFrame
            waves: List of detected waves
            current_wave: Current wave for projections
            scenarios: List of advanced future wave scenarios
            fibonacci_analysis: Optional Fibonacci analysis
            title: Chart title
            save_path: Optional path to save the chart
            
        Returns:
            Plotly figure
        """
        try:
            # Create subplots
            if self.show_volume:
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02,
                    row_heights=[0.7, 0.3],
                    subplot_titles=(title, "Volume")
                )
            else:
                fig = go.Figure()
            
            # Add candlestick chart
            candlestick = go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name="Price",
                increasing_line_color='#00ff00',
                decreasing_line_color='#ff0000'
            )
            
            if self.show_volume:
                fig.add_trace(candlestick, row=1, col=1)
            else:
                fig.add_trace(candlestick)
            
            # Add wave annotations
            if waves:
                self._add_wave_annotations(fig, waves, data, row=1 if self.show_volume else None)
            
            # Add advanced projections
            if scenarios:
                self.add_advanced_projections(fig, current_wave, scenarios, data, row=1 if self.show_volume else None)
            
            # Add Fibonacci levels
            if fibonacci_analysis:
                self._add_fibonacci_levels(fig, fibonacci_analysis, data, row=1 if self.show_volume else None)
            
            # Add volume chart
            if self.show_volume:
                self._add_volume_chart(fig, data, row=2)
            
            # Update layout
            self._update_layout(fig, title)
            
            # Save if path provided
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Advanced projection chart saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating advanced projection chart: {e}")
            raise

    def plot_waves_with_validation(
        self, 
        data: pd.DataFrame, 
        waves: List[Wave],
        validation_result: WaveValidation,
        fibonacci_analysis: Optional[FibonacciAnalysis] = None,
        title: str = "Elliott Wave Analysis with Validation",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create a chart showing waves with validation results.
        
        Args:
            data: OHLCV DataFrame
            waves: List of detected waves
            validation_result: WaveValidation object
            fibonacci_analysis: Optional Fibonacci analysis
            title: Chart title
            save_path: Optional path to save the chart
            
        Returns:
            Plotly figure
        """
        try:
            # Create subplots
            if self.show_volume:
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02,
                    row_heights=[0.7, 0.3],
                    subplot_titles=(title, "Volume")
                )
            else:
                fig = go.Figure()
            
            # Add candlestick chart
            candlestick = go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name="Price",
                increasing_line_color='#00ff00',
                decreasing_line_color='#ff0000'
            )
            
            if self.show_volume:
                fig.add_trace(candlestick, row=1, col=1)
            else:
                fig.add_trace(candlestick)
            
            # Add wave annotations with validation styling
            if waves:
                self._add_validated_wave_annotations(fig, waves, validation_result, data, row=1 if self.show_volume else None)
            
            # Add validation indicators
            self._add_validation_indicators(fig, validation_result, data, row=1 if self.show_volume else None)
            
            # Add Fibonacci levels
            if fibonacci_analysis:
                self._add_fibonacci_levels(fig, fibonacci_analysis, data, row=1 if self.show_volume else None)
            
            # Add volume chart
            if self.show_volume:
                self._add_volume_chart(fig, data, row=2)
            
            # Update layout with validation info
            self._update_layout_with_validation(fig, title, validation_result)
            
            # Save if path provided
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Validation chart saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating validation chart: {e}")
            raise

    def _add_validated_wave_annotations(
        self, 
        fig: go.Figure, 
        waves: List[Wave], 
        validation_result: WaveValidation, 
        data: pd.DataFrame, 
        row: Optional[int] = None
    ):
        """
        Add wave annotations with validation-based styling.
        
        Args:
            fig: Plotly figure
            waves: List of waves
            validation_result: WaveValidation object
            data: OHLCV DataFrame
            row: Row number for subplot
        """
        for wave in waves:
            # Determine color based on validation status
            color = self._get_validated_wave_color(wave, validation_result)
            line_width = self._get_validated_line_width(wave, validation_result)
            
            # Draw wave line
            fig.add_trace(
                go.Scatter(
                    x=[wave.start_point.timestamp, wave.end_point.timestamp],
                    y=[wave.start_point.price, wave.end_point.price],
                    mode='lines',
                    line=dict(color=color, width=line_width),
                    name=f"Wave {wave.wave_type.value}",
                    showlegend=False,
                    hovertemplate=f"Wave {wave.wave_type.value}<br>" +
                                 f"Confidence: {wave.confidence:.2f}<br>" +
                                 f"Start: {wave.start_point.price:.2f}<br>" +
                                 f"End: {wave.end_point.price:.2f}<br>" +
                                 f"Change: {wave.price_change_pct:.2%}<br>" +
                                 f"Validation Score: {validation_result.overall_score:.2f}<extra></extra>"
                ),
                row=row, col=1
            )
            
            # Add wave label with validation status
            mid_timestamp = wave.start_point.timestamp + (wave.end_point.timestamp - wave.start_point.timestamp) / 2
            mid_price = (wave.start_point.price + wave.end_point.price) / 2
            
            # Add validation status indicator
            status_symbol = self._get_validation_status_symbol(wave, validation_result)
            
            fig.add_annotation(
                x=mid_timestamp,
                y=mid_price,
                text=f"<b>{wave.wave_type.value}</b> {status_symbol}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor=color,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor=color,
                font=dict(size=12, color='black'),
                row=row,
                col=1
            )

    def _get_validated_wave_color(self, wave: Wave, validation_result: WaveValidation) -> str:
        """Get color for wave based on validation status."""
        base_color = self._get_wave_color(wave)
        
        # Check if wave has validation issues
        if not validation_result.passed_critical_rules:
            # Add red tint for failed validation
            return self._adjust_color_for_validation(base_color, 'red')
        elif validation_result.overall_score < 0.7:
            # Add orange tint for warnings
            return self._adjust_color_for_validation(base_color, 'orange')
        else:
            return base_color

    def _get_validated_line_width(self, wave: Wave, validation_result: WaveValidation) -> int:
        """Get line width based on validation status."""
        base_width = 2
        
        if not validation_result.passed_critical_rules:
            return base_width + 1  # Thicker line for failed validation
        elif validation_result.overall_score < 0.7:
            return base_width + 1  # Thicker line for warnings
        else:
            return base_width

    def _get_validation_status_symbol(self, wave: Wave, validation_result: WaveValidation) -> str:
        """Get validation status symbol for wave label."""
        if not validation_result.passed_critical_rules:
            return "❌"  # Failed validation
        elif validation_result.overall_score < 0.7:
            return "⚠️"   # Warning
        else:
            return "✅"   # Passed validation

    def _adjust_color_for_validation(self, base_color: str, validation_color: str) -> str:
        """Adjust color based on validation status."""
        # Simple color adjustment - in practice, you might want more sophisticated color mixing
        if validation_color == 'red':
            return '#FF4444'  # Red tint
        elif validation_color == 'orange':
            return '#FF8800'  # Orange tint
        else:
            return base_color

    def _add_validation_indicators(
        self, 
        fig: go.Figure, 
        validation_result: WaveValidation, 
        data: pd.DataFrame, 
        row: Optional[int] = None
    ):
        """Add validation indicators to the chart."""
        # Add validation score indicator
        score_color = 'green' if validation_result.overall_score >= 0.8 else \
                    'orange' if validation_result.overall_score >= 0.6 else 'red'
        
        fig.add_annotation(
            x=data.index[-1] - pd.Timedelta(days=5),
            y=data['high'].max(),
            text=f"<b>Validation Score: {validation_result.overall_score:.1%}</b><br>" +
                 f"Pattern: {validation_result.pattern_type.title()}<br>" +
                 f"Critical Rules: {'✓' if validation_result.passed_critical_rules else '✗'}",
            showarrow=False,
            bgcolor=f'rgba({self._get_rgb_from_color(score_color)}, 0.9)',
            bordercolor=score_color,
            borderwidth=2,
            font=dict(size=12, color='white'),
            row=row,
            col=1
        )
        
        # Add rule violation markers
        self._add_rule_violation_markers(fig, validation_result, data, row)

    def _add_rule_violation_markers(
        self, 
        fig: go.Figure, 
        validation_result: WaveValidation, 
        data: pd.DataFrame, 
        row: Optional[int] = None
    ):
        """Add markers for rule violations."""
        for result in validation_result.validation_results:
            if not result.passed and result.severity == ValidationSeverity.CRITICAL:
                # Add violation marker
                fig.add_annotation(
                    x=data.index[-1] - pd.Timedelta(days=2),
                    y=data['low'].min() + (data['high'].max() - data['low'].min()) * 0.1,
                    text=f"❌ {result.rule.value}",
                    showarrow=False,
                    bgcolor='rgba(255, 0, 0, 0.9)',
                    bordercolor='red',
                    borderwidth=1,
                    font=dict(size=10, color='white'),
                    row=row,
                    col=1
                )

    def _get_rgb_from_color(self, color_name: str) -> str:
        """Get RGB string from color name."""
        color_map = {
            'green': '0, 255, 0',
            'orange': '255, 165, 0',
            'red': '255, 0, 0',
            'blue': '0, 0, 255',
            'purple': '128, 0, 128'
        }
        return color_map.get(color_name, '128, 128, 128')

    def _update_layout_with_validation(self, fig: go.Figure, title: str, validation_result: WaveValidation):
        """Update layout with validation information."""
        # Update title with validation status
        status_emoji = "✅" if validation_result.passed_critical_rules else "❌"
        updated_title = f"{title} {status_emoji}"
        
        fig.update_layout(
            title=updated_title,
            height=self.chart_height,
            width=self.chart_width,
            template=self.theme,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add validation summary to layout
        if validation_result.warnings or validation_result.recommendations:
            fig.add_annotation(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=f"<b>Validation Summary:</b><br>" +
                     f"Score: {validation_result.overall_score:.1%}<br>" +
                     f"Warnings: {len(validation_result.warnings)}<br>" +
                     f"Recommendations: {len(validation_result.recommendations)}",
                showarrow=False,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=1,
                font=dict(size=10, color='black')
            )

    def create_validation_dashboard(
        self, 
        data: pd.DataFrame, 
        waves: List[Wave],
        validation_result: WaveValidation,
        fibonacci_analysis: Optional[FibonacciAnalysis] = None,
        title: str = "Elliott Wave Validation Dashboard",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create a comprehensive validation dashboard.
        
        Args:
            data: OHLCV DataFrame
            waves: List of detected waves
            validation_result: WaveValidation object
            fibonacci_analysis: Optional Fibonacci analysis
            title: Chart title
            save_path: Optional path to save the chart
            
        Returns:
            Plotly figure
        """
        try:
            # Create subplot structure for dashboard
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    "Elliott Wave Analysis with Validation",
                    "Validation Score Breakdown",
                    "Rule Violations",
                    "Wave Characteristics",
                    "Fibonacci Analysis",
                    "Recommendations"
                ),
                specs=[
                    [{"colspan": 2}, None],
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}]
                ],
                vertical_spacing=0.08,
                horizontal_spacing=0.05
            )
            
            # 1. Main chart with validation (top row, full width)
            candlestick = go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name="Price"
            )
            fig.add_trace(candlestick, row=1, col=1)
            
            if waves:
                self._add_validated_wave_annotations(fig, waves, validation_result, data, row=1)
            
            if fibonacci_analysis:
                self._add_fibonacci_levels(fig, fibonacci_analysis, data, row=1)
            
            # 2. Validation score breakdown (row 2, col 1)
            self._add_validation_score_chart(fig, validation_result, row=2, col=1)
            
            # 3. Rule violations (row 2, col 2)
            self._add_rule_violations_chart(fig, validation_result, row=2, col=2)
            
            # 4. Wave characteristics (row 3, col 1)
            self._add_wave_characteristics_chart(fig, waves, row=3, col=1)
            
            # 5. Fibonacci analysis (row 3, col 2)
            if fibonacci_analysis:
                self._add_fibonacci_analysis_chart(fig, fibonacci_analysis, row=3, col=2)
            
            # Update layout
            fig.update_layout(
                title=title,
                height=1000,
                width=1400,
                template=self.theme,
                showlegend=True
            )
            
            # Save if path provided
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Validation dashboard saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating validation dashboard: {e}")
            raise

    def _add_validation_score_chart(self, fig: go.Figure, validation_result: WaveValidation, row: int, col: int):
        """Add validation score breakdown chart."""
        # Group results by severity
        critical_results = [r for r in validation_result.validation_results if r.severity == ValidationSeverity.CRITICAL]
        warning_results = [r for r in validation_result.validation_results if r.severity == ValidationSeverity.WARNING]
        info_results = [r for r in validation_result.validation_results if r.severity == ValidationSeverity.INFO]
        
        # Calculate pass rates
        critical_pass_rate = np.mean([r.passed for r in critical_results]) if critical_results else 1.0
        warning_pass_rate = np.mean([r.passed for r in warning_results]) if warning_results else 1.0
        info_pass_rate = np.mean([r.passed for r in info_results]) if info_results else 1.0
        
        # Create bar chart
        categories = ['Critical', 'Warning', 'Info']
        pass_rates = [critical_pass_rate, warning_pass_rate, info_pass_rate]
        colors = ['red' if rate < 0.8 else 'orange' if rate < 1.0 else 'green' for rate in pass_rates]
        
        fig.add_trace(
            go.Bar(
                x=categories,
                y=pass_rates,
                marker_color=colors,
                name="Pass Rate",
                hovertemplate="%{y:.1%}<extra></extra>"
            ),
            row=row, col=col
        )
        
        fig.update_yaxes(title_text="Pass Rate", range=[0, 1], row=row, col=col)

    def _add_rule_violations_chart(self, fig: go.Figure, validation_result: WaveValidation, row: int, col: int):
        """Add rule violations chart."""
        failed_rules = [r for r in validation_result.validation_results if not r.passed]
        
        if failed_rules:
            rule_names = [r.rule.value[:30] + "..." if len(r.rule.value) > 30 else r.rule.value for r in failed_rules]
            severities = [r.severity.value for r in failed_rules]
            
            # Color by severity
            colors = ['red' if s == 'critical' else 'orange' if s == 'warning' else 'yellow' for s in severities]
            
            fig.add_trace(
                go.Bar(
                    x=rule_names,
                    y=[1] * len(failed_rules),
                    marker_color=colors,
                    name="Failed Rules",
                    hovertemplate="%{x}<extra></extra>"
                ),
                row=row, col=col
            )
        
        fig.update_yaxes(title_text="Rule Status", range=[0, 1], row=row, col=col)

    def _add_wave_characteristics_chart(self, fig: go.Figure, waves: List[Wave], row: int, col: int):
        """Add wave characteristics chart."""
        if not waves:
            return
        
        # Wave durations
        durations = [w.duration for w in waves]
        wave_labels = [w.wave_type.value for w in waves]
        
        fig.add_trace(
            go.Bar(
                x=wave_labels,
                y=durations,
                name="Duration",
                hovertemplate="%{y} periods<extra></extra>"
            ),
            row=row, col=col
        )
        
        fig.update_yaxes(title_text="Duration (periods)", row=row, col=col)

    def _add_fibonacci_analysis_chart(self, fig: go.Figure, fibonacci_analysis: FibonacciAnalysis, row: int, col: int):
        """Add Fibonacci analysis chart."""
        if not fibonacci_analysis.key_levels:
            return
        
        levels = [level.ratio for level in fibonacci_analysis.key_levels]
        prices = [level.price for level in fibonacci_analysis.key_levels]
        
        fig.add_trace(
            go.Scatter(
                x=levels,
                y=prices,
                mode='markers+lines',
                name="Fibonacci Levels",
                hovertemplate="Ratio: %{x}<br>Price: %{y:.2f}<extra></extra>"
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Fibonacci Ratio", row=row, col=col)
        fig.update_yaxes(title_text="Price", row=row, col=col)

    def plot_waves_with_pattern_memory(
        self,
        data: pd.DataFrame,
        waves: List[Wave],
        pattern_matches: List[PatternMatch],
        validation_result: Optional[WaveValidation] = None,
        fibonacci_analysis: Optional[FibonacciAnalysis] = None,
        title: str = "Elliott Wave Analysis with Pattern Memory",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create a chart showing waves with historical pattern matches.
        
        Args:
            data: OHLCV DataFrame
            waves: List of detected waves
            pattern_matches: List of pattern matches from memory
            validation_result: Optional validation result
            fibonacci_analysis: Optional Fibonacci analysis
            title: Chart title
            save_path: Optional path to save the chart
            
        Returns:
            Plotly figure
        """
        try:
            # Create subplots
            if self.show_volume:
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02,
                    row_heights=[0.7, 0.3],
                    subplot_titles=(title, "Volume")
                )
            else:
                fig = go.Figure()
            
            # Add candlestick chart
            candlestick = go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name="Price",
                increasing_line_color='#00ff00',
                decreasing_line_color='#ff0000'
            )
            
            if self.show_volume:
                fig.add_trace(candlestick, row=1, col=1)
            else:
                fig.add_trace(candlestick)
            
            # Add wave annotations
            if waves:
                if validation_result:
                    self._add_validated_wave_annotations(fig, waves, validation_result, data, row=1 if self.show_volume else None)
                else:
                    self._add_wave_annotations(fig, waves, data, row=1 if self.show_volume else None)
            
            # Add pattern memory information
            if pattern_matches:
                self._add_pattern_memory_info(fig, pattern_matches, data, row=1 if self.show_volume else None)
            
            # Add Fibonacci levels
            if fibonacci_analysis:
                self._add_fibonacci_levels(fig, fibonacci_analysis, data, row=1 if self.show_volume else None)
            
            # Add volume chart
            if self.show_volume:
                self._add_volume_chart(fig, data, row=2)
            
            # Update layout
            self._update_layout_with_pattern_memory(fig, title, pattern_matches)
            
            # Save if path provided
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Pattern memory chart saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating pattern memory chart: {e}")
            raise

    def _add_pattern_memory_info(
        self,
        fig: go.Figure,
        pattern_matches: List[PatternMatch],
        data: pd.DataFrame,
        row: Optional[int] = None
    ):
        """Add pattern memory information to the chart."""
        if not pattern_matches:
            return
        
        # Get best match
        best_match = pattern_matches[0]
        
        # Add pattern match annotation
        self._add_pattern_match_annotation(fig, best_match, data, row)
        
        # Add similarity breakdown
        self._add_similarity_breakdown(fig, best_match, data, row)
        
        # Add historical outcome prediction
        if best_match.outcome_prediction:
            self._add_outcome_prediction(fig, best_match, data, row)
        
        # Add predicted targets
        if best_match.predicted_targets:
            self._add_predicted_targets(fig, best_match, data, row)

    def _add_pattern_match_annotation(
        self,
        fig: go.Figure,
        pattern_match: PatternMatch,
        data: pd.DataFrame,
        row: Optional[int] = None
    ):
        """Add pattern match annotation to the chart."""
        historical_pattern = pattern_match.historical_pattern
        
        # Create annotation text
        annotation_text = f"<b>Historical Pattern Match</b><br>" + \
                         f"Similarity: {pattern_match.similarity_score:.1%}<br>" + \
                         f"Quality: {pattern_match.match_quality.title()}<br>" + \
                         f"Category: {historical_pattern.pattern_category.value}<br>" + \
                         f"Symbol: {historical_pattern.symbol}<br>" + \
                         f"Date: {historical_pattern.start_date.strftime('%Y-%m-%d')}<br>" + \
                         f"Confidence Adj: {pattern_match.confidence_adjustment:+.1%}"
        
        # Color based on match quality
        bg_color_map = {
            'excellent': 'rgba(0, 255, 0, 0.9)',
            'good': 'rgba(255, 255, 0, 0.9)',
            'fair': 'rgba(255, 165, 0, 0.9)',
            'poor': 'rgba(255, 0, 0, 0.9)'
        }
        bg_color = bg_color_map.get(pattern_match.match_quality, 'rgba(128, 128, 128, 0.9)')
        
        fig.add_annotation(
            x=data.index[-1] - pd.Timedelta(days=3),
            y=data['high'].max(),
            text=annotation_text,
            showarrow=False,
            bgcolor=bg_color,
            bordercolor='black',
            borderwidth=2,
            font=dict(size=10, color='black'),
            row=row,
            col=1
        )

    def _add_similarity_breakdown(
        self,
        fig: go.Figure,
        pattern_match: PatternMatch,
        data: pd.DataFrame,
        row: Optional[int] = None
    ):
        """Add similarity breakdown to the chart."""
        breakdown = pattern_match.similarity_breakdown
        
        # Create breakdown text
        breakdown_text = "<b>Similarity Breakdown:</b><br>"
        for metric, score in breakdown.items():
            if metric != 'composite':
                color = self._get_similarity_color(score)
                breakdown_text += f"<span style='color:{color}'>{metric.title()}: {score:.1%}</span><br>"
        
        fig.add_annotation(
            x=data.index[-1] - pd.Timedelta(days=5),
            y=data['high'].max() * 0.8,
            text=breakdown_text,
            showarrow=False,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='gray',
            borderwidth=1,
            font=dict(size=9, color='black'),
            row=row,
            col=1
        )

    def _add_outcome_prediction(
        self,
        fig: go.Figure,
        pattern_match: PatternMatch,
        data: pd.DataFrame,
        row: Optional[int] = None
    ):
        """Add historical outcome prediction to the chart."""
        if not pattern_match.outcome_prediction:
            return
        
        # Create prediction text
        prediction_text = f"<b>Historical Outcome:</b><br>" + \
                         f"{pattern_match.outcome_prediction}<br>"
        
        if pattern_match.predicted_timeframe:
            prediction_text += f"Expected: {pattern_match.predicted_timeframe}"
        
        fig.add_annotation(
            x=data.index[-1] - pd.Timedelta(days=7),
            y=data['low'].min(),
            text=prediction_text,
            showarrow=False,
            bgcolor='rgba(0, 255, 255, 0.9)',
            bordercolor='cyan',
            borderwidth=2,
            font=dict(size=10, color='black'),
            row=row,
            col=1
        )

    def _add_predicted_targets(
        self,
        fig: go.Figure,
        pattern_match: PatternMatch,
        data: pd.DataFrame,
        row: Optional[int] = None
    ):
        """Add predicted targets based on historical pattern."""
        if not pattern_match.predicted_targets:
            return
        
        current_time = data.index[-1]
        x_range = [current_time, current_time + pd.Timedelta(days=7)]
        
        for i, target in enumerate(pattern_match.predicted_targets):
            # Add target line
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=[target, target],
                    mode='lines',
                    line=dict(
                        color='purple',
                        width=2,
                        dash='dot'
                    ),
                    name=f"Historical Target {i+1}",
                    showlegend=True,
                    hovertemplate=f"Historical Target {i+1}<br>" +
                                 f"Price: {target:.2f}<br>" +
                                 f"Based on: {pattern_match.historical_pattern.pattern_id}<extra></extra>"
                ),
                row=row, col=1
            )
            
            # Add target marker
            fig.add_trace(
                go.Scatter(
                    x=[current_time + pd.Timedelta(days=7)],
                    y=[target],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='purple',
                        symbol='star'
                    ),
                    name=f"Target {i+1}",
                    showlegend=False,
                    hovertemplate=f"Historical Target {i+1}: {target:.2f}<extra></extra>"
                ),
                row=row, col=1
            )

    def _get_similarity_color(self, score: float) -> str:
        """Get color for similarity score."""
        if score >= 0.8:
            return 'green'
        elif score >= 0.6:
            return 'orange'
        else:
            return 'red'

    def _update_layout_with_pattern_memory(self, fig: go.Figure, title: str, pattern_matches: List[PatternMatch]):
        """Update layout with pattern memory information."""
        if pattern_matches:
            # Add match count to title
            match_count = len(pattern_matches)
            best_similarity = pattern_matches[0].similarity_score if pattern_matches else 0
            updated_title = f"{title} ({match_count} matches, best: {best_similarity:.1%})"
        else:
            updated_title = title
        
        fig.update_layout(
            title=updated_title,
            height=self.chart_height,
            width=self.chart_width,
            template=self.theme,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

    def create_pattern_memory_dashboard(
        self,
        data: pd.DataFrame,
        waves: List[Wave],
        pattern_matches: List[PatternMatch],
        pattern_memory_stats: Optional[Dict[str, Any]] = None,
        validation_result: Optional[WaveValidation] = None,
        fibonacci_analysis: Optional[FibonacciAnalysis] = None,
        title: str = "Pattern Memory Dashboard",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create a comprehensive pattern memory dashboard.
        
        Args:
            data: OHLCV DataFrame
            waves: List of detected waves
            pattern_matches: List of pattern matches
            pattern_memory_stats: Pattern memory statistics
            validation_result: Optional validation result
            fibonacci_analysis: Optional Fibonacci analysis
            title: Chart title
            save_path: Optional path to save the chart
            
        Returns:
            Plotly figure
        """
        try:
            # Create subplot structure for dashboard
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    "Elliott Wave Analysis with Pattern Memory",
                    "Pattern Match Similarities",
                    "Historical Pattern Categories",
                    "Pattern Memory Statistics",
                    "Similarity Breakdown",
                    "Historical Outcomes"
                ),
                specs=[
                    [{"colspan": 2}, None],
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}]
                ],
                vertical_spacing=0.08,
                horizontal_spacing=0.05
            )
            
            # 1. Main chart with pattern memory (top row, full width)
            candlestick = go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name="Price"
            )
            fig.add_trace(candlestick, row=1, col=1)
            
            if waves:
                if validation_result:
                    self._add_validated_wave_annotations(fig, waves, validation_result, data, row=1)
                else:
                    self._add_wave_annotations(fig, waves, data, row=1)
            
            if pattern_matches:
                self._add_pattern_memory_info(fig, pattern_matches, data, row=1)
            
            if fibonacci_analysis:
                self._add_fibonacci_levels(fig, fibonacci_analysis, data, row=1)
            
            # 2. Pattern match similarities (row 2, col 1)
            self._add_pattern_similarities_chart(fig, pattern_matches, row=2, col=1)
            
            # 3. Historical pattern categories (row 2, col 2)
            self._add_pattern_categories_chart(fig, pattern_matches, row=2, col=2)
            
            # 4. Pattern memory statistics (row 3, col 1)
            self._add_memory_statistics_chart(fig, pattern_memory_stats, row=3, col=1)
            
            # 5. Similarity breakdown (row 3, col 2)
            self._add_similarity_breakdown_chart(fig, pattern_matches, row=3, col=2)
            
            # Update layout
            fig.update_layout(
                title=title,
                height=1000,
                width=1400,
                template=self.theme,
                showlegend=True
            )
            
            # Save if path provided
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Pattern memory dashboard saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating pattern memory dashboard: {e}")
            raise

    def _add_pattern_similarities_chart(self, fig: go.Figure, pattern_matches: List[PatternMatch], row: int, col: int):
        """Add pattern similarities chart."""
        if not pattern_matches:
            return
        
        # Extract similarity scores and pattern IDs
        similarities = [match.similarity_score for match in pattern_matches]
        pattern_ids = [match.historical_pattern.pattern_id[:20] + "..." for match in pattern_matches]  # Truncate long IDs
        colors = [self._get_similarity_color(score) for score in similarities]
        
        fig.add_trace(
            go.Bar(
                x=pattern_ids,
                y=similarities,
                marker_color=colors,
                name="Pattern Similarities",
                hovertemplate="Pattern: %{x}<br>Similarity: %{y:.1%}<extra></extra>"
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Historical Patterns", row=row, col=col)
        fig.update_yaxes(title_text="Similarity Score", row=row, col=col)

    def _add_pattern_categories_chart(self, fig: go.Figure, pattern_matches: List[PatternMatch], row: int, col: int):
        """Add pattern categories chart."""
        if not pattern_matches:
            return
        
        # Count by category
        category_counts = {}
        for match in pattern_matches:
            category = match.historical_pattern.pattern_category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Create pie chart
        categories = list(category_counts.keys())
        counts = list(category_counts.values())
        
        fig.add_trace(
            go.Pie(
                labels=categories,
                values=counts,
                name="Pattern Categories",
                hovertemplate="Category: %{label}<br>Count: %{value}<extra></extra>"
            ),
            row=row, col=col
        )

    def _add_memory_statistics_chart(self, fig: go.Figure, stats: Optional[Dict[str, Any]], row: int, col: int):
        """Add memory statistics chart."""
        if not stats:
            return
        
        # Create bar chart of statistics
        stat_names = list(stats.keys())
        stat_values = list(stats.values())
        
        fig.add_trace(
            go.Bar(
                x=stat_names,
                y=stat_values,
                name="Memory Statistics",
                hovertemplate="Statistic: %{x}<br>Value: %{y}<extra></extra>"
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Statistics", row=row, col=col)
        fig.update_yaxes(title_text="Value", row=row, col=col)

    def _add_similarity_breakdown_chart(self, fig: go.Figure, pattern_matches: List[PatternMatch], row: int, col: int):
        """Add similarity breakdown chart."""
        if not pattern_matches:
            return
        
        # Get breakdown from best match
        best_match = pattern_matches[0]
        breakdown = best_match.similarity_breakdown
        
        # Remove composite score for breakdown chart
        breakdown_filtered = {k: v for k, v in breakdown.items() if k != 'composite'}
        
        metrics = list(breakdown_filtered.keys())
        scores = list(breakdown_filtered.values())
        colors = [self._get_similarity_color(score) for score in scores]
        
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=scores,
                marker_color=colors,
                name="Similarity Breakdown",
                hovertemplate="Metric: %{x}<br>Score: %{y:.1%}<extra></extra>"
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Similarity Metrics", row=row, col=col)
        fig.update_yaxes(title_text="Score", row=row, col=col)

    def plot_comprehensive_analysis(
        self,
        data: pd.DataFrame,
        waves: List[Wave],
        projection_scenarios: List[ProjectionScenario],
        pattern_matches: List[PatternMatch],
        validation_result: Optional[WaveValidation] = None,
        fibonacci_analysis: Optional[FibonacciAnalysis] = None,
        title: str = "Comprehensive Elliott Wave Analysis",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create a comprehensive analysis chart combining all features.
        
        Args:
            data: OHLCV DataFrame
            waves: List of detected waves
            projection_scenarios: List of projection scenarios
            pattern_matches: List of pattern matches
            validation_result: Optional validation result
            fibonacci_analysis: Optional Fibonacci analysis
            title: Chart title
            save_path: Optional path to save the chart
            
        Returns:
            Plotly figure
        """
        try:
            # Create subplots
            if self.show_volume:
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02,
                    row_heights=[0.7, 0.3],
                    subplot_titles=(title, "Volume")
                )
            else:
                fig = go.Figure()
            
            # Add candlestick chart
            candlestick = go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name="Price",
                increasing_line_color='#00ff00',
                decreasing_line_color='#ff0000'
            )
            
            if self.show_volume:
                fig.add_trace(candlestick, row=1, col=1)
            else:
                fig.add_trace(candlestick)
            
            # Add wave annotations
            if waves:
                if validation_result:
                    self._add_validated_wave_annotations(fig, waves, validation_result, data, row=1 if self.show_volume else None)
                else:
                    self._add_wave_annotations(fig, waves, data, row=1 if self.show_volume else None)
            
            # Add projection scenarios
            if projection_scenarios:
                self._add_projection_scenarios(fig, projection_scenarios, data, row=1 if self.show_volume else None)
            
            # Add pattern memory information
            if pattern_matches:
                self._add_pattern_memory_info(fig, pattern_matches, data, row=1 if self.show_volume else None)
            
            # Add Fibonacci levels
            if fibonacci_analysis:
                self._add_fibonacci_levels(fig, fibonacci_analysis, data, row=1 if self.show_volume else None)
            
            # Add volume chart
            if self.show_volume:
                self._add_volume_chart(fig, data, row=2)
            
            # Update layout
            self._update_comprehensive_layout(fig, title, projection_scenarios, pattern_matches)
            
            # Save if path provided
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Comprehensive analysis chart saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating comprehensive analysis chart: {e}")
            raise

    def _update_comprehensive_layout(
        self,
        fig: go.Figure,
        title: str,
        projection_scenarios: List[ProjectionScenario],
        pattern_matches: List[PatternMatch]
    ):
        """Update layout for comprehensive analysis."""
        # Build comprehensive title
        title_parts = [title]
        
        if projection_scenarios:
            projection_count = len(projection_scenarios)
            alt_count = sum(len(s.alternative_projections) for s in projection_scenarios)
            title_parts.append(f"({projection_count} projections, {alt_count} alternatives)")
        
        if pattern_matches:
            best_similarity = pattern_matches[0].similarity_score if pattern_matches else 0
            title_parts.append(f"({len(pattern_matches)} historical matches, best: {best_similarity:.1%})")
        
        comprehensive_title = " - ".join(title_parts)
        
        fig.update_layout(
            title=comprehensive_title,
            height=self.chart_height,
            width=self.chart_width,
            template=self.theme,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )


if __name__ == "__main__":
    # Example usage
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.data.data_loader import DataLoader
    from src.analysis.wave_detector import WaveDetector
    from src.analysis.fibonacci import FibonacciAnalyzer
    
    # Load sample data
    loader = DataLoader()
    data = loader.get_yahoo_data("AAPL", period="6mo")
    
    # Detect waves
    detector = WaveDetector()
    waves = detector.detect_waves(data)
    
    # Fibonacci analysis
    fib_analyzer = FibonacciAnalyzer()
    if len(data) > 50:
        high_price = data['high'].rolling(50).max().iloc[-1]
        low_price = data['low'].rolling(50).min().iloc[-1]
        current_price = data['close'].iloc[-1]
        
        fib_analysis = fib_analyzer.analyze_retracement(high_price, low_price, current_price, 'up')
    else:
        fib_analysis = None
    
    # Create visualizations
    visualizer = WaveVisualizer()
    
    # Main wave chart
    fig = visualizer.plot_waves(data, waves, fib_analysis)
    fig.show()
    
    # Wave progression chart
    if waves:
        progression_fig = visualizer.plot_wave_progression(waves)
        progression_fig.show()
    
    # Dashboard
    dashboard = visualizer.create_dashboard(data, waves, fib_analysis)
    dashboard.show()
    
    print("Visualizations created successfully!")
