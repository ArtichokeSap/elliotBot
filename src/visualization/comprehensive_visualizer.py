"""
Comprehensive Elliott Wave Visualizer
Displays ALL waves (1,2,3,4,5,A,B,C) with validation indicators and subwaves
"""

import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ComprehensiveWaveVisualizer:
    """
    Comprehensive visualizer for Elliott Wave analysis with validation
    """
    
    def __init__(self):
        self.colors = {
            'bullish_wave': '#00C851',
            'bearish_wave': '#FF4444',
            'corrective_wave': '#FF8800',
            'subwave': '#6C757D',
            'fibonacci': '#007BFF',
            'validation_excellent': '#00C851',
            'validation_good': '#FFC107',
            'validation_poor': '#DC3545',
            'background': '#1E1E1E',
            'grid': '#404040',
            'text': '#FFFFFF'
        }
        
        self.wave_styles = {
            'impulse': dict(line=dict(width=3, dash='solid')),
            'corrective': dict(line=dict(width=2, dash='dot')),
            'subwave': dict(line=dict(width=1, dash='dash'))
        }
    
    def create_comprehensive_chart(self, price_data: pd.DataFrame, 
                                 analysis_result: Dict[str, Any],
                                 title: str = "Elliott Wave Analysis") -> str:
        """
        Create comprehensive chart showing all waves and validation
        """
        try:
            # Create subplot with secondary y-axis for validation
            fig = make_subplots(
                rows=3, cols=1,
                row_heights=[0.7, 0.15, 0.15],
                subplot_titles=[
                    'Elliott Wave Analysis with Validation',
                    'Wave Confidence Scores',
                    'Rule Compliance Status'
                ],
                vertical_spacing=0.05,
                specs=[[{"secondary_y": False}],
                       [{"secondary_y": False}],
                       [{"secondary_y": False}]]
            )
            
            # Main price chart with waves
            self._add_price_chart(fig, price_data, row=1)
            self._add_main_waves(fig, analysis_result, row=1)
            self._add_subwaves(fig, analysis_result, row=1)
            self._add_fibonacci_levels(fig, analysis_result, price_data, row=1)
            self._add_validation_markers(fig, analysis_result, row=1)
            
            # Confidence scores chart
            self._add_confidence_chart(fig, analysis_result, row=2)
            
            # Rule compliance chart
            self._add_compliance_chart(fig, analysis_result, row=3)
            
            # Update layout
            self._update_layout(fig, title, analysis_result)
            
            # Generate HTML
            html_content = self._generate_html_report(fig, analysis_result)
            
            return html_content
            
        except Exception as e:
            logger.error(f"Error creating comprehensive chart: {e}")
            return self._create_error_chart(str(e))
            
    def create_web_figure(self, price_data: pd.DataFrame, 
                         analysis_result: Dict[str, Any],
                         title: str = "Elliott Wave Analysis"):
        """
        Create comprehensive figure for web applications (returns Plotly figure object)
        """
        try:
            # Create subplot with secondary y-axis for validation
            fig = make_subplots(
                rows=3, cols=1,
                row_heights=[0.7, 0.15, 0.15],
                subplot_titles=[
                    'Elliott Wave Analysis with Validation',
                    'Wave Confidence Scores',
                    'Rule Compliance Status'
                ],
                vertical_spacing=0.05,
                specs=[[{"secondary_y": False}],
                       [{"secondary_y": False}],
                       [{"secondary_y": False}]]
            )
            
            # Price chart with waves
            self._add_price_chart(fig, price_data, row=1)
            self._add_main_waves(fig, analysis_result, row=1)
            self._add_subwaves(fig, analysis_result, row=1)
            self._add_fibonacci_levels(fig, analysis_result, price_data, row=1)
            self._add_validation_markers(fig, analysis_result, row=1)
            
            # Confidence scores chart
            self._add_confidence_chart(fig, analysis_result, row=2)
            
            # Rule compliance chart
            self._add_compliance_chart(fig, analysis_result, row=3)
            
            # Update layout
            self._update_layout(fig, title, analysis_result)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating web figure: {e}")
            # Return a simple error figure
            from plotly.graph_objects import Figure, Scatter
            error_fig = Figure()
            error_fig.add_trace(Scatter(x=[0], y=[0], text=[f"Error: {e}"], mode='text'))
            return error_fig
    
    def _add_price_chart(self, fig, price_data: pd.DataFrame, row: int):
        """Add main price candlestick chart"""
        fig.add_trace(
            go.Candlestick(
                x=price_data.index,
                open=price_data['open'],
                high=price_data['high'],
                low=price_data['low'],
                close=price_data['close'],
                name='Price',
                increasing_line_color='#00C851',
                decreasing_line_color='#FF4444',
                increasing_fillcolor='rgba(0, 200, 81, 0.3)',
                decreasing_fillcolor='rgba(255, 68, 68, 0.3)'
            ),
            row=row, col=1
        )
    
    def _add_main_waves(self, fig, analysis_result: Dict[str, Any], row: int):
        """Add main Elliott waves (1,2,3,4,5 or A,B,C)"""
        waves = analysis_result.get('waves', [])
        
        for i, wave in enumerate(waves):
            # Determine wave color based on type and validation
            confidence = wave.get('confidence', 0.5)
            direction = wave.get('direction', 'bullish')
            wave_type = wave.get('wave_type', 'impulse')
            
            if confidence >= 0.8:
                color = self.colors['validation_excellent']
            elif confidence >= 0.6:
                color = self.colors['validation_good']
            else:
                color = self.colors['validation_poor']
            
            # Add wave line
            fig.add_trace(
                go.Scatter(
                    x=[wave['start_time'], wave['end_time']],
                    y=[wave['start_price'], wave['end_price']],
                    mode='lines+markers+text',
                    name=f"Wave {wave['wave']}",
                    line=dict(color=color, width=3),
                    marker=dict(
                        size=8,
                        color=color,
                        symbol='circle'
                    ),
                    text=[f"Wave {wave['wave']}", ""],
                    textposition="top center",
                    textfont=dict(
                        size=12,
                        color=color,
                        family="Arial Black"
                    ),
                    hovertemplate=(
                        f"<b>Wave {wave['wave']}</b><br>"
                        f"Type: {wave_type}<br>"
                        f"Direction: {direction}<br>"
                        f"Length: {wave.get('length', 0):.4f}<br>"
                        f"Confidence: {confidence:.1%}<br>"
                        f"<extra></extra>"
                    )
                ),
                row=row, col=1
            )
    
    def _add_subwaves(self, fig, analysis_result: Dict[str, Any], row: int):
        """Add subwaves for detailed analysis"""
        subwaves_data = analysis_result.get('subwaves', {})
        
        for parent_wave, subwaves in subwaves_data.items():
            for subwave in subwaves:
                confidence = subwave.get('confidence', 0.5)
                
                # Subwaves are shown with thinner, dashed lines
                fig.add_trace(
                    go.Scatter(
                        x=[subwave['start_time'], subwave['end_time']],
                        y=[subwave['start_price'], subwave['end_price']],
                        mode='lines+text',
                        name=f"Subwave {subwave['label']}",
                        line=dict(
                            color=self.colors['subwave'],
                            width=1,
                            dash='dash'
                        ),
                        text=[subwave['label'].split('.')[-1], ""],
                        textposition="middle center",
                        textfont=dict(
                            size=8,
                            color=self.colors['subwave']
                        ),
                        showlegend=False,
                        hovertemplate=(
                            f"<b>{subwave['label']}</b><br>"
                            f"Type: {subwave.get('wave_type', 'unknown')}<br>"
                            f"Confidence: {confidence:.1%}<br>"
                            f"<extra></extra>"
                        )
                    ),
                    row=row, col=1
                )
    
    def _add_fibonacci_levels(self, fig, analysis_result: Dict[str, Any], 
                            price_data: pd.DataFrame, row: int):
        """Add Fibonacci retracement and extension levels"""
        fibonacci_levels = analysis_result.get('fibonacci_levels', {})
        
        if not fibonacci_levels:
            return
        
        price_min = price_data['low'].min()
        price_max = price_data['high'].max()
        time_range = [price_data.index[0], price_data.index[-1]]
        
        # Common Fibonacci levels
        fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618]
        
        for ratio in fib_ratios:
            level_price = price_min + (price_max - price_min) * ratio
            
            fig.add_trace(
                go.Scatter(
                    x=time_range,
                    y=[level_price, level_price],
                    mode='lines',
                    name=f'Fib {ratio:.1%}',
                    line=dict(
                        color=self.colors['fibonacci'],
                        width=1,
                        dash='dot'
                    ),
                    opacity=0.5,
                    showlegend=False,
                    hovertemplate=f"Fibonacci {ratio:.1%}: {level_price:.4f}<extra></extra>"
                ),
                row=row, col=1
            )
            
            # Add level label
            fig.add_annotation(
                x=time_range[-1],
                y=level_price,
                text=f"{ratio:.1%}",
                showarrow=False,
                xanchor="left",
                font=dict(size=8, color=self.colors['fibonacci']),
                row=row, col=1
            )
    
    def _add_validation_markers(self, fig, analysis_result: Dict[str, Any], row: int):
        """Add validation status markers"""
        validation_score = analysis_result.get('validation_score', 0)
        
        # Overall validation indicator
        score_color = (
            self.colors['validation_excellent'] if validation_score >= 0.8
            else self.colors['validation_good'] if validation_score >= 0.6
            else self.colors['validation_poor']
        )
        
        # Add validation score indicator
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=f"<b>Validation: {validation_score:.1%}</b>",
            showarrow=False,
            font=dict(size=14, color=score_color),
            bgcolor="rgba(0,0,0,0.7)",
            bordercolor=score_color,
            borderwidth=2,
            row=row, col=1
        )
    
    def _add_confidence_chart(self, fig, analysis_result: Dict[str, Any], row: int):
        """Add wave confidence scores bar chart"""
        waves = analysis_result.get('waves', [])
        
        if not waves:
            return
        
        wave_labels = [f"Wave {wave['wave']}" for wave in waves]
        confidences = [wave.get('confidence', 0) * 100 for wave in waves]
        
        # Color bars based on confidence level
        colors = []
        for conf in confidences:
            if conf >= 80:
                colors.append(self.colors['validation_excellent'])
            elif conf >= 60:
                colors.append(self.colors['validation_good'])
            else:
                colors.append(self.colors['validation_poor'])
        
        fig.add_trace(
            go.Bar(
                x=wave_labels,
                y=confidences,
                name='Wave Confidence',
                marker_color=colors,
                text=[f"{conf:.0f}%" for conf in confidences],
                textposition='auto',
                hovertemplate="<b>%{x}</b><br>Confidence: %{y:.0f}%<extra></extra>"
            ),
            row=row, col=1
        )
    
    def _add_compliance_chart(self, fig, analysis_result: Dict[str, Any], row: int):
        """Add rule compliance status chart"""
        rule_compliance = analysis_result.get('rule_compliance', {})
        
        if not rule_compliance:
            return
        
        rule_names = []
        rule_scores = []
        rule_colors = []
        
        for rule_name, rule_data in rule_compliance.items():
            if isinstance(rule_data, dict) and 'score' in rule_data:
                score = rule_data['score'] * 100
                rule_names.append(rule_name.replace('_', ' ').title())
                rule_scores.append(score)
                
                if score >= 80:
                    rule_colors.append(self.colors['validation_excellent'])
                elif score >= 60:
                    rule_colors.append(self.colors['validation_good'])
                else:
                    rule_colors.append(self.colors['validation_poor'])
        
        if rule_names:
            fig.add_trace(
                go.Bar(
                    x=rule_names,
                    y=rule_scores,
                    name='Rule Compliance',
                    marker_color=rule_colors,
                    text=[f"{score:.0f}%" for score in rule_scores],
                    textposition='auto',
                    hovertemplate="<b>%{x}</b><br>Compliance: %{y:.0f}%<extra></extra>"
                ),
                row=row, col=1
            )
    
    def _update_layout(self, fig, title: str, analysis_result: Dict[str, Any]):
        """Update chart layout with professional styling"""
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                x=0.5,
                font=dict(size=24, color=self.colors['text'])
            ),
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text']),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=10)
            ),
            margin=dict(l=50, r=50, t=100, b=50),
            height=800
        )
        
        # Update x-axes
        for i in range(1, 4):
            fig.update_xaxes(
                gridcolor=self.colors['grid'],
                gridwidth=1,
                row=i, col=1
            )
        
        # Update y-axes
        fig.update_yaxes(
            gridcolor=self.colors['grid'],
            gridwidth=1,
            title="Price",
            row=1, col=1
        )
        
        fig.update_yaxes(
            title="Confidence %",
            range=[0, 100],
            row=2, col=1
        )
        
        fig.update_yaxes(
            title="Compliance %",
            range=[0, 100],
            row=3, col=1
        )
    
    def _generate_html_report(self, fig, analysis_result: Dict[str, Any]) -> str:
        """Generate comprehensive HTML report"""
        # Convert plotly figure to HTML
        chart_html = pyo.plot(fig, output_type='div', include_plotlyjs=True)
        
        # Generate detailed analysis
        summary = analysis_result.get('summary', 'No analysis available')
        validation_score = analysis_result.get('validation_score', 0) * 100
        recommendations = analysis_result.get('recommendations', [])
        issues = analysis_result.get('issues', [])
        
        # Create full HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Comprehensive Elliott Wave Analysis</title>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background-color: #1E1E1E;
                    color: #FFFFFF;
                    margin: 0;
                    padding: 20px;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .summary-box {{
                    background: linear-gradient(135deg, #2D3436, #636E72);
                    border-radius: 10px;
                    padding: 20px;
                    margin: 20px 0;
                    border-left: 5px solid {'#00C851' if validation_score >= 80 else '#FFC107' if validation_score >= 60 else '#DC3545'};
                }}
                .validation-score {{
                    font-size: 2em;
                    font-weight: bold;
                    color: {'#00C851' if validation_score >= 80 else '#FFC107' if validation_score >= 60 else '#DC3545'};
                }}
                .section {{
                    margin: 20px 0;
                    padding: 15px;
                    background-color: #2D3436;
                    border-radius: 8px;
                }}
                .section h3 {{
                    margin-top: 0;
                    color: #74B9FF;
                }}
                .recommendation, .issue {{
                    margin: 10px 0;
                    padding: 10px;
                    border-radius: 5px;
                }}
                .recommendation {{
                    background-color: rgba(0, 200, 81, 0.1);
                    border-left: 3px solid #00C851;
                }}
                .issue {{
                    background-color: rgba(255, 68, 68, 0.1);
                    border-left: 3px solid #FF4444;
                }}
                .chart-container {{
                    margin: 30px 0;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🌊 Comprehensive Elliott Wave Analysis</h1>
                <div class="validation-score">Validation Score: {validation_score:.1f}%</div>
            </div>
            
            <div class="summary-box">
                <h2>📊 Analysis Summary</h2>
                <p>{summary}</p>
            </div>
            
            <div class="chart-container">
                {chart_html}
            </div>
            
            <div class="section">
                <h3>💡 Recommendations</h3>
                {''.join([f'<div class="recommendation">• {rec}</div>' for rec in recommendations]) if recommendations else '<p>No specific recommendations at this time.</p>'}
            </div>
            
            <div class="section">
                <h3>⚠️ Issues Identified</h3>
                {''.join([f'<div class="issue">• {issue}</div>' for issue in issues]) if issues else '<p>No critical issues identified.</p>'}
            </div>
            
            <div class="section">
                <h3>📈 Wave Structure Details</h3>
                <p><strong>Pattern Type:</strong> {analysis_result.get('wave_structure', 'Unknown')}</p>
                <p><strong>Direction:</strong> {analysis_result.get('direction', 'Unknown')}</p>
                <p><strong>Wave Count:</strong> {len(analysis_result.get('waves', []))}</p>
            </div>
            
            <div class="section">
                <h3>📋 Detailed Report</h3>
                <pre style="background-color: #1E1E1E; padding: 15px; border-radius: 5px; font-size: 12px; overflow-x: auto;">
{analysis_result.get('detailed_report', 'No detailed report available')}
                </pre>
            </div>
            
            <footer style="text-align: center; margin-top: 50px; color: #74B9FF;">
                <p>Generated by Comprehensive Elliott Wave Analyzer • {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </footer>
        </body>
        </html>
        """
        
        return html_content
    
    def _create_error_chart(self, error_message: str) -> str:
        """Create error chart when visualization fails"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Elliott Wave Analysis Error</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #1E1E1E;
                    color: #FFFFFF;
                    padding: 20px;
                    text-align: center;
                }}
                .error-box {{
                    background-color: #FF4444;
                    padding: 20px;
                    border-radius: 10px;
                    margin: 50px auto;
                    max-width: 600px;
                }}
            </style>
        </head>
        <body>
            <div class="error-box">
                <h2>❌ Visualization Error</h2>
                <p>{error_message}</p>
                <p>Please check the input data and try again.</p>
            </div>
        </body>
        </html>
        """
