"""
Technical Analysis API
Main API for running comprehensive Elliott Wave + Technical Confluence Analysis
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import our modules
from src.data.enhanced_data_fetcher import EnhancedDataFetcher
from src.analysis.technical_confluence import TechnicalConfluenceAnalyzer
from src.analysis.enhanced_wave_detector import EnhancedWaveDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.secret_key = 'technical_analysis_secret_2025'

# Initialize components
data_fetcher = EnhancedDataFetcher()
confluence_analyzer = TechnicalConfluenceAnalyzer()
wave_detector = EnhancedWaveDetector()

# Supported exchanges and timeframes
SUPPORTED_EXCHANGES = ['binance', 'bybit']
SUPPORTED_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']

@app.route('/api/analyze', methods=['POST'])
def analyze_symbol():
    """
    Main analysis endpoint that combines Elliott Wave detection with technical confluence
    """
    try:
        data = request.get_json()
        
        # Validate input parameters
        symbol = data.get('symbol', '').upper()
        timeframe = data.get('timeframe', '1h')
        exchange = data.get('exchange', 'binance').lower()
        limit = data.get('limit', 500)
        
        if not symbol:
            return jsonify({'error': 'Symbol is required'}), 400
        
        if exchange not in SUPPORTED_EXCHANGES:
            return jsonify({'error': f'Unsupported exchange. Supported: {SUPPORTED_EXCHANGES}'}), 400
        
        if timeframe not in SUPPORTED_TIMEFRAMES:
            return jsonify({'error': f'Unsupported timeframe. Supported: {SUPPORTED_TIMEFRAMES}'}), 400
        
        logger.info(f"🚀 Starting analysis for {symbol} on {exchange} ({timeframe})")
        
        # Step 1: Fetch market data
        logger.info(f"📊 Fetching market data...")
        market_data = data_fetcher.fetch_ohlcv_data(symbol, timeframe, exchange, limit)
        
        if market_data.empty:
            return jsonify({'error': f'No data available for {symbol} on {exchange}'}), 404
        
        # Add extended market calculations
        market_data = data_fetcher.calculate_extended_market_data(market_data)
        
        logger.info(f"✅ Fetched {len(market_data)} candles")
        
        # Step 2: Elliott Wave Analysis
        logger.info(f"🌊 Running Elliott Wave analysis...")
        elliott_analysis = wave_detector.detect_elliott_waves(market_data, symbol)
        
        if not elliott_analysis or elliott_analysis.get('validation_score', 0) < 0.1:
            return jsonify({
                'error': 'No valid Elliott Wave structures detected',
                'suggestion': 'Try a different timeframe or symbol'
            }), 200
        
        logger.info(f"✅ Elliott Wave analysis complete: {elliott_analysis.get('validation_score', 0):.1%} confidence")
        
        # Step 3: Technical Confluence Analysis
        logger.info(f"🧩 Running technical confluence analysis...")
        target_zones = confluence_analyzer.analyze_target_zones(market_data, elliott_analysis, timeframe)
        
        if not target_zones:
            return jsonify({
                'error': 'No target zones identified',
                'elliott_analysis': _format_elliott_analysis(elliott_analysis),
                'market_summary': _get_market_summary(market_data, symbol, exchange)
            }), 200
        
        logger.info(f"✅ Found {len(target_zones)} target zones")
        
        # Step 4: Get market summary
        market_summary = data_fetcher.get_market_summary(symbol, exchange)
        if not market_summary:
            market_summary = _get_market_summary(market_data, symbol, exchange)
        
        # Step 5: Format results
        results = confluence_analyzer.format_analysis_results(target_zones, market_summary)
        
        # Add Elliott Wave analysis to results
        results['elliott_analysis'] = _format_elliott_analysis(elliott_analysis)
        results['data_quality'] = _assess_data_quality(market_data)
        
        logger.info(f"🎯 Analysis complete: {len(target_zones)} targets, best confidence: {target_zones[0].confidence_level if target_zones else 'None'}")
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'timeframe': timeframe,
            'exchange': exchange,
            'analysis': results
        })
        
    except Exception as e:
        logger.error(f"❌ Analysis error: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/symbols/<exchange>', methods=['GET'])
def get_symbols(exchange):
    """Get available trading symbols for an exchange"""
    try:
        if exchange.lower() not in SUPPORTED_EXCHANGES:
            return jsonify({'error': f'Unsupported exchange: {exchange}'}), 400
        
        symbols = data_fetcher.get_available_symbols(exchange.lower())
        
        # Filter for popular pairs
        popular_symbols = []
        for symbol in symbols:
            if any(quote in symbol for quote in ['/USDT', '/BTC', '/ETH']):
                popular_symbols.append(symbol)
        
        return jsonify({
            'exchange': exchange,
            'total_symbols': len(symbols),
            'popular_symbols': popular_symbols[:50],  # Top 50 popular pairs
            'all_symbols': symbols
        })
        
    except Exception as e:
        logger.error(f"Error getting symbols: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/market-summary/<exchange>/<symbol>', methods=['GET'])
def get_market_summary_endpoint(exchange, symbol):
    """Get market summary for a specific symbol"""
    try:
        if exchange.lower() not in SUPPORTED_EXCHANGES:
            return jsonify({'error': f'Unsupported exchange: {exchange}'}), 400
        
        summary = data_fetcher.get_market_summary(symbol.upper(), exchange.lower())
        
        if not summary:
            return jsonify({'error': f'No data available for {symbol} on {exchange}'}), 404
        
        return jsonify(summary)
        
    except Exception as e:
        logger.error(f"Error getting market summary: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/multi-timeframe', methods=['POST'])
def multi_timeframe_analysis():
    """Analyze multiple timeframes for better confluence"""
    try:
        data = request.get_json()
        
        symbol = data.get('symbol', '').upper()
        timeframes = data.get('timeframes', ['1h', '4h', '1d'])
        exchange = data.get('exchange', 'binance').lower()
        
        if not symbol:
            return jsonify({'error': 'Symbol is required'}), 400
        
        if exchange not in SUPPORTED_EXCHANGES:
            return jsonify({'error': f'Unsupported exchange: {exchange}'}), 400
        
        # Validate timeframes
        invalid_tfs = [tf for tf in timeframes if tf not in SUPPORTED_TIMEFRAMES]
        if invalid_tfs:
            return jsonify({'error': f'Invalid timeframes: {invalid_tfs}'}), 400
        
        logger.info(f"🚀 Multi-timeframe analysis: {symbol} on {timeframes}")
        
        results = {}
        overall_confluence_zones = []
        
        for tf in timeframes:
            try:
                logger.info(f"📊 Analyzing {tf} timeframe...")
                
                # Fetch data for this timeframe
                market_data = data_fetcher.fetch_ohlcv_data(symbol, tf, exchange, 500)
                
                if market_data.empty:
                    results[tf] = {'error': 'No data available'}
                    continue
                
                # Elliott Wave analysis
                elliott_analysis = wave_detector.detect_elliott_waves(market_data, symbol)
                
                if elliott_analysis and elliott_analysis.get('validation_score', 0) >= 0.1:
                    # Technical confluence analysis
                    target_zones = confluence_analyzer.analyze_target_zones(market_data, elliott_analysis, tf)
                    
                    # Format results for this timeframe
                    market_summary = _get_market_summary(market_data, symbol, exchange)
                    tf_results = confluence_analyzer.format_analysis_results(target_zones, market_summary)
                    tf_results['elliott_analysis'] = _format_elliott_analysis(elliott_analysis)
                    
                    results[tf] = tf_results
                    
                    # Collect high-confidence zones for overall analysis
                    high_conf_zones = [tz for tz in target_zones if tz.confidence_level == "HIGH"]
                    overall_confluence_zones.extend(high_conf_zones)
                    
                else:
                    results[tf] = {'error': 'No valid Elliott Wave structures detected'}
                    
            except Exception as e:
                logger.error(f"Error analyzing {tf}: {e}")
                results[tf] = {'error': str(e)}
        
        # Cross-timeframe confluence analysis
        cross_tf_zones = _analyze_cross_timeframe_confluence(overall_confluence_zones)
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'exchange': exchange,
            'timeframes_analyzed': timeframes,
            'individual_results': results,
            'cross_timeframe_confluence': cross_tf_zones,
            'analysis_timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Multi-timeframe analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'supported_exchanges': SUPPORTED_EXCHANGES,
        'supported_timeframes': SUPPORTED_TIMEFRAMES,
        'version': '1.0.0'
    })

@app.route('/api/confluence-details', methods=['POST'])
def get_confluence_details():
    """Get detailed confluence analysis for a specific price level"""
    try:
        data = request.get_json()
        
        symbol = data.get('symbol', '').upper()
        price_level = data.get('price_level')
        timeframe = data.get('timeframe', '1h')
        exchange = data.get('exchange', 'binance').lower()
        
        if not symbol or not price_level:
            return jsonify({'error': 'Symbol and price_level are required'}), 400
        
        # Fetch market data
        market_data = data_fetcher.fetch_ohlcv_data(symbol, timeframe, exchange, 500)
        
        if market_data.empty:
            return jsonify({'error': f'No data available for {symbol}'}), 404
        
        # Analyze confluence at specific level
        confluence_analysis = confluence_analyzer._analyze_confluence_at_level(
            market_data, float(price_level), {}, timeframe
        )
        
        current_price = market_data['close'].iloc[-1]
        price_distance = ((float(price_level) - current_price) / current_price) * 100
        
        return jsonify({
            'symbol': symbol,
            'price_level': float(price_level),
            'current_price': float(current_price),
            'price_distance_pct': round(price_distance, 2),
            'confluence_score': confluence_analysis['score'],
            'confidence_level': confluence_analyzer._get_confidence_level(confluence_analysis['score']),
            'confluences': confluence_analysis['methods'],
            'detailed_analysis': confluence_analysis['details'],
            'probability': confluence_analyzer._calculate_probability(confluence_analysis['score']),
            'timeframe': timeframe,
            'analysis_timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Confluence details error: {e}")
        return jsonify({'error': str(e)}), 500

# Helper functions

def _format_elliott_analysis(elliott_analysis: Dict) -> Dict:
    """Format Elliott Wave analysis for API response"""
    if not elliott_analysis:
        return {}
    
    return {
        'wave_structure': elliott_analysis.get('wave_structure', 'unknown'),
        'validation_score': round(elliott_analysis.get('validation_score', 0), 3),
        'direction': elliott_analysis.get('direction', 'neutral'),
        'waves_detected': len(elliott_analysis.get('waves', [])),
        'current_wave': elliott_analysis.get('current_wave', 'unknown'),
        'next_expected': elliott_analysis.get('next_expected', 'unknown'),
        'fibonacci_levels': elliott_analysis.get('fibonacci_levels', {}),
        'rule_compliance': elliott_analysis.get('rule_compliance', {}),
        'recommendations': elliott_analysis.get('recommendations', [])
    }

def _get_market_summary(market_data: pd.DataFrame, symbol: str, exchange: str) -> Dict:
    """Generate market summary from OHLCV data"""
    try:
        current_price = market_data['close'].iloc[-1]
        open_price = market_data['open'].iloc[0]
        high_price = market_data['high'].max()
        low_price = market_data['low'].min()
        
        total_volume = market_data['volume'].sum()
        avg_volume = market_data['volume'].mean()
        
        # Calculate price changes
        price_change = current_price - market_data['close'].iloc[-2] if len(market_data) > 1 else 0
        price_change_pct = (price_change / market_data['close'].iloc[-2] * 100) if len(market_data) > 1 else 0
        
        return {
            'symbol': symbol,
            'exchange': exchange,
            'current_price': round(float(current_price), 6),
            'open_price': round(float(open_price), 6),
            'high_price': round(float(high_price), 6),
            'low_price': round(float(low_price), 6),
            'price_change': round(float(price_change), 6),
            'price_change_pct': round(price_change_pct, 2),
            'total_volume': round(float(total_volume), 2),
            'avg_volume': round(float(avg_volume), 2),
            'data_points': len(market_data),
            'timespan': f"{market_data.index[0]} to {market_data.index[-1]}",
            'last_update': market_data.index[-1].isoformat()
        }
    except Exception as e:
        logger.error(f"Error creating market summary: {e}")
        return {'error': str(e)}

def _assess_data_quality(market_data: pd.DataFrame) -> Dict:
    """Assess the quality of market data"""
    try:
        total_points = len(market_data)
        
        # Check for missing data
        missing_data = market_data.isnull().sum().sum()
        
        # Check for zero volume periods
        zero_volume = (market_data['volume'] == 0).sum()
        
        # Check for price anomalies (gaps)
        price_changes = market_data['close'].pct_change().abs()
        large_gaps = (price_changes > 0.1).sum()  # >10% price changes
        
        # Calculate volatility
        volatility = market_data['close'].pct_change().std()
        
        # Data quality score
        quality_score = 100
        if missing_data > 0:
            quality_score -= min(missing_data * 2, 20)
        if zero_volume > total_points * 0.1:
            quality_score -= 15
        if large_gaps > total_points * 0.05:
            quality_score -= 10
        
        quality_level = "HIGH" if quality_score >= 85 else "MEDIUM" if quality_score >= 70 else "LOW"
        
        return {
            'total_data_points': total_points,
            'missing_data_points': int(missing_data),
            'zero_volume_periods': int(zero_volume),
            'large_price_gaps': int(large_gaps),
            'volatility': round(float(volatility), 4),
            'quality_score': quality_score,
            'quality_level': quality_level,
            'timespan_hours': (market_data.index[-1] - market_data.index[0]).total_seconds() / 3600
        }
    except Exception as e:
        logger.error(f"Error assessing data quality: {e}")
        return {'error': str(e)}

def _analyze_cross_timeframe_confluence(zones_list: List) -> Dict:
    """Analyze confluence across multiple timeframes"""
    try:
        if not zones_list:
            return {'message': 'No cross-timeframe zones to analyze'}
        
        # Group zones by similar price levels (within 1% tolerance)
        tolerance = 0.01
        confluence_groups = []
        
        for zone in zones_list:
            price = zone.price_level
            
            # Find existing group for this price level
            found_group = None
            for group in confluence_groups:
                group_avg_price = np.mean([z.price_level for z in group])
                if abs(price - group_avg_price) / group_avg_price <= tolerance:
                    found_group = group
                    break
            
            if found_group:
                found_group.append(zone)
            else:
                confluence_groups.append([zone])
        
        # Score confluence groups
        scored_groups = []
        for group in confluence_groups:
            if len(group) >= 2:  # At least 2 timeframes agree
                avg_price = np.mean([z.price_level for z in group])
                total_score = sum(z.confluence_score for z in group)
                timeframes = [z.timeframe for z in group]
                
                scored_groups.append({
                    'price_level': round(avg_price, 6),
                    'timeframes_count': len(group),
                    'timeframes': timeframes,
                    'total_confluence_score': total_score,
                    'avg_confluence_score': round(total_score / len(group), 1),
                    'wave_targets': [z.wave_target for z in group],
                    'confluence_strength': 'VERY_HIGH' if len(group) >= 3 else 'HIGH'
                })
        
        # Sort by confluence strength
        scored_groups.sort(key=lambda x: (x['timeframes_count'], x['total_confluence_score']), reverse=True)
        
        return {
            'total_groups_found': len(scored_groups),
            'high_confluence_zones': scored_groups,
            'analysis_summary': {
                'strongest_confluence': scored_groups[0] if scored_groups else None,
                'avg_timeframes_per_zone': round(np.mean([g['timeframes_count'] for g in scored_groups]), 1) if scored_groups else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error in cross-timeframe analysis: {e}")
        return {'error': str(e)}

if __name__ == '__main__':
    print("🚀 Technical Analysis API Starting...")
    print("=" * 60)
    print("📊 Supported Exchanges:", SUPPORTED_EXCHANGES)
    print("⏰ Supported Timeframes:", SUPPORTED_TIMEFRAMES)
    print("🌊 Elliott Wave Detection: Enabled")
    print("🧩 Technical Confluence: Enabled")
    print("🎯 Target Zone Analysis: Enabled")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5001, debug=True)
