#!/usr/bin/env python3
"""
Command-Line Technical Analysis Tool
Quick Elliott Wave + Technical Confluence analysis from command line
"""

import sys
import os
import argparse
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

def analyze_symbol(symbol, timeframe='1h', exchange='binance', limit=500, detailed=False):
    """Perform technical analysis on a symbol"""
    try:
        from src.data.enhanced_data_fetcher import EnhancedDataFetcher
        from src.analysis.technical_confluence import TechnicalConfluenceAnalyzer
        from src.analysis.enhanced_wave_detector import EnhancedWaveDetector
        
        print(f"🚀 Analyzing {symbol} on {exchange} ({timeframe})")
        print("=" * 60)
        
        # Initialize components
        data_fetcher = EnhancedDataFetcher()
        confluence_analyzer = TechnicalConfluenceAnalyzer()
        wave_detector = EnhancedWaveDetector()
        
        # Fetch market data
        print(f"📊 Fetching market data...")
        market_data = data_fetcher.fetch_ohlcv_data(symbol, timeframe, exchange, limit)
        
        if market_data.empty:
            print(f"❌ No data available for {symbol} on {exchange}")
            return
        
        print(f"✅ Fetched {len(market_data)} candles")
        print(f"💰 Current Price: ${market_data['close'].iloc[-1]:.6f}")
        print(f"📈 24h Change: {((market_data['close'].iloc[-1] / market_data['close'].iloc[-24] - 1) * 100):+.2f}%" if len(market_data) >= 24 else "N/A")
        
        # Elliott Wave Analysis
        print(f"\n🌊 Elliott Wave Analysis...")
        elliott_analysis = wave_detector.detect_elliott_waves(market_data, symbol)
        
        if not elliott_analysis or elliott_analysis.get('validation_score', 0) < 0.1:
            print(f"❌ No valid Elliott Wave structures detected")
            print(f"💡 Try a different timeframe or symbol")
            return
        
        print(f"✅ Elliott Wave Structure: {elliott_analysis.get('wave_structure', 'Unknown').upper()}")
        print(f"🎯 Validation Score: {elliott_analysis.get('validation_score', 0):.1%}")
        print(f"📊 Direction: {elliott_analysis.get('direction', 'neutral').upper()}")
        print(f"🌊 Waves Detected: {len(elliott_analysis.get('waves', []))}")
        
        # Technical Confluence Analysis
        print(f"\n🧩 Technical Confluence Analysis...")
        target_zones = confluence_analyzer.analyze_target_zones(market_data, elliott_analysis, timeframe)
        
        if not target_zones:
            print(f"❌ No target zones identified")
            return
        
        print(f"✅ Found {len(target_zones)} target zones")
        
        # Categorize by confidence
        high_confidence = [tz for tz in target_zones if tz.confidence_level == "HIGH"]
        medium_confidence = [tz for tz in target_zones if tz.confidence_level == "MEDIUM"]
        low_confidence = [tz for tz in target_zones if tz.confidence_level == "LOW"]
        
        print(f"🔥 High Confidence: {len(high_confidence)}")
        print(f"⚠️ Medium Confidence: {len(medium_confidence)}")
        print(f"🔽 Low Confidence: {len(low_confidence)}")
        
        # Display results
        current_price = market_data['close'].iloc[-1]
        
        print(f"\n🎯 TARGET ZONES ANALYSIS")
        print("=" * 60)
        
        if high_confidence:
            print(f"🔥 HIGH CONFIDENCE TARGETS:")
            for i, zone in enumerate(high_confidence[:5], 1):  # Top 5
                price_change = ((zone.price_level - current_price) / current_price) * 100
                print(f"   {i}. ${zone.price_level:.6f} ({price_change:+.2f}%)")
                print(f"      Wave: {zone.wave_target}")
                print(f"      Basis: {zone.elliott_basis}")
                print(f"      Confluence Score: {zone.confluence_score}/10")
                print(f"      Probability: {zone.probability:.1%}")
                print(f"      Risk/Reward: {zone.risk_reward_ratio}")
                if detailed:
                    print(f"      Confluences: {', '.join(zone.confluences[:3])}...")
                print()
        
        if medium_confidence and not high_confidence:
            print(f"⚠️ MEDIUM CONFIDENCE TARGETS:")
            for i, zone in enumerate(medium_confidence[:3], 1):  # Top 3
                price_change = ((zone.price_level - current_price) / current_price) * 100
                print(f"   {i}. ${zone.price_level:.6f} ({price_change:+.2f}%)")
                print(f"      Wave: {zone.wave_target}")
                print(f"      Confluence Score: {zone.confluence_score}/10")
                print(f"      Probability: {zone.probability:.1%}")
                print()
        
        # Summary
        best_zone = target_zones[0]
        print(f"🏆 BEST TARGET RECOMMENDATION:")
        print(f"   Target: ${best_zone.price_level:.6f}")
        print(f"   Expected Move: {((best_zone.price_level - current_price) / current_price) * 100:+.2f}%")
        print(f"   Confidence: {best_zone.confidence_level}")
        print(f"   Wave: {best_zone.wave_target}")
        print(f"   Probability: {best_zone.probability:.1%}")
        
        # Market Context
        print(f"\n📊 MARKET CONTEXT:")
        volatility = market_data['close'].pct_change().std()
        volume_trend = "HIGH" if market_data['volume'].iloc[-1] > market_data['volume'].mean() * 1.5 else "NORMAL"
        
        print(f"   Volatility: {volatility:.3f} ({'HIGH' if volatility > 0.03 else 'MODERATE' if volatility > 0.015 else 'LOW'})")
        print(f"   Volume: {volume_trend}")
        print(f"   Data Quality: HIGH ({len(market_data)} data points)")
        
        print(f"\n📅 Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if detailed:
            # Save detailed JSON output
            output_file = f"{symbol.replace('/', '_')}_{timeframe}_analysis.json"
            
            market_summary = data_fetcher.get_market_summary(symbol, exchange)
            results = confluence_analyzer.format_analysis_results(target_zones, market_summary or {})
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"💾 Detailed analysis saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

def list_symbols(exchange='binance', limit=20):
    """List available symbols"""
    try:
        from src.data.enhanced_data_fetcher import EnhancedDataFetcher
        
        data_fetcher = EnhancedDataFetcher()
        
        print(f"📋 Getting symbols from {exchange.upper()}...")
        symbols = data_fetcher.get_available_symbols(exchange)
        
        if not symbols:
            print(f"❌ No symbols found for {exchange}")
            return
        
        # Filter for popular USDT pairs
        usdt_pairs = [s for s in symbols if s.endswith('/USDT')]
        
        print(f"✅ Found {len(symbols)} total symbols")
        print(f"🔥 Popular USDT pairs (showing {min(limit, len(usdt_pairs))}):")
        
        for i, symbol in enumerate(usdt_pairs[:limit], 1):
            print(f"   {i:2d}. {symbol}")
        
        if len(usdt_pairs) > limit:
            print(f"   ... and {len(usdt_pairs) - limit} more")
        
    except Exception as e:
        print(f"❌ Failed to get symbols: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Elliott Wave + Technical Confluence Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s BTC/USDT                    # Analyze BTC/USDT on 1h timeframe
  %(prog)s ETH/USDT -t 4h              # Analyze ETH on 4h timeframe
  %(prog)s SOL/USDT -e bybit -t 1d     # Analyze SOL on Bybit daily
  %(prog)s --list-symbols              # Show available symbols
  %(prog)s BTC/USDT --detailed         # Save detailed JSON output

Supported Exchanges: binance, bybit
Supported Timeframes: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w
        """
    )
    
    parser.add_argument('symbol', nargs='?', help='Trading pair symbol (e.g., BTC/USDT)')
    parser.add_argument('-t', '--timeframe', default='1h', 
                       choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'],
                       help='Timeframe (default: 1h)')
    parser.add_argument('-e', '--exchange', default='binance',
                       choices=['binance', 'bybit'],
                       help='Exchange (default: binance)')
    parser.add_argument('-l', '--limit', type=int, default=500,
                       help='Number of candles to fetch (default: 500)')
    parser.add_argument('--detailed', action='store_true',
                       help='Save detailed JSON output')
    parser.add_argument('--list-symbols', action='store_true',
                       help='List available symbols')
    
    args = parser.parse_args()
    
    print("🚀 Elliott Wave Technical Analysis Tool")
    print("🌊 Elliott Wave Theory + Technical Confluence")
    print("🎯 High-Probability Target Zone Detection")
    print()
    
    if args.list_symbols:
        list_symbols(args.exchange)
    elif args.symbol:
        analyze_symbol(
            args.symbol.upper(),
            args.timeframe,
            args.exchange,
            args.limit,
            args.detailed
        )
    else:
        parser.print_help()
        print("\n💡 Quick start: python analyze.py BTC/USDT")

if __name__ == "__main__":
    main()
