#!/usr/bin/env python3
"""
Test script to verify confluence analysis display in web app
This script tests the enhanced web interface with comprehensive confluence information
"""

import requests
import json
import time

def test_confluence_api():
    """Test the confluence analysis API endpoint."""
    print("🧪 Testing Enhanced Confluence Analysis API...")
    
    # Test data
    test_symbols = ['AAPL', 'BTC-USD', 'EURUSD=X']
    
    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol}...")
        
        try:
            # Make API request
            response = requests.post('http://localhost:5000/api/analyze', json={
                'symbol': symbol,
                'timeframe': '1d'
            }, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success'):
                    print(f"✅ Analysis successful for {symbol}")
                    
                    # Check for confluence data
                    target_zones = data.get('target_zones', [])
                    confluence_summary = data.get('confluence_summary', {})
                    
                    print(f"📈 Target Zones: {len(target_zones)}")
                    print(f"🎯 High Confidence: {confluence_summary.get('high_confidence', 0)}")
                    print(f"⚠️  Medium Confidence: {confluence_summary.get('medium_confidence', 0)}")
                    print(f"📊 Low Confidence: {confluence_summary.get('low_confidence', 0)}")
                    
                    # Display detailed confluence information
                    for i, zone in enumerate(target_zones[:3]):  # Show top 3
                        print(f"\n  🎯 Target {i+1}: {zone.get('wave_target', 'Unknown')}")
                        print(f"     💰 Price: ${zone.get('price_level', 0):.4f}")
                        print(f"     📊 Change: {zone.get('price_change_pct', 0):+.2f}%")
                        print(f"     🔥 Confidence: {zone.get('confidence_level', 'Unknown')}")
                        print(f"     📈 Probability: {zone.get('probability', 0):.0%}")
                        print(f"     ⚡ Confluence Score: {zone.get('confluence_score', 0):.0f}")
                        
                        # Display all confluences
                        all_confluences = zone.get('all_confluences', zone.get('confluences', []))
                        print(f"     🧩 Confluences ({len(all_confluences)}):")
                        for conf in all_confluences:
                            print(f"        • {conf}")
                        
                        # Display confluence methods breakdown
                        if 'confluence_methods' in zone:
                            methods = zone['confluence_methods']
                            for method_type, method_list in methods.items():
                                if method_list:
                                    print(f"     📊 {method_type.replace('_', ' ').title()}: {len(method_list)} methods")
                    
                    # Check Elliott Wave analysis
                    waves = data.get('waves', [])
                    validation_score = data.get('validation_score', 0)
                    
                    print(f"\n🌊 Elliott Waves Detected: {len(waves)}")
                    print(f"🔍 Validation Score: {validation_score:.1%}")
                    
                    # Check analysis mode
                    analysis_mode = data.get('analysis_mode', 'unknown')
                    print(f"🤖 Analysis Mode: {analysis_mode}")
                    
                else:
                    print(f"❌ Analysis failed for {symbol}: {data.get('error', 'Unknown error')}")
                    
            else:
                print(f"❌ HTTP Error {response.status_code} for {symbol}")
                
        except Exception as e:
            print(f"❌ Exception testing {symbol}: {e}")
        
        time.sleep(2)  # Rate limiting

def test_confluence_web_interface():
    """Test the web interface confluence display."""
    print("\n🌐 Testing Web Interface Confluence Display...")
    
    try:
        # Check main page
        response = requests.get('http://localhost:5000', timeout=10)
        
        if response.status_code == 200:
            print("✅ Web interface accessible")
            
            # Check for confluence-related HTML elements
            html_content = response.text
            
            confluence_elements = [
                'confluenceResults',
                'confluenceSummary', 
                'targetZonesTable',
                'confluenceDetails',
                'Technical Confluence Analysis'
            ]
            
            for element in confluence_elements:
                if element in html_content:
                    print(f"✅ Found confluence element: {element}")
                else:
                    print(f"❌ Missing confluence element: {element}")
        else:
            print(f"❌ Web interface error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Web interface test failed: {e}")

def main():
    """Main test function."""
    print("🚀 Enhanced Confluence Analysis Web App Test")
    print("=" * 60)
    
    # Test API confluence functionality
    test_confluence_api()
    
    # Test web interface confluence display
    test_confluence_web_interface()
    
    print("\n" + "=" * 60)
    print("🏁 Test Complete!")
    print("\n💡 How to use the enhanced confluence display:")
    print("   1. Open http://localhost:5000 in your browser")
    print("   2. Select a trading pair and timeframe")
    print("   3. Click 'Analyze' to run Elliott Wave + Confluence Analysis")
    print("   4. Scroll down to see 'Technical Confluence Analysis' section")
    print("   5. Review target zones table with all confluence details")
    print("   6. Click on target zone rows for detailed breakdown")
    print("   7. Explore confluence categories and methods")

if __name__ == "__main__":
    main()
