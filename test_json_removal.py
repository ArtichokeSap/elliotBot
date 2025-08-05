#!/usr/bin/env python3
"""
Test script to verify that JSON wave data display has been removed from web app
"""

import requests
import json

def test_json_removal():
    """Test that JSON wave data is no longer displayed."""
    print("🧪 Testing JSON Wave Data Removal...")
    
    try:
        # Test API response structure
        response = requests.post('http://localhost:5000/api/analyze', json={
            'symbol': 'AAPL',
            'timeframe': '1d'
        }, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('success'):
                print("✅ API response successful")
                
                # Check that enhanced_waves and wave_labels are no longer in response
                has_enhanced_waves = 'enhanced_waves' in data
                has_wave_labels = 'wave_labels' in data
                has_waves = 'waves' in data
                has_ascii_table = 'ascii_table' in data
                
                print(f"📊 Regular waves data: {'✅ Present' if has_waves else '❌ Missing'}")
                print(f"📝 ASCII table: {'✅ Present' if has_ascii_table else '❌ Missing'}")
                print(f"🚫 Enhanced waves (should be removed): {'❌ Still present' if has_enhanced_waves else '✅ Successfully removed'}")
                print(f"🚫 Wave labels (should be removed): {'❌ Still present' if has_wave_labels else '✅ Successfully removed'}")
                
                # Check confluence data is still present
                has_target_zones = 'target_zones' in data
                has_confluence_summary = 'confluence_summary' in data
                
                print(f"🎯 Target zones: {'✅ Present' if has_target_zones else '❌ Missing'}")
                print(f"📈 Confluence summary: {'✅ Present' if has_confluence_summary else '❌ Missing'}")
                
                if not has_enhanced_waves and not has_wave_labels:
                    print("\n🎉 SUCCESS: JSON wave data display has been successfully removed!")
                else:
                    print("\n⚠️  WARNING: Some JSON wave data elements are still present")
                
            else:
                print(f"❌ API error: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Test error: {e}")

def test_web_interface():
    """Test that web interface no longer has JSON display elements."""
    print("\n🌐 Testing Web Interface...")
    
    try:
        response = requests.get('http://localhost:5000', timeout=10)
        
        if response.status_code == 200:
            html_content = response.text
            
            # Check for removed elements
            removed_elements = [
                'jsonDataContainer',
                'enhancedWaveJson',
                'waveLabelsJson',
                'Wave Data (JSON Format)',
                'Enhanced Wave Data:',
                'Wave Labels & Positions:'
            ]
            
            # Check for elements that should still be present
            present_elements = [
                'confluenceResults',
                'targetZonesTable',
                'Technical Confluence Analysis',
                'asciiTableContainer'
            ]
            
            print("🚫 Checking removed elements:")
            for element in removed_elements:
                if element in html_content:
                    print(f"   ❌ Found (should be removed): {element}")
                else:
                    print(f"   ✅ Not found (correctly removed): {element}")
            
            print("\n✅ Checking present elements:")
            for element in present_elements:
                if element in html_content:
                    print(f"   ✅ Found (correctly present): {element}")
                else:
                    print(f"   ❌ Missing (should be present): {element}")
                    
        else:
            print(f"❌ Web interface error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Web interface test error: {e}")

def main():
    """Main test function."""
    print("🚀 Testing JSON Wave Data Removal")
    print("=" * 50)
    
    test_json_removal()
    test_web_interface()
    
    print("\n" + "=" * 50)
    print("🏁 Test Complete!")
    print("\n💡 Expected result:")
    print("   • Enhanced wave data JSON should be removed")
    print("   • Wave labels JSON should be removed")
    print("   • ASCII table should still be present")
    print("   • Confluence analysis should still be present")
    print("   • Regular wave data should still be present")

if __name__ == "__main__":
    main()
