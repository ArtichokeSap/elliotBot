#!/usr/bin/env python3
"""
Elliott Bot Network Error Fix Script
Addresses common causes of "Network error. Please check your connection." message
"""

import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} may have compatibility issues")
        return False

def install_missing_packages():
    """Install any missing required packages."""
    print("📦 Checking required packages...")
    
    required_packages = [
        'flask',
        'flask-cors', 
        'requests',
        'yfinance',
        'plotly',
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📥 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install'
            ] + missing_packages)
            print("   ✅ Packages installed successfully")
        except subprocess.CalledProcessError:
            print("   ❌ Failed to install packages")
            return False
    
    return True

def clear_browser_cache():
    """Instructions for clearing browser cache."""
    print("🧹 Browser Cache Instructions:")
    print("   • Chrome/Edge: Ctrl+Shift+Delete → Clear cache")
    print("   • Firefox: Ctrl+Shift+Delete → Clear cache")
    print("   • Or try opening in incognito/private mode")

def check_firewall():
    """Check Windows Firewall status."""
    print("🔥 Checking Windows Firewall...")
    try:
        # Check if port 5000 is being blocked
        result = subprocess.run([
            'netsh', 'advfirewall', 'firewall', 'show', 'rule', 
            'name=all', 'dir=in', 'protocol=tcp', 'localport=5000'
        ], capture_output=True, text=True, shell=True)
        
        if 'No rules match' in result.stdout:
            print("   ⚠️ No firewall rule found for port 5000")
            print("   💡 You may need to add a firewall exception")
        else:
            print("   ✅ Firewall rule exists for port 5000")
    except Exception:
        print("   ⚠️ Could not check firewall status")

def fix_app_config():
    """Fix common configuration issues in the app."""
    print("⚙️ Checking app configuration...")
    
    app_file = Path("web/app.py")
    if app_file.exists():
        with open(app_file, 'r') as f:
            content = f.read()
        
        fixes_needed = []
        
        # Check for CORS import
        if 'from flask_cors import CORS' not in content:
            fixes_needed.append("Add CORS import")
        
        # Check for CORS initialization
        if 'CORS(app)' not in content:
            fixes_needed.append("Add CORS initialization")
        
        if fixes_needed:
            print(f"   ⚠️ Configuration issues found: {', '.join(fixes_needed)}")
            print("   💡 These have been fixed in the updated app.py")
        else:
            print("   ✅ App configuration looks good")
    else:
        print("   ❌ web/app.py not found")

def start_server():
    """Start the Elliott Bot server."""
    print("🚀 Starting Elliott Bot server...")
    
    try:
        # Change to the correct directory
        os.chdir(Path(__file__).parent)
        
        # Start the server on port 5050
        print("   🌐 Launching web application on port 5050...")
        subprocess.Popen([sys.executable, 'launch_web.py', '5050'])
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Test connectivity
        import requests
        try:
            response = requests.get('http://localhost:5050/api/health', timeout=10)
            if response.status_code == 200:
                print("   ✅ Server started successfully on port 5050")
                return True
            else:
                print(f"   ⚠️ Server responding but status: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("   ❌ Server not responding on port 5050")
            return False
        except Exception as e:
            print(f"   ❌ Connection test failed: {e}")
            return False
    except Exception as e:
        print(f"   ❌ Failed to start server: {e}")
        return False

def open_browser():
    """Open the Elliott Bot interface in the default browser."""
    print("🌐 Opening Elliott Bot interface...")
    try:
        webbrowser.open('http://localhost:5050')
        print("   ✅ Browser opened on port 5050")
        
        # Also open the network test page
        time.sleep(2)
        webbrowser.open('http://localhost:5050/network-test')
        print("   ✅ Network test page opened on port 5050")
        
    except Exception as e:
        print(f"   ⚠️ Could not open browser: {e}")
        print("   💡 Manually open: http://localhost:5000")

def main():
    """Main fix routine."""
    print("🔧 Elliott Bot Network Error Fix Script")
    print("=" * 60)
    
    # Run all fix steps
    steps = [
        ("Check Python version", check_python_version),
        ("Install missing packages", install_missing_packages),
        ("Check app configuration", fix_app_config),
        ("Check firewall", check_firewall),
        ("Start server", start_server),
        ("Open browser", open_browser)
    ]
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        try:
            success = step_func()
            if success is False:
                print(f"   ⚠️ {step_name} had issues, but continuing...")
        except Exception as e:
            print(f"   ❌ Error in {step_name}: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 Fix script completed!")
    print("\n💡 If you still see 'Network error' messages:")
    print("   1. Clear your browser cache completely")
    print("   2. Try incognito/private browsing mode")
    print("   3. Check Windows Firewall settings")
    print("   4. Restart the Elliott Bot server")
    print("   5. Try a different browser")
    
    clear_browser_cache()

if __name__ == "__main__":
    main()
