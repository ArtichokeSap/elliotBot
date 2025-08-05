#!/usr/bin/env python3
"""
Elliott Wave Bot - Offline Launcher
Handles all network issues by running completely offline
"""

import os
import sys
import subprocess
import importlib.util

def check_package(package_name):
    """Check if a package is installed"""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_package(package_name):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("🌊 Elliott Wave Bot - Offline Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('web/app_offline.py'):
        print("❌ Error: app_offline.py not found!")
        print(f"📁 Current directory: {os.getcwd()}")
        print("🔧 Please run this from the Elliott Bot project directory.")
        return False
    
    print("✅ Offline application found")
    
    # Check required packages
    required_packages = ['flask', 'pandas', 'numpy', 'plotly']
    missing_packages = []
    
    print("🔍 Checking required packages...")
    for package in required_packages:
        if check_package(package):
            print(f"✅ {package}")
        else:
            print(f"❌ {package} - missing")
            missing_packages.append(package)
    
    # Install missing packages
    if missing_packages:
        print(f"\n🔧 Installing missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            print(f"Installing {package}...")
            if install_package(package):
                print(f"✅ {package} installed")
            else:
                print(f"❌ Failed to install {package}")
                print("💡 Try manually: pip install", package)
                return False
    
    print("\n🚀 Starting Elliott Wave Bot (Offline Mode)...")
    print("📊 No network connection required")
    print("🎯 Built-in sample data ready")
    print("\n" + "=" * 50)
    
    # Change to the correct directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Import and run the offline app
    try:
        sys.path.insert(0, '.')
        from web.app_offline import app
        
        print("✅ Offline app loaded successfully")
        print("🌐 Starting server at http://localhost:5000")
        print("🔌 Completely offline - no network errors possible!")
        print("\n" + "=" * 50)
        
        # Start the Flask app
        app.run(
            debug=False,
            host='127.0.0.1',
            port=5000,
            use_reloader=False,
            threaded=True
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("🔧 Check that all files are in the correct location")
        return False
    except OSError as e:
        if "Address already in use" in str(e):
            print("❌ Port 5000 is already in use!")
            print("🌐 Elliott Wave Bot may already be running")
            print("📱 Try accessing: http://localhost:5000")
        else:
            print(f"❌ Network error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Elliott Wave Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        print("🔧 Please check your Python installation")
