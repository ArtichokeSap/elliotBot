#!/usr/bin/env python3
"""
Quick Web Launcher for Elliott Wave Bot
Simple script to start the web interface
"""

import os
import sys
import subprocess

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def main():
    print("🌐 Elliott Wave Bot - Quick Web Launcher")
    print("=" * 50)
    
    # Change to web directory
    web_dir = os.path.join(project_root, "web")
    os.chdir(web_dir)
    
    print("🚀 Starting web server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("⛔ Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        # Change to web directory and run the Flask app
        sys.path.insert(0, web_dir)
        os.chdir(web_dir)
        
        # Import and run the app
        import app
        app.app.run(debug=True, host='0.0.0.0', port=5000)
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try installing dependencies: pip install flask")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
