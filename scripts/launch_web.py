"""
Elliott Wave Bot - Web Application Launcher
Quick start script for the web interface
"""

import subprocess
import sys
import os
import webbrowser
import time
from pathlib import Path

def install_requirements():
    """Install required packages if needed."""
    print("🔧 Checking dependencies...")
    try:
        import flask
        print("✅ Flask is installed")
    except ImportError:
        print("📦 Installing Flask...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'flask', 'flask-cors'])
        print("✅ Flask installed successfully")

def check_port(port=5000):
    """Check if port is available."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result != 0

def main():
    """Main launcher function."""
    print("🚀 Elliott Wave Bot - Web Application Launcher")
    print("=" * 60)
    
    # Change to web directory
    web_dir = Path(__file__).parent / "web"
    if not web_dir.exists():
        print("❌ Web directory not found!")
        return
    
    os.chdir(web_dir)
    print(f"📁 Working directory: {web_dir}")
    
    # Install requirements
    install_requirements()
    
    # Check port availability
    port = 5000
    if not check_port(port):
        print(f"⚠️  Port {port} is already in use. Trying port {port + 1}...")
        port += 1
    
    print(f"🌐 Starting web server on port {port}...")
    print("🔗 Web interface will be available at:")
    print(f"   http://localhost:{port}")
    print("   http://127.0.0.1:{0}".format(port))
    
    # Start the Flask application
    try:
        print("\n🎯 Starting Elliott Wave Bot Web Interface...")
        print("   • Press Ctrl+C to stop the server")
        print("   • The browser will open automatically in 3 seconds")
        print("=" * 60)
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(3)
            webbrowser.open(f'http://localhost:{port}')
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.start()
        
        # Set environment variables
        os.environ['FLASK_APP'] = 'app.py'
        os.environ['FLASK_ENV'] = 'development'
        
        # Run Flask app
        subprocess.run([
            sys.executable, '-m', 'flask', 'run', 
            '--host', '0.0.0.0', 
            '--port', str(port),
            '--debug'
        ])
        
    except KeyboardInterrupt:
        print("\n\n🛑 Elliott Wave Bot Web Interface stopped")
        print("👋 Thank you for using Elliott Wave Bot!")
    
    except Exception as e:
        print(f"\n❌ Error starting web server: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure you're in the correct directory")
        print("   2. Check if all dependencies are installed:")
        print("      pip install -r requirements-minimal.txt")
        print("   3. Try running directly:")
        print("      python web/app.py")

if __name__ == "__main__":
    main()
