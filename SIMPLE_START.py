#!/usr/bin/env python3
"""
SIMPLE ELLIOTT WAVE LAUNCHER - No complications
"""
import os
import sys

# Change to project directory
project_dir = r"c:\Users\Emre Yılmaz\Desktop\projects\elliottBot"
os.chdir(project_dir)

print("🌊 Starting Elliott Wave Bot (Offline)")
print("📍 Project Directory:", os.getcwd())
print("🚀 Launching offline application...")

# Simple direct execution
try:
    import subprocess
    subprocess.run([sys.executable, "web/app_offline.py"])
except Exception as e:
    print(f"Error: {e}")
    print("\n🔧 Manual command:")
    print("python web/app_offline.py")
    input("Press Enter to exit...")
