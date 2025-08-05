#!/usr/bin/env python3

import sys
import os
import logging

# Set up logging to see all messages
logging.basicConfig(level=logging.DEBUG)

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("Testing MLWaveAccuracy import...")

try:
    from src.analysis.ml_wave_accuracy import MLWaveAccuracy
    print("✅ MLWaveAccuracy imported successfully!")
    
    # Try to create an instance
    ml_accuracy = MLWaveAccuracy()
    print("✅ MLWaveAccuracy instance created successfully!")
    
except Exception as e:
    print(f"❌ Error importing MLWaveAccuracy: {e}")
    import traceback
    traceback.print_exc()
