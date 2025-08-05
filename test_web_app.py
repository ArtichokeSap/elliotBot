#!/usr/bin/env python3

import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(__file__))

print("Testing Flask web app import...")

try:
    # Add web directory to path and import
    web_dir = os.path.join(os.path.dirname(__file__), 'web')
    sys.path.insert(0, web_dir)
    
    # Import the Flask app
    from app import app, ml_features_available
    print("✅ Flask app imported successfully!")
    print(f"✅ ML features available: {ml_features_available}")
    
    # Test if we can create the app context
    with app.app_context():
        print("✅ Flask app context created successfully!")
    
except Exception as e:
    print(f"❌ Error importing Flask app: {e}")
    import traceback
    traceback.print_exc()
