#!/usr/bin/env python3
"""
Quick local runner for the NCZarr Viewer
Run this script to start the viewer locally without Docker.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from main import ViewerApp

    print("Starting NCZarr Viewer...")
    print("The viewer will be available at: http://localhost:8050")
    print("Press Ctrl+C to stop the server")

    app = ViewerApp()
    app.run()

except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("\nPlease install the required dependencies first:")
    print("pip install -e .")
    print("\nOr if you're using uv:")
    print("uv sync")
    sys.exit(1)
except Exception as e:
    print(f"Error starting the viewer: {e}")
    sys.exit(1)
