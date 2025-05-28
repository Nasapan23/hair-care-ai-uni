#!/usr/bin/env python3
"""
Streamlit App Entry Point for Hair Care AI

Run this file with: streamlit run streamlit_app.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the Streamlit app
from src.gui.streamlit_app import run_streamlit_app

if __name__ == "__main__":
    run_streamlit_app() 