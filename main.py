#!/usr/bin/env python3
"""
Hair Care AI - Professional Scalp Health Analysis
Main Entry Point

This script provides the primary interface for running the Hair Care AI application
in various modes: GUI, CLI, and batch processing.

Author: Nisipeanu Ionut
Version: 1.0.0
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the main application
from src.gui.main_app import main

if __name__ == "__main__":
    # Set up environment
    os.environ['PYTHONPATH'] = str(project_root)
    
    # Run the main application
    main() 