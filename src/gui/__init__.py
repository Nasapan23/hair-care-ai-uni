"""
GUI Package for Hair Care AI Application

This package contains the graphical user interface components
for the scalp health analysis application.
"""

from .main_app import HairCareApp
from .streamlit_app import run_streamlit_app

__all__ = [
    'HairCareApp',
    'run_streamlit_app'
] 