"""
Hair Care AI Models Package

This package contains all the AI models used for scalp health analysis:
- YOLOv11 model for object detection
- Additional CNN architectures for enhanced analysis
- Model ensemble and fusion capabilities
"""

from .yolo_model import YOLOModel
from .cnn_models import ScalpCNN, HairHealthCNN, EnsembleModel
from .model_manager import ModelManager

__all__ = [
    'YOLOModel',
    'ScalpCNN', 
    'HairHealthCNN',
    'EnsembleModel',
    'ModelManager'
] 