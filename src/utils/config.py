"""Configuration management for the Hair Care AI application."""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for the application."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.yaml
        """
        if config_path is None:
            # Get the project root directory
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'models.confidence_threshold')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self) -> None:
        """Save current configuration to file."""
        with open(self.config_path, 'w', encoding='utf-8') as file:
            yaml.dump(self._config, file, default_flow_style=False, indent=2)
    
    @property
    def model_path(self) -> str:
        """Get the YOLOv11 model path."""
        return self.get('models.yolo_model_path', 'best.pt')
    
    @property
    def confidence_threshold(self) -> float:
        """Get the confidence threshold for detections."""
        return self.get('models.confidence_threshold', 0.25)
    
    @property
    def image_size(self) -> int:
        """Get the input image size for the model."""
        return self.get('models.image_size', 640)
    
    @property
    def classes(self) -> Dict[str, str]:
        """Get the class definitions."""
        return self.get('classes', {})
    
    @property
    def supported_formats(self) -> list:
        """Get supported image formats."""
        return self.get('app.supported_formats', ['jpg', 'jpeg', 'png', 'bmp'])
    
    @property
    def max_file_size(self) -> int:
        """Get maximum file size in MB."""
        return self.get('app.max_file_size', 50)


# Global configuration instance
config = Config() 