"""
Additional CNN Models for Enhanced Scalp Health Analysis

This module implements specialized CNN architectures to complement the YOLOv11 model:
- ScalpCNN: Focused on scalp condition classification
- HairHealthCNN: Specialized for hair health assessment
- EnsembleModel: Combines multiple models for robust predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
import cv2

from ..utils.logger import logger
from ..utils.config import config


class ScalpCNN(nn.Module):
    """
    Specialized CNN for scalp condition classification.
    
    This model focuses on identifying and classifying scalp conditions
    with high accuracy using a custom architecture optimized for scalp imagery.
    """
    
    def __init__(self, num_classes: int = 6, input_size: int = 224, dropout_rate: float = 0.3):
        """
        Initialize the ScalpCNN model.
        
        Args:
            num_classes: Number of scalp condition classes
            input_size: Input image size (assumes square images)
            dropout_rate: Dropout rate for regularization
        """
        super(ScalpCNN, self).__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        self.dropout_rate = dropout_rate
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature representations from the model."""
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        # Return features before final classification layer
        x = self.classifier[:-1](x)
        return x


class HairHealthCNN(nn.Module):
    """
    Specialized CNN for hair health assessment.
    
    This model focuses on analyzing hair quality, density, and overall health
    using a lightweight but effective architecture.
    """
    
    def __init__(self, num_health_classes: int = 5, input_size: int = 224):
        """
        Initialize the HairHealthCNN model.
        
        Args:
            num_health_classes: Number of hair health categories
            input_size: Input image size
        """
        super(HairHealthCNN, self).__init__()
        
        self.num_health_classes = num_health_classes
        self.input_size = input_size
        
        # Efficient feature extraction using depthwise separable convolutions
        self.features = nn.Sequential(
            # Initial conv layer
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Depthwise separable blocks
            self._make_separable_block(32, 64, stride=1),
            self._make_separable_block(64, 128, stride=2),
            self._make_separable_block(128, 128, stride=1),
            self._make_separable_block(128, 256, stride=2),
            self._make_separable_block(256, 256, stride=1),
            self._make_separable_block(256, 512, stride=2),
            
            # Final layers
            self._make_separable_block(512, 512, stride=1),
            self._make_separable_block(512, 1024, stride=2),
            self._make_separable_block(1024, 1024, stride=1),
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_health_classes)
        )
        
        self._initialize_weights()
    
    def _make_separable_block(self, in_channels: int, out_channels: int, stride: int = 1):
        """Create a depthwise separable convolution block."""
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, 
                     padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class EnsembleModel(nn.Module):
    """
    Ensemble model combining multiple CNN architectures for robust predictions.
    
    This model combines predictions from ScalpCNN and HairHealthCNN to provide
    comprehensive scalp and hair health analysis.
    """
    
    def __init__(self, scalp_model: ScalpCNN, hair_model: HairHealthCNN, 
                 fusion_method: str = 'weighted_average'):
        """
        Initialize the ensemble model.
        
        Args:
            scalp_model: Trained ScalpCNN model
            hair_model: Trained HairHealthCNN model
            fusion_method: Method to combine predictions ('weighted_average', 'attention', 'concat')
        """
        super(EnsembleModel, self).__init__()
        
        self.scalp_model = scalp_model
        self.hair_model = hair_model
        self.fusion_method = fusion_method
        
        # Freeze base models during ensemble training
        for param in self.scalp_model.parameters():
            param.requires_grad = False
        for param in self.hair_model.parameters():
            param.requires_grad = False
        
        # Fusion layers based on method
        if fusion_method == 'weighted_average':
            self.scalp_weight = nn.Parameter(torch.tensor(0.6))
            self.hair_weight = nn.Parameter(torch.tensor(0.4))
        elif fusion_method == 'attention':
            feature_dim = 1024  # Combined feature dimension
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 2),  # Attention weights for 2 models
                nn.Softmax(dim=1)
            )
        elif fusion_method == 'concat':
            # Concatenate features and add fusion layers
            combined_features = 1024 + 512  # ScalpCNN + HairHealthCNN features
            self.fusion_classifier = nn.Sequential(
                nn.Linear(combined_features, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, max(scalp_model.num_classes, hair_model.num_health_classes))
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the ensemble model.
        
        Returns:
            Dict containing individual and combined predictions
        """
        # Get predictions from individual models
        scalp_pred = self.scalp_model(x)
        hair_pred = self.hair_model(x)
        
        # Apply fusion method
        if self.fusion_method == 'weighted_average':
            # Normalize weights
            total_weight = self.scalp_weight + self.hair_weight
            w1 = self.scalp_weight / total_weight
            w2 = self.hair_weight / total_weight
            
            # Weighted average (assuming same output dimensions)
            if scalp_pred.shape[1] == hair_pred.shape[1]:
                combined_pred = w1 * scalp_pred + w2 * hair_pred
            else:
                # Handle different output dimensions
                combined_pred = scalp_pred  # Default to scalp prediction
        
        elif self.fusion_method == 'attention':
            # Extract features for attention mechanism
            scalp_features = self.scalp_model.extract_features(x)
            hair_features = self.hair_model.extract_features(x)
            
            # Combine features
            combined_features = torch.cat([scalp_features, hair_features], dim=1)
            attention_weights = self.attention(combined_features)
            
            # Apply attention weights
            combined_pred = (attention_weights[:, 0:1] * scalp_pred + 
                           attention_weights[:, 1:2] * hair_pred)
        
        elif self.fusion_method == 'concat':
            # Extract and concatenate features
            scalp_features = self.scalp_model.extract_features(x)
            hair_features = self.hair_model.extract_features(x)
            
            combined_features = torch.cat([scalp_features, hair_features], dim=1)
            combined_pred = self.fusion_classifier(combined_features)
        
        return {
            'scalp_prediction': scalp_pred,
            'hair_prediction': hair_pred,
            'combined_prediction': combined_pred,
            'scalp_probabilities': F.softmax(scalp_pred, dim=1),
            'hair_probabilities': F.softmax(hair_pred, dim=1),
            'combined_probabilities': F.softmax(combined_pred, dim=1)
        }


class CNNModelWrapper:
    """
    Wrapper class for CNN models to handle inference and preprocessing.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu', input_size: int = 224):
        """
        Initialize the CNN model wrapper.
        
        Args:
            model: PyTorch model instance
            device: Device to run inference on
            input_size: Expected input image size
        """
        self.model = model
        self.device = device
        self.input_size = input_size
        
        # Move model to device
        self.model.to(device)
        self.model.eval()
        
        # Define preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"CNN model wrapper initialized on device: {device}")
    
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for model inference.
        
        Args:
            image: Input image in various formats
            
        Returns:
            Preprocessed tensor ready for inference
        """
        # Convert to PIL Image if needed
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Unsupported image format")
        
        # Apply transforms
        tensor = self.transform(image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def predict(self, image: Union[str, np.ndarray, Image.Image]) -> Dict:
        """
        Run inference on an image.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary containing predictions and probabilities
        """
        try:
            # Preprocess image
            input_tensor = self.preprocess_image(image)
            
            # Run inference
            with torch.no_grad():
                if isinstance(self.model, EnsembleModel):
                    outputs = self.model(input_tensor)
                    return {
                        'scalp_prediction': outputs['scalp_prediction'].cpu().numpy(),
                        'hair_prediction': outputs['hair_prediction'].cpu().numpy(),
                        'combined_prediction': outputs['combined_prediction'].cpu().numpy(),
                        'scalp_probabilities': outputs['scalp_probabilities'].cpu().numpy(),
                        'hair_probabilities': outputs['hair_probabilities'].cpu().numpy(),
                        'combined_probabilities': outputs['combined_probabilities'].cpu().numpy()
                    }
                else:
                    outputs = self.model(input_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    
                    return {
                        'prediction': outputs.cpu().numpy(),
                        'probabilities': probabilities.cpu().numpy(),
                        'predicted_class': torch.argmax(outputs, dim=1).cpu().numpy()
                    }
        
        except Exception as e:
            logger.error(f"CNN prediction failed: {str(e)}")
            raise
    
    def load_weights(self, weights_path: str):
        """Load model weights from file."""
        try:
            checkpoint = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            logger.info(f"Model weights loaded from: {weights_path}")
        except Exception as e:
            logger.error(f"Failed to load weights: {str(e)}")
            raise
    
    def save_weights(self, save_path: str):
        """Save model weights to file."""
        try:
            torch.save(self.model.state_dict(), save_path)
            logger.info(f"Model weights saved to: {save_path}")
        except Exception as e:
            logger.error(f"Failed to save weights: {str(e)}")
            raise 