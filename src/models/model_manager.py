"""
Model Manager for Hair Care AI Application

This module provides a unified interface for managing and coordinating
all AI models used in the scalp health analysis system.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from PIL import Image

from .yolo_model import YOLOModel
from .cnn_models import ScalpCNN, HairHealthCNN, EnsembleModel, CNNModelWrapper
from ..utils.config import config
from ..utils.logger import logger
from ..utils.helpers import validate_image_file, generate_recommendations


class ModelManager:
    """
    Centralized manager for all AI models in the hair care analysis system.
    
    This class coordinates between YOLOv11, ScalpCNN, HairHealthCNN, and ensemble models
    to provide comprehensive scalp health analysis.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the model manager.
        
        Args:
            device: Device to run models on ('cpu', 'cuda', 'mps'). If None, auto-detects.
        """
        self.device = device or self._get_device()
        
        # Class names for scalp conditions
        self.class_names = ['d', 'o', 's', 'ds', 'os', 'dss']  # dandruff, oiliness, sensitivity, combinations
        
        # Model instances
        self.yolo_model = None
        self.scalp_cnn = None
        self.hair_cnn = None
        self.ensemble_model = None
        
        # Model wrappers
        self.scalp_wrapper = None
        self.hair_wrapper = None
        self.ensemble_wrapper = None
        
        # Model status
        self.models_loaded = {
            'yolo': False,
            'scalp_cnn': False,
            'hair_cnn': False,
            'ensemble': False
        }
        
        # Analysis settings
        self.analysis_mode = 'comprehensive'  # 'yolo_only', 'cnn_only', 'comprehensive'
        self.confidence_threshold = config.confidence_threshold
        
        logger.info(f"ModelManager initialized on device: {self.device}")
    
    def _get_device(self) -> str:
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def load_yolo_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load the YOLOv11 model.
        
        Args:
            model_path: Path to the YOLO model file
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            self.yolo_model = YOLOModel(model_path, self.device)
            success = self.yolo_model.load_model()
            self.models_loaded['yolo'] = success
            
            if success:
                logger.info("YOLO model loaded successfully")
            else:
                logger.error("Failed to load YOLO model")
            
            return success
            
        except Exception as e:
            logger.error(f"Error loading YOLO model: {str(e)}")
            return False
    
    def load_cnn_models(self, scalp_cnn_path: Optional[str] = None,
                        hair_cnn_path: Optional[str] = None) -> Dict[str, bool]:
        """
        Load CNN models for additional analysis.
        
        Args:
            scalp_cnn_path: Path to ScalpCNN weights
            hair_cnn_path: Path to HairHealthCNN weights
            
        Returns:
            Dict with loading status for each model
        """
        results = {'scalp_cnn': False, 'hair_cnn': False}
        
        try:
            # Load ScalpCNN
            self.scalp_cnn = CNNModelWrapper(
                ScalpCNN(num_classes=len(self.class_names)), 
                self.device
            )
            self.scalp_wrapper = self.scalp_cnn  # Assign wrapper
            
            if scalp_cnn_path and Path(scalp_cnn_path).exists():
                self.scalp_cnn.load_weights(scalp_cnn_path)
                logger.info(f"ScalpCNN loaded from: {scalp_cnn_path}")
            else:
                # Initialize with random weights for demonstration
                logger.info("ScalpCNN initialized with random weights (no pre-trained weights available)")
            
            results['scalp_cnn'] = True
            self.models_loaded['scalp_cnn'] = True
            
        except Exception as e:
            logger.error(f"Failed to load ScalpCNN: {str(e)}")
        
        try:
            # Load HairHealthCNN
            self.hair_cnn = CNNModelWrapper(
                HairHealthCNN(num_health_classes=len(self.class_names)), 
                self.device
            )
            self.hair_wrapper = self.hair_cnn  # Assign wrapper
            
            if hair_cnn_path and Path(hair_cnn_path).exists():
                self.hair_cnn.load_weights(hair_cnn_path)
                logger.info(f"HairHealthCNN loaded from: {hair_cnn_path}")
            else:
                # Initialize with random weights for demonstration
                logger.info("HairHealthCNN initialized with random weights (no pre-trained weights available)")
            
            results['hair_cnn'] = True
            self.models_loaded['hair_cnn'] = True
            
        except Exception as e:
            logger.error(f"Failed to load HairHealthCNN: {str(e)}")
        
        return results
    
    def create_ensemble_model(self) -> bool:
        """
        Create ensemble model combining ScalpCNN and HairHealthCNN.
        
        Returns:
            True if ensemble created successfully
        """
        try:
            if self.scalp_cnn is None or self.hair_cnn is None:
                logger.warning("Cannot create ensemble: CNN models not loaded")
                return False
            
            # Create ensemble model
            self.ensemble_model = EnsembleModel(
                self.scalp_cnn.model,
                self.hair_cnn.model,
                fusion_method='weighted_average'
            ).to(self.device)
            
            # Wrap in CNNModelWrapper for consistent interface
            self.ensemble_wrapper = CNNModelWrapper(
                self.ensemble_model,
                self.device
            )
            
            self.models_loaded['ensemble'] = True
            logger.info("Ensemble model created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create ensemble model: {str(e)}")
            return False
    
    def analyze_image(self, image: Union[str, np.ndarray, Image.Image],
                     save_results: bool = False,
                     output_dir: Optional[str] = None) -> Dict:
        """
        Perform comprehensive scalp health analysis on an image.
        
        Args:
            image: Input image
            save_results: Whether to save analysis results
            output_dir: Directory to save results
            
        Returns:
            Comprehensive analysis results
        """
        # Validate image
        if isinstance(image, str):
            validation_result = validate_image_file(image)
            if not validation_result['valid']:
                raise ValueError(f"Invalid image: {validation_result['error']}")
        
        results = {
            'image_info': self._get_image_info(image),
            'yolo_results': None,
            'scalp_cnn_results': None,
            'hair_cnn_results': None,
            'ensemble_results': None,
            'combined_analysis': None,
            'recommendations': None,
            'analysis_metadata': {
                'device': self.device,
                'analysis_mode': self.analysis_mode,
                'models_used': []
            }
        }
        
        try:
            # YOLO Analysis
            if self.models_loaded['yolo'] and self.analysis_mode in ['yolo_only', 'comprehensive']:
                logger.info("Running YOLO analysis...")
                yolo_results = self.yolo_model.predict(
                    image, save_results, output_dir
                )
                results['yolo_results'] = yolo_results
                results['analysis_metadata']['models_used'].append('yolo')
                
                # Generate annotated image for visualization
                if isinstance(image, str):
                    # Load image for annotation
                    img_array = cv2.imread(image)
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                elif isinstance(image, Image.Image):
                    img_array = np.array(image)
                else:
                    img_array = image
                
                # Create annotated image
                annotated_image = self.yolo_model.visualize_results(
                    img_array, yolo_results, show_confidence=True, show_labels=True
                )
                results['annotated_image'] = annotated_image
            
            # ScalpCNN Analysis
            if (self.models_loaded['scalp_cnn'] and 
                self.analysis_mode in ['cnn_only', 'comprehensive']):
                logger.info("Running ScalpCNN analysis...")
                results['scalp_cnn_results'] = self.scalp_wrapper.predict(image)
                results['analysis_metadata']['models_used'].append('scalp_cnn')
            
            # HairHealthCNN Analysis
            if (self.models_loaded['hair_cnn'] and 
                self.analysis_mode in ['cnn_only', 'comprehensive']):
                logger.info("Running HairHealthCNN analysis...")
                results['hair_cnn_results'] = self.hair_wrapper.predict(image)
                results['analysis_metadata']['models_used'].append('hair_cnn')
            
            # Ensemble Analysis
            if (self.models_loaded['ensemble'] and 
                self.analysis_mode == 'comprehensive'):
                logger.info("Running ensemble analysis...")
                results['ensemble_results'] = self.ensemble_wrapper.predict(image)
                results['analysis_metadata']['models_used'].append('ensemble')
            
            # Combine results and generate final analysis
            results['combined_analysis'] = self._combine_analysis_results(results)
            results['recommendations'] = self._generate_comprehensive_recommendations(results)
            
            logger.info("Image analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error during image analysis: {str(e)}")
            raise
    
    def _get_image_info(self, image: Union[str, np.ndarray, Image.Image]) -> Dict:
        """Extract basic information about the input image."""
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image from: {image}")
            height, width = img.shape[:2]
            file_path = image
        elif isinstance(image, np.ndarray):
            height, width = image.shape[:2]
            file_path = None
        elif isinstance(image, Image.Image):
            width, height = image.size
            file_path = None
        else:
            raise ValueError("Unsupported image format")
        
        return {
            'width': width,
            'height': height,
            'aspect_ratio': width / height,
            'file_path': file_path,
            'total_pixels': width * height
        }
    
    def _combine_analysis_results(self, results: Dict) -> Dict:
        """
        Combine results from different models into a unified analysis.
        
        Args:
            results: Dictionary containing individual model results
            
        Returns:
            Combined analysis results
        """
        combined = {
            'overall_health_score': 0.0,
            'confidence_level': 'Low',
            'primary_conditions': [],
            'severity_assessment': 'Unknown',
            'affected_area_percentage': 0.0,
            'model_agreement': 'Unknown'
        }
        
        # Start with YOLO results as baseline
        if results['yolo_results']:
            yolo_data = results['yolo_results']
            combined['overall_health_score'] = yolo_data['health_score']
            combined['affected_area_percentage'] = yolo_data['total_area_affected']
            
            # Extract primary conditions
            for detection in yolo_data['detections']:
                condition_info = {
                    'condition': detection['class_name'],
                    'confidence': detection['confidence'],
                    'severity': detection['severity'],
                    'area_percentage': detection['area_percentage']
                }
                combined['primary_conditions'].append(condition_info)
        
        # Enhance with CNN results
        if results['scalp_cnn_results']:
            scalp_data = results['scalp_cnn_results']
            # Use CNN confidence to adjust overall confidence
            max_prob = np.max(scalp_data['probabilities'])
            if max_prob > 0.8:
                combined['confidence_level'] = 'High'
            elif max_prob > 0.6:
                combined['confidence_level'] = 'Medium'
            else:
                combined['confidence_level'] = 'Low'
        
        # Factor in ensemble results if available
        if results['ensemble_results']:
            ensemble_data = results['ensemble_results']
            # Use ensemble for final confidence assessment
            combined_probs = ensemble_data.get('combined_probabilities', [[]])[0]
            if len(combined_probs) > 0:
                max_ensemble_prob = np.max(combined_probs)
                if max_ensemble_prob > 0.85:
                    combined['confidence_level'] = 'Very High'
                elif max_ensemble_prob > 0.7:
                    combined['confidence_level'] = 'High'
        
        # Determine overall severity
        if combined['primary_conditions']:
            severity_scores = {'Mild': 1, 'Moderate': 2, 'Severe': 3}
            max_severity = max(
                severity_scores.get(cond['severity'], 0) 
                for cond in combined['primary_conditions']
            )
            
            severity_map = {1: 'Mild', 2: 'Moderate', 3: 'Severe'}
            combined['severity_assessment'] = severity_map.get(max_severity, 'Unknown')
        
        # Calculate model agreement
        models_used = results['analysis_metadata']['models_used']
        if len(models_used) > 1:
            # Simple agreement metric based on consistency of findings
            agreement_score = self._calculate_model_agreement(results)
            if agreement_score > 0.8:
                combined['model_agreement'] = 'High'
            elif agreement_score > 0.6:
                combined['model_agreement'] = 'Medium'
            else:
                combined['model_agreement'] = 'Low'
        else:
            combined['model_agreement'] = 'Single Model'
        
        return combined
    
    def _calculate_model_agreement(self, results: Dict) -> float:
        """Calculate agreement score between different models."""
        # Simplified agreement calculation
        # In a real implementation, this would be more sophisticated
        
        yolo_score = results['yolo_results']['health_score'] if results['yolo_results'] else 50
        
        # Normalize scores to 0-100 scale
        scores = [yolo_score]
        
        if results['scalp_cnn_results']:
            # Convert CNN probabilities to health score
            probs = results['scalp_cnn_results']['probabilities'][0]
            # Assume higher probability of healthy class means better health
            cnn_score = probs[0] * 100 if len(probs) > 0 else 50
            scores.append(cnn_score)
        
        if len(scores) < 2:
            return 1.0  # Perfect agreement with single model
        
        # Calculate coefficient of variation (lower = more agreement)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        if mean_score == 0:
            return 1.0
        
        cv = std_score / mean_score
        # Convert to agreement score (0-1, higher = more agreement)
        agreement = max(0, 1 - cv)
        
        return agreement
    
    def _generate_comprehensive_recommendations(self, results: Dict) -> Dict:
        """
        Generate comprehensive care recommendations based on all analysis results.
        
        Args:
            results: Complete analysis results
            
        Returns:
            Comprehensive recommendations
        """
        recommendations = {
            'immediate_actions': [],
            'care_routine': [],
            'product_suggestions': [],
            'lifestyle_recommendations': [],
            'follow_up_advice': [],
            'severity_based_actions': []
        }
        
        combined_analysis = results['combined_analysis']
        
        # Base recommendations on primary conditions
        if combined_analysis['primary_conditions']:
            for condition in combined_analysis['primary_conditions']:
                condition_name = condition['condition']
                severity = condition['severity']
                
                # Get condition-specific recommendations
                condition_recs = generate_recommendations(condition_name)
                
                # Categorize recommendations based on severity
                if severity == 'Severe':
                    recommendations['immediate_actions'].extend([
                        f"Seek professional consultation for {condition_name}",
                        "Consider medicated treatments",
                        "Avoid harsh hair products temporarily"
                    ])
                    recommendations['severity_based_actions'].append(
                        f"Severe {condition_name} detected - professional treatment recommended"
                    )
                
                elif severity == 'Moderate':
                    recommendations['care_routine'].extend([
                        f"Use specialized products for {condition_name}",
                        "Maintain consistent hair care routine",
                        "Monitor condition progress"
                    ])
                
                else:  # Mild
                    recommendations['care_routine'].extend([
                        f"Gentle care for {condition_name}",
                        "Preventive measures recommended"
                    ])
                
                # Add condition-specific recommendations
                recommendations['product_suggestions'].extend(condition_recs)
        
        # General recommendations based on overall health score
        health_score = combined_analysis['overall_health_score']
        
        if health_score >= 80:
            recommendations['care_routine'].extend([
                "Maintain current hair care routine",
                "Regular scalp massage for continued health",
                "Balanced diet for optimal hair health"
            ])
        elif health_score >= 60:
            recommendations['care_routine'].extend([
                "Improve hair care routine consistency",
                "Consider scalp-nourishing treatments",
                "Monitor for changes in condition"
            ])
        else:
            recommendations['immediate_actions'].extend([
                "Comprehensive hair care routine overhaul needed",
                "Consider professional consultation",
                "Address underlying scalp health issues"
            ])
        
        # Lifestyle recommendations
        recommendations['lifestyle_recommendations'].extend([
            "Maintain a balanced diet rich in vitamins and minerals",
            "Stay hydrated for optimal scalp health",
            "Manage stress levels as they affect hair health",
            "Protect hair from environmental damage",
            "Get adequate sleep for hair regeneration"
        ])
        
        # Follow-up advice
        if combined_analysis['confidence_level'] in ['Low', 'Medium']:
            recommendations['follow_up_advice'].append(
                "Consider retaking analysis with better lighting or image quality"
            )
        
        recommendations['follow_up_advice'].extend([
            "Re-analyze in 2-4 weeks to track progress",
            "Keep a hair care diary to monitor improvements",
            "Consult a dermatologist if conditions worsen"
        ])
        
        # Remove duplicates and empty entries
        for key in recommendations:
            recommendations[key] = list(set(filter(None, recommendations[key])))
        
        return recommendations
    
    def set_analysis_mode(self, mode: str):
        """
        Set the analysis mode.
        
        Args:
            mode: 'yolo_only', 'cnn_only', or 'comprehensive'
        """
        valid_modes = ['yolo_only', 'cnn_only', 'comprehensive']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode. Must be one of: {valid_modes}")
        
        self.analysis_mode = mode
        logger.info(f"Analysis mode set to: {mode}")
    
    def get_model_status(self) -> Dict:
        """Get the current status of all models."""
        return {
            'models_loaded': self.models_loaded.copy(),
            'device': self.device,
            'analysis_mode': self.analysis_mode,
            'yolo_info': self.yolo_model.get_model_info() if self.yolo_model else None
        }
    
    def batch_analyze(self, image_paths: List[str], 
                     output_dir: Optional[str] = None) -> List[Dict]:
        """
        Analyze multiple images in batch.
        
        Args:
            image_paths: List of image file paths
            output_dir: Directory to save results
            
        Returns:
            List of analysis results for each image
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                logger.info(f"Analyzing image {i+1}/{len(image_paths)}: {image_path}")
                result = self.analyze_image(
                    image_path,
                    save_results=bool(output_dir),
                    output_dir=output_dir
                )
                result['image_path'] = image_path
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to analyze {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'analysis_successful': False
                })
        
        return results
    
    def update_confidence_threshold(self, threshold: float):
        """Update confidence threshold for all models."""
        self.confidence_threshold = threshold
        
        if self.yolo_model:
            self.yolo_model.update_confidence_threshold(threshold)
        
        logger.info(f"Confidence threshold updated to: {threshold}")
    
    def cleanup(self):
        """Clean up model resources."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model resources cleaned up")
    
    def initialize_models(self, yolo_path: Optional[str] = None,
                         scalp_cnn_path: Optional[str] = None,
                         hair_cnn_path: Optional[str] = None) -> Dict[str, bool]:
        """
        Initialize all AI models.
        
        Args:
            yolo_path: Path to YOLO model
            scalp_cnn_path: Path to ScalpCNN weights
            hair_cnn_path: Path to HairHealthCNN weights
            
        Returns:
            Dict with initialization status for each model
        """
        logger.info("Initializing AI models...")
        
        results = {
            'yolo': False,
            'scalp_cnn': False,
            'hair_cnn': False,
            'ensemble': False
        }
        
        try:
            # Load YOLO model
            results['yolo'] = self.load_yolo_model(yolo_path)
            self.models_loaded['yolo'] = results['yolo']
            
            # Load CNN models
            cnn_results = self.load_cnn_models(scalp_cnn_path, hair_cnn_path)
            results.update(cnn_results)
            self.models_loaded['scalp_cnn'] = cnn_results['scalp_cnn']
            self.models_loaded['hair_cnn'] = cnn_results['hair_cnn']
            
            # Create ensemble if both CNNs are available
            if results['scalp_cnn'] and results['hair_cnn']:
                results['ensemble'] = self.create_ensemble_model()
                self.models_loaded['ensemble'] = results['ensemble']
            
            logger.info(f"Model initialization completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            return results
    
    def _analyze_with_scalp_cnn(self, image: np.ndarray) -> Dict:
        """Analyze image with ScalpCNN."""
        try:
            if self.scalp_cnn is None:
                return {'error': 'ScalpCNN not loaded'}
            
            # Run inference
            results = self.scalp_cnn.predict(image)
            
            # Format results
            return {
                'predictions': results['predictions'],
                'probabilities': results['probabilities'],
                'confidence': float(np.max(results['probabilities'])),
                'predicted_class': self.class_names[results['predictions'][0]],
                'model_info': {
                    'name': 'ScalpCNN',
                    'architecture': 'VGG-inspired CNN',
                    'input_size': '224x224',
                    'note': 'Using random weights - for demonstration only'
                }
            }
            
        except Exception as e:
            logger.error(f"ScalpCNN analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_with_hair_cnn(self, image: np.ndarray) -> Dict:
        """Analyze image with HairHealthCNN."""
        try:
            if self.hair_cnn is None:
                return {'error': 'HairHealthCNN not loaded'}
            
            # Run inference
            results = self.hair_cnn.predict(image)
            
            # Format results
            return {
                'predictions': results['predictions'],
                'probabilities': results['probabilities'],
                'confidence': float(np.max(results['probabilities'])),
                'predicted_class': self.class_names[results['predictions'][0]],
                'model_info': {
                    'name': 'HairHealthCNN',
                    'architecture': 'Lightweight CNN with depthwise separable convolutions',
                    'input_size': '224x224',
                    'note': 'Using random weights - for demonstration only'
                }
            }
            
        except Exception as e:
            logger.error(f"HairHealthCNN analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_with_ensemble(self, image: np.ndarray) -> Dict:
        """Analyze image with ensemble model."""
        try:
            if self.ensemble_wrapper is None:
                return {'error': 'Ensemble model not loaded'}
            
            # Run inference
            results = self.ensemble_wrapper.predict(image)
            
            # Format results
            return {
                'combined_predictions': results['predictions'],
                'combined_probabilities': results['probabilities'],
                'confidence': float(np.max(results['probabilities'])),
                'predicted_class': self.class_names[results['predictions'][0]],
                'fusion_method': 'weighted_average',
                'model_info': {
                    'name': 'Ensemble Model',
                    'components': ['ScalpCNN', 'HairHealthCNN'],
                    'fusion_method': 'weighted_average',
                    'note': 'Using random weights - for demonstration only'
                }
            }
            
        except Exception as e:
            logger.error(f"Ensemble analysis failed: {str(e)}")
            return {'error': str(e)} 