"""
YOLOv11 Model Wrapper for Scalp Health Analysis

This module provides a comprehensive wrapper around the YOLOv11 model
for detecting scalp conditions including dandruff, oiliness, and sensitivity.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont

from ..utils.config import config
from ..utils.logger import logger
from ..utils.helpers import preprocess_image, calculate_area_percentage, get_severity_level


class YOLOModel:
    """
    YOLOv11 model wrapper for scalp condition detection.
    
    Handles model loading, inference, post-processing, and visualization
    of scalp health conditions including dandruff, oiliness, and sensitivity.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the YOLO model.
        
        Args:
            model_path: Path to the trained model file. If None, uses config default.
            device: Device to run inference on ('cpu', 'cuda', 'mps'). If None, auto-detects.
        """
        self.model_path = model_path or config.model_path
        self.device = device or self._get_device()
        self.confidence_threshold = config.confidence_threshold
        self.image_size = config.image_size
        self.classes = config.classes
        
        # Initialize model
        self.model = None
        self.is_loaded = False
        
        # Class colors for visualization
        self.class_colors = {
            'd': (255, 0, 0),      # Red for dandruff
            'o': (0, 255, 0),      # Green for oiliness  
            's': (0, 0, 255),      # Blue for sensitive
            'ds': (255, 255, 0),   # Yellow for dandruff + sensitive
            'os': (255, 0, 255),   # Magenta for oiliness + sensitive
            'dss': (0, 255, 255)   # Cyan for dandruff + sensitive + sensitive
        }
        
        logger.info(f"YOLOModel initialized with device: {self.device}")
    
    def _get_device(self) -> str:
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def load_model(self) -> bool:
        """
        Load the YOLO model from file.
        
        Returns:
            bool: True if model loaded successfully, False otherwise.
        """
        try:
            if not Path(self.model_path).exists():
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            logger.info(f"Loading YOLO model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # Move model to specified device
            if self.device != 'cpu':
                self.model.to(self.device)
            
            self.is_loaded = True
            logger.info("YOLO model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {str(e)}")
            self.is_loaded = False
            return False
    
    def predict(self, image: Union[str, np.ndarray, Image.Image], 
                save_results: bool = False, 
                output_dir: Optional[str] = None) -> Dict:
        """
        Run inference on an image.
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
            save_results: Whether to save annotated results
            output_dir: Directory to save results (if save_results=True)
            
        Returns:
            Dict containing detection results and metadata
        """
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError("Model not loaded and failed to load")
        
        try:
            # Preprocess image if needed
            if isinstance(image, str):
                image_path = image
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Could not load image from: {image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(image, Image.Image):
                image = np.array(image)
            
            original_shape = image.shape[:2]
            
            # Run inference
            logger.info("Running YOLO inference...")
            results = self.model(
                image,
                conf=self.confidence_threshold,
                imgsz=self.image_size,
                device=self.device,
                verbose=False
            )
            
            # Process results
            processed_results = self._process_results(results[0], original_shape)
            
            # Save annotated image if requested
            if save_results and output_dir:
                self._save_annotated_image(image, processed_results, output_dir)
            
            logger.info(f"Detection completed. Found {len(processed_results['detections'])} objects")
            return processed_results
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def _process_results(self, result, original_shape: Tuple[int, int]) -> Dict:
        """
        Process raw YOLO results into structured format.
        
        Args:
            result: Raw YOLO result object
            original_shape: Original image dimensions (height, width)
            
        Returns:
            Dict with processed detection results
        """
        detections = []
        class_counts = {class_name: 0 for class_name in self.classes.keys()}
        total_area_affected = 0.0
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            # Get class names from model
            class_names = result.names
            
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                x1, y1, x2, y2 = box
                class_name = class_names[cls_id]
                
                # Calculate area percentage
                area_percentage = calculate_area_percentage(
                    (x1, y1, x2, y2), original_shape
                )
                
                # Determine severity
                severity = get_severity_level(conf, area_percentage)
                
                detection = {
                    'id': i,
                    'class_name': class_name,
                    'class_id': cls_id,
                    'confidence': float(conf),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'area_percentage': area_percentage,
                    'severity': severity,
                    'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                    'width': x2 - x1,
                    'height': y2 - y1
                }
                
                detections.append(detection)
                class_counts[class_name] += 1
                total_area_affected += area_percentage
        
        # Calculate overall health score
        health_score = self._calculate_health_score(detections, total_area_affected)
        
        # Generate summary
        summary = self._generate_summary(detections, class_counts, health_score)
        
        return {
            'detections': detections,
            'class_counts': class_counts,
            'total_detections': len(detections),
            'total_area_affected': min(total_area_affected, 100.0),  # Cap at 100%
            'health_score': health_score,
            'summary': summary,
            'image_shape': original_shape,
            'model_info': {
                'confidence_threshold': self.confidence_threshold,
                'image_size': self.image_size,
                'device': self.device
            }
        }
    
    def _calculate_health_score(self, detections: List[Dict], total_area: float) -> float:
        """
        Calculate overall scalp health score (0-100, higher is better).
        
        Args:
            detections: List of detection dictionaries
            total_area: Total area affected by conditions
            
        Returns:
            Health score between 0 and 100
        """
        if not detections:
            return 100.0  # Perfect health if no issues detected
        
        # Base score reduction based on area affected
        area_penalty = min(total_area * 2, 80)  # Max 80% penalty for area
        
        # Additional penalty based on severity and number of conditions
        severity_penalty = 0
        for detection in detections:
            if detection['severity'] == 'Severe':
                severity_penalty += 15
            elif detection['severity'] == 'Moderate':
                severity_penalty += 8
            else:  # Mild
                severity_penalty += 3
        
        # Penalty for multiple condition types
        unique_conditions = len(set(d['class_name'] for d in detections))
        multi_condition_penalty = (unique_conditions - 1) * 5
        
        total_penalty = area_penalty + severity_penalty + multi_condition_penalty
        health_score = max(100 - total_penalty, 0)
        
        return round(health_score, 1)
    
    def _generate_summary(self, detections: List[Dict], class_counts: Dict, health_score: float) -> str:
        """Generate a human-readable summary of the analysis."""
        if not detections:
            return "No scalp conditions detected. Your scalp appears healthy!"
        
        summary_parts = []
        
        # Health score
        if health_score >= 80:
            health_status = "Good"
        elif health_score >= 60:
            health_status = "Fair"
        elif health_score >= 40:
            health_status = "Poor"
        else:
            health_status = "Critical"
        
        summary_parts.append(f"Overall scalp health: {health_status} (Score: {health_score}/100)")
        
        # Detected conditions
        detected_conditions = [name for name, count in class_counts.items() if count > 0]
        if detected_conditions:
            condition_names = {
                'd': 'Dandruff',
                'o': 'Oiliness', 
                's': 'Sensitivity',
                'ds': 'Dandruff with Sensitivity',
                'os': 'Oiliness with Sensitivity',
                'dss': 'Multiple Sensitivities with Dandruff'
            }
            
            conditions_text = ", ".join([condition_names.get(cond, cond) for cond in detected_conditions])
            summary_parts.append(f"Detected conditions: {conditions_text}")
        
        # Severity breakdown
        severity_counts = {}
        for detection in detections:
            severity = detection['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        if severity_counts:
            severity_text = ", ".join([f"{count} {severity.lower()}" for severity, count in severity_counts.items()])
            summary_parts.append(f"Severity breakdown: {severity_text}")
        
        return ". ".join(summary_parts) + "."
    
    def visualize_results(self, image: np.ndarray, results: Dict, 
                         save_path: Optional[str] = None, 
                         show_confidence: bool = True,
                         show_labels: bool = True) -> np.ndarray:
        """
        Create visualization of detection results.
        
        Args:
            image: Original image as numpy array
            results: Detection results from predict()
            save_path: Path to save the visualization
            show_confidence: Whether to show confidence scores
            show_labels: Whether to show class labels
            
        Returns:
            Annotated image as numpy array
        """
        annotated_image = image.copy()
        
        for detection in results['detections']:
            x1, y1, x2, y2 = [int(coord) for coord in detection['bbox']]
            class_name = detection['class_name']
            confidence = detection['confidence']
            severity = detection['severity']
            
            # Get color for this class
            color = self.class_colors.get(class_name, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label_parts = []
            if show_labels:
                # Map class names to readable labels
                readable_names = {
                    'd': 'Dandruff',
                    'o': 'Oily',
                    's': 'Sensitive', 
                    'ds': 'Dandruff+Sensitive',
                    'os': 'Oily+Sensitive',
                    'dss': 'Multi-Sensitive+Dandruff'
                }
                label_parts.append(readable_names.get(class_name, class_name))
            
            if show_confidence:
                label_parts.append(f"{confidence:.2f}")
            
            label_parts.append(f"({severity})")
            
            label = " ".join(label_parts)
            
            # Draw label background
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            cv2.rectangle(
                annotated_image,
                (x1, y1 - label_height - 10),
                (x1 + label_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        # Add summary text
        summary_text = f"Health Score: {results['health_score']}/100"
        cv2.putText(
            annotated_image,
            summary_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            logger.info(f"Visualization saved to: {save_path}")
        
        return annotated_image
    
    def _save_annotated_image(self, image: np.ndarray, results: Dict, output_dir: str):
        """Save annotated image with detection results."""
        from ..utils.helpers import create_output_filename, ensure_dir
        
        ensure_dir(output_dir)
        filename = create_output_filename("detection", "jpg")
        save_path = Path(output_dir) / filename
        
        annotated_image = self.visualize_results(image, results)
        cv2.imwrite(str(save_path), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        
        logger.info(f"Annotated image saved to: {save_path}")
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if not self.is_loaded:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_path": self.model_path,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "image_size": self.image_size,
            "classes": self.classes,
            "class_colors": self.class_colors
        }
    
    def update_confidence_threshold(self, threshold: float):
        """Update the confidence threshold for detections."""
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
            logger.info(f"Confidence threshold updated to: {threshold}")
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
    
    def batch_predict(self, image_paths: List[str], 
                     output_dir: Optional[str] = None) -> List[Dict]:
        """
        Run inference on multiple images.
        
        Args:
            image_paths: List of image file paths
            output_dir: Directory to save results
            
        Returns:
            List of detection results for each image
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
                result = self.predict(
                    image_path, 
                    save_results=bool(output_dir),
                    output_dir=output_dir
                )
                result['image_path'] = image_path
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'detections': [],
                    'health_score': 0
                })
        
        return results 