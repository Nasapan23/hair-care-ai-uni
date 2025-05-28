"""
Image Processor for Hair Care AI Application

This module provides comprehensive image processing capabilities including
preprocessing, enhancement, quality assessment, and preparation for AI analysis.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from pathlib import Path

from ..utils.logger import logger
from ..utils.config import config
from ..utils.helpers import validate_image_file


class ImageProcessor:
    """
    Comprehensive image processor for scalp health analysis.
    
    Handles image preprocessing, enhancement, quality assessment,
    and preparation for AI model inference.
    """
    
    def __init__(self):
        """Initialize the image processor."""
        self.target_size = config.image_size
        self.supported_formats = config.supported_formats
        
        # Quality assessment thresholds
        self.quality_thresholds = {
            'blur_threshold': 100.0,  # Laplacian variance threshold
            'brightness_min': 50,     # Minimum brightness
            'brightness_max': 200,    # Maximum brightness
            'contrast_min': 30,       # Minimum contrast
            'saturation_min': 20      # Minimum saturation
        }
        
        logger.info("ImageProcessor initialized")
    
    def load_image(self, image_path: str) -> Dict:
        """
        Load and validate an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict containing image data and metadata
        """
        # Validate image file
        validation = validate_image_file(image_path)
        if not validation['valid']:
            raise ValueError(f"Invalid image: {validation['error']}")
        
        try:
            # Load image using PIL for better format support
            pil_image = Image.open(image_path).convert('RGB')
            
            # Convert to numpy array for processing
            image_array = np.array(pil_image)
            
            # Get image metadata
            metadata = {
                'original_size': pil_image.size,  # (width, height)
                'array_shape': image_array.shape,  # (height, width, channels)
                'file_path': image_path,
                'file_size': Path(image_path).stat().st_size,
                'format': pil_image.format,
                'mode': pil_image.mode
            }
            
            logger.info(f"Image loaded successfully: {image_path}")
            logger.info(f"Image size: {metadata['original_size']}, Format: {metadata['format']}")
            
            return {
                'image': image_array,
                'pil_image': pil_image,
                'metadata': metadata,
                'valid': True
            }
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {str(e)}")
            raise
    
    def assess_image_quality(self, image: np.ndarray) -> Dict:
        """
        Assess the quality of an image for scalp analysis.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dict containing quality metrics and assessment
        """
        quality_metrics = {}
        
        try:
            # Convert to grayscale for some metrics
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 1. Blur detection using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            quality_metrics['blur_score'] = laplacian_var
            quality_metrics['is_blurry'] = laplacian_var < self.quality_thresholds['blur_threshold']
            
            # 2. Brightness assessment
            brightness = np.mean(gray)
            quality_metrics['brightness'] = brightness
            quality_metrics['brightness_ok'] = (
                self.quality_thresholds['brightness_min'] <= brightness <= 
                self.quality_thresholds['brightness_max']
            )
            
            # 3. Contrast assessment
            contrast = gray.std()
            quality_metrics['contrast'] = contrast
            quality_metrics['contrast_ok'] = contrast >= self.quality_thresholds['contrast_min']
            
            # 4. Saturation assessment (for color images)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            saturation = np.mean(hsv[:, :, 1])
            quality_metrics['saturation'] = saturation
            quality_metrics['saturation_ok'] = saturation >= self.quality_thresholds['saturation_min']
            
            # 5. Noise assessment using standard deviation of Laplacian
            noise_score = cv2.Laplacian(gray, cv2.CV_64F).std()
            quality_metrics['noise_score'] = noise_score
            quality_metrics['low_noise'] = noise_score < 50  # Threshold for acceptable noise
            
            # 6. Overall quality score (0-100)
            quality_score = self._calculate_overall_quality(quality_metrics)
            quality_metrics['overall_quality'] = quality_score
            
            # 7. Quality category
            if quality_score >= 80:
                quality_category = 'Excellent'
            elif quality_score >= 60:
                quality_category = 'Good'
            elif quality_score >= 40:
                quality_category = 'Fair'
            else:
                quality_category = 'Poor'
            
            quality_metrics['quality_category'] = quality_category
            
            # 8. Recommendations for improvement
            recommendations = self._generate_quality_recommendations(quality_metrics)
            quality_metrics['recommendations'] = recommendations
            
            logger.info(f"Image quality assessed: {quality_category} (Score: {quality_score:.1f})")
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {str(e)}")
            raise
    
    def _calculate_overall_quality(self, metrics: Dict) -> float:
        """Calculate overall quality score from individual metrics."""
        score = 100.0
        
        # Penalize for poor metrics
        if metrics['is_blurry']:
            score -= 30
        
        if not metrics['brightness_ok']:
            if metrics['brightness'] < self.quality_thresholds['brightness_min']:
                score -= 20  # Too dark
            else:
                score -= 15  # Too bright
        
        if not metrics['contrast_ok']:
            score -= 20
        
        if not metrics['saturation_ok']:
            score -= 10
        
        if not metrics['low_noise']:
            score -= 15
        
        return max(0, score)
    
    def _generate_quality_recommendations(self, metrics: Dict) -> List[str]:
        """Generate recommendations for improving image quality."""
        recommendations = []
        
        if metrics['is_blurry']:
            recommendations.append("Image appears blurry - ensure camera is focused and stable")
        
        if not metrics['brightness_ok']:
            if metrics['brightness'] < self.quality_thresholds['brightness_min']:
                recommendations.append("Image is too dark - improve lighting conditions")
            else:
                recommendations.append("Image is too bright - reduce lighting or adjust exposure")
        
        if not metrics['contrast_ok']:
            recommendations.append("Low contrast detected - ensure good lighting and avoid shadows")
        
        if not metrics['saturation_ok']:
            recommendations.append("Low color saturation - check camera settings and lighting")
        
        if not metrics['low_noise']:
            recommendations.append("High noise detected - improve lighting or use lower ISO settings")
        
        if not recommendations:
            recommendations.append("Image quality is good for analysis")
        
        return recommendations
    
    def enhance_image(self, image: np.ndarray, enhancement_level: str = 'moderate') -> np.ndarray:
        """
        Enhance image quality for better analysis results.
        
        Args:
            image: Input image as numpy array
            enhancement_level: 'mild', 'moderate', or 'aggressive'
            
        Returns:
            Enhanced image as numpy array
        """
        try:
            # Convert to PIL for easier enhancement
            pil_image = Image.fromarray(image)
            
            # Enhancement parameters based on level
            if enhancement_level == 'mild':
                brightness_factor = 1.1
                contrast_factor = 1.1
                sharpness_factor = 1.1
                saturation_factor = 1.05
            elif enhancement_level == 'moderate':
                brightness_factor = 1.2
                contrast_factor = 1.3
                sharpness_factor = 1.2
                saturation_factor = 1.1
            else:  # aggressive
                brightness_factor = 1.3
                contrast_factor = 1.5
                sharpness_factor = 1.4
                saturation_factor = 1.2
            
            # Apply enhancements
            enhanced = pil_image
            
            # Brightness enhancement
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(brightness_factor)
            
            # Contrast enhancement
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(contrast_factor)
            
            # Sharpness enhancement
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(sharpness_factor)
            
            # Color saturation enhancement
            enhancer = ImageEnhance.Color(enhanced)
            enhanced = enhancer.enhance(saturation_factor)
            
            # Convert back to numpy array
            enhanced_array = np.array(enhanced)
            
            logger.info(f"Image enhanced with {enhancement_level} level")
            return enhanced_array
            
        except Exception as e:
            logger.error(f"Image enhancement failed: {str(e)}")
            return image  # Return original if enhancement fails
    
    def preprocess_for_analysis(self, image: np.ndarray, 
                               target_size: Optional[int] = None,
                               normalize: bool = True) -> np.ndarray:
        """
        Preprocess image for AI model analysis.
        
        Args:
            image: Input image as numpy array
            target_size: Target size for resizing (if None, uses config default)
            normalize: Whether to normalize pixel values
            
        Returns:
            Preprocessed image ready for model inference
        """
        try:
            target_size = target_size or self.target_size
            
            # Resize image while maintaining aspect ratio
            processed_image = self._resize_with_padding(image, target_size)
            
            # Normalize if requested
            if normalize:
                processed_image = processed_image.astype(np.float32) / 255.0
            
            logger.info(f"Image preprocessed to size: {processed_image.shape}")
            return processed_image
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise
    
    def _resize_with_padding(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """
        Resize image to target size while maintaining aspect ratio using padding.
        
        Args:
            image: Input image
            target_size: Target square size
            
        Returns:
            Resized and padded image
        """
        h, w = image.shape[:2]
        
        # Calculate scaling factor
        scale = min(target_size / w, target_size / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create padded image
        padded = np.zeros((target_size, target_size, 3), dtype=image.dtype)
        
        # Calculate padding offsets
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        
        # Place resized image in center
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return padded
    
    def extract_scalp_region(self, image: np.ndarray, 
                            method: str = 'adaptive') -> Dict:
        """
        Extract scalp region from the image for focused analysis.
        
        Args:
            image: Input image
            method: Extraction method ('adaptive', 'threshold', 'contour')
            
        Returns:
            Dict containing extracted region and metadata
        """
        try:
            if method == 'adaptive':
                return self._extract_scalp_adaptive(image)
            elif method == 'threshold':
                return self._extract_scalp_threshold(image)
            elif method == 'contour':
                return self._extract_scalp_contour(image)
            else:
                raise ValueError(f"Unknown extraction method: {method}")
                
        except Exception as e:
            logger.error(f"Scalp region extraction failed: {str(e)}")
            # Return original image if extraction fails
            return {
                'scalp_region': image,
                'mask': np.ones(image.shape[:2], dtype=np.uint8) * 255,
                'extraction_success': False,
                'method_used': method
            }
    
    def _extract_scalp_adaptive(self, image: np.ndarray) -> Dict:
        """Extract scalp region using adaptive thresholding."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply mask to original image
        scalp_region = image.copy()
        scalp_region[mask == 0] = [0, 0, 0]
        
        return {
            'scalp_region': scalp_region,
            'mask': mask,
            'extraction_success': True,
            'method_used': 'adaptive'
        }
    
    def _extract_scalp_threshold(self, image: np.ndarray) -> Dict:
        """Extract scalp region using simple thresholding."""
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply mask
        scalp_region = image.copy()
        scalp_region[mask == 0] = [0, 0, 0]
        
        return {
            'scalp_region': scalp_region,
            'mask': mask,
            'extraction_success': True,
            'method_used': 'threshold'
        }
    
    def _extract_scalp_contour(self, image: np.ndarray) -> Dict:
        """Extract scalp region using contour detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask from largest contour
        mask = np.zeros(gray.shape, dtype=np.uint8)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.fillPoly(mask, [largest_contour], 255)
        
        # Apply mask
        scalp_region = image.copy()
        scalp_region[mask == 0] = [0, 0, 0]
        
        return {
            'scalp_region': scalp_region,
            'mask': mask,
            'extraction_success': len(contours) > 0,
            'method_used': 'contour'
        }
    
    def create_analysis_patches(self, image: np.ndarray, 
                               patch_size: int = 224,
                               overlap: float = 0.2) -> List[Dict]:
        """
        Create overlapping patches for detailed analysis.
        
        Args:
            image: Input image
            patch_size: Size of each patch
            overlap: Overlap ratio between patches
            
        Returns:
            List of patch dictionaries with metadata
        """
        try:
            h, w = image.shape[:2]
            patches = []
            
            # Calculate step size based on overlap
            step_size = int(patch_size * (1 - overlap))
            
            patch_id = 0
            for y in range(0, h - patch_size + 1, step_size):
                for x in range(0, w - patch_size + 1, step_size):
                    # Extract patch
                    patch = image[y:y + patch_size, x:x + patch_size]
                    
                    # Create patch metadata
                    patch_info = {
                        'id': patch_id,
                        'patch': patch,
                        'position': (x, y),
                        'size': (patch_size, patch_size),
                        'center': (x + patch_size // 2, y + patch_size // 2)
                    }
                    
                    patches.append(patch_info)
                    patch_id += 1
            
            logger.info(f"Created {len(patches)} analysis patches")
            return patches
            
        except Exception as e:
            logger.error(f"Patch creation failed: {str(e)}")
            return []
    
    def save_processed_image(self, image: np.ndarray, 
                           output_path: str,
                           quality: int = 95) -> bool:
        """
        Save processed image to file.
        
        Args:
            image: Image to save
            output_path: Output file path
            quality: JPEG quality (if applicable)
            
        Returns:
            True if saved successfully
        """
        try:
            # Convert to PIL Image
            if image.dtype == np.float32 or image.dtype == np.float64:
                # Denormalize if needed
                image = (image * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(image)
            
            # Save with appropriate format
            if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
                pil_image.save(output_path, 'JPEG', quality=quality)
            else:
                pil_image.save(output_path)
            
            logger.info(f"Processed image saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save image: {str(e)}")
            return False
    
    def batch_process_images(self, image_paths: List[str],
                           output_dir: str,
                           enhancement_level: str = 'moderate') -> List[Dict]:
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of input image paths
            output_dir: Output directory for processed images
            enhancement_level: Enhancement level to apply
            
        Returns:
            List of processing results
        """
        results = []
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for i, image_path in enumerate(image_paths):
            try:
                logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
                
                # Load image
                loaded = self.load_image(image_path)
                
                # Assess quality
                quality = self.assess_image_quality(loaded['image'])
                
                # Enhance if needed
                if quality['overall_quality'] < 70:
                    enhanced = self.enhance_image(loaded['image'], enhancement_level)
                else:
                    enhanced = loaded['image']
                
                # Preprocess for analysis
                preprocessed = self.preprocess_for_analysis(enhanced, normalize=False)
                
                # Save processed image
                output_path = Path(output_dir) / f"processed_{Path(image_path).name}"
                self.save_processed_image(preprocessed, str(output_path))
                
                result = {
                    'input_path': image_path,
                    'output_path': str(output_path),
                    'original_quality': quality,
                    'enhanced': quality['overall_quality'] < 70,
                    'processing_successful': True
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {str(e)}")
                results.append({
                    'input_path': image_path,
                    'error': str(e),
                    'processing_successful': False
                })
        
        return results 