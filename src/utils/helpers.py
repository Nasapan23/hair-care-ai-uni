"""Helper functions for the Hair Care AI application."""

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Union, List
import hashlib
from datetime import datetime


def validate_image_file(file_path: Union[str, Path], 
                       max_size_mb: int = 50,
                       supported_formats: List[str] = None) -> Tuple[bool, str]:
    """Validate image file.
    
    Args:
        file_path: Path to image file
        max_size_mb: Maximum file size in MB
        supported_formats: List of supported formats
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if supported_formats is None:
        supported_formats = ['jpg', 'jpeg', 'png', 'bmp']
    
    file_path = Path(file_path)
    
    # Check if file exists
    if not file_path.exists():
        return False, "File does not exist"
    
    # Check file extension
    ext = file_path.suffix.lower().lstrip('.')
    if ext not in supported_formats:
        return False, f"Unsupported format. Supported: {', '.join(supported_formats)}"
    
    # Check file size
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return False, f"File too large. Maximum size: {max_size_mb}MB"
    
    # Try to open image
    try:
        with Image.open(file_path) as img:
            # Check image dimensions
            width, height = img.size
            if width < 224 or height < 224:
                return False, "Image resolution too low. Minimum: 224x224"
            if width > 4096 or height > 4096:
                return False, "Image resolution too high. Maximum: 4096x4096"
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"
    
    return True, ""


def preprocess_image(image: Union[str, Path, np.ndarray, Image.Image],
                    target_size: int = 640) -> np.ndarray:
    """Preprocess image for model inference.
    
    Args:
        image: Input image (path, numpy array, or PIL Image)
        target_size: Target size for resizing
        
    Returns:
        Preprocessed image as numpy array
    """
    # Load image if path is provided
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(image, Image.Image):
        image = np.array(image)
        if image.shape[-1] == 4:  # RGBA
            image = image[:, :, :3]  # Remove alpha channel
    
    # Ensure image is RGB
    if len(image.shape) == 3 and image.shape[-1] == 3:
        pass  # Already RGB
    elif len(image.shape) == 3 and image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    
    # Resize image while maintaining aspect ratio
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad to target size
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    image = cv2.copyMakeBorder(image, top, bottom, left, right, 
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    return image


def calculate_image_hash(image_path: Union[str, Path]) -> str:
    """Calculate MD5 hash of image file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        MD5 hash string
    """
    hash_md5 = hashlib.md5()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def create_output_filename(original_path: Union[str, Path], 
                          suffix: str = "_analyzed",
                          extension: Optional[str] = None) -> str:
    """Create output filename based on original path.
    
    Args:
        original_path: Original file path
        suffix: Suffix to add to filename
        extension: New extension (if different)
        
    Returns:
        New filename
    """
    original_path = Path(original_path)
    stem = original_path.stem
    ext = extension or original_path.suffix
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stem}{suffix}_{timestamp}{ext}"


def ensure_dir(directory: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def format_confidence(confidence: float) -> str:
    """Format confidence score as percentage.
    
    Args:
        confidence: Confidence score (0-1)
        
    Returns:
        Formatted percentage string
    """
    return f"{confidence * 100:.1f}%"


def calculate_area_percentage(bbox: Tuple[int, int, int, int], 
                            image_shape: Tuple[int, int]) -> float:
    """Calculate percentage of image area covered by bounding box.
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        image_shape: Image shape (height, width)
        
    Returns:
        Area percentage (0-100)
    """
    x1, y1, x2, y2 = bbox
    bbox_area = (x2 - x1) * (y2 - y1)
    image_area = image_shape[0] * image_shape[1]
    return (bbox_area / image_area) * 100


def get_severity_level(confidence: float, area_percentage: float) -> str:
    """Determine severity level based on confidence and area.
    
    Args:
        confidence: Detection confidence
        area_percentage: Area percentage covered
        
    Returns:
        Severity level string
    """
    if area_percentage > 30 or confidence > 0.8:
        return "Severe"
    elif area_percentage > 15 or confidence > 0.6:
        return "Moderate"
    else:
        return "Mild"


def generate_recommendations(detections: dict) -> List[str]:
    """Generate care recommendations based on detections.
    
    Args:
        detections: Dictionary of detected conditions
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    if 'd' in detections or 'ds' in detections or 'dss' in detections:
        recommendations.extend([
            "Use anti-dandruff shampoo with zinc pyrithione or ketoconazole",
            "Avoid harsh hair products and excessive heat styling",
            "Maintain good scalp hygiene with regular washing"
        ])
    
    if 'o' in detections or 'os' in detections:
        recommendations.extend([
            "Use clarifying shampoo to remove excess oil",
            "Wash hair more frequently (every other day)",
            "Avoid heavy hair products and oils"
        ])
    
    if 's' in detections or 'ds' in detections or 'os' in detections or 'dss' in detections:
        recommendations.extend([
            "Use gentle, fragrance-free hair products",
            "Avoid scratching or irritating the scalp",
            "Consider consulting a dermatologist for persistent issues"
        ])
    
    if not detections:
        recommendations.append("Maintain current hair care routine - scalp appears healthy")
    
    return recommendations 