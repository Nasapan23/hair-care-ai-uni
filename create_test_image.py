#!/usr/bin/env python3
"""
Create a test scalp image for demonstration purposes.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
from pathlib import Path

def create_test_scalp_image(width=640, height=640, save_path="data/sample_images/test_scalp.jpg"):
    """Create a realistic-looking test scalp image."""
    
    # Ensure directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create base scalp color (pinkish-beige)
    base_color = (220, 180, 160)
    
    # Create the base image
    img = Image.new('RGB', (width, height), base_color)
    draw = ImageDraw.Draw(img)
    
    # Add some texture variation
    for _ in range(1000):
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        # Slight color variations
        r = base_color[0] + random.randint(-20, 20)
        g = base_color[1] + random.randint(-20, 20)
        b = base_color[2] + random.randint(-20, 20)
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        draw.point((x, y), fill=(r, g, b))
    
    # Add some hair follicles (small dark dots)
    for _ in range(200):
        x = random.randint(5, width-5)
        y = random.randint(5, height-5)
        size = random.randint(1, 3)
        draw.ellipse([x-size, y-size, x+size, y+size], fill=(80, 60, 40))
    
    # Add some potential "conditions" for testing
    # Simulate some oily areas (slightly darker/shinier)
    for _ in range(5):
        x = random.randint(50, width-50)
        y = random.randint(50, height-50)
        size = random.randint(20, 40)
        # Draw semi-transparent darker area
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.ellipse([x-size, y-size, x+size, y+size], 
                           fill=(150, 120, 100, 30))
        img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    
    # Add some texture with slight blur
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # Save the image
    img.save(save_path, 'JPEG', quality=85)
    print(f"Test scalp image created: {save_path}")
    print(f"Image size: {width}x{height}")
    
    return save_path

if __name__ == "__main__":
    # Create a test image
    image_path = create_test_scalp_image()
    
    # Also create a smaller version for faster testing
    small_path = create_test_scalp_image(320, 320, "data/sample_images/test_scalp_small.jpg")
    
    print("\nTest images created successfully!")
    print("You can now upload these images in the Streamlit app to test the analysis.") 