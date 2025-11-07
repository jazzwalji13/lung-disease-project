\# data_preprocessing.py (Lightweight - no TensorFlow)
import cv2
import numpy as np

def enhance_pneumonia_features(image_path):
    """Enhanced preprocessing for pneumonia - works without TensorFlow"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not load image")
        
        # Convert to grayscale for processing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Convert back to 3 channels
        if len(image.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return enhanced
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None

def prepare_image_for_model(image, target_size=(224, 224)):
    """Resize and normalize image for model"""
    # Resize
    image = cv2.resize(image, target_size)
    # Normalize
    image = image / 255.0
    return image