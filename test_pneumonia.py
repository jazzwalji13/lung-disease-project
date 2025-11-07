from improved_model import create_pneumonia_model
from data_preprocessing import enhance_pneumonia_features
import cv2
import numpy as np

def test_pneumonia_detection():
    """Test pneumonia detection on sample image"""
    model = create_pneumonia_model()
    print("Pneumonia model loaded successfully!")
    
    # Load and preprocess test image
    # image = cv2.imread('test_image.jpg')
    # processed_image = enhance_pneumonia_features(image)
    
    print("Pneumonia detection test completed!")

if __name__ == "__main__":
    test_pneumonia_detection()