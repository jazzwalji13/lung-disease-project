# test_image_reading.py
import cv2
import numpy as np
from PIL import Image
import os

def test_image_reading():
    """Test if we can read the dataset images"""
    test_folder = 'chest_xray/train/NORMAL'
    
    if os.path.exists(test_folder):
        files = [f for f in os.listdir(test_folder) if f.endswith('.jpeg')][:3]
        
        for file in files:
            filepath = os.path.join(test_folder, file)
            print(f"\n🔍 Testing: {file}")
            
            # Try OpenCV
            img_cv = cv2.imread(filepath)
            print(f"   OpenCV result: {img_cv is not None}")
            
            # Try PIL
            try:
                img_pil = Image.open(filepath)
                print(f"   PIL result: Can open")
                print(f"   PIL format: {img_pil.format}")
                print(f"   PIL mode: {img_pil.mode}")
                print(f"   PIL size: {img_pil.size}")
            except Exception as e:
                print(f"   PIL failed: {e}")

if __name__ == '__main__':
    test_image_reading()