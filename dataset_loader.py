# dataset_loader.py
import os
import random
import shutil

def prepare_test_images(num_images=50):
    """Copy random images from datasets to uploads folder for testing"""
    source_folders = [
        'datasets/pneumonia/train/NORMAL',
        'datasets/pneumonia/train/PNEUMONIA'
    ]
    
    all_images = []
    for folder in source_folders:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_images.append(os.path.join(folder, file))
    
    # Select random images
    selected_images = random.sample(all_images, min(num_images, len(all_images)))
    
    # Copy to uploads folder
    for i, image_path in enumerate(selected_images):
        shutil.copy2(image_path, f'static/uploads/dataset_image_{i:03d}.jpg')
    
    print(f"✅ Copied {len(selected_images)} images to uploads folder")
    return selected_images

def get_random_dataset_image():
    """Get a random image path from datasets for quick testing"""
    all_images = []
    for folder in ['datasets/pneumonia/train/NORMAL', 'datasets/pneumonia/train/PNEUMONIA']:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_images.append(os.path.join(folder, file))
    
    return random.choice(all_images) if all_images else None