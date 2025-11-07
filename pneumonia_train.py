import tensorflow as tf
from improved_model import create_pneumonia_model
from data_preprocessing import create_pneumonia_datagen
import numpy as np

def train_pneumonia_model():
    """Training script focused on pneumonia accuracy"""
    # Create model
    model = create_pneumonia_model()
    
    # Compile with class weights
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    # Higher weight for pneumonia class
    class_weights = {0: 1.0, 1: 3.0}
    
    print("Pneumonia-focused training started...")
    return model

if __name__ == "__main__":
    train_pneumonia_model()