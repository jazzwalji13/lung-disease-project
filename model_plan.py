# model_plan.py - Planning and structure without TensorFlow
MODEL_ARCHITECTURE = {
    "base_model": "DenseNet201",
    "input_size": (224, 224, 3),
    "custom_layers": [
        "GlobalAveragePooling2D",
        "Dense(512, relu)",
        "Dropout(0.5)", 
        "Dense(256, relu)",
        "Dropout(0.3)",
        "Dense(2, softmax)"
    ],
    "optimizer": "adam",
    "loss": "categorical_crossentropy"
}

def get_training_plan():
    """Returns the training strategy"""
    return {
        "class_weights": {0: 1.0, 1: 3.0},  # Higher weight for pneumonia
        "batch_size": 32,
        "epochs": 50,
        "augmentation": True
    }