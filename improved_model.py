import tensorflow as tf
from tensorflow.keras import layers, Model

def create_pneumonia_model():
    """Improved model architecture for pneumonia detection"""
    base_model = tf.keras.applications.DenseNet201(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze initial layers
    base_model.trainable = False
    
    # Add custom layers
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    predictions = layers.Dense(2, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model