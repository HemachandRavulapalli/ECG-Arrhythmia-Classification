import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, initializers
from tensorflow.keras.applications import EfficientNetB0
import numpy as np

# ------------------------------------
# 1️⃣ Enhanced 1D CNN for ECG Signals with Residual Connections
# ------------------------------------
def build_cnn_1d(input_shape, num_classes, dropout_rate=0.2):
    """
    Ultra-advanced 1D CNN model for ECG classification with multiple attention mechanisms.
    Designed for 99%+ accuracy with sophisticated feature extraction.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Multi-scale feature extraction
    # Scale 1: Fine details
    x1 = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x1)
    x1 = layers.BatchNormalization()(x1)
    
    # Scale 2: Medium patterns
    x2 = layers.Conv1D(64, kernel_size=7, padding='same', activation='relu')(inputs)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Conv1D(64, kernel_size=7, padding='same', activation='relu')(x2)
    x2 = layers.BatchNormalization()(x2)
    
    # Scale 3: Coarse patterns
    x3 = layers.Conv1D(64, kernel_size=15, padding='same', activation='relu')(inputs)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Conv1D(64, kernel_size=15, padding='same', activation='relu')(x3)
    x3 = layers.BatchNormalization()(x3)
    
    # Concatenate multi-scale features
    x = layers.Concatenate()([x1, x2, x3])
    x = layers.Conv1D(192, kernel_size=1, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # Residual blocks with attention
    for i, filters in enumerate([256, 512, 1024]):
        # Attention mechanism
        attention = layers.GlobalAveragePooling1D()(x)
        attention = layers.Dense(filters//4, activation='relu')(attention)
        attention = layers.Dense(filters, activation='sigmoid')(attention)
        attention = layers.Reshape((1, filters))(attention)
        
        # Residual block
        residual = layers.Conv1D(filters, kernel_size=1, padding='same')(x)
        
        x = layers.Conv1D(filters, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(filters, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Apply attention
        x = layers.Multiply()([x, attention])
        
        # Add residual
        x = layers.Add()([x, residual])
        x = layers.Activation('relu')(x)
        
        if i < 2:  # Don't pool on last block
            x = layers.MaxPooling1D(pool_size=2)(x)
            x = layers.Dropout(dropout_rate)(x)
    
    # Global attention pooling
    attention_weights = layers.Dense(1, activation='sigmoid')(x)
    attention_weights = layers.Softmax(axis=1)(attention_weights)
    x = layers.Multiply()([x, attention_weights])
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers with advanced regularization
    x = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model


# ------------------------------------
# 2️⃣ Enhanced 2D CNN for Spectrograms with Attention
# ------------------------------------
def build_cnn_2d(input_shape, num_classes, dropout_rate=0.2):
    """
    Ultra-advanced 2D CNN for ECG spectrogram-like representations with multiple attention mechanisms.
    Designed for 99%+ accuracy with sophisticated feature extraction.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Multi-scale feature extraction
    # Scale 1: Fine details
    x1 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x1)
    x1 = layers.BatchNormalization()(x1)
    
    # Scale 2: Medium patterns
    x2 = layers.Conv2D(64, (5, 5), padding='same', activation='relu')(inputs)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Conv2D(64, (5, 5), padding='same', activation='relu')(x2)
    x2 = layers.BatchNormalization()(x2)
    
    # Scale 3: Coarse patterns
    x3 = layers.Conv2D(64, (7, 7), padding='same', activation='relu')(inputs)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Conv2D(64, (7, 7), padding='same', activation='relu')(x3)
    x3 = layers.BatchNormalization()(x3)
    
    # Concatenate multi-scale features
    x = layers.Concatenate()([x1, x2, x3])
    x = layers.Conv2D(192, (1, 1), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Residual blocks with attention
    for i, filters in enumerate([256, 512, 1024]):
        # Get current number of channels
        current_channels = x.shape[-1]
        
        # Spatial attention
        spatial_attention = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
        x = layers.Multiply()([x, spatial_attention])
        
        # Channel attention
        channel_attention = layers.GlobalAveragePooling2D()(x)
        channel_attention = layers.Dense(max(current_channels//4, 16), activation='relu')(channel_attention)
        channel_attention = layers.Dense(current_channels, activation='sigmoid')(channel_attention)
        channel_attention = layers.Reshape((1, 1, current_channels))(channel_attention)
        x = layers.Multiply()([x, channel_attention])
        
        # Residual block
        residual = layers.Conv2D(filters, (1, 1), padding='same')(x)
        
        x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Add residual
        x = layers.Add()([x, residual])
        x = layers.Activation('relu')(x)
        
        if i < 2:  # Don't pool on last block
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(dropout_rate)(x)
    
    # Global attention pooling
    attention_weights = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    attention_weights = layers.Softmax(axis=[1, 2])(attention_weights)
    x = layers.Multiply()([x, attention_weights])
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers with advanced regularization
    x = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model
