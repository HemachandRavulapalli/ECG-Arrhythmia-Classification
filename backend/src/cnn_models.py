# backend/src/cnn_models.py
"""
CNN architectures for the ECG classification system.

Branch 1 — CNN1D  : raw 1D signal   (1000, 1)
Branch 2 — CNN2D  : REAL spectrogram → computed here, NOT a reshape

Design rules:
  - fs = 100 Hz for ALL spectrograms
  - nperseg = 128, noverlap = 64  → consistent shape
  - No hard-coded (100, 10) reshape
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from scipy.signal import spectrogram as scipy_spectrogram


# ======================================================
# Spectrogram helper (must match inference exactly)
# ======================================================
def signal_to_spectrogram(signal: np.ndarray, fs: float = 100,
                           nperseg: int = 128, noverlap: int = 64) -> np.ndarray:
    """
    Convert a 1D ECG signal to a log-power spectrogram.

    Parameters
    ----------
    signal  : (1000,) array at `fs` Hz
    fs      : sampling frequency (always 100 in this system)
    nperseg : STFT window length
    noverlap: STFT overlap

    Returns
    -------
    np.ndarray  shape (n_freq, n_time, 1)  — ready for CNN2D
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()
    _, _, Sxx = scipy_spectrogram(signal, fs=fs,
                                  nperseg=nperseg,
                                  noverlap=noverlap)
    spec = np.log1p(Sxx).astype(np.float32)     # log-power
    return spec[..., np.newaxis]                 # (freq, time, 1)


def batch_to_spectrograms(X: np.ndarray, fs: float = 100) -> np.ndarray:
    """
    Convert a batch of signals (N, 1000) to spectrograms (N, freq, time, 1).
    Used during training to pre-convert the whole dataset.
    """
    specs = [signal_to_spectrogram(x, fs=fs) for x in X]
    return np.array(specs, dtype=np.float32)


def get_spectrogram_shape(fs: float = 100, nperseg: int = 128,
                          noverlap: int = 64, n_samples: int = 1000) -> tuple:
    """
    Compute the exact spectrogram shape that will be produced.
    Used to build the CNN2D with the correct Input shape.
    """
    dummy = np.zeros(n_samples)
    spec  = signal_to_spectrogram(dummy, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return spec.shape   # (freq, time, 1)


# ======================================================
# CNN-1D  — Raw temporal signal
# ======================================================
def build_cnn_1d(input_shape: tuple = (1000, 1),
                 num_classes: int = 5,
                 dropout_rate: float = 0.3) -> tf.keras.Model:
    """
    Multi-scale residual CNN for 1D ECG.
    Input: (1000, 1)
    """
    inputs = layers.Input(shape=input_shape, name="ecg_1d_input")

    # ── Multi-scale stem ──────────────────────────────
    x1 = layers.Conv1D(32, 3, padding="same", activation="relu")(inputs)
    x2 = layers.Conv1D(32, 7, padding="same", activation="relu")(inputs)
    x3 = layers.Conv1D(32, 15, padding="same", activation="relu")(inputs)
    x  = layers.Concatenate()([x1, x2, x3])
    x  = layers.BatchNormalization()(x)
    x  = layers.MaxPooling1D(2)(x)          # (500, 96)

    # ── Residual block 1 ─────────────────────────────
    shortcut = layers.Conv1D(128, 1, padding="same")(x)
    x = layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 5, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)           # (250, 128)
    x = layers.Dropout(dropout_rate)(x)

    # ── Residual block 2 ─────────────────────────────
    shortcut2 = layers.Conv1D(256, 1, padding="same")(x)
    x = layers.Conv1D(256, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(256, 5, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut2])
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)           # (125, 256)
    x = layers.Dropout(dropout_rate)(x)

    # ── Global pooling + head ─────────────────────────
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="cnn1d_out")(x)

    return models.Model(inputs, outputs, name="CNN1D_Residual")


# ======================================================
# CNN-2D  — Real spectrogram
# ======================================================
def build_cnn_2d(input_shape: tuple | None = None,
                 num_classes: int = 5,
                 dropout_rate: float = 0.3) -> tf.keras.Model:
    """
    Lightweight residual 2D CNN for ECG spectrogram classification.
    Design: 3 residual blocks max, filters 32 → 64 → 128.
    """
    if input_shape is None:
        input_shape = get_spectrogram_shape()

    inputs = layers.Input(shape=input_shape, name="spec_2d_input")

    # stem
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)

    def residual_block(x, filters, dropout):
        shortcut = layers.Conv2D(filters, (1, 1), padding="same")(x)
        
        y = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
        y = layers.BatchNormalization()(y)
        y = layers.Conv2D(filters, (3, 3), padding="same")(y)
        y = layers.BatchNormalization()(y)
        
        y = layers.Add()([y, shortcut])
        y = layers.Activation("relu")(y)
        y = layers.MaxPooling2D((2, 2))(y)
        y = layers.Dropout(dropout)(y)
        return y

    # 3 Residual blocks
    x = residual_block(x, 32, dropout_rate)
    x = residual_block(x, 64, dropout_rate)
    x = residual_block(x, 128, dropout_rate)

    # Global Average Pooling + Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="cnn2d_out")(x)

    return models.Model(inputs, outputs, name="CNN2D_Residual_Spectrogram")


# ======================================================
# Advanced Hybrid sub-models
# ======================================================
def build_residual_cnn(input_shape=(1000, 1), num_classes=5) -> tf.keras.Model:
    """Deep residual 1D CNN (used inside AdvancedHybridModel)."""
    inp = layers.Input(shape=input_shape)
    x   = layers.Conv1D(64, 7, padding="same", activation="relu")(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling1D(2)(x)

    for filters in [128, 256]:
        sc = layers.Conv1D(filters, 1, padding="same")(x)
        x  = layers.Conv1D(filters, 5, padding="same", activation="relu")(x)
        x  = layers.BatchNormalization()(x)
        x  = layers.Conv1D(filters, 5, padding="same")(x)
        x  = layers.BatchNormalization()(x)
        x  = layers.Add()([x, sc])
        x  = layers.Activation("relu")(x)
        x  = layers.MaxPooling1D(2)(x)
        x  = layers.Dropout(0.3)(x)

    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dense(128, activation="relu")(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out, name="ResidualCNN")


def build_densenet_cnn(input_shape=(1000, 1), num_classes=5) -> tf.keras.Model:
    """DenseNet-style 1D CNN."""
    inp = layers.Input(shape=input_shape)
    x   = layers.Conv1D(32, 7, padding="same", activation="relu")(inp)

    # Three dense blocks
    for _ in range(3):
        branch = layers.Conv1D(32, 3, padding="same", activation="relu")(x)
        branch = layers.BatchNormalization()(branch)
        x      = layers.Concatenate()([x, branch])
        x      = layers.MaxPooling1D(2)(x)
        x      = layers.Dropout(0.2)(x)

    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out, name="DenseNetCNN")


def build_attention_cnn(input_shape=(1000, 1), num_classes=5) -> tf.keras.Model:
    """1D CNN with self-attention."""
    inp = layers.Input(shape=input_shape)
    x   = layers.Conv1D(64, 5, padding="same", activation="relu")(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling1D(2)(x)

    x   = layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling1D(2)(x)

    # Lightweight self-attention
    attn = layers.Dense(128, activation="softmax")(x)
    x    = layers.Multiply()([x, attn])

    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dense(128, activation="relu")(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out, name="AttentionCNN")


def build_multiscale_cnn(input_shape=(1000, 1), num_classes=5) -> tf.keras.Model:
    """Multi-scale 1D CNN for feature extraction at different rhythms."""
    inp = layers.Input(shape=input_shape)

    # Parallel branches at different kernel sizes
    b1 = layers.Conv1D(32, 3,  padding="same", activation="relu")(inp)
    b2 = layers.Conv1D(32, 7,  padding="same", activation="relu")(inp)
    b3 = layers.Conv1D(32, 15, padding="same", activation="relu")(inp)
    b4 = layers.Conv1D(32, 31, padding="same", activation="relu")(inp)

    x  = layers.Concatenate()([b1, b2, b3, b4])
    x  = layers.BatchNormalization()(x)
    x  = layers.MaxPooling1D(2)(x)

    x  = layers.Conv1D(256, 5, padding="same", activation="relu")(x)
    x  = layers.BatchNormalization()(x)
    x  = layers.MaxPooling1D(2)(x)
    x  = layers.Dropout(0.3)(x)

    x  = layers.GlobalAveragePooling1D()(x)
    x  = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out, name="MultiScaleCNN")
