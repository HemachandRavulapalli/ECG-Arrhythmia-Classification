# backend/src/preprocessing.py
"""
Research-grade ECG preprocessing pipeline.

Design rules (MANDATORY):
  - All signals must be at 100 Hz BEFORE any processing
  - All windows must be exactly 1000 samples (10 sec)
  - PCA aggregation is done AFTER resampling, BEFORE preprocessing
  - This file is used identically during training AND inference
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, medfilt, resample
from sklearn.decomposition import PCA
import pywt


# =====================================================
# 1. Resampling
# =====================================================
def resample_signal(signal: np.ndarray, original_fs: float, target_fs: float = 100) -> np.ndarray:
    """
    Resample a 1D or 2D (N, leads) signal from original_fs → target_fs.
    MIT-BIH: 360 → 100
    PTB-XL LR: 100 → 100 (no-op)
    """
    signal = np.asarray(signal, dtype=np.float64)

    if original_fs == target_fs:
        return signal

    duration = signal.shape[0] / original_fs
    target_length = max(1, int(round(duration * target_fs)))

    if signal.ndim == 1:
        return resample(signal, target_length)
    else:
        # Multi-lead: resample each lead independently
        resampled = np.zeros((target_length, signal.shape[1]), dtype=np.float64)
        for ch in range(signal.shape[1]):
            resampled[:, ch] = resample(signal[:, ch], target_length)
        return resampled


# =====================================================
# 2. PCA Lead Aggregation
# =====================================================
def aggregate_leads(signal_multi: np.ndarray) -> np.ndarray:
    """
    Aggregate multi-lead ECG into a single channel via PCA.
    Input:  (N, n_leads)  – already resampled
    Output: (N,)          – first principal component

    IMPORTANT: Always resample FIRST, then call this.
    """
    signal_multi = np.asarray(signal_multi, dtype=np.float64)

    if signal_multi.ndim == 1:
        return signal_multi  # already single lead

    if signal_multi.shape[1] == 1:
        return signal_multi[:, 0]

    # Remove zero-variance leads before PCA
    std = np.std(signal_multi, axis=0)
    valid = std > 1e-10
    if valid.sum() == 0:
        return signal_multi[:, 0]
    if valid.sum() == 1:
        return signal_multi[:, valid].flatten()

    pca = PCA(n_components=1)
    agg = pca.fit_transform(signal_multi[:, valid]).flatten()
    return agg


# =====================================================
# 3. Individual Filter Functions  (fs=100 aware)
# =====================================================
def bandpass_filter(signal: np.ndarray, lowcut: float = 0.5,
                    highcut: float = 40.0, fs: float = 100, order: int = 4) -> np.ndarray:
    """Bandpass 0.5–40 Hz (ECG diagnostics range at 100 Hz)."""
    nyq = 0.5 * fs
    highcut = min(highcut, 0.45 * fs)   # guard against instability
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, signal)


def notch_filter(signal: np.ndarray, freq: float = 50.0,
                 fs: float = 100, Q: float = 30) -> np.ndarray:
    """Remove powerline interference (50 Hz or 60 Hz)."""
    w0 = freq / (fs / 2)
    if w0 >= 1.0:
        return signal   # frequency above Nyquist – skip
    b, a = iirnotch(w0, Q)
    return filtfilt(b, a, signal)


def remove_baseline(signal: np.ndarray, kernel_size: int = 201) -> np.ndarray:
    """Baseline wander removal via median filter subtraction."""
    # kernel_size must be odd and < signal length
    ks = min(kernel_size, len(signal) - 1)
    if ks % 2 == 0:
        ks -= 1
    if ks < 3:
        return signal
    baseline = medfilt(signal, ks)
    return signal - baseline


def wavelet_denoise(signal: np.ndarray, wavelet: str = "db4", level: int = 2) -> np.ndarray:
    """Soft-threshold wavelet denoising."""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thresh = sigma * np.sqrt(2 * np.log(max(len(signal), 1)))
    coeffs[1:] = [pywt.threshold(c, thresh, mode="soft") for c in coeffs[1:]]
    rec = pywt.waverec(coeffs, wavelet)

    # Guarantee length consistency
    if len(rec) > len(signal):
        rec = rec[:len(signal)]
    elif len(rec) < len(signal):
        rec = np.pad(rec, (0, len(signal) - len(rec)))
    return rec


def normalize(signal: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalisation."""
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)


# =====================================================
# 4. Canonical Preprocessing Pipeline  (fs=100)
# =====================================================
def preprocess_pipeline(signal: np.ndarray, fs: float = 100) -> np.ndarray:
    """
    Canonical preprocessing.  Input must already be at `fs` Hz.
    Returns a 1D float32 array.

    Order:
      bandpass → notch → baseline → wavelet → normalize
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()
    signal = bandpass_filter(signal, fs=fs)
    signal = notch_filter(signal, fs=fs)
    signal = remove_baseline(signal)
    signal = wavelet_denoise(signal)
    signal = normalize(signal)
    return signal.astype(np.float32)


# =====================================================
# 5. Full Window Extraction Helper
# =====================================================
def extract_window(signal: np.ndarray, window_size: int = 1000) -> np.ndarray:
    """
    Crop / pad signal to exactly `window_size` samples.
    Always crop from the first sample.
    """
    signal = np.asarray(signal).flatten()
    if len(signal) >= window_size:
        return signal[:window_size]
    return np.pad(signal, (0, window_size - len(signal))).astype(np.float32)


# =====================================================
# 6. Legacy shim  (keeps old callers working)
# =====================================================
def preprocess_ecg(signal, fs=100, window_size=1000):
    """
    Backward-compatible wrapper.
    Returns dict with 'filtered_signal' key.
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()
    signal = extract_window(signal, window_size)
    processed = preprocess_pipeline(signal, fs=fs)
    return {
        "filtered_signal": processed,
    }
