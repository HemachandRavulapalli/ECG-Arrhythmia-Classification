# backend/src/feature_extraction.py
"""
Research-grade ECG feature extraction.

Features extracted (all at fs=100):
  1.  RR mean (sec)
  2.  RR std  (HRV proxy)
  3.  RMSSD   (short-term HRV)
  4.  Heart rate (BPM from mean RR)
  5.  QRS width estimate (sec)
  6.  Spectral entropy
  7.  Signal energy
  8.  Signal skewness
  9.  Signal kurtosis
  10. Band energy: LF  (0.5–4 Hz)
  11. Band energy: MF  (4–15 Hz,  QRS region)
  12. Band energy: HF  (15–40 Hz)
  13. LF/MF ratio
  14. Peak amplitude (max of signal)
  15. P-R interval proxy
  16. n_beats (number of R-peaks detected)

All features are returned as a (16,) float32 array.
Function signature mirrors the legacy `extract_ecg_features(signal, fs)`.
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.stats  import skew, kurtosis


# ======================================================
# Helpers
# ======================================================
def _safe_rr_features(peaks: np.ndarray, fs: float):
    """
    Compute RR-based features from peak sample indices.
    Returns (rr_mean, rr_std, rmssd, hr_bpm) — all float.
    """
    if len(peaks) < 2:
        return 0.0, 0.0, 0.0, 0.0

    rr = np.diff(peaks).astype(np.float64) / fs   # seconds

    rr_mean = float(np.mean(rr))
    rr_std  = float(np.std(rr))
    rmssd   = float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) > 1 else 0.0
    hr_bpm  = 60.0 / rr_mean if rr_mean > 0 else 0.0

    return rr_mean, rr_std, rmssd, hr_bpm


def _qrs_width(signal: np.ndarray, peaks: np.ndarray, fs: float) -> float:
    """
    Rough QRS width: median duration of the peak envelope above signal mean.
    Returns width in seconds.
    """
    if len(peaks) == 0:
        return 0.0

    threshold = np.mean(signal)
    widths = []
    for p in peaks:
        lo, hi = p, p
        while lo > 0 and signal[lo] > threshold:
            lo -= 1
        while hi < len(signal) - 1 and signal[hi] > threshold:
            hi += 1
        widths.append((hi - lo) / fs)

    return float(np.median(widths)) if widths else 0.0


def _spectral_entropy(signal: np.ndarray) -> float:
    """Normalised spectral entropy of the power spectrum."""
    fft_mag = np.abs(np.fft.rfft(signal)) ** 2
    total = fft_mag.sum()
    if total < 1e-12:
        return 0.0
    p = fft_mag / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)) / np.log(len(p) + 1))


def _band_energy(signal: np.ndarray, fs: float, low: float, high: float) -> float:
    """Energy in a frequency band [low, high] Hz."""
    fft_mag = np.abs(np.fft.rfft(signal)) ** 2
    freqs   = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    mask    = (freqs >= low) & (freqs <= high)
    return float(np.sum(fft_mag[mask]))


# ======================================================
# Main feature extraction
# ======================================================
def extract_ecg_features(signal: np.ndarray, fs: float = 100) -> np.ndarray:
    """
    Extract 16 research-grade features from a preprocessed 1D ECG window.

    Parameters
    ----------
    signal : np.ndarray  shape (1000,)
        Preprocessed, normalised ECG at `fs` Hz.
    fs : float
        Sampling frequency (default 100 for all training/inference data).

    Returns
    -------
    np.ndarray  shape (16,)  dtype float32
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()

    # ── R-peak detection ──────────────────────────────
    min_distance = max(int(fs * 0.3), 1)  # at least 0.3 s between beats (≥200 BPM cap)
    height_thr   = np.std(signal) * 0.3

    peaks, _ = find_peaks(signal, distance=min_distance, height=height_thr)

    # ── RR features ───────────────────────────────────
    rr_mean, rr_std, rmssd, hr_bpm = _safe_rr_features(peaks, fs)

    # ── QRS width ─────────────────────────────────────
    qrs_w = _qrs_width(signal, peaks, fs)

    # ── Spectral entropy ──────────────────────────────
    sp_ent = _spectral_entropy(signal)

    # ── Signal energy ─────────────────────────────────
    energy = float(np.sum(signal ** 2))

    # ── Higher-order statistics ───────────────────────
    skewness = float(skew(signal))
    kurt     = float(kurtosis(signal))

    # ── Band energies ─────────────────────────────────
    lf_energy = _band_energy(signal, fs, 0.5, 4.0)
    mf_energy = _band_energy(signal, fs, 4.0, 15.0)
    hf_energy = _band_energy(signal, fs, 15.0, 40.0)

    lf_mf_ratio = lf_energy / (mf_energy + 1e-8)

    # ── Amplitude ─────────────────────────────────────
    peak_amplitude = float(np.max(signal))

    # ── PR interval proxy (mean distance from prior zero-cross to peak) ─
    try:
        zero_crossings = np.where(np.diff(np.sign(signal)))[0]
        if len(zero_crossings) > 0 and len(peaks) > 0:
            pr_vals = []
            for p in peaks:
                prior = zero_crossings[zero_crossings < p]
                if len(prior) > 0:
                    pr_vals.append((p - prior[-1]) / fs)
            pr_proxy = float(np.mean(pr_vals)) if pr_vals else 0.0
        else:
            pr_proxy = 0.0
    except Exception:
        pr_proxy = 0.0

    n_beats = float(len(peaks))

    # ── Pack features ─────────────────────────────────
    features = np.array([
        rr_mean,         # 0
        rr_std,          # 1
        rmssd,           # 2
        hr_bpm,          # 3
        qrs_w,           # 4
        sp_ent,          # 5
        energy,          # 6
        skewness,        # 7
        kurt,            # 8
        lf_energy,       # 9
        mf_energy,       # 10
        hf_energy,       # 11
        lf_mf_ratio,     # 12
        peak_amplitude,  # 13
        pr_proxy,        # 14
        n_beats,         # 15
    ], dtype=np.float32)

    # Replace any NaN/Inf with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


# ======================================================
# Feature names (for interpretability / logging)
# ======================================================
FEATURE_NAMES = [
    "rr_mean", "rr_std", "rmssd", "hr_bpm",
    "qrs_width", "spectral_entropy", "energy",
    "skewness", "kurtosis",
    "lf_energy", "mf_energy", "hf_energy", "lf_mf_ratio",
    "peak_amplitude", "pr_proxy", "n_beats",
]
