#!/usr/bin/env python3
"""
pdf_to_signal.py
Scientifically correct ECG waveform extraction from Kardia 6L PDFs and images.

Architecture (v2 — Sensor Layer Rewrite):
  1. Render PDF/image to high-resolution bitmap
  2. Adaptive threshold → binary waveform mask
  3. Band detection: only accept bands with height >= 60px AND sufficient variance
  4. Column tracing: use np.min(ys) — traces darkest pixel = actual waveform
  5. Resampling: scipy.signal.resample (mathematically correct, preserves R-peaks)
  6. Lead quality filter: reject bands with low variance, low spectral energy, < 4 peaks
  7. Return single BEST lead (no PCA — avoids amplifying garbage leads)

Model is frozen. Only extraction changes.
"""

import numpy as np
import cv2
import fitz  # PyMuPDF
import os
from scipy.signal import resample, welch, find_peaks


# ======================================================
# Constants
# ======================================================
PDF_RENDER_SCALE  = 5        # 5x zoom for sharper lines at print resolution
MIN_BAND_HEIGHT   = 60       # px at 5x scale (~12px original) — filters text rows
MIN_BAND_ACTIVE   = 0.40     # ≥40% of columns must have a signal pixel
GAP_THRESHOLD     = 25       # px gap between rows to start a new lead band
MIN_LEAD_VARIANCE = 0.02     # leads flatter than this are likely flatlines or text
MIN_SPECTRAL_RATIO = 0.60    # leads with < 60% energy in 0.5-40 Hz band are noise
MIN_PEAKS         = 4        # leads with < 4 detectable peaks are skipped


# ======================================================
# PDF → High-Resolution Image
# ======================================================
def pdf_to_image(pdf_path, scale=PDF_RENDER_SCALE):
    """
    Render the best ECG page from a PDF to a high-res numpy array (RGB).

    Kardia PDFs have 5 pages: page 0 is a blank/logo cover, pages 1-4 contain
    ECG lead strips. We scan all pages and pick the one with the most valid
    lead bands to ensure each unique file renders its own unique waveform.
    A quick low-res scan is used for band counting; the winner is rendered at
    full scale for accurate extraction.
    """
    doc = fitz.open(pdf_path)
    best_page_idx = 0
    best_band_count = 0

    # Quick low-res scan to find the page with the most ECG lead bands
    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # low-res for speed
        img_lr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 4:
            img_lr = img_lr[:, :, :3]
        gray = cv2.cvtColor(img_lr, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        bw = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 15, 5
        )
        # Count bands at 2x scale (MIN_BAND_HEIGHT scales proportionally)
        bands = detect_lead_bands(bw, min_height=max(1, MIN_BAND_HEIGHT // (PDF_RENDER_SCALE // 2)))
        if len(bands) > best_band_count:
            best_band_count = len(bands)
            best_page_idx = i

    # Render the best page at full resolution
    page = doc.load_page(best_page_idx)
    pix  = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    img  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img


# ======================================================
# Band Detection
# ======================================================
def detect_lead_bands(bw: np.ndarray, min_height: int = MIN_BAND_HEIGHT) -> list[tuple[int, int]]:
    """
    Identify horizontal bands likely containing ECG waveforms.

    Criteria (per band):
      - height  >= MIN_BAND_HEIGHT pixels
      - active columns (columns with >= 1 dark pixel) >= MIN_BAND_ACTIVE fraction

    Returns list of (row_start, row_end) tuples.
    """
    h, w = bw.shape
    row_sum = np.sum(bw > 0, axis=1)  # number of dark pixels per row

    # A row is "active" if it has > 2% dark pixels
    active_rows = np.where(row_sum > w * 0.02)[0]
    if len(active_rows) < 20:
        return []

    # Group consecutive active rows into candidate bands
    raw_bands = []
    start = active_rows[0]
    for i in range(1, len(active_rows)):
        if active_rows[i] - active_rows[i - 1] > GAP_THRESHOLD:
            raw_bands.append((start, active_rows[i - 1]))
            start = active_rows[i]
    raw_bands.append((start, active_rows[-1]))

    # Filter: must be tall enough and have sufficient horizontal coverage
    valid_bands = []
    for (lo, hi) in raw_bands:
        band_height = hi - lo
        if band_height < MIN_BAND_HEIGHT:
            continue
        band_slice = bw[lo:hi, :]
        # Fraction of columns that contain at least one dark pixel
        col_coverage = np.mean(np.any(band_slice > 0, axis=0))
        if col_coverage < MIN_BAND_ACTIVE:
            continue
        valid_bands.append((lo, hi))

    return valid_bands


# ======================================================
# Lead Quality Scoring
# ======================================================
def score_lead(signal: np.ndarray, fs: float = 100.0, is_pdf: bool = False) -> float:
    """
    Score a candidate lead (0.0 = useless noise, higher = better ECG quality).
    is_pdf=True skips the signal-density filter (PDF waveforms are denser by nature
    due to multi-lead overlap; they are already protected by the keyword gate).
    """
    if len(signal) < 100:
        return 0.0

    var = np.var(signal)
    if var < MIN_LEAD_VARIANCE:
        return 0.0

    try:
        freqs, power = welch(signal, fs=fs)
        tp = np.sum(power)
        bp = np.sum(power[(freqs >= 0.5) & (freqs <= 40)])
        ratio = bp / (tp + 1e-8)
        if ratio < MIN_SPECTRAL_RATIO:
            return 0.0
    except Exception:
        return 0.0

    # Count R-peaks
    threshold = np.mean(signal) + 0.5 * np.std(signal)
    peaks, _ = find_peaks(signal, height=threshold, distance=int(fs * 0.25))
    if len(peaks) < MIN_PEAKS:
        return 0.0

    # ── Morphology Check: Signal Density (images only) ───────────────
    # ECG waveforms have a quiet baseline: only 10-30% of samples exceed threshold.
    # Resume/photo text-column patterns are dense: 30-70% exceed threshold.
    # PDF waveforms are skipped here — they are denser by nature and already
    # protected by the strict medical-keyword gate in verify_medical_content().
    if not is_pdf:
        fraction_above = np.mean(signal > threshold)
        if fraction_above > 0.32:
            return 0.0  # Dense oscillation — not ECG spikes

    # Score = spectral quality × peak regularity bonus
    intervals = np.diff(peaks)
    cv = np.std(intervals) / (np.mean(intervals) + 1e-8) if len(intervals) > 1 else 9.9
    regularity_bonus = max(0.0, 1.0 - cv)
    return ratio + regularity_bonus


# ======================================================
# Core Waveform Extraction
# ======================================================
def extract_waveform_from_image(img: np.ndarray, target_length: int = 1000, is_pdf: bool = False) -> np.ndarray:
    """
    Extract a single, high-quality ECG waveform from an ECG image or PDF page.
    is_pdf=True skips the image-specific density filter in lead scoring.
      2. Adaptive threshold to binary
      3. Band detection (height + coverage filters)
      4. Column tracing with np.min (traces actual waveform, not noise average)
      5. scipy.resample (preserves peak morphology)
      6. Quality scoring → return single best lead
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold — dark waveform on light grid background
    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 5
    )

    # ── Step 1: Detect valid lead bands ──────────────────────────────
    bands = detect_lead_bands(bw)
    print(f"  [extraction] {len(bands)} valid bands detected (h>={MIN_BAND_HEIGHT}px, cov>={MIN_BAND_ACTIVE:.0%})")

    if not bands:
        raise ValueError("No valid ECG lead bands detected in image")

    # ── Step 2: Trace each band and score it ─────────────────────────
    candidate_leads = []

    for band_idx, (lo, hi) in enumerate(bands):
        band_img = bw[lo:hi, :]
        bh, bw_len = band_img.shape
        lead_signal = np.zeros(bw_len, dtype=np.float64)

        for x in range(bw_len):
            col = band_img[:, x]
            ys = np.where(col > 128)[0]
            if len(ys) > 0:
                # Use median of dark pixels to trace the waveform midpoint.
                # np.min picked the topmost dark row = text/axis labels at band top.
                # np.max picked the bottommost = grid lines at band bottom.
                # np.median robustly traces the waveform body, discarding outlier artifacts.
                lead_signal[x] = bh - np.median(ys)

        # Interpolate small gaps (up to ~5% of width)
        nonzero_idx = np.where(lead_signal > 0)[0]
        if len(nonzero_idx) < bw_len * 0.3:
            # Too sparse — likely text or a border, skip
            continue

        if len(nonzero_idx) > 10:
            lead_signal = np.interp(
                np.arange(bw_len),
                nonzero_idx,
                lead_signal[nonzero_idx]
            )

        # FIX: scipy.signal.resample — preserves R-peak amplitudes
        # cv2.INTER_AREA was averaging R-peaks away
        lead_resampled = resample(lead_signal, target_length)

        # Local z-score normalization
        mu, sd = np.mean(lead_resampled), np.std(lead_resampled)
        if sd < 1e-6:
            continue
        lead_resampled = (lead_resampled - mu) / sd

        score = score_lead(lead_resampled, is_pdf=is_pdf)
        if score > 0.0:  # Only accept leads that pass ALL quality gates
            candidate_leads.append((score, band_idx, lead_resampled))

    if not candidate_leads:
        raise ValueError("No leads passed quality filter (low variance / low spectral energy / too few peaks)")

    # ── Step 3: Return single BEST lead — no PCA ─────────────────────
    # PCA was mixing garbage leads with signal leads, destroying morphology.
    candidate_leads.sort(key=lambda x: x[0], reverse=True)
    best_score, best_idx, best_signal = candidate_leads[0]
    print(f"  [extraction] Best lead: band {best_idx}, quality score = {best_score:.3f} (of {len(candidate_leads)} candidates)")

    return best_signal


# ======================================================
# Basic Validation (length, NaN, flat signal)
# ======================================================
def validate_ecg_signal(signal: np.ndarray) -> tuple[bool, str]:
    if signal is None or len(signal) < 200:
        return False, "Signal too short"
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        return False, "NaN/Inf detected"
    if np.std(signal) < 0.05:
        return False, "Low signal variance (likely flat)"
    return True, "OK"


# ======================================================
# Main API
# ======================================================
def extract_signal_from_file(file_path: str, target_length: int = 1000) -> np.ndarray:
    """
    Public entry point. Returns a 1D float64 numpy array of length `target_length`.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    ext = os.path.splitext(file_path)[-1].lower()
    is_pdf = ext == ".pdf"
    if is_pdf:
        img = pdf_to_image(file_path)
    else:
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Could not read image: {file_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    signal = extract_waveform_from_image(img, target_length, is_pdf=is_pdf)

    ok, reason = validate_ecg_signal(signal)
    if not ok:
        raise ValueError(f"Invalid ECG signal: {reason}")

    return signal
