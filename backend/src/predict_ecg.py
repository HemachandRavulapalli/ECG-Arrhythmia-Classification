#!/usr/bin/env python3
"""
predict_ecg.py — Research-Grade ECG Inference Pipeline

Inference pipeline (identical to training preprocessing):
  1. Validate file type and ECG keywords
  2. Extract raw image/pixels from PDF or image file
  3. Extract up to 6 lead waveforms
  4. Resample each lead to 100 Hz → stack → (N, n_leads)
  5. PCA aggregate (multi-lead → 1D)
  6. Extract exactly 1000 samples (10 sec)
  7. Apply canonical preprocess_pipeline (fs=100)
  8. Extract 16-dim ML features
  9. Compute real spectrogram (fs=100, nperseg=128, noverlap=64)
  10. Run HybridEnsemble  →  return probabilities
"""

import os
import sys
import json
import numpy as np
import joblib
import mimetypes
from pathlib   import Path
from scipy.signal import find_peaks, welch

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pdf_to_signal      import extract_signal_from_file
from preprocessing      import (resample_signal, aggregate_leads,
                                 preprocess_pipeline, extract_window)
from feature_extraction import extract_ecg_features
from cnn_models         import signal_to_spectrogram
from hybrid_model       import HybridEnsemble


# ======================================================
# Paths & constants
# ======================================================
MODEL_DIR    = os.environ.get("MODEL_DIR", os.path.join(SRC_DIR, "saved_models"))
LOG_DIR      = os.path.join(SRC_DIR, "..", "logs")
RESULTS_FILE = os.path.join(LOG_DIR, "results_history.csv")

TARGET_FS    = 100
WINDOW_SIZE  = 1000

TARGET_CLASSES = [
    "Normal Sinus Rhythm",
    "Atrial Fibrillation",
    "Bradycardia",
    "Tachycardia",
    "Ventricular Arrhythmias",
]


# ======================================================
# Run selection helpers
# ======================================================
def get_latest_run() -> str | None:
    if not os.path.isdir(MODEL_DIR):
        return None
    runs = sorted(
        [os.path.join(MODEL_DIR, d)
         for d in os.listdir(MODEL_DIR) if d.startswith("run_")],
        key=os.path.getmtime,
    )
    return runs[-1] if runs else None


def get_best_run(by: str = "hybrid_macro_f1") -> str | None:
    """Return run_dir of the historically best model by macro F1."""
    if not os.path.exists(RESULTS_FILE):
        return get_latest_run()
    try:
        import pandas as pd
        df = pd.read_csv(RESULTS_FILE)
        # Fallback column names for backward compat
        col = by if by in df.columns else (
              "hybrid_acc" if "hybrid_acc" in df.columns else df.columns[-2])
        if df.empty:
            return get_latest_run()
        best = df.loc[df[col].idxmax(), "run_folder"]
        d    = os.path.join(MODEL_DIR, os.path.basename(best))
        return d if os.path.isdir(d) else get_latest_run()
    except Exception:
        return get_latest_run()


# ======================================================
# Model loading
# ======================================================
def load_models(run_dir: str) -> tuple[dict, dict]:
    import glob
    import tensorflow as tf

    ml_models, dl_models = {}, {}

    for path in glob.glob(os.path.join(run_dir, "*.joblib")):
        name = Path(path).stem
        try:
            ml_models[name] = joblib.load(path)
        except Exception as e:
            print(f"  ⚠️  Could not load ML model {name}: {e}")

    for path in glob.glob(os.path.join(run_dir, "*.keras")):
        name = Path(path).stem
        try:
            dl_models[name] = tf.keras.models.load_model(path, safe_mode=False)
        except Exception as e:
            print(f"  ⚠️  Could not load DL model {name}: {e}")

    return ml_models, dl_models


def load_scores(run_dir: str) -> dict:
    """Load per-model macro F1 scores saved during training."""
    path = os.path.join(run_dir, "scores.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    # Merge all sub-score dicts
    weights = {}
    for key in ("ml_scores", "dl_scores"):
        weights.update(data.get(key, {}))
    
    # Also include the hybrid scores if we want to use them for something,
    # but the ensemble usually needs per-model scores for weighting.
    return weights


# ======================================================
# ECG keyword validation  (PDF / image)
# ======================================================
def verify_medical_content(file_path: str) -> tuple[bool, float]:
    """
    Fast heuristic: check if the file likely contains ECG content.
    """
    ext = Path(file_path).suffix.lower()
    # Broaden but keep specific. 'lead' alone is too common.
    # Hardened Keywords: Resumes have 'ecg' and 'arrhythmia', but lack these medical headers.
    MUST_HAVE = {"kardia", "normal sinus rhythm", "sinus rhythm", "unclassified", 
                 "atrial fibrillation", "tachycardia", "bradycardia", "ekg recording",
                 "instant ekg", "personal ekg"}
    SECONDARY = {"bpm", "heart rate", "recorded:", "lead i", "lead ii", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"}

    try:
        text = ""
        if ext == ".pdf":
            import fitz
            doc  = fitz.open(file_path)
            text = " ".join(p.get_text()[:2000] for p in doc).lower() # limit read
            
            # OCR Fallback for Image-PDFs (many Kardia reports are scans)
            if not text.strip() and len(doc) > 0:
                try:
                    import cv2, pytesseract
                    # Render first page at 2x zoom for OCR
                    pix = doc[0].get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                    if pix.n == 4: # RGBA
                        img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
                    gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
                    text = pytesseract.image_to_string(gray, config='--psm 11').lower()
                except Exception:
                    pass
        else:
            try:
                import cv2, pytesseract
                img  = cv2.imread(file_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray, config='--psm 11').lower()
            except Exception:
                return False # Strict rejection

        secondary_count = sum(1 for kw in SECONDARY if kw in text)
        has_must = any(kw in text for kw in MUST_HAVE)

        # 1. High Trust: Clear medical headers
        if has_must:
            return True, 0.20
        
        # 2. Medium Trust: Strong secondary indicators or sparse/no text (pure waveform images)
        if secondary_count >= 2:
            return True, 0.35
            
        if len(text.strip()) < 100:
            return True, 0.35
            
        # 3. Low Trust: Text-heavy document lacking medical context
        if ext == ".pdf":
            return False, 1.0 # PDFs without headers are rejected
        else:
            # Text-heavy images (e.g. Resume photos) must have very high model confidence
            return True, 0.55

    except Exception:
        return False, 1.0


# ======================================================
# Signal validation
# ======================================================
def validate_ecg_signal(signal: np.ndarray,
                         fs: float = TARGET_FS) -> tuple[bool, str]:
    """
    Physiological sanity checks on a normalised 1D ECG window.
    """
    if signal is None or len(signal) < WINDOW_SIZE // 2:
        return False, "Signal too short"
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        return False, "NaN/Inf in signal"
    
    # Check for flatlines or extreme noise
    variance = np.var(signal)
    if variance < 1e-4:
        return False, "Signal is practically flat"
    if variance > 5.0:
        return False, "Signal variance excessively high (likely pure noise)"

    # R-peak based physiological check
    # distance=int(fs * 0.25) ensures we don't pick up P or T waves as R peaks.
    # Refined Morphology: R-peaks are rarely < 25ms (2.5 samples at 100Hz).
    # Single-pixel scanning artifacts are usually < 15ms.
    peaks, properties = find_peaks(signal, height=0.4, 
                                   distance=int(fs * 0.25),
                                   width=(fs * 0.025, fs * 0.15)) 
    
    duration  = len(signal) / fs
    bpm       = len(peaks) * 60.0 / duration if duration > 0 else 0

    if len(peaks) < 4:
        return False, f"Too few rhythmic peaks: {len(peaks)} (need >= 4)"
    if bpm < 35 or bpm > 220:
        return False, f"Implausible HR: {bpm:.1f} BPM"
    
    # NOTE: spectral energy is already validated at the extraction level (score_lead
    # enforces spectral_ratio >= 0.60 on the raw extracted signal). Applying an
    # additional Welch PSD check here on the preprocessed signal causes false
    # rejections because preprocessing changes the spectral profile.

    # Regularity Check: Real heartbeats have somewhat consistent intervals.
    # Natural heartbeats never have 0.0 standard deviation in intervals.
    if len(peaks) >= 4:
        intervals = np.diff(peaks)
        cv = np.std(intervals) / (np.mean(intervals) + 1e-8)
        
        # Balanced Regularity: Scanned PDFs often have high CV (1.0 - 2.0) due to extraction jitter.
        # We increase the upper limit significantly to restore sensitivity.
        if cv > 2.5:
            return False, f"Signal extremely chaotic (CV={cv:.2f}) - likely non-physiological noise"
            
        # CV < 0.005 suggests perfectly spaced digital lines (e.g. table borders)
        if cv < 0.005:
            return False, "Signal is too perfectly regular (likely digital grid/table lines)"
            

    return True, f"Valid ECG ({len(peaks)} peaks, {bpm:.1f} BPM)"


# ======================================================
# Main prediction function
# ======================================================
def predict_ecg(file_path: str) -> dict:
    """
    Full inference pipeline.

    Parameters
    ----------
    file_path : path to an ECG PDF or image

    Returns
    -------
    dict with keys:
      predicted_class, confidence, probabilities
    """
    print(f"📄 predict_ecg: {os.path.basename(file_path)}")

    # ── File type check ────────────────────────────────
    mime, _ = mimetypes.guess_type(file_path)
    allowed = {"image/png", "image/jpeg", "image/bmp",
               "image/tiff", "application/pdf"}
    if mime is None or mime not in allowed:
        raise ValueError(f"Unsupported file type: {mime}")

    # ── 1. Text/Keyword Verification (Adaptive Thresholding) ──
    is_supported, adaptive_threshold = verify_medical_content(file_path)
    if not is_supported:
        raise ValueError(
            "File does not appear to contain ECG content. "
            "Only ECG PDFs / images are supported."
        )

    # ── Extract raw signal from file ───────────────────
    print("📊 Extracting waveform...")
    try:
        signal_raw = extract_signal_from_file(file_path, target_length=WINDOW_SIZE)
    except Exception as e:
        raise ValueError(f"Waveform extraction failed: {e}")

    # ── Ensure 1D and correct length ───────────────────
    signal_raw = np.asarray(signal_raw, dtype=np.float64).flatten()

    # BUG 6 FIX: pdf_to_signal should already return WINDOW_SIZE samples.
    # If it doesn't (e.g. very short source), use extract_window to pad/crop.
    # Do NOT resample with original_fs==target_fs — that's a no-op and does
    # NOT enforce length. Only extract_window guarantees (1000,) exactly.
    if len(signal_raw) != WINDOW_SIZE:
        signal_raw = extract_window(signal_raw, WINDOW_SIZE)

    # ── Canonical preprocessing ────────────────────────
    signal = preprocess_pipeline(signal_raw, fs=TARGET_FS)

    # ── Validate ECG signal ────────────────────────────
    ok, reason = validate_ecg_signal(signal, fs=TARGET_FS)
    if not ok:
        raise ValueError(f"Invalid ECG signal: {reason}")
    print(f"  ✅ Signal OK: {reason}")

    # ── Prepare ML feature vector ──────────────────────
    X_ml   = extract_ecg_features(signal, fs=TARGET_FS).reshape(1, -1)

    # ── Prepare CNN1D input ────────────────────────────
    X_1d   = signal.reshape(1, WINDOW_SIZE, 1)

    # ── Prepare spectrogram (CNN2D input) ─────────────
    spec   = signal_to_spectrogram(signal, fs=TARGET_FS)  # (freq, time, 1)
    X_spec = spec[np.newaxis, ...]                         # (1, freq, time, 1)

    # ── Load models ────────────────────────────────────
    print("🤖 Loading models...")
    run_dir = get_best_run()
    if run_dir is None:
        raise ValueError("No trained models found. Run train_pipeline.py first.")
    print(f"  📁 Using: {os.path.basename(run_dir)}")

    ml_models, dl_models = load_models(run_dir)
    weights              = load_scores(run_dir)

    if not ml_models and not dl_models:
        raise ValueError("No models loaded from run directory.")

    print(f"  ✅ Loaded {len(ml_models)} ML + {len(dl_models)} DL models")

    # ── Load class list ────────────────────────────────
    classes_path = os.path.join(run_dir, "classes.json")
    if os.path.exists(classes_path):
        with open(classes_path) as f:
            classes = json.load(f)
    else:
        classes = TARGET_CLASSES

    # ── Check which DL models are CNN2D ────────────────
    # CNN2D models need X_spec; CNN1D models need X_1d
    cnn1d_models = {n: m for n, m in dl_models.items()
                    if "cnn2d" not in n.lower() and "spec" not in n.lower()}
    cnn2d_models = {n: m for n, m in dl_models.items()
                    if "cnn2d" in n.lower() or "spec" in n.lower()}

    # ── HybridEnsemble prediction ──────────────────────
    print("🧠 Running Hybrid Ensemble...")
    ensemble = HybridEnsemble(
        ml_models = ml_models,
        dl_models = {**cnn1d_models, **cnn2d_models},
        classes   = classes,
        weights   = weights,
    )

    probs  = ensemble.predict_proba(X_ml, X_1d, X_spec=X_spec)
    probs  = probs / (probs.sum(axis=1, keepdims=True) + 1e-8)  # re-normalise

    idx    = int(np.argmax(probs))
    label  = classes[idx] if idx < len(classes) else "Unknown"
    conf   = float(np.max(probs))

    # Use adaptive threshold (0.20 to 0.55) based on document verification
    if conf < adaptive_threshold:
        raise ValueError(f"Low confidence prediction ({conf:.2f} < {adaptive_threshold:.2f}). The signal is either too noisy or does not appear to be a standard ECG.")

    print(f"🎯 Predicted: {label}  (confidence: {conf:.4f})")

    return {
        "predicted_class": label,
        "confidence":      round(conf, 4),
        "probabilities":   {
            classes[i]: round(float(p), 4)
            for i, p in enumerate(probs[0])
            if i < len(classes)
        },
    }


# ======================================================
# CLI
# ======================================================
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser("ECG Prediction")
    p.add_argument("file", help="Path to ECG PDF or image")
    a = p.parse_args()

    try:
        result = predict_ecg(a.file)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)
