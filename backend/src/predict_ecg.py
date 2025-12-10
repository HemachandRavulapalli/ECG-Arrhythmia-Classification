# backend/src/predict_ecg.py
import os
import sys
import numpy as np
import joblib
import json
from pathlib import Path
import pandas as pd  # For reading results_history.csv
import PyPDF2        # PDF text extraction
from scipy.signal import find_peaks  # Peak detection for ECG validation

try:
    from pdf_to_signal import extract_signal_from_file
except ImportError as e:
    print(f"⚠️ Warning: Cannot import pdf_to_signal: {e}")
    extract_signal_from_file = None

try:
    from hybrid_model import HybridEnsemble
    HYBRID_ENSEMBLE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Cannot import HybridEnsemble: {e}")
    HybridEnsemble = None
    HYBRID_ENSEMBLE_AVAILABLE = False

# ------------------------
# Label map (5 target classes)
# ------------------------
LABEL_MAP = {
    "Class_0": "Normal Sinus Rhythm",
    "Class_1": "Atrial Fibrillation",
    "Class_2": "Bradycardia",
    "Class_3": "Tachycardia",
    "Class_4": "Ventricular Arrhythmias",
}

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "saved_models"))
LOG_DIR = os.path.join(BASE_DIR, "..", "logs")
RESULTS_FILE = os.path.join(LOG_DIR, "results_history.csv")

def get_latest_run():
    """Get most recent training run."""
    if not os.path.exists(MODEL_DIR):
        return None
    runs = sorted(
        [os.path.join(MODEL_DIR, d) for d in os.listdir(MODEL_DIR) if d.startswith("run_")],
        key=os.path.getmtime,
    )
    return runs[-1] if runs else None

def get_best_run(by: str = "advanced_hybrid_acc"):
    """
    Select the best training run based on accuracy in results_history.csv.
    """
    if not os.path.exists(RESULTS_FILE):
        print("⚠️ results_history.csv not found, falling back to latest run")
        return get_latest_run()

    try:
        df = pd.read_csv(RESULTS_FILE)
    except Exception as e:
        print(f"⚠️ Failed to read results_history.csv: {e}, using latest run")
        return get_latest_run()

    if df.empty or by not in df.columns:
        print("⚠️ No valid accuracy column in results_history.csv, using latest run")
        return get_latest_run()

    best_row = df.loc[df[by].idxmax()]
    run_folder = best_row["run_folder"]
    best_run_path = os.path.join(MODEL_DIR, os.path.basename(run_folder))
    print(f"🏆 Selected best run by {by}: {os.path.basename(best_run_path)} (acc={best_row[by]:.4f})")
    return best_run_path

def load_models(run_dir):
    """Load ALL models - TensorFlow imported HERE to avoid early failure."""
    try:
        import tensorflow as tf
        TF_AVAILABLE = True
        print("✅ TensorFlow loaded in load_models()")
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        TF_AVAILABLE = False
        tf = None

    ml_models = {}
    dl_models = {}

    print(f"📂 Scanning models in: {run_dir}")

    # Priority 1: *_best.keras
    best_files = sorted(
        [f for f in os.listdir(run_dir) if f.endswith("_best.keras")],
        key=lambda x: os.path.getmtime(os.path.join(run_dir, x)),
        reverse=True,
    )

    # Priority 2: regular .keras
    regular_files = [
        f
        for f in os.listdir(run_dir)
        if f.endswith(".keras") and "_best" not in f
    ]

    # Priority 3: ML models (.joblib)
    joblib_files = [f for f in os.listdir(run_dir) if f.endswith(".joblib")]

    all_files = best_files + regular_files + joblib_files

    for f in all_files:
        file_path = os.path.join(run_dir, f)

        if f.endswith(".joblib"):
            name = f.replace(".joblib", "")
            try:
                ml_models[name] = joblib.load(file_path)
                print(f"✅ Loaded ML model: {name}")
            except Exception as e:
                print(f"⚠️ Skipping ML {f}: {e}")
                continue

        elif f.endswith(".keras") or f.endswith(".h5"):
            if not TF_AVAILABLE or tf is None:
                print(f"⚠️ Skipping DL {f}: TensorFlow not available")
                continue

            name = Path(f).stem.replace("_best", "").upper()

            try:
                model = tf.keras.models.load_model(
                    file_path,
                    safe_mode=False,
                    custom_objects={
                        "Add": tf.keras.layers.Add,
                        "Multiply": tf.keras.layers.Multiply,
                        "Concatenate": tf.keras.layers.Concatenate,
                        "Reshape": tf.keras.layers.Reshape,
                    },
                )
                dl_models[name] = model
                print(f"✅ Loaded DL model: {name} ({f})")
            except Exception as e:
                print(f"⚠️ Detailed DL error {f}: {str(e)[:100]}...")
                try:
                    model = tf.keras.models.load_model(file_path, safe_mode=False)
                    dl_models[name] = model
                    print(f"✅ Loaded DL (fallback): {name}")
                except Exception as e2:
                    print(f"❌ Final DL fail {f}: {str(e2)[:80]}")
                    continue

    print(f"📦 Loaded ML models: {list(ml_models.keys())}")
    print(f"📦 Loaded DL models: {list(dl_models.keys())}")
    return ml_models, dl_models

def has_ecg_keywords(pdf_path):
    """Check if PDF/image contains ECG-specific keywords or lead labels using text extraction + OCR."""
    
    # ECG keywords and lead labels
    ecg_keywords = [
        "kardia", "ekg recording", "ecg recording",
        "atrial fibrillation", "normal sinus rhythm", "bradycardia",
        "tachycardia", "ventricular arrhythmias",
        "lead i", "lead ii", "lead iii",
        "avr", "avl", "avf"
    ]
    
    # For PDFs: try text extraction first, then OCR
    if pdf_path.lower().endswith('.pdf'):
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages[:3]:
                    page_text = page.extract_text() or ""
                    text += page_text.lower()
                
                keyword_hits = sum(1 for kw in ecg_keywords if kw in text)
                if keyword_hits >= 1:
                    print(f"✅ ECG keywords found in PDF text: {keyword_hits}")
                    return True
                else:
                    print("📋 No text found, trying PDF → OCR...")
        except Exception as e:
            print(f"⚠️ PDF text extraction failed: {e}")
        
        # Fallback: OCR on PDF pages
        try:
            import pytesseract
            from pdf2image import convert_from_path
            from PIL import Image
            
            print("🔍 Converting PDF to images for OCR...")
            images = convert_from_path(pdf_path, first_page=1, last_page=2)
            
            for img in images:
                text = pytesseract.image_to_string(img).lower()
                keyword_hits = sum(1 for kw in ecg_keywords if kw in text)
                if keyword_hits >= 1:
                    print(f"✅ ECG keywords found via OCR: {keyword_hits}")
                    return True
            
            print("⚠️ No ECG keywords found via OCR")
            return False
        except ImportError:
            print("ℹ️ pytesseract or pdf2image not installed - cannot OCR PDF")
            print("   Install: pip install pytesseract pdf2image")
            return False
        except Exception as e:
            print(f"⚠️ PDF OCR failed: {e}")
            return False
    
    # For images: try OCR
    if pdf_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(pdf_path)
            text = pytesseract.image_to_string(img).lower()
            keyword_hits = sum(1 for kw in ecg_keywords if kw in text)
            if keyword_hits >= 1:
                print(f"✅ ECG keywords found in image: {keyword_hits}")
                return True
            else:
                print("⚠️ No ECG keywords in image - relying on signal validation")
                return None  # None = proceed without keyword validation
        except ImportError:
            print("ℹ️ pytesseract not installed - skipping image OCR")
            print("   Install: pip install pytesseract")
            return None
        except Exception as e:
            print(f"⚠️ Image OCR failed: {e}")
            return None
    
    return False

def predict_ecg(pdf_path):
    if extract_signal_from_file is None:
        raise ValueError("❌ PDF signal extraction not available.")
    if not HYBRID_ENSEMBLE_AVAILABLE or HybridEnsemble is None:
        raise ValueError("❌ Hybrid ensemble not available.")

    print(f"📄 Processing: {os.path.basename(pdf_path)}")
    
    # 🚨 STRICT: Check for ECG keywords first
    ecg_check = has_ecg_keywords(pdf_path)
    
    # For PDFs: MUST have ECG keywords
    if pdf_path.lower().endswith('.pdf') and ecg_check is False:
        raise ValueError(
            "❌ No ECG keywords found in PDF. "
            "Must contain: Kardia, EKG Recording, Normal Sinus Rhythm, "
            "Bradycardia, Tachycardia, Ventricular Arrhythmias, "
            "or Lead labels (I, II, III, aVR, aVL, aVF)"
        )
    
    # Extract signal
    print("📄 Converting PDF → ECG signal...")
    signal = extract_signal_from_file(pdf_path)
    if signal is None:
        raise ValueError("❌ Could not extract signal from PDF")

    print(f"📈 Extracted waveform of length {len(signal)}")

    # Signal validation based on peaks and BPM
    peaks, _ = find_peaks(np.abs(signal), height=np.std(signal)*0.3, distance=40)
    peak_rate = len(peaks) * 60 / 10  # BPM equivalent

    # For images with confirmed ECG keywords, be more lenient
        # For images with confirmed ECG keywords, be more lenient
    if ecg_check is True and pdf_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        min_peaks = 2  # Very lenient for keyword-confirmed images
        min_bpm = 5    # Very lenient BPM for extracted image signals
        max_bpm = 500  # Allow any reasonable extraction
        print(f"🔓 Using lenient validation (keyword-confirmed ECG image)")
    else:
        min_peaks = 8  # Normal threshold for PDFs and non-keyword images
        min_bpm = 30
        max_bpm = 200

    if len(peaks) < min_peaks or peak_rate < min_bpm or peak_rate > max_bpm:
        raise ValueError(
            f"❌ Invalid ECG signal: {len(peaks)} peaks, {peak_rate:.1f} BPM "
            f"(need {min_peaks}+ peaks, {min_bpm}-{max_bpm} BPM)"
        )

    print(f"✅ Signal validated: {len(peaks)} peaks, {peak_rate:.1f} BPM")

    # Standardize to 1000 samples
    orig_len = len(signal)
    target_len = 1000

    if orig_len != target_len:
        if orig_len < target_len:
            signal = np.pad(signal, (0, target_len - orig_len), mode="constant")
        else:
            signal = signal[:target_len]
        print(f"📏 Signal: {orig_len} → {target_len} samples")

    # Z-score normalization
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    # Prepare inputs
    X_ml = signal.reshape(1, -1)
    X_dl_1d = signal.reshape(1, 1000, 1)       # for 1D CNNs
    X_dl_2d = signal.reshape(1, 100, 10, 1)    # for 2D CNNs

    # For backward compatibility with current HybridEnsemble
    X_dl = X_dl_1d

    # Load models from BEST run
    best_run = get_best_run(by="advanced_hybrid_acc")
    if best_run is None:
        raise ValueError("❌ No trained models found. Run training first.")

    print(f"📂 Loading models from: {Path(best_run).name}")
    ml_models, dl_models = load_models(best_run)

    if not ml_models and not dl_models:
        raise ValueError("❌ No models loaded. Check saved_models/")

    # Use saved classes or default
    classes = [
        "Normal Sinus Rhythm",
        "Atrial Fibrillation",
        "Bradycardia",
        "Tachycardia",
        "Ventricular Arrhythmias",
    ]
    class_file = os.path.join(best_run, "classes.json")
    if os.path.exists(class_file):
        try:
            with open(class_file, "r") as f:
                saved_classes = json.load(f)
            if len(saved_classes) == 5:
                classes = saved_classes
                print(f"📋 Using saved classes: {classes[:3]}...")
        except Exception as e:
            print(f"⚠️ Could not load classes: {e}")

    # Create ensemble
    hybrid = HybridEnsemble(
        ml_models=ml_models,
        dl_models=dl_models,
        classes=classes,
        weights={},
    )

    print("🧠 Predicting...")
    try:
        probs = hybrid.predict_proba(X_ml, X_dl)
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        raise

    # Ensure 5 classes
    if probs.shape[1] > 5:
        probs = probs[:, :5]
    elif probs.shape[1] < 5:
        pad_probs = np.zeros((probs.shape[0], 5))
        pad_probs[:, :probs.shape[1]] = probs
        probs = pad_probs

    # Normalize
    probs = probs / (np.sum(probs, axis=1, keepdims=True) + 1e-8)

    pred_idx = int(np.argmax(probs))
    pred_class = classes[pred_idx] if pred_idx < len(classes) else classes[0]
    pred_conf = float(np.max(probs))

    # Confidence check: warn on medium confidence
    if pred_conf < 0.8:
        print(f"⚠️ Medium confidence: {pred_conf:.3f}")

    results = {
        "predicted_class": pred_class,
        "confidence": round(pred_conf, 4),
        "probabilities": {
            classes[i]: round(float(p), 4) for i, p in enumerate(probs[0])
        },
    }

    print(json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 predict_ecg.py <path_to_pdf>")
        sys.exit(1)
    pdf_path = sys.argv[1]
    try:
        predict_ecg(pdf_path)
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)
