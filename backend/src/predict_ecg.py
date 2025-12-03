# backend/src/predict_ecg.py
import os
import sys
import numpy as np
import joblib
import json

# Try to import tensorflow, but handle if missing
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ Warning: TensorFlow not available. Deep learning models will not work.")

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
    "Class_4": "Ventricular Arrhythmias"
}

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

def get_latest_run():
    runs = sorted(
        [os.path.join(MODEL_DIR, d) for d in os.listdir(MODEL_DIR) if d.startswith("run_")],
        key=os.path.getmtime,
    )
    return runs[-1] if runs else None


def load_models(run_dir):
    ml_models = {}
    dl_models = {}

    for f in os.listdir(run_dir):
        if f.endswith(".joblib"):
            name = f.replace(".joblib", "")
            try:
                ml_models[name] = joblib.load(os.path.join(run_dir, f))
                print(f"✅ Loaded ML model: {name}")
            except Exception as e:
                print(f"⚠️ Skipping {f}: Could not load model - {e}")
                continue
        elif f.endswith(".keras") or f.endswith(".h5"):
            if not TENSORFLOW_AVAILABLE:
                print(f"⚠️ Skipping {f}: TensorFlow not available")
                continue
            name = f.replace(".keras", "").replace(".h5", "")
            try:
                dl_models[name.upper()] = tf.keras.models.load_model(os.path.join(run_dir, f))
                print(f"✅ Loaded DL model: {name}")
            except Exception as e:
                print(f"⚠️ Skipping {f}: Could not load model - {e}")
                continue

    return ml_models, dl_models


def predict_ecg(pdf_path):
    if extract_signal_from_file is None:
        raise ValueError("❌ PDF signal extraction not available. Missing dependencies.")
    
    if not HYBRID_ENSEMBLE_AVAILABLE or HybridEnsemble is None:
        raise ValueError("❌ Hybrid ensemble model not available. Missing dependencies.")
    
    print("📄 Converting PDF → ECG signal...")
    signal = extract_signal_from_file(pdf_path)
    if signal is None:
        raise ValueError("❌ Could not extract signal from PDF")

    # preprocess for models
    X_ml = signal.reshape(1, -1)
    X_dl = signal.reshape(1, -1, 1)

    # normalization (z-score)
    X_ml = (X_ml - np.mean(X_ml)) / (np.std(X_ml) + 1e-8)
    X_dl = (X_dl - np.mean(X_dl)) / (np.std(X_dl) + 1e-8)

    # load models
    best_run = get_latest_run()
    if best_run is None:
        raise ValueError("❌ No trained models found. Please train models first.")
    print(f"📂 Loading models from: {best_run}")

    # Sanity check: ensure all models belong to new 4-class setup
    ml_models, dl_models = load_models(best_run)
    print(f"📦 Loaded ML models: {list(ml_models.keys())}")
    print(f"📦 Loaded DL models: {list(dl_models.keys())}")
    
    # Check if we have at least some models
    if not ml_models and not dl_models:
        raise ValueError("❌ No models loaded. Please check that model files exist in the saved_models directory.")
    
    if not ml_models:
        print("⚠️ Warning: No ML models loaded. Predictions may be less accurate.")
    if not dl_models:
        print("⚠️ Warning: No DL models loaded (TensorFlow not available). Using ML models only.")

    # force 5-class labels for consistent predictions
    classes = ["Normal Sinus Rhythm", "Atrial Fibrillation", "Bradycardia", "Tachycardia", "Ventricular Arrhythmias"]
    num_classes = len(classes)


    # load class list (fallback to 5)
    class_file = os.path.join(best_run, "classes.json")
    if os.path.exists(class_file):
        with open(class_file, "r") as f:
            classes = json.load(f)
    else:
        classes = ["Class_0", "Class_1", "Class_2", "Class_3", "Class_4"]

    # hybrid ensemble
    hybrid = HybridEnsemble(ml_models=ml_models, dl_models=dl_models, classes=classes)

    print("🧠 Predicting...")
    probs = hybrid.predict_proba(X_ml, X_dl)
    probs = probs[:, :len(classes)]  # truncate if model output > number of classes
    pred_idx = int(np.argmax(probs))
    pred_class = classes[pred_idx]
    pred_conf = float(np.max(probs))

    # use human-readable labels
    readable_pred = LABEL_MAP.get(pred_class, pred_class)
    readable_probs = {
        LABEL_MAP.get(cls, cls): round(float(p), 4)
        for cls, p in zip(classes, probs[0].tolist())
    }

    results = {
        "predicted_class": readable_pred,
        "confidence": round(pred_conf, 4),
        "probabilities": readable_probs
    }

    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 predict_ecg.py <path_to_pdf>")
        sys.exit(1)
    pdf_path = sys.argv[1]
    predict_ecg(pdf_path)
