# model.py
import os
import numpy as np
import joblib
import tensorflow as tf
from hybrid_model import HybridEnsemble

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
ARTIFACTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../artifacts"))

# --------------------------------------------------
# Load ML + DL models
# --------------------------------------------------
def load_saved_models():
    ml_models = {}
    dl_models = {}

    # ---------- ML models ----------
    for name in ["SVM", "RandomForest", "KNN", "XGBoost"]:
        path = os.path.join(ARTIFACTS_DIR, f"{name}.joblib")
        if os.path.exists(path):
            ml_models[name] = joblib.load(path)
            print(f"✅ Loaded ML model: {name}")

    # ---------- CNN 1D ----------
    cnn1d_path = os.path.join(ARTIFACTS_DIR, "cnn1d.keras")
    if os.path.exists(cnn1d_path):
        dl_models["CNN1D"] = tf.keras.models.load_model(cnn1d_path)
        print("✅ Loaded DL model: CNN1D")

    # ---------- CNN 2D ----------
    cnn2d_path = os.path.join(ARTIFACTS_DIR, "cnn2d.keras")
    if os.path.exists(cnn2d_path):
        dl_models["CNN2D"] = tf.keras.models.load_model(cnn2d_path)
        print("✅ Loaded DL model: CNN2D")

    return ml_models, dl_models


# --------------------------------------------------
# Predict single ECG signal
# --------------------------------------------------
def predict_signal(signal, classes):
    """
    Predict class for a single ECG signal using ML + DL hybrid
    """
    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim != 1 or signal.shape[0] != 1000:
        raise ValueError("❌ ECG signal must be 1D with 1000 samples")

    # Load models
    ml_models, dl_models = load_saved_models()

    if not ml_models and not dl_models:
        raise RuntimeError("❌ No trained models found in artifacts/")

    # Prepare inputs
    X_ml = signal.reshape(1, -1)          # (1, 1000)
    X_dl_1d = signal.reshape(1, 1000, 1)  # (1, 1000, 1)

    # CNN2D reshape (optional)
    X_dl_2d = signal.reshape(1, 40, 25, 1) if "CNN2D" in dl_models else None

    # Build hybrid ensemble dynamically
    hybrid = HybridEnsemble(
        ml_models=ml_models,
        dl_models=dl_models,
        classes=classes,
        weights={}  # equal weighting for now
    )

    # Use CNN1D input by default
    probs = hybrid.predict_proba(X_ml, X_dl_1d)

    pred_idx = int(np.argmax(probs))
    confidence = float(np.max(probs))

    return classes[pred_idx], confidence


# --------------------------------------------------
# Quick test
# --------------------------------------------------
if __name__ == "__main__":
    dummy_signal = np.random.randn(1000)

    classes = [
        "Normal Sinus Rhythm",
        "Atrial Fibrillation",
        "Bradycardia",
        "Tachycardia",
        "Ventricular Arrhythmias",
    ]

    label, conf = predict_signal(dummy_signal, classes)
    print(f"🫀 Prediction: {label} | Confidence: {conf:.3f}")
