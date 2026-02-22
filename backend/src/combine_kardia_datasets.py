#!/usr/bin/env python3
import os
import json
import numpy as np
import scipy.io

from preprocessing import preprocess_ecg

# ======================================================
# Paths
# ======================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RAW_KARDIA_DIR = os.path.join(BASE_DIR, "data", "raw", "kardia")
OUT_DIR = os.path.join(BASE_DIR, "data", "processed", "kardia")

os.makedirs(OUT_DIR, exist_ok=True)

X_OUT = os.path.join(OUT_DIR, "X.npy")
Y_OUT = os.path.join(OUT_DIR, "y.npy")
LABELS_OUT = os.path.join(OUT_DIR, "labels.json")

# ======================================================
# Label mapping (FINAL 5 classes)
# ======================================================
LABEL_MAP = {
    "normal": "Normal Sinus Rhythm",
    "nsr": "Normal Sinus Rhythm",
    "atrial fibrillation": "Atrial Fibrillation",
    "af": "Atrial Fibrillation",
    "bradycardia": "Bradycardia",
    "tachycardia": "Tachycardia",
    "ventricular arrhythmia": "Ventricular Arrhythmia",
    "vt": "Ventricular Arrhythmia",
    "vf": "Ventricular Arrhythmia",
}

FINAL_CLASSES = [
    "Normal Sinus Rhythm",
    "Atrial Fibrillation",
    "Bradycardia",
    "Tachycardia",
    "Ventricular Arrhythmia",
]

CLASS_TO_IDX = {c: i for i, c in enumerate(FINAL_CLASSES)}

# ======================================================
# Extract ECG from .mat
# ======================================================
def extract_ecg_from_mat(mat_path):
    try:
        mat = scipy.io.loadmat(mat_path)
        data_struct = mat.get("Data")
        if data_struct is None:
            return None
        nested = data_struct[0, 0]
        ecg_struct = nested[4][0, 0]
        if "ECG_1" in ecg_struct.dtype.names:
            return ecg_struct["ECG_1"][0, 0].flatten()
    except Exception as e:
        print(f"⚠️ {mat_path}: {e}")
    return None

# ======================================================
# Scan files
# ======================================================
def scan_files(base):
    pairs = []
    for root, _, files in os.walk(base):
        for f in files:
            if f.endswith(".mat"):
                mat_path = os.path.join(root, f)
                txt_path = mat_path.replace(".mat", ".txt")
                pairs.append((mat_path, txt_path))
    return pairs

# ======================================================
# Main
# ======================================================
def main():
    pairs = scan_files(RAW_KARDIA_DIR)
    print(f"🔍 Found {len(pairs)} Kardia records")

    X, y = [], []

    for mat_path, txt_path in pairs:
        ecg = extract_ecg_from_mat(mat_path)
        if ecg is None or len(ecg) < 100:
            continue

        # Resize to 1000 samples
        if len(ecg) < 1000:
            ecg = np.pad(ecg, (0, 1000 - len(ecg)))
        else:
            ecg = ecg[:1000]

        # Preprocess
        processed = preprocess_ecg(ecg)
        signal = processed["filtered_signal"]

        # Read label
        label = None
        if os.path.exists(txt_path):
            with open(txt_path) as f:
                raw = f.read().strip().lower()
                label = LABEL_MAP.get(raw)

        if label not in CLASS_TO_IDX:
            continue

        X.append(signal)
        y.append(CLASS_TO_IDX[label])

    if not X:
        print("❌ No valid Kardia data found")
        return

    X = np.array(X)
    y = np.array(y)

    np.save(X_OUT, X)
    np.save(Y_OUT, y)
    json.dump(FINAL_CLASSES, open(LABELS_OUT, "w"), indent=2)

    print("✅ Kardia dataset prepared")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")

if __name__ == "__main__":
    main()
