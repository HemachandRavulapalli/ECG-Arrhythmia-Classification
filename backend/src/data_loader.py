# backend/src/data_loader.py
"""
Research-grade data loader with patient-wise GroupShuffleSplit and SMOTE.

Correct order (MANDATORY):
  1. Load all NPZ files (both MIT-BIH and PTB-XL)
  2. Patient-wise GroupShuffleSplit  → no inter-patient leakage
  3. Extract features for ML branch
  4. Pre-compute spectrograms for CNN2D branch
  5. SMOTE on training SET only  (never on test)
  6. Return all three inputs  (X_ml, X_1d, X_spec) + labels
"""

import os
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

from feature_extraction import extract_ecg_features
from cnn_models         import batch_to_spectrograms
from joblib             import Parallel, delayed


# ======================================================
# Paths
# ======================================================
BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MIT_DIR       = os.path.join(PROCESSED_DIR, "mitdb")
PTBXL_DIR     = os.path.join(PROCESSED_DIR, "ptbxl")

# ======================================================
# Target class definitions
# ======================================================
TARGET_CLASSES = [
    "Normal Sinus Rhythm",
    "Atrial Fibrillation",
    "Bradycardia",
    "Tachycardia",
    "Ventricular Arrhythmias",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(TARGET_CLASSES)}


# ======================================================
# NPZ loader
# ======================================================
def _load_npz_folder(folder: str, limit: int | None = None):
    """
    Load all NPZ files from a folder.
    Each NPZ must have: signal (1000,), label (str), patient_id (str).

    Returns
    -------
    X          : (N, 1000)  float32
    y          : (N,)       int
    patient_ids: (N,)       str
    """
    if not os.path.isdir(folder):
        print(f"  ⚠️  Folder not found: {folder}")
        return np.array([]), np.array([]), np.array([])

    files = [f for f in os.listdir(folder) if f.endswith(".npz")]
    if not files:
        print(f"  ⚠️  No NPZ files in: {folder}")
        return np.array([]), np.array([]), np.array([])

    X, y, pids = [], [], []

    for fname in files:
        data   = np.load(os.path.join(folder, fname), allow_pickle=True)
        signal = data["signal"].astype(np.float32).flatten()

        if len(signal) != 1000:
            continue

        label = str(data["label"].item()) if data["label"].ndim > 0 else str(data["label"])

        if label not in CLASS_TO_IDX:
            continue

        pid = str(data["patient_id"].item()) if "patient_id" in data else fname

        X.append(signal)
        y.append(CLASS_TO_IDX[label])
        pids.append(pid)

        if limit and len(X) >= limit:
            break

    if not X:
        return np.array([]), np.array([]), np.array([])

    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.int32),
            np.array(pids))


# ======================================================
# Data augmentation (applied ONLY to training signals)
# ======================================================
def _augment_signals(X: np.ndarray, y: np.ndarray,
                     pids: np.ndarray) -> tuple:
    """
    Lightweight augmentation: Gaussian noise + amplitude scaling.
    Produces 2× the training samples while preserving labels & patient IDs.
    """
    noise   = np.random.normal(0, 0.02, X.shape).astype(np.float32)
    scale   = np.random.uniform(0.9, 1.1, (len(X), 1)).astype(np.float32)

    X_noisy  = X + noise
    X_scaled = X * scale

    X_aug    = np.concatenate([X, X_noisy, X_scaled], axis=0)
    y_aug    = np.tile(y, 3)
    pids_aug = np.tile(pids, 3)

    return X_aug, y_aug, pids_aug


# ======================================================
# Main loader
# ======================================================
def load_all_datasets(
    limit:           int | None = None,
    one_hot:         bool       = True,
    apply_smote:     bool       = True,
    apply_augment:   bool       = True,
    compute_spec:    bool       = True,
    test_size:       float      = 0.2,
    random_state:    int        = 42,
    window_size:     int        = 1000,   # kept for API compat
    include_kardia:  bool       = False,  # ← multi-domain flag
):
    """
    Load, split (patient-wise), augment (train only),
    SMOTE (train only), and return all inputs.

    Returns
    -------
    train       : dict with keys 'X_1d', 'X_ml', 'X_spec', 'y', 'y_oh'
    test        : dict with keys 'X_1d', 'X_ml', 'X_spec', 'y', 'y_oh'  (digital only)
    classes     : list of class names
    kardia_test : dict or None — Kardia 20% held-out test (only when include_kardia=True)
    """

    # ── 1. Load MIT-BIH ────────────────────────────────
    print("📥 Loading MIT-BIH...")
    X_mit, y_mit, p_mit = _load_npz_folder(MIT_DIR, limit=limit)

    # ── 2. Load PTB-XL ─────────────────────────────────
    print("📥 Loading PTB-XL...")
    X_ptb, y_ptb, p_ptb = _load_npz_folder(PTBXL_DIR, limit=limit)

    # ── 3. Combine ─────────────────────────────────────
    parts_X   = [a for a in [X_mit, X_ptb] if len(a) > 0]
    parts_y   = [a for a in [y_mit, y_ptb] if len(a) > 0]
    parts_p   = [a for a in [p_mit, p_ptb] if len(a) > 0]

    if not parts_X:
        raise RuntimeError("❌ No data found. Run preprocess_data.py first.")

    X   = np.concatenate(parts_X, axis=0)    # (N, 1000)
    y   = np.concatenate(parts_y, axis=0)    # (N,)
    pid = np.concatenate(parts_p, axis=0)    # (N,)

    print(f"📊 Total samples: {len(X)}")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"   - {TARGET_CLASSES[u]}: {c}")

    # ── 4. Patient-wise GroupShuffleSplit ──────────────
    print(f"✂️  Patient-wise GroupShuffleSplit (test_size={test_size})...")
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size,
                            random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=pid))

    X_tr, y_tr, p_tr = X[train_idx], y[train_idx], pid[train_idx]
    X_te, y_te       = X[test_idx],  y[test_idx]

    print(f"  Train: {len(X_tr)}  |  Test: {len(X_te)}")

    # ── 5. Augment training only ────────────────────────
    if apply_augment:
        print("🔄 Augmenting training signals...")
        X_tr, y_tr, p_tr = _augment_signals(X_tr, y_tr, p_tr)
        print(f"  After augmentation: {len(X_tr)}")

    # ── 6. Feature extraction (ML branch) ──────────────
    print(f"🔬 Extracting ML features for {len(X_tr)} training and {len(X_te)} testing samples...")
    # Parallel extraction to avoid stalls
    def _extract_batch(signals):
        results = Parallel(n_jobs=-1)(delayed(extract_ecg_features)(s, fs=100) for s in signals)
        return np.array(results, dtype=np.float32)

    X_tr_ml = _extract_batch(X_tr)
    X_te_ml = _extract_batch(X_te)
    X_tr_ml = np.nan_to_num(X_tr_ml, nan=0.0, posinf=0.0, neginf=0.0)
    X_te_ml = np.nan_to_num(X_te_ml, nan=0.0, posinf=0.0, neginf=0.0)

    # ── 7. SMOTE  (flat signals + flat features) ────────
    if apply_smote and SMOTE_AVAILABLE:
        print("⚖️  Applying SMOTE to training data...")
        smote = SMOTE(random_state=random_state)

        # SMOTE on 1D signals (flat)
        X_flat  = X_tr.reshape(len(X_tr), -1)
        
        # Calculate balanced sampling strategy
        # Target: 15,000 samples per class (but don't undersample)
        unique_classes, current_counts = np.unique(y_tr, return_counts=True)
        target_per_class = 15000
        strategy = {}
        for c, count in zip(unique_classes, current_counts):
            if count < target_per_class:
                strategy[c] = target_per_class
            else:
                strategy[c] = count # Keep as is if already larger
        
        print(f"  ⚖️  SMOTE strategy: {strategy}")
        try:
            smote = SMOTE(sampling_strategy=strategy, random_state=random_state)
            X_flat, y_tr = smote.fit_resample(X_flat, y_tr)
            X_tr = X_flat.reshape(-1, 1000)
            X_tr_ml = _extract_batch(X_tr)
        except Exception as e:
            print(f"  ⚠️  SMOTE failed or was not needed: {e}")
            X_tr_ml = _extract_batch(X_tr)
        
        print(f"  After SMOTE: {len(X_tr)}")
    elif apply_smote and not SMOTE_AVAILABLE:
        print("  ⚠️  imbalanced-learn not installed, skipping SMOTE")

    # ── 8. Spectrograms (CNN2D branch) ─────────────────
    X_tr_spec, X_te_spec = None, None
    if compute_spec:
        print("🌈 Computing spectrograms (CNN2D)...")
        X_tr_spec = batch_to_spectrograms(X_tr, fs=100)
        X_te_spec = batch_to_spectrograms(X_te, fs=100)
        print(f"  Spectrogram shape: {X_tr_spec.shape[1:]}")

    # ── 9. 1D input reshape for Keras ──────────────────
    X_tr_1d = X_tr[..., np.newaxis]    # (N, 1000, 1)
    X_te_1d = X_te[..., np.newaxis]

    # ── 10. One-hot labels ──────────────────────────────
    n_classes  = len(TARGET_CLASSES)
    y_tr_oh    = np.eye(n_classes)[y_tr]
    y_te_oh    = np.eye(n_classes)[y_te]

    train = dict(X_1d=X_tr_1d, X_ml=X_tr_ml, X_spec=X_tr_spec,
                 y=y_tr, y_oh=y_tr_oh)
    test  = dict(X_1d=X_te_1d, X_ml=X_te_ml, X_spec=X_te_spec,
                 y=y_te,  y_oh=y_te_oh)

    # ── 11. (Optional) Merge Kardia training data ───────
    kardia_test_out = None
    if include_kardia:
        try:
            from kardia_loader import load_kardia_split
            k_train, k_test = load_kardia_split(
                test_size=test_size, random_state=random_state
            )

            # 2× duplication of Kardia training — no SMOTE on Kardia
            X_k = np.tile(k_train["X"], (2, 1)).astype(np.float32)  # (2N_k, 1000)
            y_k = np.tile(k_train["y"], 2)
            print(f"\n  Kardia training samples (2× dup.): {len(X_k)}")

            X_k_ml = _extract_batch(X_k)
            X_k_ml = np.nan_to_num(X_k_ml, nan=0.0, posinf=0.0, neginf=0.0)
            X_k_1d = X_k[..., np.newaxis]
            y_k_oh = np.eye(n_classes)[y_k]

            # Merge into training arrays
            train["X_1d"] = np.concatenate([train["X_1d"], X_k_1d], axis=0)
            train["X_ml"] = np.concatenate([train["X_ml"], X_k_ml], axis=0)
            train["y"]    = np.concatenate([train["y"],    y_k],    axis=0)
            train["y_oh"] = np.concatenate([train["y_oh"], y_k_oh], axis=0)
            if compute_spec and train["X_spec"] is not None:
                X_k_spec = batch_to_spectrograms(X_k, fs=100)
                train["X_spec"] = np.concatenate([train["X_spec"], X_k_spec], axis=0)

            print(f"  Combined training pool: {len(train['X_1d'])} samples")

            # Build Kardia test output dict
            X_kte    = k_test["X"]
            y_kte    = k_test["y"]
            X_kte_ml = _extract_batch(X_kte)
            X_kte_ml = np.nan_to_num(X_kte_ml, nan=0.0, posinf=0.0, neginf=0.0)
            X_kte_1d = X_kte[..., np.newaxis]
            X_kte_spec = batch_to_spectrograms(X_kte, fs=100) if compute_spec else None
            y_kte_oh = np.eye(n_classes)[y_kte]

            kardia_test_out = dict(
                X_1d=X_kte_1d, X_ml=X_kte_ml, X_spec=X_kte_spec,
                y=y_kte, y_oh=y_kte_oh, pids=k_test["pids"]
            )
            print(f"  Kardia test set : {len(X_kte)} samples (held-out, never touched during training)")

        except Exception as e:
            print(f"  ⚠️  Kardia loading failed: {e}")
            print("  Continuing with digital-only data.")

    print("✅ Data loading complete.")
    return train, test, TARGET_CLASSES, kardia_test_out
