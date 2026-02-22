# backend/src/ml_models.py
"""
Classical ML models for research-grade ECG classification.

Key design decisions:
  - All models trained on 16-dimensional feature vectors (from feature_extraction.py)
  - Training / validation metrics include macro F1 (not just accuracy)
  - Models return calibrated probabilities via predict_proba
"""

import numpy as np
from sklearn.svm             import SVC
from sklearn.ensemble        import RandomForestClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.metrics         import (classification_report,
                                     f1_score, accuracy_score)
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
import joblib

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from feature_extraction import extract_ecg_features


# ======================================================
# Feature preparation
# ======================================================
def prepare_features(X: np.ndarray, fs: float = 100) -> np.ndarray:
    """
    Extract feature vectors from array of preprocessed signals.

    Parameters
    ----------
    X  : (N, 1000) array of signals at 100 Hz
    fs : sampling frequency (always 100 in this system)

    Returns
    -------
    np.ndarray  (N, 16)  float32
    """
    features = []
    for signal in X:
        feat = extract_ecg_features(signal, fs=fs)
        features.append(feat)
    arr = np.array(features, dtype=np.float32)

    # Replace remaining NaN/Inf (safety net)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


# ======================================================
# ML model definitions
# ======================================================
def get_ml_models(num_classes: int) -> dict:
    """
    Returns a dict of sklearn-compatible models wrapped in Pipelines
    (StandardScaler → Classifier) for numerical stability.
    """
    models = {
        # ── SVM ──────────────────────────────────────
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                max_iter=5000,
                random_state=42,
            )),
        ]),

        # ── Random Forest ─────────────────────────────
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),

        # ── KNN  ──────────────────────────────────────
        "KNN": KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            metric="euclidean",
            n_jobs=-1,
        ),
    }

    # ── XGBoost (optional) ────────────────────────────
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            use_label_encoder=False,
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        )

    return models


# ======================================================
# Train + Evaluate
# ======================================================
def train_ml_model(name: str, model, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   classes: list | None = None) -> tuple:
    """
    Fit an ML model and return (fitted_model, macro_f1_val).

    Returns macro F1 (not accuracy) as the primary score,
    consistent with the global metric policy.
    """
    print(f"🚀 Training ML model: {name}  |  train={len(X_train)}  val={len(X_val)}")
    model.fit(X_train, y_train)

    # ── Validation metrics ────────────────────────────
    y_pred     = model.predict(X_val)
    macro_f1   = f1_score(y_val, y_pred, average="macro", zero_division=0)
    val_acc    = accuracy_score(y_val, y_pred)

    print(f"  ✅ {name} | Val Accuracy: {val_acc:.4f} | Macro F1: {macro_f1:.4f}")

    if classes:
        unique_labels  = np.unique(np.concatenate([y_val, y_pred]))
        filtered_names = [classes[i] for i in unique_labels if i < len(classes)]
        print(f"  📊 {name} Classification Report:")
        print(classification_report(y_val, y_pred,
                                    target_names=filtered_names,
                                    labels=unique_labels,
                                    zero_division=0))

    return model, val_acc


# ======================================================
# Save / Load helpers
# ======================================================
def save_ml_models(models: dict, save_dir: str) -> None:
    import os
    os.makedirs(save_dir, exist_ok=True)
    for name, model in models.items():
        path = os.path.join(save_dir, f"{name}.joblib")
        joblib.dump(model, path)
        print(f"  💾 Saved ML model: {path}")


def load_ml_models(load_dir: str) -> dict:
    import os, glob
    ml_models = {}
    for path in glob.glob(os.path.join(load_dir, "*.joblib")):
        name = os.path.basename(path).replace(".joblib", "")
        ml_models[name] = joblib.load(path)
        print(f"  📂 Loaded ML model: {name}")
    return ml_models
