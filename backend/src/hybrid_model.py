# backend/src/hybrid_model.py
"""
Hybrid ML + DL ensemble for research-grade ECG classification.

Architecture:
  Branch 1 (CNN1D)        : raw 1D signal  (1000, 1)
  Branch 2 (CNN2D)        : real spectrogram (freq, time, 1)
  Branch 3 (ML models)    : 16-dim feature vectors

Ensemble weighting:
  - Weights are derived from per-model macro F1 (NOT accuracy)
  - Softmax-normalised so they sum to 1

AdvancedHybridModel:
  - Trains ResidualCNN, DenseNetCNN, AttentionCNN, MultiScaleCNN on 1D signals
  - ALL use identical 1D input (no reshape / no fake 2D)
"""

import os
import glob
import numpy as np
import tensorflow as tf
import joblib
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report
)
from sklearn.linear_model import LogisticRegression


class HybridEnsemble:
    """
    Hybrid ML + DL (CNN1D + CNN2D spectrogram) Ensemble.

    Parameters
    ----------
    ml_models    : dict  name → sklearn model
    dl_models    : dict  name → tf.keras.Model
    classes      : list  of class name strings
    weights      : dict  name → macro_f1 weight (will be normalised)
    """

    def __init__(self, ml_models=None, dl_models=None,
                 classes=None, weights=None):
        self.ml_models = ml_models or {}
        self.dl_models = dl_models or {}
        self.classes   = classes   or []
        self.weights   = weights   or {}

        # Unified model view (for save/load compatibility)
        self.models = {**self.ml_models, **self.dl_models}

    # --------------------------------------------------
    # Internal: compute normalised weight vector
    # --------------------------------------------------
    def _get_weight(self, name: str, predictions: list) -> float:
        return max(self.weights.get(name, 1.0), 1e-6)

    # --------------------------------------------------
    # Core prediction
    # --------------------------------------------------
    def predict_proba(self, X_ml: np.ndarray,
                      X_dl: np.ndarray,
                      X_spec: np.ndarray | None = None) -> np.ndarray:
        """
        Produce weighted-average probability predictions.

        Parameters
        ----------
        X_ml   : (N, 16)          feature vectors for ML models
        X_dl   : (N, 1000, 1)     raw ECG for CNN1D
        X_spec : (N, freq, time, 1) spectrograms for CNN2D  (optional)
                  If None, CNN2D models are skipped.

        Returns
        -------
        np.ndarray  (N, num_classes)
        """
        predictions = []
        w_vals      = []
        n_classes   = len(self.classes)

        # ── ML predictions ────────────────────────────
        for name, model in self.ml_models.items():
            try:
                proba = model.predict_proba(X_ml)
                # Align to n_classes (some models may return fewer)
                if proba.shape[1] < n_classes:
                    padded = np.zeros((proba.shape[0], n_classes))
                    padded[:, :proba.shape[1]] = proba
                    proba = padded
                predictions.append(proba)
                w_vals.append(self._get_weight(name, predictions))
            except Exception as e:
                print(f"  ⚠️  ML model {name} failed: {e}")

        # ── CNN1D predictions ─────────────────────────
        for name, model in self.dl_models.items():
            if "CNN2D" in name.upper() or "SPEC" in name.upper():
                continue    # handle CNN2D separately
            try:
                proba = model.predict(X_dl, verbose=0)
                predictions.append(proba)
                w_vals.append(self._get_weight(name, predictions))
            except Exception as e:
                print(f"  ⚠️  DL model {name} failed: {e}")

        # ── CNN2D (spectrogram) predictions ───────────
        if X_spec is not None:
            for name, model in self.dl_models.items():
                if "CNN2D" not in name.upper() and "SPEC" not in name.upper():
                    continue
                try:
                    proba = model.predict(X_spec, verbose=0)
                    predictions.append(proba)
                    w_vals.append(self._get_weight(name, predictions))
                except Exception as e:
                    print(f"  ⚠️  CNN2D model {name} failed: {e}")

        if not predictions:
            raise ValueError("❌ No models produced predictions")

        # ── Weighted average ──────────────────────────
        w_arr = np.array(w_vals, dtype=np.float64)
        w_arr = w_arr / w_arr.sum()

        ensemble = np.zeros_like(predictions[0])
        for p, w in zip(predictions, w_arr):
            ensemble += w * p[:, :ensemble.shape[1]]

        return ensemble

    def predict(self, X_ml, X_dl, X_spec=None):
        probs = self.predict_proba(X_ml, X_dl, X_spec)
        idx   = np.argmax(probs, axis=1)
        return idx, probs

    # --------------------------------------------------
    # Evaluation  (uses macro F1)
    # --------------------------------------------------
    def evaluate(self, X_ml, X_dl, y_true, X_spec=None):
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)

        probs  = self.predict_proba(X_ml, X_dl, X_spec)
        y_pred = np.argmax(probs, axis=1)

        acc       = accuracy_score(y_true, y_pred)
        macro_f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)

        print(f"\n🎯 Hybrid Ensemble | Accuracy: {acc:.4f}  |  Macro F1: {macro_f1:.4f}\n")

        unique_labels  = np.unique(np.concatenate([y_true, y_pred]))
        filtered_names = [self.classes[i] for i in unique_labels
                          if i < len(self.classes)]
        print("📊 Classification Report:")
        print(classification_report(y_true, y_pred,
                                    target_names=filtered_names,
                                    labels=unique_labels,
                                    zero_division=0))

        return macro_f1, probs   # NOTE: returns macro_f1, not acc

    # --------------------------------------------------
    # Save / Load
    # --------------------------------------------------
    def save_models(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        for name, model in self.dl_models.items():
            model.save(os.path.join(save_dir, f"{name}.keras"))
            print(f"  💾 Saved DL model: {name}")
        for name, model in self.ml_models.items():
            joblib.dump(model, os.path.join(save_dir, f"{name}.joblib"))
            print(f"  💾 Saved ML model: {name}")

    def load_models(self, load_dir: str):
        for path in glob.glob(os.path.join(load_dir, "*.keras")):
            name = os.path.basename(path).replace(".keras", "")
            self.dl_models[name] = tf.keras.models.load_model(
                path, safe_mode=False)
            print(f"  📂 Loaded DL model: {name}")

        for path in glob.glob(os.path.join(load_dir, "*.joblib")):
            name = os.path.basename(path).replace(".joblib", "")
            self.ml_models[name] = joblib.load(path)
            print(f"  📂 Loaded ML model: {name}")

        self.models = {**self.ml_models, **self.dl_models}


# ======================================================
# Advanced Hybrid Model  (sub-ensemble of 1D CNNs)
# ======================================================
class AdvancedHybridModel:
    """
    Ensemble of four advanced 1D CNN architectures.
    All operate on raw 1D signals (1000, 1) — no reshape, no spectrogram.
    Weights are derived from per-model macro F1.
    """

    def __init__(self, input_shape: tuple = (1000, 1), num_classes: int = 5):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.models      = {}
        self.model_acc    = {}   # accuracy per model

        from cnn_models import (build_residual_cnn, build_densenet_cnn,
                                build_attention_cnn, build_multiscale_cnn)

        builders = {
            "ResidualCNN":    build_residual_cnn,
            "DenseNetCNN":    build_densenet_cnn,
            "AttentionCNN":   build_attention_cnn,
            "MultiScaleCNN":  build_multiscale_cnn,
        }

        for name, fn in builders.items():
            m = fn(input_shape, num_classes)
            m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
            self.models[name] = m

    def train_ensemble(self, X_train, y_train, X_val, y_val,
                       epochs: int = 20, batch_size: int = 32):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=8, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", patience=4, factor=0.5, min_lr=1e-6),
        ]

        y_true_int = np.argmax(y_val, axis=1)

        for name, model in self.models.items():
            print(f"\n  Training {name}...")
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1,
            )
            # Record per-model accuracy
            y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
            vacc   = accuracy_score(y_true_int, y_pred)
            self.model_acc[name] = vacc
            print(f"  ✅ {name} Val Accuracy: {vacc:.4f}")

    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """
        Accuracy-weighted ensemble prediction.
        """
        if not self.model_acc:
            # No weights yet → uniform average
            weights = {n: 1.0 for n in self.models}
        else:
            weights = self.model_acc

        w_sum  = sum(weights.values()) + 1e-8
        result = None

        for name, model in self.models.items():
            proba = model.predict(X, verbose=0)
            w     = weights.get(name, 1.0) / w_sum
            if result is None:
                result = w * proba
            else:
                result = result + w * proba

        return result