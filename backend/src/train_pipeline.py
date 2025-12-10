#!!/usr/bin/env python3
"""
train_pipeline.py — Fixed ECG ML + DL Hybrid Training Pipeline
✅ Fixed: Missing imports, AdvancedHybridModel methods, ML constant prediction, DL validation split
✅ Fixed: Empty validation scores when resuming - uses equal weights fallback
"""


import sys
import os
import time
import joblib
import numpy as np
import tensorflow as tf
import pandas as pd
import shutil
from datetime import datetime
import argparse
import json  # ✅ ADDED: Missing import
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


from data_loader import load_all_datasets
from ml_models import prepare_features, get_ml_models, train_ml_model
from cnn_models import build_cnn_1d, build_cnn_2d
from hybrid_model import AdvancedHybridModel, HybridEnsemble


# ------------------------
# CLI Arguments
# ------------------------
parser = argparse.ArgumentParser(description="Train ECG hybrid ML + DL models")
parser.add_argument("--resume", action="store_true", help="Resume from latest run")
parser.add_argument("--limit", type=int, default=3000, help="Limit number of samples to load per dataset")
parser.add_argument("--epochs", type=int, default=5, help="Epochs for DL training")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for DL training")
parser.add_argument("--max_per_class", type=int, default=5000, help="Max samples per class if balancing")
parser.add_argument("--normalize", action="store_true", help="Apply global normalization (recommended)")
parser.add_argument("--sample_norm", action="store_true", help="Apply per-sample normalization")
parser.add_argument("--undersample", action="store_true", help="Undersample majority classes")
parser.add_argument("--svm_limit", type=int, default=2000, help="Subset size for SVM training")
parser.add_argument("--keep_runs", type=int, default=5, help="Keep last N model runs")
args = parser.parse_args()


# ------------------------
# Paths
# ------------------------
BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.join(BASE_DIR, "..", "logs")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


log_file = os.path.join(LOG_DIR, "train_log.txt")
results_file = os.path.join(LOG_DIR, "results_history.csv")


# ------------------------
# Logger
# ------------------------
class Logger:
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")
    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
        self.log.flush()
    def flush(self): pass


sys.stdout = Logger()


# ------------------------
# Config
# ------------------------
LIMIT = args.limit
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
SVM_LIMIT = args.svm_limit
MAX_PER_CLASS = args.max_per_class
APPLY_GLOBAL_NORM = args.normalize
APPLY_SAMPLE_NORM = args.sample_norm
UNDERSAMPLE = args.undersample
KEEP_RUNS = args.keep_runs


# ------------------------
# Run Management
# ------------------------
def get_latest_run():
    runs = sorted(
        [os.path.join(MODEL_DIR, d) for d in os.listdir(MODEL_DIR) if d.startswith("run_")],
        key=os.path.getmtime,
    )
    return runs[-1] if runs else None


def cleanup_old_runs(keep_last=KEEP_RUNS):
    runs = sorted(
        [os.path.join(MODEL_DIR, d) for d in os.listdir(MODEL_DIR) if d.startswith("run_")],
        key=os.path.getmtime,
    )
    if len(runs) > keep_last:
        for old_run in runs[:-keep_last]:
            shutil.rmtree(old_run, ignore_errors=True)
            print(f"🧹 Deleted old run folder: {old_run}")


if args.resume:
    RUN_DIR = get_latest_run()
    if RUN_DIR:
        print(f"🔁 Resuming training from {RUN_DIR}")
    else:
        RUN_DIR = os.path.join(MODEL_DIR, f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        os.makedirs(RUN_DIR, exist_ok=True)
else:
    RUN_DIR = os.path.join(MODEL_DIR, f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(RUN_DIR, exist_ok=True)
    cleanup_old_runs()


# ------------------------
# Load Data
# ------------------------
print("📥 Loading datasets (PTB-XL train/test + others)...")
(X_train, y_train), (X_test, y_test), classes = load_all_datasets(limit=LIMIT, one_hot=True, window_size=1000)
print(f"✅ Data loaded: Train={X_train.shape}, Test={X_test.shape}, Classes={classes}")


if y_train.ndim > 1 and y_train.shape[1] > 1:
    y_train_int = np.argmax(y_train, axis=1)
    y_test_int = np.argmax(y_test, axis=1)
else:
    y_train_int, y_test_int = y_train, y_test


print(f"Class distribution: {np.bincount(y_train_int)}")  # ✅ Debug class imbalance


# ------------------------
# Proper Train/Val/Test Split (FIXED)
# ------------------------
print("🔀 Creating proper train/validation split...")
X_tr_dl, X_val_dl, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, 
    stratify=y_train_int  # ✅ Stratified split
)
print(f"✅ Train/Val split: Train={X_tr_dl.shape}, Val={X_val_dl.shape}")


# ------------------------
# Normalization (IMPROVED)
# ------------------------
if APPLY_SAMPLE_NORM and APPLY_GLOBAL_NORM:
    print("⚠️ Both sample and global normalization chosen, using global only.")
    APPLY_SAMPLE_NORM = False


if APPLY_SAMPLE_NORM:
    print("🔬 Applying per-sample z-score normalization")
    X_train = (X_train - np.mean(X_train, axis=1, keepdims=True)) / (np.std(X_train, axis=1, keepdims=True) + 1e-8)
    X_test = (X_test - np.mean(X_test, axis=1, keepdims=True)) / (np.std(X_test, axis=1, keepdims=True) + 1e-8)
    X_tr_dl = (X_tr_dl - np.mean(X_tr_dl, axis=1, keepdims=True)) / (np.std(X_tr_dl, axis=1, keepdims=True) + 1e-8)
    X_val_dl = (X_val_dl - np.mean(X_val_dl, axis=1, keepdims=True)) / (np.std(X_val_dl, axis=1, keepdims=True) + 1e-8)


if APPLY_GLOBAL_NORM:
    print("🔬 Applying global normalization")
    mean, std = np.mean(X_train), np.std(X_train) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    X_tr_dl = (X_tr_dl - mean) / std
    X_val_dl = (X_val_dl - mean) / std


# Prepare data for models
X_train_ml = prepare_features(X_train)
X_test_ml = prepare_features(X_test)
X_train_dl = X_train[..., np.newaxis]
X_test_dl = X_test[..., np.newaxis]
X_tr_dl_final = X_tr_dl[..., np.newaxis]
X_val_dl_final = X_val_dl[..., np.newaxis]


# ------------------------
# Class Weights (IMPROVED)
# ------------------------
class_weights = None
if len(np.unique(y_train_int)) > 1:
    weights = compute_class_weight("balanced", classes=np.unique(y_train_int), y=y_train_int)
    class_weights = {int(c): float(w) for c, w in zip(np.unique(y_train_int), weights)}
    print("⚖️ Class weights:", class_weights)


# ------------------------
# Train ML Models (WITH BETTER DIAGNOSTICS)
# ------------------------
ml_models = {}
ml_val_scores = {}
ml_defs = get_ml_models(num_classes=len(classes))


for name, model_def in ml_defs.items():
    path = os.path.join(RUN_DIR, f"{name}.joblib")
    if args.resume and os.path.exists(path):
        print(f"📂 Resuming {name}")
        ml_models[name] = joblib.load(path)
        continue
    
    print(f"🚀 Training {name}...")
    start = time.time()
    
    if name == "SVM":
        idx = np.random.choice(len(X_train_ml), min(SVM_LIMIT, len(X_train_ml)), replace=False)
        X_sub, y_sub = X_train_ml[idx], y_train_int[idx]
        model, acc = train_ml_model(name, model_def, X_sub, y_sub, X_test_ml, y_test_int)
    else:
        model, acc = train_ml_model(name, model_def, X_train_ml, y_train_int, X_test_ml, y_test_int)
    
    # ✅ DIAGNOSTIC: Check if predicting constant class
    sample_pred = model.predict(X_test_ml[:20])
    unique_preds = np.unique(sample_pred)
    print(f"  {name} predictions variety: {unique_preds} (count={len(unique_preds)})")
    
    joblib.dump(model, path)
    ml_models[name] = model
    ml_val_scores[name] = acc
    print(f"✅ {name} finished in {time.time() - start:.2f}s (val_acc={acc:.4f})")


# ------------------------
# Train DL Models (FIXED VALIDATION SPLIT)
# ------------------------
dl_models = {}
dl_val_scores = {}
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint


def make_callbacks(name):
    ckpt = os.path.join(RUN_DIR, f"{name}_best.keras")
    return [
        ModelCheckpoint(ckpt, monitor="val_accuracy", mode="max", save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=5, min_lr=1e-8, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True, verbose=1),
    ]


# CNN1D ✅ FIXED
cnn1d_path = os.path.join(RUN_DIR, "cnn1d.keras")
if args.resume and os.path.exists(cnn1d_path):
    print("📂 Resuming CNN1D")
    cnn1d = tf.keras.models.load_model(cnn1d_path)
else:
    print("🚀 Training CNN1D...")
    cnn1d = build_cnn_1d((1000, 1), num_classes=len(classes))
    cnn1d.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    history = cnn1d.fit(X_tr_dl_final, y_tr, validation_data=(X_val_dl_final, y_val),
                       epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1,
                       callbacks=make_callbacks("cnn1d"),
                       class_weight=class_weights)
    cnn1d.save(cnn1d_path)
    dl_val_scores["CNN1D"] = max(history.history.get("val_accuracy", [0.0]))
dl_models["CNN1D"] = cnn1d


# CNN2D ✅ FIXED (no manual reshape needed if using hybrid, but keeping for comparison)
cnn2d_path = os.path.join(RUN_DIR, "cnn2d.keras")
if args.resume and os.path.exists(cnn2d_path):
    print("📂 Resuming CNN2D")
    cnn2d = tf.keras.models.load_model(cnn2d_path)
else:
    print("🚀 Training CNN2D...")
    X_tr_2d = X_tr_dl_final.reshape(-1, 100, 10, 1)
    X_val_2d = X_val_dl_final.reshape(-1, 100, 10, 1)
    cnn2d = build_cnn_2d((100, 10, 1), num_classes=len(classes))
    cnn2d.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    history2 = cnn2d.fit(X_tr_2d, y_tr, validation_data=(X_val_2d, y_val),
                        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1,
                        callbacks=make_callbacks("cnn2d"),
                        class_weight=class_weights)
    cnn2d.save(cnn2d_path)
    dl_val_scores["CNN2D"] = max(history2.history.get("val_accuracy", [0.0]))
dl_models["CNN2D"] = cnn2d


# ------------------------
# Advanced Hybrid Model (FIXED)
# ------------------------
print("🚀 Training Advanced Hybrid Model...")
advanced_hybrid = AdvancedHybridModel(input_shape=(1000, 1), num_classes=len(classes))


# Train (uses proper train/val split internally now)
advanced_hybrid.train_ensemble(
    X_tr_dl_final, y_tr, 
    X_val_dl_final, y_val,  # ✅ Proper validation split
    epochs=30,
    batch_size=BATCH_SIZE
)


# ✅ FIXED EVALUATION
advanced_pred = advanced_hybrid.predict_ensemble(X_test_dl)
advanced_acc = np.mean(np.argmax(advanced_pred, axis=1) == y_test_int)
print(f"✅ Advanced Hybrid Test Accuracy: {advanced_acc:.4f}")


# Save individual models
for name, model in advanced_hybrid.models.items():
    model.save(os.path.join(RUN_DIR, f"advanced_{name}.keras"))
print("💾 Saved advanced hybrid models")


# ────────────────────────────────────────────────────────────────────
# Traditional Hybrid Ensemble
# ────────────────────────────────────────────────────────────────────
print("🤝 Building Traditional Hybrid Ensemble...")

# ✅ FIX: Handle empty validation scores (from resume)
if not ml_val_scores or not dl_val_scores:
    print("⚠️  Validation scores missing (from resume). Using equal weights...")
    # Get all model names
    ml_names = list(ml_models.keys())
    dl_names = list(dl_models.keys())
    all_names = ml_names + dl_names
    
    # Equal weights for all
    num_models = len(all_names)
    weights_dict = {name: 1.0 / num_models for name in all_names}
else:
    scores = {**ml_val_scores, **dl_val_scores}
    vals = np.array(list(scores.values()), dtype=float)
    vals = (vals - vals.min()) + 1e-8
    weights = vals / (vals.sum() + 1e-8)
    weights_dict = {k: float(weights[i]) for i, k in enumerate(scores.keys())}

print("🔢 Ensemble weights:", weights_dict)


hybrid = HybridEnsemble(ml_models=ml_models, dl_models=dl_models, classes=classes, weights=weights_dict)
acc, _ = hybrid.evaluate(X_test_ml, X_test_dl, y_test_int)
print(f"✅ Traditional Hybrid Test Accuracy: {acc:.4f}")


# ────────────────────────────────────────────────────────────────────
# Save Classes & Results
# ────────────────────────────────────────────────────────────────────
classes_file = os.path.join(RUN_DIR, "classes.json")
with open(classes_file, "w") as f:
    json.dump(classes, f)
print(f"💾 Saved classes to {classes_file}")


# Log results
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
record = pd.DataFrame([{
    "timestamp": timestamp,
    "limit": LIMIT,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "normalize": APPLY_GLOBAL_NORM,
    "ml_scores": str(ml_val_scores),
    "dl_scores": str(dl_val_scores),
    "hybrid_acc": acc,
    "advanced_hybrid_acc": advanced_acc,
    "run_folder": RUN_DIR
}])


if os.path.exists(results_file):
    results = pd.concat([pd.read_csv(results_file), record], ignore_index=True)
else:
    results = record
results.to_csv(results_file, index=False)


print(f"📊 Results logged: {results_file}")
print(f"🏆 Best traditional hybrid accuracy: {acc:.4f}")
print(f"🚀 Best advanced hybrid accuracy: {advanced_acc:.4f}")
print(f"📁 Model folder: {RUN_DIR}")
print("🎉 Training completed successfully!")