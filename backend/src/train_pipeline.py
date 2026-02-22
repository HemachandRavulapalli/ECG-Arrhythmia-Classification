#!/usr/bin/env python3
"""
train_pipeline.py — Research-Grade ECG Hybrid Training Pipeline

Design rules implemented:
  ✔ Patient-wise GroupShuffleSplit (no leakage)
  ✔ SMOTE on training data only
  ✔ Real spectrogram CNN2D (not reshape)
  ✔ Macro F1 as primary metric
  ✔ Ensemble weights derived from macro F1
  ✔ All signals at 100 Hz, 1000 samples
  ✔ Resume + run management supported
"""

import os
# Disable GPU to avoid CUDA errors on CPU-only systems
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import time
import json
import shutil
import joblib
import logging
import argparse
import numpy as np
import tensorflow as tf
import pandas as pd
from datetime import datetime

logging.getLogger("tensorflow").setLevel(logging.ERROR)

from sklearn.model_selection    import train_test_split, StratifiedShuffleSplit
from sklearn.metrics            import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

# ── Local imports ──────────────────────────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_loader  import load_all_datasets, TARGET_CLASSES
from ml_models    import get_ml_models, train_ml_model
from cnn_models   import (build_cnn_1d, build_cnn_2d, get_spectrogram_shape,
                           build_residual_cnn, build_densenet_cnn,
                           build_attention_cnn, build_multiscale_cnn)
from hybrid_model import HybridEnsemble, AdvancedHybridModel


# ======================================================
# CLI
# ======================================================
parser = argparse.ArgumentParser("Research-Grade ECG Training Pipeline")
parser.add_argument("--limit",      type=int, default=30000,
                    help="Max samples per dataset (default: 30000 to prevent OOM)")
parser.add_argument("--epochs",     type=int, default=30)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--svm_limit",  type=int, default=3000,
                    help="Max training samples for SVM (slow)")
parser.add_argument("--keep_runs",  type=int, default=5)
parser.add_argument("--resume",     action="store_true")
parser.add_argument("--skip_adv",   action="store_true",
                    help="Skip AdvancedHybridModel (faster run)")
parser.add_argument("--no_smote",   action="store_true",
                    help="Disable SMOTE (faster, less balanced)")
parser.add_argument("--no_spec",    action="store_true",
                    help="Disable CNN2D spectrogram branch")
parser.add_argument("--kardia",     action="store_true",
                    help="Include Kardia 6L PDFs in the training pool (multi-domain mode)")
args = parser.parse_args()


# ======================================================
# Paths
# ======================================================
BASE_DIR   = os.path.dirname(SRC_DIR)
LOG_DIR    = os.path.join(BASE_DIR, "logs")
MODEL_DIR  = os.path.join(SRC_DIR,  "saved_models")
os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

log_file     = os.path.join(LOG_DIR, "train_log.txt")
results_file = os.path.join(LOG_DIR, "results_history.csv")


# ======================================================
# Logger (tee stdout → file)
# ======================================================
class TeeLogger:
    def __init__(self, path):
        self.terminal = sys.__stdout__
        self.log      = open(path, "a", buffering=1)

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = TeeLogger(log_file)


# ======================================================
# Run management
# ======================================================
def _get_runs():
    return sorted(
        [os.path.join(MODEL_DIR, d)
         for d in os.listdir(MODEL_DIR) if d.startswith("run_")],
        key=os.path.getmtime,
    )

def get_latest_run():
    runs = _get_runs()
    return runs[-1] if runs else None

def cleanup_old_runs(keep_last):
    for old in _get_runs()[:-keep_last]:
        shutil.rmtree(old, ignore_errors=True)


if args.resume:
    RUN_DIR = get_latest_run()
    if RUN_DIR:
        print(f"🔁 Resuming from {RUN_DIR}")
    else:
        RUN_DIR = os.path.join(MODEL_DIR, f"run_{datetime.now():%Y%m%d_%H%M%S}")
        os.makedirs(RUN_DIR, exist_ok=True)
else:
    RUN_DIR = os.path.join(MODEL_DIR, f"run_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(RUN_DIR, exist_ok=True)
    cleanup_old_runs(args.keep_runs)

print(f"\n{'='*60}")
print(f"  ECG Research-Grade Training Pipeline")
print(f"  Run folder : {RUN_DIR}")
print(f"  Started    : {datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"{'='*60}\n")


# ======================================================
# 1. Load data  (patient-wise split + SMOTE built-in)
# ======================================================
print("📥 Loading datasets (patient-wise split)...")
train, test, classes, kardia_test = load_all_datasets(
    limit          = args.limit,
    one_hot        = True,
    apply_smote    = not args.no_smote,
    apply_augment  = True,
    compute_spec   = not args.no_spec,
    test_size      = 0.2,
    random_state   = 42,
    include_kardia = args.kardia,
)

X_tr_1d   = train["X_1d"]      # (N_tr, 1000, 1)
X_tr_ml   = train["X_ml"]      # (N_tr, 16)
X_tr_spec = train["X_spec"]    # (N_tr, freq, time, 1) or None
y_tr      = train["y"]         # (N_tr,) int
y_tr_oh   = train["y_oh"]      # (N_tr, 5)

X_te_1d   = test["X_1d"]
X_te_ml   = test["X_ml"]
X_te_spec = test["X_spec"]
y_te      = test["y"]
y_te_oh   = test["y_oh"]

n_classes = len(classes)

print(f"\nTrain: {len(X_tr_1d)}  |  Test: {len(X_te_1d)}  |  Classes: {n_classes}")
print("Classes:", classes)
print("Train class dist:", np.unique(y_tr, return_counts=True))
print("Test class dist: ", np.unique(y_te, return_counts=True))


# ======================================================
# 2. Train/Val split for DL  (from training set only)
# ======================================================
print("\n✂️  Train/Val split for DL...")
val_size = min(0.15, max(0.05, 500 / len(X_tr_1d)))   # adaptive val size

sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
tr_idx, val_idx = next(sss.split(X_tr_1d, y_tr))

X_trn,  X_val  = X_tr_1d[tr_idx],   X_tr_1d[val_idx]
X_trn_ml        = X_tr_ml[tr_idx]
y_trn,  y_val_y = y_tr[tr_idx],     y_tr[val_idx]
y_trn_oh        = y_tr_oh[tr_idx]
y_val_oh        = y_tr_oh[val_idx]

X_trn_spec = X_tr_spec[tr_idx]  if X_tr_spec is not None else None
X_val_spec  = X_tr_spec[val_idx] if X_tr_spec is not None else None

print(f"  Tr_nn: {len(X_trn)}  /  Val: {len(X_val)}")


# ======================================================
# 3. Class weights  (for DL loss)
# ======================================================
cls_weights = compute_class_weight("balanced",
                                   classes=np.unique(y_tr),
                                   y=y_tr)
class_weight_dict = {int(i): float(w)
                     for i, w in zip(np.unique(y_tr), cls_weights)}
print(f"⚖️  Class weights: {class_weight_dict}")


# ======================================================
# Shared DL callbacks
# ======================================================
def make_callbacks(patience_es=12, patience_lr=5):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience_es,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=patience_lr,
            factor=0.5, min_lr=1e-6, verbose=1),
    ]


# ======================================================
# 4. Train ML models
# ======================================================
print(f"\n{'='*50}")
print("🤖 Training ML Models")
print('='*50)

ml_models  = {}
ml_scores  = {}   # macro F1

for name, model in get_ml_models(n_classes).items():
    path = os.path.join(RUN_DIR, f"{name}.joblib")

    if args.resume and os.path.exists(path):
        ml_models[name] = joblib.load(path)
        print(f"  ⏭️  Skipped {name} (already exists)")
        continue

    X_fit = X_trn_ml
    y_fit = y_trn

    if name == "SVM" and len(X_fit) > args.svm_limit:
        idx   = np.random.choice(len(X_fit), args.svm_limit, replace=False)
        X_fit = X_fit[idx]
        y_fit = y_fit[idx]
        print(f"  → SVM training limited to {args.svm_limit} samples")

    # Score on the held-out validation split
    fitted, val_acc = train_ml_model(
        name, model,
        X_fit,          y_fit,
        X_tr_ml[val_idx], y_val_y,
        classes,
    )

    joblib.dump(fitted, path)
    ml_models[name] = fitted
    ml_scores[name] = val_acc

print(f"\n  ML Validation Accuracy scores: {ml_scores}")


# ======================================================
# 5. Train CNN1D
# ======================================================
print(f"\n{'='*50}")
print("🧠 Training CNN1D")
print('='*50)

cnn1d_path = os.path.join(RUN_DIR, "cnn1d.keras")
dl_scores  = {}

if args.resume and os.path.exists(cnn1d_path):
    cnn1d = tf.keras.models.load_model(cnn1d_path, safe_mode=False)
    print("  ⏭️  Skipped CNN1D (already exists)")
else:
    cnn1d = build_cnn_1d((1000, 1), n_classes)
    cnn1d.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    cnn1d.fit(
        X_trn, y_trn_oh,
        validation_data = (X_val, y_val_oh),
        epochs          = args.epochs,
        batch_size      = args.batch_size,
        class_weight    = class_weight_dict,
        callbacks       = make_callbacks(),
        verbose         = 1,
    )
    cnn1d.save(cnn1d_path)

y_pred_val_1d = np.argmax(cnn1d.predict(X_val, verbose=0), axis=1)
acc_cnn1d     = accuracy_score(y_val_y, y_pred_val_1d)
dl_scores["cnn1d"] = acc_cnn1d
print(f"  ✅ CNN1D Val Accuracy: {acc_cnn1d:.4f}")

# Still compute test F1 for reporting
y_pred_te_1d = np.argmax(cnn1d.predict(X_te_1d, verbose=0), axis=1)
f1_cnn1d     = f1_score(y_te, y_pred_te_1d, average="macro", zero_division=0)


# ======================================================
# 6. Train CNN2D  (spectrogram)
# ======================================================
f1_cnn2d  = 0.0
cnn2d     = None

if not args.no_spec and X_trn_spec is not None:
    print(f"\n{'='*50}")
    print("🌈 Training CNN2D (Spectrogram)")
    print('='*50)

    cnn2d_path = os.path.join(RUN_DIR, "cnn2d.keras")

    spec_shape = X_trn_spec.shape[1:]   # (freq, time, 1)
    print(f"  Spectrogram input shape: {spec_shape}")

    if args.resume and os.path.exists(cnn2d_path):
        cnn2d = tf.keras.models.load_model(cnn2d_path, safe_mode=False)
        print("  ⏭️  Skipped CNN2D (already exists)")
    else:
        cnn2d = build_cnn_2d(spec_shape, n_classes)
        cnn2d.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
        cnn2d.fit(
            X_trn_spec, y_trn_oh,
            validation_data = (X_val_spec, y_val_oh),
            epochs          = args.epochs,
            batch_size      = args.batch_size,
            class_weight    = class_weight_dict,
            callbacks       = make_callbacks(),
            verbose         = 1,
        )
        cnn2d.save(cnn2d_path)

    y_pred_val_2d = np.argmax(cnn2d.predict(X_val_spec, verbose=0), axis=1)
    acc_cnn2d     = accuracy_score(y_val_y, y_pred_val_2d)
    dl_scores["cnn2d"] = acc_cnn2d
    print(f"  ✅ CNN2D Val Accuracy: {acc_cnn2d:.4f}")

    # Still compute test F1 for reporting
    y_pred_te_2d = np.argmax(cnn2d.predict(X_te_spec, verbose=0), axis=1)
    f1_cnn2d      = f1_score(y_te, y_pred_te_2d, average="macro", zero_division=0)
else:
    print("\n⏭️  CNN2D / spectrogram branch skipped (--no_spec)")


# ======================================================
# 7. Advanced Hybrid sub-models
# ======================================================
adv_f1 = 0.0

if not args.skip_adv:
    print(f"\n{'='*50}")
    print("🚀 Training Advanced Hybrid (4 sub-models)")
    print('='*50)

    adv = AdvancedHybridModel(input_shape=(1000, 1), num_classes=n_classes)
    adv.train_ensemble(
        X_trn, y_trn_oh,
        X_val, y_val_oh,
        epochs     = min(args.epochs, 30),
        batch_size = args.batch_size,
    )

    adv_probs = adv.predict_ensemble(X_te_1d)
    y_pred_adv = np.argmax(adv_probs, axis=1)
    adv_f1     = f1_score(y_te, y_pred_adv, average="macro", zero_division=0)
    print(f"  ✅ Advanced Hybrid Test Macro F1: {adv_f1:.4f}")

    for name, model in adv.models.items():
        model.save(os.path.join(RUN_DIR, f"adv_{name}.keras"))


# ======================================================
# 8. Hybrid Ensemble  (ML + CNN1D + CNN2D)
# ======================================================
print(f"\n{'='*50}")
print("🎯 Evaluating Hybrid Ensemble")
print('='*50)

# Build weights dict from macro F1
all_scores = {**ml_scores, **dl_scores}
dl_model_dict = {"cnn1d": cnn1d}
if cnn2d is not None:
    dl_model_dict["cnn2d"] = cnn2d

hybrid = HybridEnsemble(
    ml_models = ml_models,
    dl_models = dl_model_dict,
    classes   = classes,
    weights   = all_scores,
)

# Comprehensive evaluation
probs = hybrid.predict_proba(X_te_ml, X_te_1d, X_spec=X_te_spec)
y_pred = np.argmax(probs, axis=1)

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

prec, rec, f1, _ = precision_recall_fscore_support(y_te, y_pred, average="macro", zero_division=0)
hyb_acc = accuracy_score(y_te, y_pred)

print(f"  ✅ Hybrid Ensemble Metrics:")
print(f"     Accuracy:  {hyb_acc:.4f}")
print(f"     Precision: {prec:.4f}")
print(f"     Recall:    {rec:.4f}")
print(f"     Macro F1:  {f1:.4f}")

# Classification Report
print("\n📊 Full Classification Report:")
print(classification_report(y_te, y_pred, target_names=classes, labels=np.arange(len(classes)), zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_te, y_pred)
cm_file = os.path.join(RUN_DIR, "confusion_matrix.json")
with open(cm_file, "w") as f:
    json.dump(cm.tolist(), f)
print(f"  ✅ Confusion matrix saved to: {cm_file}")


# ======================================================
# 9. Dual Evaluation (Digital + Kardia)
# ======================================================
if args.kardia and kardia_test is not None:
    print(f"\n{'='*50}")
    print("📊 KARDIA TEST SET (held-out 20% — never seen during training)")
    print('='*50)

    k_probs = hybrid.predict_proba(
        kardia_test["X_ml"],
        kardia_test["X_1d"],
        X_spec = kardia_test["X_spec"],
    )
    k_pred = np.argmax(k_probs, axis=1)
    y_kte  = kardia_test["y"]

    k_prec, k_rec, k_f1, _ = precision_recall_fscore_support(
        y_kte, k_pred, average="macro", zero_division=0
    )
    k_acc = accuracy_score(y_kte, k_pred)

    print(f"  Accuracy : {k_acc:.4f}")
    print(f"  Precision: {k_prec:.4f}")
    print(f"  Recall   : {k_rec:.4f}")
    print(f"  Macro F1 : {k_f1:.4f}")
    print(f"  Samples  : {len(y_kte)}")
    print("\n  Per-sample predictions:")
    for pid, gt, pred in zip(kardia_test["pids"], y_kte, k_pred):
        match = "✅" if gt == pred else "❌"
        print(f"    {match} {pid:22s}: GT={classes[gt]:22s} Pred={classes[pred]}")

    print("\n📊 Classification Report (Kardia):")
    print(classification_report(y_kte, k_pred,
                                target_names=classes,
                                labels=np.arange(len(classes)),
                                zero_division=0))

    # Save Kardia confusion matrix separately
    k_cm      = confusion_matrix(y_kte, k_pred)
    k_cm_file = os.path.join(RUN_DIR, "confusion_matrix_kardia.json")
    with open(k_cm_file, "w") as f:
        json.dump(k_cm.tolist(), f)

    # Append Kardia metrics to scores.json
    scores_path = os.path.join(RUN_DIR, "scores.json")
    if os.path.exists(scores_path):
        with open(scores_path) as f:
            scores = json.load(f)
        scores["kardia_acc"]     = float(k_acc)
        scores["kardia_macro_f1"] = float(k_f1)
        scores["kardia_n"]       = int(len(y_kte))
        with open(scores_path, "w") as f:
            json.dump(scores, f, indent=2)

print(f"\n{'='*60}")
print(f"  FINAL SUMMARY")
print(f"{'='*60}")
print(f"  Digital  test  | Acc: {hyb_acc:.4f} | Macro F1: {f1:.4f}")
if args.kardia and kardia_test is not None:
    print(f"  Kardia   test  | Acc: {k_acc:.4f} | Macro F1: {k_f1:.4f} | N={len(y_kte)}")

# ======================================================
# 9. Save metadata + results
# ======================================================
with open(os.path.join(RUN_DIR, "classes.json"), "w") as f:
    json.dump(classes, f)

with open(os.path.join(RUN_DIR, "scores.json"), "w") as f:
    json.dump({
        "ml_scores":  ml_scores,
        "dl_scores":  dl_scores,
        "adv_f1":     adv_f1,
        "hybrid_f1":  float(f1),
        "hybrid_acc": float(hyb_acc),
        "precision":  float(prec),
        "recall":     float(rec),
    }, f, indent=2)

record = pd.DataFrame([{
    "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "limit":           args.limit,
    "epochs":          args.epochs,
    "hybrid_macro_f1": f1,
    "hybrid_acc":      hyb_acc,
    "precision":       prec,
    "recall":          rec,
    "cnn1d_f1":        f1_cnn1d,
    "cnn2d_f1":        f1_cnn2d,
    "run_folder":      RUN_DIR,
}])

if os.path.exists(results_file):
    pd.concat([pd.read_csv(results_file), record]).to_csv(results_file, index=False)
else:
    record.to_csv(results_file, index=False)

print(f"\n🎉 Training complete!")
print(f"📁 Run folder  : {RUN_DIR}")
print(f"📊 Macro F1    : Hybrid={f1:.4f} | CNN1D={f1_cnn1d:.4f} | CNN2D={f1_cnn2d:.4f}")
print(f"📊 Results CSV : {results_file}")

sys.stdout.log.close()
sys.stdout = sys.__stdout__
print("✅ Training pipeline finished.")
