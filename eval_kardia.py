#!/usr/bin/env python3
"""
eval_kardia.py — External Kardia 6L Validation Script

Loads a trained model (MIT+PTB only) and evaluates it on ALL 66 Kardia
6L PDFs as an external, held-out, never-seen-during-training dataset.

Scientific framing:
  "The Kardia 6L dataset was excluded from training due to extreme class
   imbalance (94% NSR) and used exclusively for external real-world
   cross-domain generalization assessment."

Usage:
    python3 eval_kardia.py               # uses latest trained run
    python3 eval_kardia.py --run run_YYYYMMDD_HHMMSS
    python3 eval_kardia.py --force_extract   # re-extract even if cached
"""

import os
import sys
import json
import argparse
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

SRC_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend", "src")
PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

import tensorflow as tf
import joblib
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from joblib import Parallel, delayed

from kardia_loader     import (load_label_map, extract_and_cache,
                                _records_to_arrays, TARGET_CLASSES)
from feature_extraction import extract_ecg_features
from hybrid_model      import HybridEnsemble
from cnn_models        import batch_to_spectrograms


# ======================================================
# CLI
# ======================================================
parser = argparse.ArgumentParser("Kardia 6L External Validator")
parser.add_argument("--run", default=None,
                    help="Run folder name inside saved_models/ (default: latest)")
parser.add_argument("--force_extract", action="store_true",
                    help="Re-extract all PDFs even if NPZ cache exists")
args = parser.parse_args()


# ======================================================
# Find model run folder
# ======================================================
MODEL_DIR = os.path.join(SRC_DIR, "saved_models")

def _latest_run():
    runs = sorted(
        [os.path.join(MODEL_DIR, d)
         for d in os.listdir(MODEL_DIR) if d.startswith("run_")],
        key=os.path.getmtime,
    )
    return runs[-1] if runs else None

RUN_DIR = os.path.join(MODEL_DIR, args.run) if args.run else _latest_run()
if not RUN_DIR or not os.path.isdir(RUN_DIR):
    print("❌ No trained run found. Run train_pipeline.py first.")
    sys.exit(1)

print(f"\n📁 Model run : {RUN_DIR}")

classes_file = os.path.join(RUN_DIR, "classes.json")
with open(classes_file) as f:
    classes = json.load(f)
n_classes = len(classes)


# ======================================================
# Load ALL Kardia 6L files (external validation)
# ======================================================
print("\n📂 Loading ALL Kardia 6L files (external validation)...")
label_map = load_label_map()
print(f"  Labeled files : {len(label_map)}")

# Extract all (idempotent: uses cache)
records = extract_and_cache(label_map, force=args.force_extract)
if not records:
    print("❌ No records extracted. Check PDF paths and kardia_labels.csv.")
    sys.exit(1)

X_all, y_all, pids = _records_to_arrays(records)
print(f"  Total samples : {len(X_all)}")

from collections import Counter
dist = Counter(r["label"] for r in records)
print("  Class distribution:")
for cls in TARGET_CLASSES:
    n = dist.get(cls, 0)
    print(f"    {cls:35s}: {n:3d}  ({100*n/len(records):.1f}%)")


# ======================================================
# Feature extraction
# ======================================================
print("\n🔬 Extracting ML features...")
X_ml = np.array(
    Parallel(n_jobs=-1)(delayed(extract_ecg_features)(s, fs=100) for s in X_all),
    dtype=np.float32
)
X_ml = np.nan_to_num(X_ml, nan=0.0, posinf=0.0, neginf=0.0)
X_1d = X_all[..., np.newaxis]

print("🌈 Computing spectrograms...")
X_spec = batch_to_spectrograms(X_all, fs=100)


# ======================================================
# Load models
# ======================================================
print("\n⬇️  Loading trained models...")
ml_models, dl_models = {}, {}

for name in ["KNN", "SVM", "RandomForest", "XGBoost"]:
    path = os.path.join(RUN_DIR, f"{name}.joblib")
    if os.path.isfile(path):
        ml_models[name] = joblib.load(path)
        print(f"  ✅ {name}")

for name in ["cnn1d", "cnn2d"]:
    path = os.path.join(RUN_DIR, f"{name}.keras")
    if os.path.isfile(path):
        dl_models[name] = tf.keras.models.load_model(path, safe_mode=False)
        print(f"  ✅ {name}")

weights = {}
scores_path = os.path.join(RUN_DIR, "scores.json")
if os.path.isfile(scores_path):
    with open(scores_path) as f:
        s = json.load(f)
    weights = {**s.get("ml_scores", {}), **s.get("dl_scores", {})}

hybrid = HybridEnsemble(
    ml_models=ml_models,
    dl_models=dl_models,
    classes=classes,
    weights=weights,
)


# ======================================================
# Predict
# ======================================================
print(f"\n🔮 Running ensemble predictions on {len(X_all)} Kardia samples...")
probs  = hybrid.predict_proba(X_ml, X_1d, X_spec=X_spec)
y_pred = np.argmax(probs, axis=1)


# ======================================================
# Results
# ======================================================
acc  = accuracy_score(y_all, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(
    y_all, y_pred, average="macro", zero_division=0
)

print(f"\n{'='*62}")
print(f"  📊 TABLE 2 — EXTERNAL KARDIA 6L VALIDATION (N={len(y_all)})")
print(f"{'='*62}")
print(f"  Accuracy  : {acc:.4f}  ({acc:.1%})")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  Macro F1  : {f1:.4f}")

print(f"\n  Per-sample breakdown:")
for pid, gt, pred in zip(pids, y_all, y_pred):
    mark = "✅" if gt == pred else "❌"
    print(f"    {mark} {pid:22s}: GT={classes[gt]:25s}  Pred={classes[pred]}")

print("\n📊 Classification Report (Kardia External):")
print(classification_report(y_all, y_pred,
                            target_names=classes,
                            labels=np.arange(n_classes),
                            zero_division=0))

cm = confusion_matrix(y_all, y_pred)
print("📊 Confusion Matrix:")
header = " " * 26 + "  ".join(f"{c[:6]:>6}" for c in classes)
print(header)
for i, row in enumerate(cm):
    print(f"  {classes[i]:24s}  " + "  ".join(f"{v:>6}" for v in row))


# ======================================================
# Save results
# ======================================================
out = {
    "external_validation": "Kardia 6L",
    "n_samples":   int(len(y_all)),
    "accuracy":    float(acc),
    "precision":   float(prec),
    "recall":      float(rec),
    "macro_f1":    float(f1),
    "class_dist":  {k: int(v) for k, v in dist.items()},
    "per_sample":  [
        {
            "patient":  str(p),
            "gt":       classes[int(g)],
            "pred":     classes[int(pr)],
            "correct":  bool(g == pr),
            "confidence": float(np.max(probs[i])),
        }
        for i, (p, g, pr) in enumerate(zip(pids, y_all, y_pred))
    ],
}

out_path = os.path.join(RUN_DIR, "kardia_external_eval.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"\n✅ Results saved to: {out_path}")
print(f"\n{'='*62}")
print(f"  SUMMARY")
print(f"{'='*62}")
print(f"  Digital test (MIT+PTB)  → see scores.json")
print(f"  Kardia external         → Acc={acc:.4f}  F1={f1:.4f}  N={len(y_all)}")
