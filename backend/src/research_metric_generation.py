#!/usr/bin/env python3
"""
research_metric_generation.py — Academic Deliverables Generator (Scientific Revision)

Generates all 9 tables and figures requested for the research paper.
TARGET: Digital Test Set (MIT-BIH + PTB-XL) ONLY.
Kardia data is excluded to maintain statistical rigor.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                             confusion_matrix, roc_curve, auc, classification_report)
from sklearn.preprocessing import label_binarize
import joblib
import tensorflow as tf

# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Path setup
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(SRC_DIR)
OUTPUT_DIR = os.path.join(SRC_DIR, "research_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, SRC_DIR)

from data_loader import load_all_datasets, TARGET_CLASSES
from hybrid_model import HybridEnsemble
from cnn_models import batch_to_spectrograms
from feature_extraction import extract_ecg_features, FEATURE_NAMES
from joblib import Parallel, delayed

# ======================================================
# 1. Config & Run Selection
# ======================================================
MODEL_DIR = os.path.join(SRC_DIR, "saved_models")

def get_latest_run():
    runs = sorted([os.path.join(MODEL_DIR, d) for d in os.listdir(MODEL_DIR) if d.startswith("run_")],
                  key=os.path.getmtime)
    return runs[-1] if runs else None

RUN_DIR = get_latest_run()
if not RUN_DIR:
    print("❌ No run folder found.")
    sys.exit(1)

print(f"📊 Generating Scientific Metrics from: {RUN_DIR}")

# ======================================================
# 2. Data Loading (Digital Test Set ONLY)
# ======================================================
print("📥 Loading Digital Data (Patient-wise Split)...")
# We load a representative slice for metrics generation
_, test_digital, classes, _ = load_all_datasets(limit=5000, include_kardia=False)

X_dig_1d = test_digital["X_1d"]
X_dig_ml = test_digital["X_ml"]
X_dig_spec = test_digital["X_spec"]
y_dig = test_digital["y"]

# ======================================================
# 3. Model Loading
# ======================================================
print("⬇️ Loading models from run folder...")
ml_models, dl_models = {}, {}
for name in ["KNN", "SVM", "RandomForest", "XGBoost"]:
    path = os.path.join(RUN_DIR, f"{name}.joblib")
    if os.path.exists(path): ml_models[name] = joblib.load(path)

for name in ["cnn1d", "cnn2d"]:
    path = os.path.join(RUN_DIR, f"{name}.keras")
    if os.path.exists(path): dl_models[name] = tf.keras.models.load_model(path, safe_mode=False)

scores_path = os.path.join(RUN_DIR, "scores.json")
with open(scores_path) as f:
    s = json.load(f)
weights = {**s.get("ml_scores", {}), **s.get("dl_scores", {})}

ensemble = HybridEnsemble(ml_models=ml_models, dl_models=dl_models, classes=classes, weights=weights)

# ======================================================
# 4. Inference
# ======================================================
print(f"🔮 Running ensemble inference on {len(y_dig)} digital test samples...")
probs_dig = ensemble.predict_proba(X_dig_ml, X_dig_1d, X_spec=X_dig_spec)
y_pred_dig = np.argmax(probs_dig, axis=1)

# ======================================================
# 5. GENERATE DELIVERABLES
# ======================================================

def save_fig(name):
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  🖼️ Saved: {name}.png")

def calculate_specificity(y_true, y_pred, n_classes):
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    specs = []
    for i in range(n_classes):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        specs.append(tn / (tn + fp + 1e-8))
    return np.mean(specs)

def calculate_metrics_per_class(y_true, probs, n_classes):
    y_pred = np.argmax(probs, axis=1)
    y_bin = label_binarize(y_true, classes=range(n_classes))
    
    metrics = []
    for i in range(n_classes):
        # Accuracy for class i (one-vs-rest)
        acc = accuracy_score(y_true == i, y_pred == i)
        
        p, r, f1, _ = precision_recall_fscore_support(y_true == i, y_pred == i, average='binary', zero_division=0)
        
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
        except:
            roc_auc = 0.5
            
        metrics.append({
            "Class": classes[i],
            "Accuracy": acc,
            "Precision": p,
            "Recall": r,
            "F1-Score": f1,
            "AUC": roc_auc
        })
    
    # Micro Avg
    p_mic, r_mic, f1_mic, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
    # Simple micro accuracy is same as overall accuracy
    acc_mic = accuracy_score(y_true, y_pred)
    # Average AUC as micro-repr
    auc_mic = np.mean([m["AUC"] for m in metrics])
    
    metrics.append({
        "Class": "Micro Avg",
        "Accuracy": acc_mic,
        "Precision": p_mic,
        "Recall": r_mic,
        "F1-Score": f1_mic,
        "AUC": auc_mic
    })
    
    return pd.DataFrame(metrics).set_index("Class")

# --- 1. TABLE I. OVERALL PERFORMANCE METRICS ---
def generate_table_1():
    acc = accuracy_score(y_dig, y_pred_dig)
    p, r, f1, _ = precision_recall_fscore_support(y_dig, y_pred_dig, average='macro')
    spec = calculate_specificity(y_dig, y_pred_dig, len(classes))
    
    y_bin = label_binarize(y_dig, classes=range(len(classes)))
    auc_scores = []
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs_dig[:, i])
        auc_scores.append(auc(fpr, tpr))
    auc_macro = np.mean(auc_scores)

    df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "AUC", "Specificity"],
        "Value": [acc, p, r, f1, auc_macro, spec]
    })
    df.to_csv(os.path.join(OUTPUT_DIR, "table_1_overall.csv"), index=False)
    print("\n✅ TABLE I Generated.")

# --- 2. TABLE II. ACCURACY UNDER SCENARIOS ---
def generate_table_2():
    report = classification_report(y_dig, y_pred_dig, output_dict=True)
    macro_acc = report['macro avg']['recall']
    weighted_acc = accuracy_score(y_dig, y_pred_dig)
    
    nsr_idx = classes.index("Normal Sinus Rhythm")
    af_idx  = classes.index("Atrial Fibrillation")
    mask = (y_dig == nsr_idx) | (y_dig == af_idx)
    screening_acc = accuracy_score(y_dig[mask], y_pred_dig[mask])

    df = pd.DataFrame([
        {"Scenario": "Macro Accuracy", "Value": macro_acc},
        {"Scenario": "Weighted Accuracy", "Value": weighted_acc},
        {"Scenario": "Screening Accuracy (NSR+AF)", "Value": screening_acc}
    ])
    df.to_csv(os.path.join(OUTPUT_DIR, "table_2_scenarios.csv"), index=False)
    print("✅ TABLE II Generated.")

# --- 3 & 4. PER-CLASS PERFORMANCE BAR CHART ---
def generate_table_3_fig_2():
    # Mapping for shorter labels in plot
    short_classes = {
        "Normal Sinus Rhythm": "NSR",
        "Atrial Fibrillation": "AF",
        "Bradycardia": "Bradycardia",
        "Tachycardia": "Tachycardia",
        "Ventricular Arrhythmias": "VA",
        "Micro Avg": "Micro Avg"
    }
    
    df = calculate_metrics_per_class(y_dig, probs_dig, len(classes))
    df.to_csv(os.path.join(OUTPUT_DIR, "table_3_per_class.csv"))
    
    plot_df = df.copy()
    plot_df.index = [short_classes.get(i, i) for i in plot_df.index]
    
    ax = (plot_df * 100).plot(kind='bar', figsize=(14, 8), color=['red', 'orange', 'silver', 'indigo', 'dodgerblue'], width=0.85)
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.1f}%", 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', xytext=(0, 3), 
                        textcoords='offset points', fontsize=8, rotation=90)

    plt.title("Per Class Evaluation Metrics")
    plt.ylabel("Performance (%)")
    plt.ylim(0, 120)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='-', alpha=0.3)
    
    # Custom Legend
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5, frameon=False)
    
    save_fig("fig_2_per_class")
    print("✅ TABLE III & Fig 2 Generated.")

# --- 5. NORMALIZED CONFUSION MATRIX ---
def generate_fig_3():
    cm = confusion_matrix(y_dig, y_pred_dig, normalize='true')
    plt.figure(figsize=(10, 8))
    im = plt.imshow(cm, cmap="Blues", interpolation='nearest')
    plt.colorbar(im, label="Proportion")
    
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if cm[i, j] > 0.5 else "black"
            plt.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color=color)

    plt.title("Normalized Confusion Matrix for ECG Classification")
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    plt.yticks(range(len(classes)), classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    save_fig("fig_3_confusion_matrix")
    print("✅ Fig 3 Generated.")

# --- 6. FEATURE IMPORTANCE ---
def generate_table_4():
    rf = ml_models.get("RandomForest")
    if rf:
        importances = rf.feature_importances_
        df = pd.DataFrame({"Feature": FEATURE_NAMES, "Importance": importances})
        df = df.sort_values(by="Importance", ascending=False).head(10)
        df.to_csv(os.path.join(OUTPUT_DIR, "table_4_features.csv"), index=False)
    print("✅ TABLE IV Generated.")

# --- 7. CONFIDENCE BOXPLOT (Per Class Correct) ---
def generate_fig_4():
    conf_scores = np.max(probs_dig, axis=1)
    correct_mask = (y_dig == y_pred_dig)
    
    data_to_plot = [conf_scores[correct_mask & (y_dig == i)] for i in range(len(classes))]
    
    plt.figure(figsize=(12, 8))
    # Colors matching the reference image style
    colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3']
    
    bp = plt.boxplot(data_to_plot, patch_artist=True, widths=0.7)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor('gray')
    
    for whisker in bp['whiskers']:
        whisker.set(color='gray', linewidth=1)
    for cap in bp['caps']:
        cap.set(color='gray', linewidth=1)
    for median in bp['medians']:
        median.set(color='black', linewidth=1)

    # Add Jitter/Stripplot manually
    for i, data in enumerate(data_to_plot):
        if len(data) > 0:
            x = np.random.normal(i + 1, 0.04, size=len(data))
            plt.scatter(x, data, alpha=0.3, s=2, color='gray')

    plt.title("Model Confidence (Max Probability) for Correct Predictions")
    plt.ylabel("Confidence Score")
    plt.xlabel("ECG Arrhythmia Class")
    plt.ylim(0, 1.05)
    plt.xticks(range(1, len(classes) + 1), classes, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    save_fig("fig_4_confidence")
    print("✅ Fig 4 Generated.")

# --- 8. ROC CURVES ---
def generate_fig_5():
    y_bin = label_binarize(y_dig, classes=range(len(classes)))
    plt.figure(figsize=(10, 8))
    colors = ['tab:blue', 'tab:green', 'tab:brown', 'tab:gray', 'tab:cyan']
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs_dig[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, color=colors[i], label=f'{classes[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.2, linestyle='--')
    save_fig("fig_5_roc")
    print("✅ Fig 5 Generated.")

# --- 9. RADAR CHART ---
def generate_fig_6():
    report = classification_report(y_dig, y_pred_dig, target_names=classes, output_dict=True)
    
    metrics = ['precision', 'recall', 'f1-score']
    N = len(classes)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], classes, size=10)
    
    radar_colors = ['tab:blue', 'tab:orange', 'tab:green']
    for metric, color in zip(metrics, radar_colors):
        vals = [report[c][metric] for c in classes]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, linestyle='solid', label=metric.capitalize(), color=color)
        ax.fill(angles, vals, color, alpha=0.1)
    
    plt.title("Performance Metrics Radar Chart by Class")
    plt.legend(loc='lower left', bbox_to_anchor=(0.1, 0.1))
    save_fig("fig_6_radar")
    print("✅ Fig 6 Generated.")

# RUN ALL
generate_table_1()
generate_table_2()
generate_table_3_fig_2()
generate_fig_3()
generate_table_4()
generate_fig_4()
generate_fig_5()
generate_fig_6()

print(f"\n🎉 Scientific Deliverables (Digital ONLY) generated in {OUTPUT_DIR}.")
