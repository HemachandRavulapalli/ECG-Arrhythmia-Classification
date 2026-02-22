#!/usr/bin/env python3
"""
ptbxl_preprocess.py — PTB-XL Database Preprocessor (Research-Grade)

Correct implementation:
  1. Load ptbxl_database.csv  (contains scp_codes + patient_id)
  2. Map scp_codes → one of 5 classes  (skip ambiguous records)
  3. Load filename_lr  →  already 100 Hz, 12 leads, (1000, 12)
  4. PCA aggregate (12 leads → 1)
  5. Apply canonical preprocess_pipeline (fs=100)
  6. Save (1000,) + label + patient_id
"""

import os
import sys
import ast
import json
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocessing import aggregate_leads, preprocess_pipeline, extract_window

# ======================================================
# Paths
# ======================================================
BASE_DIR     = os.path.abspath(os.path.join(SRC_DIR, "..", ".."))
PTB_RAW_DIR  = os.path.join(BASE_DIR, "data", "raw",       "ptbxl")
PTB_SAVE_DIR = os.path.join(BASE_DIR, "data", "processed", "ptbxl")
os.makedirs(PTB_SAVE_DIR, exist_ok=True)

# ======================================================
# Constants
# ======================================================
TARGET_FS   = 100
WINDOW_SIZE = 1000   # filename_lr records are exactly 1000 samples at 100 Hz

# ======================================================
# SCP Code → Class Mapping
# Priority order matters: VA > AF > Brady > Tachy > Normal
# ======================================================
SCP_MAP = {
    # Ventricular Arrhythmias
    "VEB":  "Ventricular Arrhythmias",
    "VT":   "Ventricular Arrhythmias",
    "PVC":  "Ventricular Arrhythmias",
    "BIGU": "Ventricular Arrhythmias",
    "TRIGU":"Ventricular Arrhythmias",

    # Atrial Fibrillation / Flutter
    "AFIB": "Atrial Fibrillation",
    "AFLT": "Atrial Fibrillation",

    # Bradycardia
    "SBRAD": "Bradycardia",
    "PACE":  "Bradycardia",

    # Tachycardia
    "STACH": "Tachycardia",
    "SVTAC": "Tachycardia",
    "PSVT":  "Tachycardia",

    # Normal Sinus Rhythm
    "NORM":  "Normal Sinus Rhythm",
    "SR":    "Normal Sinus Rhythm",
}

PRIORITY_ORDER = [
    "Ventricular Arrhythmias",
    "Atrial Fibrillation",
    "Bradycardia",
    "Tachycardia",
    "Normal Sinus Rhythm",
]


# ======================================================
# SCP Code Parser
# ======================================================
def parse_scp_codes(raw: str) -> dict:
    """
    Parse the scp_codes column.
    PTB-XL stores it as a Python-dict literal, e.g.
      {'NORM': 100.0, 'SR': 0.0}
    """
    try:
        return ast.literal_eval(raw)
    except Exception:
        try:
            return json.loads(raw.replace("'", '"'))
        except Exception:
            return {}


def map_label(scp_codes: dict) -> str | None:
    """
    Return the highest-priority class present in scp_codes.
    Returns None if no known class found (ambiguous / unlabelled).
    """
    found_classes = set()
    for code in scp_codes:
        if code in SCP_MAP:
            found_classes.add(SCP_MAP[code])

    for cls in PRIORITY_ORDER:
        if cls in found_classes:
            return cls
    return None


# ======================================================
# Worker: process one PTB-XL record
# ======================================================
def process_ptbxl_record(record_path: str, patient_id: str,
                          label: str) -> dict | None:
    """
    Load a single PTB-XL filename_lr record, aggregate leads,
    preprocess, and return a dict ready for saving.
    Returns None on failure.
    """
    try:
        rec    = wfdb.rdrecord(record_path)
        signal = rec.p_signal.astype(np.float64)   # (1000, 12)
    except Exception as e:
        return None

    # Guard: must have at least 1000 samples
    if signal.shape[0] < WINDOW_SIZE:
        extra = np.zeros((WINDOW_SIZE - signal.shape[0], signal.shape[1]))
        signal = np.vstack([signal, extra])

    # Take exactly WINDOW_SIZE samples (should already be 1000)
    signal = signal[:WINDOW_SIZE]                  # (1000, 12)

    # PCA aggregate → (1000,)
    signal_1d = aggregate_leads(signal)

    # Ensure exact window size
    signal_1d = extract_window(signal_1d, WINDOW_SIZE)

    # Canonical preprocessing (fs already 100)
    signal_1d = preprocess_pipeline(signal_1d, fs=TARGET_FS)

    return {
        "signal":     signal_1d,
        "label":      label,
        "patient_id": str(patient_id),
    }


# ======================================================
# Runner
# ======================================================
def process_ptbxl(limit: int = None, verbose: bool = False) -> None:
    """
    Process PTB-XL database and save NPZ files.
    Each record is uniquely identified by ecg_id.
    Patient ID is stored for group-split during training.
    """
    db_path = os.path.join(PTB_RAW_DIR, "ptbxl_database.csv")
    if not os.path.exists(db_path):
        print(f"❌ ptbxl_database.csv not found at {db_path}")
        return

    df = pd.read_csv(db_path)
    print(f"📊 PTB-XL records in CSV: {len(df)}")

    saved_count = 0
    skipped     = 0
    label_dist  = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="PTB-XL", unit="rec"):

        # ── Parse label ────────────────────────────────
        scp_codes = parse_scp_codes(row["scp_codes"])
        label     = map_label(scp_codes)
        if label is None:
            continue

        # ── Build record path (filename_lr, no extension) ─
        record_rel  = row["filename_lr"]              # e.g. records100/00001/00001_lr
        record_path = os.path.join(PTB_RAW_DIR, record_rel)

        if not os.path.exists(record_path + ".dat"):
            if verbose:
                print(f"  ⚠️  Missing .dat: {record_path}")
            continue

        # ── Build save path ────────────────────────────
        ecg_id    = int(row["ecg_id"])
        save_path = os.path.join(PTB_SAVE_DIR, f"{ecg_id:06d}.npz")

        if os.path.exists(save_path):
            skipped += 1
            continue

        # ── Patient ID for group-split ─────────────────
        patient_id = str(int(row["patient_id"])) if "patient_id" in row else str(ecg_id)

        # ── Process ───────────────────────────────────
        result = process_ptbxl_record(record_path, patient_id, label)
        if result is None:
            if verbose:
                print(f"  ⚠️  Failed to process ecg_id={ecg_id}")
            continue

        # ── Save ──────────────────────────────────────
        np.savez_compressed(
            save_path,
            signal     = result["signal"],
            label      = result["label"],
            patient_id = result["patient_id"],
            fs         = TARGET_FS,
            dataset    = "PTB-XL",
        )
        saved_count += 1
        label_dist[label] = label_dist.get(label, 0) + 1

        if limit and saved_count >= limit:
            break

    print(f"\n✅ PTB-XL done. Saved: {saved_count}  Skipped (existing): {skipped}")
    print("   Label distribution:", label_dist)


# ======================================================
# CLI
# ======================================================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser("PTB-XL Preprocessor")
    p.add_argument("--limit",   type=int, default=None, help="Max records to save")
    p.add_argument("--verbose", action="store_true",     help="Print per-record info")
    a = p.parse_args()
    process_ptbxl(limit=a.limit, verbose=a.verbose)
