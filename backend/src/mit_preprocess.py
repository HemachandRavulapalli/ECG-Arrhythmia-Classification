#!/usr/bin/env python3
"""
mit_preprocess.py — MIT-BIH Arrhythmia Database Preprocessor (Research-Grade)

Correct implementation:
  1. Read full 2-lead (N, 2) signal at 360 Hz
  2. Resample entire record to 100 Hz
  3. Scale beat positions: pos_new = pos_old * (100 / 360)
  4. Map beat symbols → ECG class (Brady/Tachy via RR intervals)
  5. Extract 1000-sample windows centred on each beat
  6. PCA aggregate (2 leads → 1)
  7. Apply canonical preprocess_pipeline (at fs=100)
  8. Save (1000,) + label + patient_id
"""

import os
import sys
import numpy as np
import wfdb
from tqdm import tqdm

# Resolve paths so this script can be run from any CWD
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocessing import resample_signal, aggregate_leads, preprocess_pipeline, extract_window

# ======================================================
# Paths
# ======================================================
BASE_DIR = os.path.abspath(os.path.join(SRC_DIR, "..", ".."))
MIT_RAW_DIR  = os.path.join(BASE_DIR, "data", "raw",       "mitdb")
MIT_SAVE_DIR = os.path.join(BASE_DIR, "data", "processed", "mitdb")
os.makedirs(MIT_SAVE_DIR, exist_ok=True)

# ======================================================
# Constants
# ======================================================
TARGET_FS   = 100
WINDOW_SIZE = 1000
HALF        = WINDOW_SIZE // 2       # 500 samples on each side of beat

# ======================================================
# Beat Symbol Mapping (symbol → base class)
# ======================================================
MIT_SYMBOL_MAP = {
    # Normal beats
    "N": "Normal Sinus Rhythm",
    "L": "Normal Sinus Rhythm",   # Left bundle branch block
    "R": "Normal Sinus Rhythm",   # Right bundle branch block
    "e": "Normal Sinus Rhythm",   # Atrial escape
    "j": "Normal Sinus Rhythm",   # Nodal (junctional) escape

    # Atrial Fibrillation / Supraventricular
    "A": "Atrial Fibrillation",
    "a": "Atrial Fibrillation",   # Aberrant atrial
    "S": "Atrial Fibrillation",   # Supraventricular premature

    # Ventricular
    "V": "Ventricular Arrhythmias",
    "E": "Ventricular Arrhythmias",  # Ventricular escape
    "F": "Ventricular Arrhythmias",  # Fusion of ventricular and normal
}

# ======================================================
# RR-Based Brady/Tachy Override
# ======================================================
RR_WINDOW = 5   # number of adjacent beats used to compute local HR

def rr_override(base_label: str, rr_hr: float) -> str:
    """
    Override Normal Sinus Rhythm with Brady/Tachy when HR is outside 60–100.
    AF and VA labels are never overridden.
    """
    if base_label not in ("Normal Sinus Rhythm",):
        return base_label
    if rr_hr < 60:
        return "Bradycardia"
    if rr_hr > 100:
        return "Tachycardia"
    return base_label


# ======================================================
# Worker: process one MIT-BIH record
# ======================================================
def process_mit_record(record_id: str, verbose: bool = False) -> list[dict]:
    """
    Process a single MIT-BIH record.
    Returns a list of dicts: {signal, label, patient_id}.
    """
    record_path = os.path.join(MIT_RAW_DIR, record_id)

    # ── 1. Read raw record ──────────────────────────────
    try:
        record = wfdb.rdrecord(record_path)
        ann    = wfdb.rdann(record_path, "atr")
    except Exception as e:
        if verbose:
            print(f"  ⚠️  Cannot read {record_id}: {e}")
        return []

    signal_raw   = record.p_signal          # (N, 2)  float64
    fs_orig      = float(record.fs)         # typically 360

    beat_pos_raw = ann.sample               # sample indices at original fs
    beat_syms    = ann.symbol

    # ── 2. Resample entire signal to 100 Hz ────────────
    signal_100 = resample_signal(signal_raw, fs_orig, TARGET_FS)  # (M, 2)

    # ── 3. Scale beat positions ────────────────────────
    scale = TARGET_FS / fs_orig
    beat_pos = (beat_pos_raw * scale).astype(int)

    n_samples = signal_100.shape[0]

    # ── 4. Compute per-beat HR from local RR intervals ─
    rr_intervals = np.diff(beat_pos) / TARGET_FS   # in seconds
    rr_hr_per_beat = np.full(len(beat_pos), np.nan)

    # assign each beat its local HR from the surrounding RR_WINDOW beats
    for i in range(len(beat_pos)):
        lo = max(0, i - RR_WINDOW // 2)
        hi = min(len(rr_intervals), i + RR_WINDOW // 2 + 1)
        local_rr = rr_intervals[lo:hi]
        if len(local_rr) > 0:
            rr_hr_per_beat[i] = 60.0 / np.mean(local_rr)

    # ── 5. Extract windows ─────────────────────────────
    samples = []
    for i, (pos, sym) in enumerate(zip(beat_pos, beat_syms)):
        if len(samples) >= 100:  # LIMIT beats per record to 100
            break

        if sym not in MIT_SYMBOL_MAP:
            continue

        start = pos - HALF
        end   = pos + HALF   # exclusive upper bound (Python slicing)

        # BUG 10 FIX: `end > n_samples` is correct.
        # signal[start:end] where end==n_samples yields exactly WINDOW_SIZE
        # rows, so end==n_samples must NOT be excluded.
        if start < 0 or end > n_samples:
            continue

        segment = signal_100[start:end]    # must be (1000, 2)

        # Hard shape assertion — catches any fs/scale drift immediately
        if segment.shape[0] != WINDOW_SIZE:
            continue

        # ── 6. PCA aggregation ─────────────────────────
        # BUG 2 ORDER: Resample (done above) → PCA → Preprocess
        segment_1d = aggregate_leads(segment)   # (1000,)

        # Enforce exact window (handles rare PCA output length drift)
        segment_1d = extract_window(segment_1d, WINDOW_SIZE)

        # ── 7. Preprocess (fs=100, identical to inference) ─
        # BUG 9 NOTE: normalization is per-signal z-score inside
        # preprocess_pipeline — NEVER global. Consistent with inference.
        segment_1d = preprocess_pipeline(segment_1d, fs=TARGET_FS)

        # Final length guard before save
        if len(segment_1d) != WINDOW_SIZE:
            continue

        # ── 8. Determine label ─────────────────────────
        base_label = MIT_SYMBOL_MAP[sym]
        hr_val     = rr_hr_per_beat[i]
        label      = rr_override(base_label, hr_val) if not np.isnan(hr_val) else base_label

        samples.append({
            "signal":     segment_1d,
            "label":      label,
            "patient_id": record_id,
        })

    return samples


# ======================================================
# Runner: iterate all MIT records
# ======================================================
def process_mitdb(limit: int = None, verbose: bool = False) -> None:
    """
    Process all MIT-BIH records and save NPZ files.
    """
    if not os.path.isdir(MIT_RAW_DIR):
        print(f"❌ MIT raw directory not found: {MIT_RAW_DIR}")
        return

    # Collect available record IDs
    record_ids = sorted({
        f.replace(".dat", "")
        for f in os.listdir(MIT_RAW_DIR)
        if f.endswith(".dat")
    })

    if not record_ids:
        print(f"❌ No .dat files found in {MIT_RAW_DIR}")
        return

    print(f"📊 MIT-BIH records found: {len(record_ids)}")

    saved_count = 0
    skipped     = 0
    label_dist  = {}

    for rec_id in tqdm(record_ids, desc="MIT-BIH", unit="record"):
        samples = process_mit_record(rec_id, verbose=verbose)

        for j, s in enumerate(samples):
            save_path = os.path.join(MIT_SAVE_DIR, f"{rec_id}_{j:05d}.npz")

            if os.path.exists(save_path):
                skipped += 1
                continue

            np.savez_compressed(
                save_path,
                signal     = s["signal"],
                label      = s["label"],
                patient_id = s["patient_id"],
                fs         = TARGET_FS,
                dataset    = "MIT-BIH",
            )
            saved_count  += 1
            label_dist[s["label"]] = label_dist.get(s["label"], 0) + 1

            if limit and saved_count >= limit:
                break

        if limit and saved_count >= limit:
            break

    print(f"\n✅ MIT-BIH done. Saved: {saved_count}  Skipped (existing): {skipped}")
    print("   Label distribution:", label_dist)


# ======================================================
# CLI
# ======================================================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser("MIT-BIH Preprocessor")
    p.add_argument("--limit",   type=int, default=None,  help="Max windows to save")
    p.add_argument("--verbose", action="store_true",      help="Print per-record info")
    a = p.parse_args()
    process_mitdb(limit=a.limit, verbose=a.verbose)
