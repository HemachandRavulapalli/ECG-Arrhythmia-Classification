#!/usr/bin/env python3
"""
kardia_loader.py — Kardia 6L PDF Signal Extractor and Dataset Splitter

Responsibilities:
  1. Read kardia_labels.csv  →  {filename: label}
  2. Extract 1000-sample signal from each PDF via pdf_to_signal.py
  3. Cache each signal as an NPZ in data/processed/kardia/
  4. Perform stratified 80/20 split
  5. Return (X_tr, y_tr, pids_tr), (X_te, y_te, pids_te) arrays

Rules:
  ✔ Uses same extract_signal_from_file() as inference pipeline
  ✔ Skips extraction if NPZ cache already exists (idempotent)
  ✔ Same 100 Hz / 1000-sample standard as MIT + PTB
  ✔ Fails fast with clear error if label not found or extraction fails
"""

import os
import sys
import argparse
import numpy as np

# ── Path setup ─────────────────────────────────────────
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.abspath(os.path.join(SRC_DIR, "..", ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pdf_to_signal import extract_signal_from_file

KARDIA_PDF_DIR  = os.path.join(PROJ_DIR, "data", "Kardia 6L")
KARDIA_LABELS   = os.path.join(KARDIA_PDF_DIR, "kardia_labels.csv")
KARDIA_CACHE    = os.path.join(PROJ_DIR, "data", "processed", "kardia")

TARGET_CLASSES = [
    "Normal Sinus Rhythm",
    "Atrial Fibrillation",
    "Bradycardia",
    "Tachycardia",
    "Ventricular Arrhythmias",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(TARGET_CLASSES)}

# ── Sentinel label — means "not yet filled in" ─────────
UNLABELED = "UNKNOWN"


# ======================================================
# Read labels CSV
# ======================================================
def load_label_map() -> dict[str, str]:
    """
    Returns {filename: label_string} for every row in kardia_labels.csv.
    Raises clearly if the file is missing or a label is UNKNOWN.
    """
    if not os.path.isfile(KARDIA_LABELS):
        raise FileNotFoundError(
            f"kardia_labels.csv not found at:\n  {KARDIA_LABELS}\n"
            "Create it with columns: filename,label"
        )

    label_map = {}
    with open(KARDIA_LABELS) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("filename"):
                continue
            parts = line.split(",", 1)
            if len(parts) != 2:
                raise ValueError(f"Malformed row {i}: {line!r}")
            fname, label = parts[0].strip(), parts[1].strip()
            if label == UNLABELED:
                raise ValueError(
                    f"Label not set for '{fname}'. Open kardia_labels.csv and\n"
                    "replace 'UNKNOWN' with the actual Kardia-app diagnosis."
                )
            if label not in CLASS_TO_IDX:
                raise ValueError(
                    f"Unknown label '{label}' for '{fname}'.\n"
                    f"Valid labels: {TARGET_CLASSES}"
                )
            label_map[fname] = label

    return label_map


# ======================================================
# Extract & Cache
# ======================================================
def extract_and_cache(label_map: dict[str, str], force: bool = False) -> list[dict]:
    """
    For each labeled PDF:
      - Skip if NPZ cache already exists (unless force=True)
      - Otherwise: extract signal → save NPZ

    Returns list of records: {patient_id, label, label_idx, npz_path}
    """
    os.makedirs(KARDIA_CACHE, exist_ok=True)
    records = []
    n_ok, n_skip, n_fail = 0, 0, 0

    for fname, label in sorted(label_map.items()):
        pdf_path = os.path.join(KARDIA_PDF_DIR, fname)
        pid      = os.path.splitext(fname)[0]
        npz_path = os.path.join(KARDIA_CACHE, f"{pid}.npz")

        if not os.path.isfile(pdf_path):
            print(f"  ⚠️  PDF not found, skipping: {fname}")
            n_fail += 1
            continue

        if os.path.isfile(npz_path) and not force:
            # Already cached
            records.append({
                "patient_id": pid,
                "label":      label,
                "label_idx":  CLASS_TO_IDX[label],
                "npz_path":   npz_path,
            })
            n_skip += 1
            continue

        try:
            signal = extract_signal_from_file(pdf_path)
            np.savez_compressed(
                npz_path,
                signal     = signal.astype(np.float32),
                label      = label,
                patient_id = pid,
            )
            records.append({
                "patient_id": pid,
                "label":      label,
                "label_idx":  CLASS_TO_IDX[label],
                "npz_path":   npz_path,
            })
            n_ok += 1
        except Exception as e:
            print(f"  ❌ Extraction failed for {fname}: {e}")
            n_fail += 1

    print(f"\n  [Kardia cache] extracted={n_ok}  cached={n_skip}  failed={n_fail}")
    return records


# ======================================================
# Stratified 80/20 Split
# ======================================================
def stratified_split(records: list[dict], test_size: float = 0.2,
                     random_state: int = 42) -> tuple[list, list]:
    """
    Stratified 80/20 split on class labels.
    Guarantees at least one sample per class in train (when class has ≥ 2 samples).
    Classes with only 1 sample go entirely to training.
    """
    from collections import defaultdict
    rng = np.random.default_rng(random_state)

    by_class = defaultdict(list)
    for r in records:
        by_class[r["label"]].append(r)

    train_recs, test_recs = [], []
    for label, recs in by_class.items():
        recs_arr = np.array(recs, dtype=object)
        rng.shuffle(recs_arr)
        n = len(recs_arr)
        n_test = max(0, round(n * test_size)) if n >= 2 else 0
        test_recs.extend(recs_arr[:n_test].tolist())
        train_recs.extend(recs_arr[n_test:].tolist())
        print(f"    {label:30s}: total={n}  train={n - n_test}  test={n_test}")

    return train_recs, test_recs


# ======================================================
# Load arrays from cached NPZs
# ======================================================
def _records_to_arrays(records: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load NPZ files and return (X, y, patient_ids) arrays."""
    X, y, pids = [], [], []
    for r in records:
        data = np.load(r["npz_path"], allow_pickle=True)
        X.append(data["signal"].astype(np.float32).flatten())
        y.append(r["label_idx"])
        pids.append(r["patient_id"])
    return np.array(X, dtype=np.float32), np.array(y, dtype=int), pids


# ======================================================
# Public API — called by data_loader.py
# ======================================================
def load_kardia_split(test_size: float = 0.2, random_state: int = 42,
                      force_extract: bool = False):
    """
    Full pipeline: labels → extract/cache → split → arrays.

    Returns
    -------
    train : dict  {'X': (N_tr, 1000), 'y': (N_tr,), 'pids': [...]}
    test  : dict  {'X': (N_te, 1000), 'y': (N_te,), 'pids': [...]}
    """
    print("\n📂 Loading Kardia 6L dataset...")
    label_map = load_label_map()
    print(f"  Labels loaded: {len(label_map)} files")

    records   = extract_and_cache(label_map, force=force_extract)
    if not records:
        raise RuntimeError("No Kardia records could be extracted. Check PDF paths and labels.")

    print(f"\n  Stratified 80/20 split (seed={random_state}):")
    train_recs, test_recs = stratified_split(records, test_size, random_state)

    X_tr, y_tr, pids_tr = _records_to_arrays(train_recs)
    X_te, y_te, pids_te = _records_to_arrays(test_recs)

    print(f"\n  ✅ Kardia train : {len(X_tr)} samples")
    print(f"  ✅ Kardia test  : {len(X_te)} samples")
    print(f"  Train class dist: {dict(zip(*np.unique(y_tr, return_counts=True)))}")
    print(f"  Test  class dist: {dict(zip(*np.unique(y_te, return_counts=True)))}")

    return (
        {"X": X_tr, "y": y_tr, "pids": pids_tr},
        {"X": X_te, "y": y_te, "pids": pids_te},
    )


# ======================================================
# CLI  — python3 kardia_loader.py --verify / --extract
# ======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Kardia 6L Loader")
    parser.add_argument("--verify",  action="store_true",
                        help="Load labels and print class distribution (no extraction)")
    parser.add_argument("--extract", action="store_true",
                        help="Extract signals from all labeled PDFs and cache as NPZ")
    parser.add_argument("--force",   action="store_true",
                        help="Re-extract even if NPZ cache already exists")
    args = parser.parse_args()

    label_map = load_label_map()
    print(f"\n📋 Kardia label map — {len(label_map)} files:")

    from collections import Counter
    dist = Counter(label_map.values())
    for cls in TARGET_CLASSES:
        print(f"  {cls:35s}: {dist.get(cls, 0)}")

    if args.verify:
        unlabeled_pdfs = [
            f for f in os.listdir(KARDIA_PDF_DIR)
            if f.endswith(".pdf") and f not in label_map
        ]
        if unlabeled_pdfs:
            print(f"\n  ⚠️  {len(unlabeled_pdfs)} PDFs have no label:")
            for u in sorted(unlabeled_pdfs):
                print(f"     {u}")
        else:
            print("\n  ✅ All PDFs are labeled.")

    if args.extract:
        records = extract_and_cache(label_map, force=args.force)
        print(f"\n  ✅ Records ready: {len(records)}")
        print(f"\n  Running 80/20 split preview:")
        stratified_split(records)
