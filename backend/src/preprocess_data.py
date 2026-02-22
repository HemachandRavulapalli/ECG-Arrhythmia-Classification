#!/usr/bin/env python3
"""
preprocess_data.py — Unified dataset preprocessor runner.

Calls:
  1. mit_preprocess.process_mitdb()
  2. ptbxl_preprocess.process_ptbxl()

Both produce NPZ files in data/processed/{mitdb,ptbxl}/ with fields:
  signal     : (1000,) float32
  label      : str  (one of TARGET_CLASSES)
  patient_id : str
  fs         : 100
  dataset    : 'MIT-BIH' | 'PTB-XL'
"""

import os
import sys
import argparse

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from mit_preprocess    import process_mitdb
from ptbxl_preprocess  import process_ptbxl


def main():
    parser = argparse.ArgumentParser("ECG Dataset Preprocessor")
    parser.add_argument("--mit_limit",   type=int, default=None,
                        help="Max MIT-BIH windows to save (default: all)")
    parser.add_argument("--ptbxl_limit", type=int, default=None,
                        help="Max PTB-XL records to save (default: all)")
    parser.add_argument("--skip_mit",    action="store_true",
                        help="Skip MIT-BIH preprocessing")
    parser.add_argument("--skip_ptbxl", action="store_true",
                        help="Skip PTB-XL preprocessing")
    parser.add_argument("--verbose",     action="store_true",
                        help="Print per-record details")
    args = parser.parse_args()

    if not args.skip_mit:
        print("\n" + "="*60)
        print("MIT-BIH Preprocessing")
        print("="*60)
        process_mitdb(limit=args.mit_limit, verbose=args.verbose)

    if not args.skip_ptbxl:
        print("\n" + "="*60)
        print("PTB-XL Preprocessing")
        print("="*60)
        process_ptbxl(limit=args.ptbxl_limit, verbose=args.verbose)

    print("\n🎉 Preprocessing completed.")


if __name__ == "__main__":
    main()
