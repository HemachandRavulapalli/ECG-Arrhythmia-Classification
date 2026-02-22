#!/usr/bin/env bash
# run_pipeline.sh — Full research-grade ECG pipeline helper
# Usage:
#   ./run_pipeline.sh preprocess   # Preprocess MIT-BIH + PTB-XL
#   ./run_pipeline.sh train        # Train all models
#   ./run_pipeline.sh predict <file>
#   ./run_pipeline.sh server       # Start FastAPI server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/backend/src"
BACKEND="$SCRIPT_DIR/backend"

case "$1" in
  preprocess)
    echo "=== Preprocessing MIT-BIH + PTB-XL ==="
    cd "$SRC"
    python3 preprocess_data.py "${@:2}"
    ;;

  mit)
    echo "=== MIT-BIH only ==="
    cd "$SRC"
    python3 mit_preprocess.py "${@:2}"
    ;;

  ptbxl)
    echo "=== PTB-XL only ==="
    cd "$SRC"
    python3 ptbxl_preprocess.py "${@:2}"
    ;;

  train)
    echo "=== Training Pipeline ==="
    cd "$SRC"
    python3 train_pipeline.py "${@:2}"
    ;;

  predict)
    if [ -z "$2" ]; then
      echo "Usage: $0 predict <ecg_file>"
      exit 1
    fi
    cd "$SRC"
    python3 predict_ecg.py "$2"
    ;;

  server)
    echo "=== Starting FastAPI server ==="
    cd "$BACKEND"
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
    ;;

  download)
    echo "=== Downloading MIT-BIH + PTB-XL ==="
    cd "$SRC"
    python3 dataset_downloader.py
    ;;

  install)
    echo "=== Installing dependencies ==="
    python3 -m pip install --break-system-packages -r "$BACKEND/requirements.txt"
    ;;

  *)
    echo "Commands:"
    echo "  install              Install Python dependencies"
    echo "  preprocess           Run MIT-BIH + PTB-XL preprocessing"
    echo "  mit [--limit N]      MIT-BIH only"
    echo "  ptbxl [--limit N]    PTB-XL only"
    echo "  train [options]      Full training pipeline"
    echo "    --limit N          Max samples per dataset"
    echo "    --epochs N         DL epochs (default 30)"
    echo "    --batch_size N     Batch size (default 32)"
    echo "    --no_smote         Disable SMOTE"
    echo "    --no_spec          Disable CNN2D spectrogram branch"
    echo "    --skip_adv         Skip AdvancedHybridModel"
    echo "    --resume           Resume from latest run"
    echo "  predict <file>       Predict from ECG PDF/image"
    echo "  server               Start FastAPI server"
    exit 1
    ;;
esac
