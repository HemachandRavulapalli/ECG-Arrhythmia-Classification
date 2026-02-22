import os
import wfdb
import subprocess
import shutil
from pathlib import Path

# ------------------------
# Setup paths (top-level /data/raw)
# ------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # D4/
DATA_DIR = BASE_DIR / "data" / "raw"
MITDB_DIR = DATA_DIR / "mitdb"
PTBXL_DIR = DATA_DIR / "ptbxl"

# Ensure folders exist
MITDB_DIR.mkdir(parents=True, exist_ok=True)
PTBXL_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------
# MIT-BIH Arrhythmia Database Downloader
# ------------------------
def download_mitdb():
    print("📥 Downloading MIT-BIH dataset...")
    try:
        wfdb.dl_database("mitdb", dl_dir=str(MITDB_DIR))
        print(f"✅ MIT-BIH downloaded to {MITDB_DIR}")
    except Exception as e:
        print(f"❌ Error downloading MITDB: {e}")

# ------------------------
# PTB-XL ECG Dataset Downloader
# ------------------------
def download_ptbxl():
    url = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
    zip_path = PTBXL_DIR / "ptbxl.zip"

    if not zip_path.exists():
        print("📥 Downloading PTB-XL dataset with wget...")
        subprocess.run(["wget", "-c", url, "-O", str(zip_path)], check=True)
    else:
        print("✅ PTB-XL zip already exists, skipping download")

    print("📂 Extracting PTB-XL dataset...")
    subprocess.run(["python3", "-m", "zipfile", "-e", str(zip_path), str(PTBXL_DIR)], check=True)

    # Find the nested folder
    nested_dir = next(PTBXL_DIR.glob("ptb-xl-*"), None)
    if nested_dir and nested_dir.is_dir():
        print(f"📂 Moving files up from {nested_dir} → {PTBXL_DIR}")
        for item in nested_dir.iterdir():
            target = PTBXL_DIR / item.name
            if target.exists():
                continue  # don’t overwrite if already exists
            shutil.move(str(item), str(PTBXL_DIR))
        shutil.rmtree(nested_dir)  # delete empty nested folder

    print(f"✅ PTB-XL ready at {PTBXL_DIR}")

if __name__ == "__main__":
    download_mitdb()
    download_ptbxl()
    print("✅ All datasets downloaded and organized.")