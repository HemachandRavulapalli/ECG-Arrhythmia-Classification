# backend/src/ecg_api.py

import os
import sys
import tempfile
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException

# -------------------------------------------------
# Ensure backend/src is in Python path
# -------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(BASE_DIR, "backend", "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from predict_ecg import predict_ecg

# -------------------------------------------------
# FastAPI app
# -------------------------------------------------
app = FastAPI(
    title="ECG Arrhythmia Classification API",
    description="Hybrid ML + DL ECG classification using PTB-XL, MIT-BIH, and Kardia data",
    version="1.0"
)

# -------------------------------------------------
# API endpoint
# -------------------------------------------------
@app.post("/analyze-ecg")
async def analyze_ecg(file: UploadFile = File(...)):
    temp_file_path = None

    # Validate file type early
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".pdf", ".png", ".jpg", ".jpeg"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Upload ECG PDF or image."
        )

    try:
        # Save uploaded file securely
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        # Run ECG prediction
        result = predict_ecg(temp_file_path)
        return result

    except ValueError as e:
        # Known validation / ECG extraction errors
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Unexpected internal errors
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

    finally:
        file.file.close()
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
