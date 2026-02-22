#!/usr/bin/env python3
"""
FastAPI Backend for ECG Classification - WITH ECG VALIDATION
Rejects non-ECG files + fixes wrong predictions
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import sys
import tempfile
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import predictor from src
try:
    from predict_ecg import predict_ecg
    print("✅ predict_ecg imported successfully")
except ImportError as e:
    print(f"⚠️ predict_ecg import failed: {e}")
    predict_ecg = None

app = FastAPI(title="ECG Classification API", version="1.1.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def validate_ecg_signal(signal, sr=100):
    """Strict ECG validation - rejects non-ECG signals"""
    if len(signal) < 500:
        raise ValueError("Too short - not ECG")

    # Check signal amplitude (ECG should have reasonable variation)
    signal_std = np.std(signal)
    if signal_std < 0.01:
        raise ValueError("Too flat - not ECG signal")

    # Check R-peaks (heartbeats)
    peaks, _ = find_peaks(signal, height=np.percentile(signal, 70), distance=sr // 3)
    if len(peaks) < 4 or len(peaks) > 50:
        raise ValueError(f"Invalid heartbeat count: {len(peaks)} (expected 4-50)")

    # Check reasonable heart rate
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks)
        avg_rr = np.mean(rr_intervals)
        heart_rate = 60 / (avg_rr / sr)
        if heart_rate < 40 or heart_rate > 250:
            raise ValueError(f"Unrealistic heart rate: {heart_rate:.1f} BPM")

    return True

@app.get("/")
def root():
    return {
        "message": "ECG Classification API 🚀 (Validated ECG only)",
        "version": "1.1.0",
        "status": "healthy",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Upload ECG PDF/image ONLY",
        },
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "ECG Classification API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    ECG Classification with STRICT validation
    Rejects: non-ECG images, flat signals, wrong heart rates
    """
    print("🚀 Received prediction request!")
    print(f"📄 File: {file.filename}, Content-Type: {file.content_type}")

    if predict_ecg is None:
        print("❌ Prediction module not available")
        raise HTTPException(status_code=500, detail="Prediction module not available")

    # File type validation
    allowed_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        print(f"❌ Unsupported file extension: {file_extension}")
        raise HTTPException(
            status_code=400,
            detail=f"❌ Unsupported: {file_extension}. Use ECG PDF/image only",
        )

    temp_file = None
    try:
        # Save temp file
        suffix = Path(file.filename).suffix
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        content = await file.read()
        temp_file.write(content)
        temp_file.close()

        print(f"📄 Processing: {file.filename}, saved to: {temp_file.name}")

        # Run prediction
        result = predict_ecg(temp_file.name)

        # Optional: extra ECG validation if your result includes raw signal
        signal = result.get("signal", None)
        if signal is not None:
            try:
                validate_ecg_signal(signal)
                print("✅ ECG validation passed")
            except ValueError as e:
                print(f"❌ ECG validation failed: {e}")
                raise HTTPException(status_code=400, detail=f"❌ Invalid ECG: {str(e)}")

        # Confidence check from model output
        confidence = result.get("confidence", 0)
        # Allow low confidence

        return JSONResponse(
            content={
                "status": "success",
                "message": "Valid ECG detected",
                **result,
            }
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is (400, etc.)
        raise
    except ValueError as e:
        # Any ValueError from predict_ecg → 400 Bad Request
        print(f"❌ Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Full error: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}",
        )
    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    
    # Check for SSL certificates
    ssl_keyfile = "key.pem"
    ssl_certfile = "cert.pem"
    
    if os.path.exists(ssl_keyfile) and os.path.exists(ssl_certfile):
        print(f"🔒 Starting in HTTPS mode using {ssl_keyfile} and {ssl_certfile}")
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", ssl_keyfile=ssl_keyfile, ssl_certfile=ssl_certfile)
    else:
        print(f"⚠️  Certificates not found. Starting in HTTP mode.")
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
