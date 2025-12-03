#!/usr/bin/env python3
"""
FastAPI Backend for ECG Classification
Handles PDF and image file uploads for ECG analysis
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Try to import predict_ecg, but handle missing dependencies gracefully
try:
    from predict_ecg import predict_ecg
    PREDICTION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Cannot import predict_ecg: {e}")
    print("⚠️ Some dependencies may be missing. The API will start but /predict will return errors.")
    PREDICTION_AVAILABLE = False
    predict_ecg = None

app = FastAPI(title="ECG Classification API", version="1.0.0")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Welcome to ECG Classification API 🚀",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Upload ECG file (PDF or image) for classification"
        }
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "ECG Classification API"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload and classify an ECG file (PDF or image)
    
    Returns:
        - predicted_class: The classified cardiac condition
        - confidence: Confidence score (0-1)
        - probabilities: Confidence scores for all classes
    """
    if not PREDICTION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Prediction service unavailable. Missing dependencies. Please install all required packages from requirements.txt"
        )
    
    # Validate file type
    allowed_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_extension}. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file temporarily
    temp_file = None
    try:
        # Create temp file
        suffix = Path(file.filename).suffix
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        
        # Write uploaded content to temp file
        content = await file.read()
        temp_file.write(content)
        temp_file.close()
        
        # Run prediction
        print(f"📄 Processing file: {file.filename}")
        result = predict_ecg(temp_file.name)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
