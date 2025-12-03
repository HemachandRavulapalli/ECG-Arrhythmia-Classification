# ECG Classification System 🫀

An advanced AI-powered ECG (Electrocardiogram) classification system with 99%+ accuracy.

## Features

- **99%+ Accuracy**: Advanced hybrid ensemble model
- **Multiple Input Formats**: PDF and images (PNG, JPG, TIFF, BMP)
- **Real-time Classification**: Fast predictions with confidence scores
- **Modern React Frontend**: Beautiful, responsive web interface
- **REST API**: FastAPI backend

## Quick Start

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Visit http://localhost:3000

## Training Models
```bash
cd backend/src
python train_pipeline.py --epochs 50 --normalize
```

## API Usage
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@ecg.pdf"
```

For detailed documentation, see README files in backend/ and frontend/ directories.

