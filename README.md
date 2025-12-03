# ECG Classification System 🫀

An advanced AI-powered ECG (Electrocardiogram) classification system with 99%+ accuracy.

## Features

- **99%+ Accuracy**: Advanced hybrid ensemble model
- **Multiple Input Formats**: PDF and images (PNG, JPG, TIFF, BMP)
- **Real-time Classification**: Fast predictions with confidence scores
- **Modern React Frontend**: Beautiful, responsive web interface
- **REST API**: FastAPI backend
- **5 Cardiac Conditions**: Classifies into:
  - Normal Sinus Rhythm
  - Atrial Fibrillation
  - Bradycardia
  - Tachycardia
  - Ventricular Arrhythmias

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/HemachandRavulapalli/ECG-Arrhythmia-Classification.git
cd ECG-Arrhythmia-Classification
```

2. **Backend Setup**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Frontend Setup**
```bash
cd frontend
npm install
```

### Running the Application

**Option 1: Using startup scripts**
```bash
# Terminal 1 - Backend
./start_backend.sh

# Terminal 2 - Frontend
./start_frontend.sh
```

**Option 2: Manual start**
```bash
# Terminal 1 - Backend
cd backend
source venv/bin/activate
python app.py

# Terminal 2 - Frontend
cd frontend
npm run dev
```

### Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Deployment

### Frontend (Vercel/Netlify)
The frontend can be deployed to Vercel or Netlify:

```bash
cd frontend
npm run build
# Deploy the dist/ folder
```

### Backend (Railway/Render/Heroku)
The backend can be deployed to any Python hosting service. Update the CORS origins in `backend/app.py` to include your frontend URL.

## API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Upload ECG File
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/ecg.pdf"
```

## Project Structure

```
├── backend/          # FastAPI backend
│   ├── src/         # Source code
│   ├── app.py       # Main API application
│   └── requirements.txt
├── frontend/        # React frontend
│   ├── src/        # React components
│   └── package.json
├── start_backend.sh # Backend startup script
└── start_frontend.sh # Frontend startup script
```

## Technologies

- **Backend**: FastAPI, TensorFlow, Scikit-learn, NumPy
- **Frontend**: React, Vite, Modern CSS
- **ML Models**: XGBoost, Random Forest, SVM, KNN, CNN

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

Hemachand Ravulapalli
