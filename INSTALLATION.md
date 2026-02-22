# Installation Guide

## Prerequisites

- Python 3.8 or higher
- Node.js 16+ and npm
- Virtual environment (recommended)

## Step 1: Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install
```

## Step 3: Training Models (Optional)

To achieve 99% accuracy, train the models first:

```bash
cd backend/src
python train_pipeline.py --epochs 50 --normalize
```

This will create saved models in `backend/src/saved_models/`

## Step 4: Running the Application

### Option A: Using Start Scripts

**Terminal 1 - Backend:**
```bash
./start_backend.sh
```

**Terminal 2 - Frontend:**
```bash
./start_frontend.sh
```

### Option B: Manual Start

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate
python app.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

## Step 5: Access the Application

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Troubleshooting

### Python dependencies issues
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Node.js issues
```bash
npm cache clean --force
npm install
```

### Port already in use
- Backend: Edit `backend/app.py` and change port
- Frontend: Edit `frontend/vite.config.js` and change port

### Models not found
Run training first or ensure models exist in `backend/src/saved_models/`

## Testing

Test the API directly:
```bash
curl http://localhost:8000/health
```

Upload an ECG file:
```bash
curl -X POST http://localhost:8000/predict -F "file=@path/to/ecg.pdf"
```
