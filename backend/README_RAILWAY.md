# Railway Deployment - Backend Service

## Required Files (All Present ✅)

- `app.py` - Main FastAPI application
- `requirements.txt` - Python dependencies
- `Procfile` - Start command: `python app.py`
- `nixpacks.toml` - Nixpacks configuration
- `railway.json` - Railway service configuration

## Railway Configuration

### Root Directory
**MUST be set to:** `backend`

### Environment Variables
- `PORT` - Automatically set by Railway (no need to add manually)

### Build Process
1. Railway detects Python from `requirements.txt`
2. Runs `pip install -r requirements.txt`
3. Starts with `python app.py` (from Procfile)

## Troubleshooting

If build fails:
1. Check Root Directory is set to `backend` in Railway Settings
2. Verify `requirements.txt` exists and is valid
3. Check build logs in Railway dashboard
4. Ensure all dependencies in requirements.txt are installable

