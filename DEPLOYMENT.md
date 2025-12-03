# Deployment Guide

This guide explains how to deploy the ECG Classification System to public hosting.

## Frontend Deployment (Vercel - Recommended)

### Option 1: Vercel CLI
```bash
cd frontend
npm install -g vercel
vercel
```

### Option 2: Vercel Dashboard
1. Go to [vercel.com](https://vercel.com)
2. Import your GitHub repository
3. Set root directory to `frontend`
4. Build command: `npm run build`
5. Output directory: `dist`
6. Deploy!

### Update API URL
After deploying, update the API URL in `frontend/src/App.jsx`:
```javascript
const response = await fetch('https://your-backend-url.railway.app/predict', {
  method: 'POST',
  body: formData,
});
```

## Backend Deployment (Railway - Recommended)

### Option 1: Railway CLI
```bash
npm install -g @railway/cli
railway login
railway init
railway up
```

### Option 2: Railway Dashboard
1. Go to [railway.app](https://railway.app)
2. New Project → Deploy from GitHub
3. Select your repository
4. Set root directory to `backend`
5. Add environment variables if needed
6. Deploy!

### Update CORS in Backend
In `backend/app.py`, update CORS origins:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.vercel.app"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Alternative Hosting Options

### Frontend Alternatives
- **Netlify**: Similar to Vercel, drag & drop `frontend/dist` folder
- **GitHub Pages**: Requires build step, update base path in `vite.config.js`

### Backend Alternatives
- **Render**: Free tier available, similar to Railway
- **Heroku**: Requires Procfile
- **Fly.io**: Good for Docker deployments
- **AWS/GCP/Azure**: For production scale

## Environment Variables

### Backend
- `PORT`: Server port (default: 8000)
- `MODEL_PATH`: Path to saved models (optional)

### Frontend
- `VITE_API_URL`: Backend API URL (optional, defaults to localhost:8000)

## Quick Deploy Commands

### Full Stack on Railway
```bash
# Railway supports monorepos
# Deploy both frontend and backend as separate services
```

### Separate Deployments
```bash
# Frontend on Vercel
cd frontend && vercel --prod

# Backend on Railway
cd backend && railway up
```

## Post-Deployment Checklist

- [ ] Update CORS origins in backend
- [ ] Update API URL in frontend
- [ ] Test file upload functionality
- [ ] Verify models are accessible
- [ ] Check API documentation at `/docs`
- [ ] Monitor logs for errors

## Troubleshooting

### CORS Errors
- Ensure backend CORS includes frontend URL
- Check that frontend is using correct API URL

### Model Loading Errors
- Ensure model files are in `backend/src/saved_models/`
- Check file paths in deployment environment

### Build Errors
- Verify all dependencies in `requirements.txt` and `package.json`
- Check Node.js and Python versions match requirements

