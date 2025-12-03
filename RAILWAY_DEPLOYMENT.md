# Railway Deployment Guide - Both Services

## Step-by-Step Instructions

### Step 1: Create Railway Project

1. Go to [railway.app](https://railway.app)
2. Sign up/login with GitHub
3. Click "New Project"
4. Select "Deploy from GitHub repo"
5. Choose your repository: `HemachandRavulapalli/ECG-Arrhythmia-Classification`

### Step 2: Add Backend Service

1. In your Railway project, click **"+ New"** → **"GitHub Repo"**
2. Select the same repository again
3. Railway will create a service
4. **IMPORTANT**: Click on the service → **Settings** → **Root Directory**
5. Set Root Directory to: `backend`
6. Railway will auto-detect Python
7. The service will automatically:
   - Detect `requirements.txt` in the backend folder
   - Run `pip install -r requirements.txt`
   - Start with `python app.py` (from Procfile)

### Step 3: Configure Backend Environment Variables

1. Go to your backend service → **Variables** tab
2. Add if needed:
   - `PORT`: Railway will set this automatically (usually 8000)
   - Any other environment variables your app needs

### Step 4: Add Frontend Service

1. In the same Railway project, click **"+ New"** → **"GitHub Repo"**
2. Select the same repository again
3. **IMPORTANT**: Click on the new service → **Settings** → **Root Directory**
4. Set Root Directory to: `frontend`
5. Railway will auto-detect Node.js
6. The service will automatically:
   - Detect `package.json`
   - Run `npm install`
   - You need to set custom start command (see below)

### Step 5: Configure Frontend Service

1. Go to your frontend service → **Settings** → **Deploy**
2. Set **Start Command** to: `npm run preview`
   - OR use a static file server: `npx serve dist -p $PORT`
3. Go to **Variables** tab
4. Add environment variable:
   - `VITE_API_URL`: Your backend service URL
     - Find it in: Backend service → Settings → Generate Domain
     - Example: `https://ecg-backend-production.up.railway.app`

### Step 6: Build Frontend Before Deploy

The frontend needs to be built first. Update the build settings:

1. Go to frontend service → **Settings** → **Build**
2. Set **Build Command** to: `npm install && npm run build`
3. Set **Output Directory** to: `dist` (if Railway asks)

### Step 7: Update Backend CORS

1. Get your frontend URL from Railway (frontend service → Settings → Generate Domain)
2. Edit `backend/app.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-frontend-service.up.railway.app",
        "http://localhost:3000"  # Keep for local dev
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
3. Commit and push - Railway will auto-deploy

### Step 8: Get Your URLs

1. **Backend URL**: Backend service → Settings → Generate Domain
2. **Frontend URL**: Frontend service → Settings → Generate Domain

### Step 9: Update Frontend API URL

1. Go to frontend service → Variables
2. Update `VITE_API_URL` to your backend URL
3. Redeploy frontend service

## Alternative: Use Railway's Monorepo Feature

Railway supports monorepos. You can:

1. Create one service
2. Set root directory to project root
3. Use `railway.json` files in each subdirectory
4. Railway will detect and deploy both

But the **two-service approach is recommended** for better separation.

## Troubleshooting

### Backend won't start
- Check Root Directory is set to `backend`
- Check that `requirements.txt` exists in backend folder
- Check logs in Railway dashboard

### Frontend shows blank page
- Check that `VITE_API_URL` is set correctly
- Check browser console for CORS errors
- Verify backend URL is accessible

### Build fails
- Check that all dependencies are in `requirements.txt` (backend) or `package.json` (frontend)
- Check Railway logs for specific error messages

### Port errors
- Railway sets `PORT` environment variable automatically
- Backend code reads `PORT` from environment (already configured)
- Frontend uses Railway's assigned port

## Quick Checklist

- [ ] Created Railway project
- [ ] Added backend service with root directory = `backend`
- [ ] Added frontend service with root directory = `frontend`
- [ ] Set frontend build command: `npm install && npm run build`
- [ ] Set frontend start command: `npm run preview` or `npx serve dist -p $PORT`
- [ ] Added `VITE_API_URL` environment variable to frontend
- [ ] Updated backend CORS to include frontend URL
- [ ] Both services deployed successfully
- [ ] Tested file upload on live site

## Cost

Railway offers:
- **Free tier**: $5 credit/month
- **Hobby plan**: $5/month for more resources
- Both services should fit in free tier for testing

## Need Help?

- Railway Docs: https://docs.railway.app
- Railway Discord: https://discord.gg/railway
- Check Railway logs in dashboard for errors

