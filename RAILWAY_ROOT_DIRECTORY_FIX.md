# Railway Root Directory Fix - CRITICAL STEP

## The Problem
Railway is trying to build from the root directory (`/`) instead of the `backend/` directory. This is why Nixpacks can't detect Python.

## The Solution - Set Root Directory in Railway Dashboard

### Step 1: Go to Your Service Settings

1. In Railway dashboard, click on your **backend service**
2. Click on **Settings** (gear icon)
3. Scroll down to **Root Directory**

### Step 2: Set Root Directory

1. Click on **Root Directory** field
2. Type: `backend`
3. Click **Save** or press Enter
4. Railway will automatically redeploy

### Step 3: Verify

After setting the root directory, Railway should:
- ✅ Detect Python automatically
- ✅ Find `requirements.txt` in `backend/requirements.txt`
- ✅ Find `Procfile` in `backend/Procfile`
- ✅ Build successfully

## Visual Guide

```
Railway Service Settings:
┌─────────────────────────────────┐
│ Service: Backend                │
├─────────────────────────────────┤
│ Name: ECG-Backend               │
│ Root Directory: backend  ← SET THIS!
│                                 │
│ [Save Changes]                  │
└─────────────────────────────────┘
```

## Alternative: Delete and Recreate Service

If setting root directory doesn't work:

1. **Delete the current service** in Railway
2. Create a **new service** from GitHub repo
3. **IMMEDIATELY** go to Settings → Root Directory
4. Set it to `backend` **BEFORE** Railway starts building
5. Save and let it deploy

## For Frontend Service

Repeat the same process:
1. Create frontend service
2. Set Root Directory to: `frontend`
3. Set Start Command to: `npm run preview`

## Still Not Working?

### Check These Files Exist:

```bash
backend/
├── app.py          ✅ Must exist
├── requirements.txt ✅ Must exist  
├── Procfile        ✅ Must exist
└── nixpacks.toml   ✅ Added for better detection
```

### Verify in Railway:

1. Go to service → **Deployments** tab
2. Click on latest deployment → **View Logs**
3. Check if it shows:
   - `Detected Python`
   - `Installing dependencies from requirements.txt`
   - `Starting with: python app.py`

## Quick Checklist

- [ ] Backend service created
- [ ] Root Directory set to `backend` in Settings
- [ ] Service redeployed after setting root directory
- [ ] Build logs show Python detection
- [ ] Frontend service created separately
- [ ] Frontend Root Directory set to `frontend`

## Important Notes

⚠️ **Root Directory MUST be set in Railway dashboard** - it cannot be set via code files alone.

⚠️ **Set it BEFORE Railway tries to build** - otherwise it will fail.

⚠️ **Each service needs its own root directory** - backend and frontend are separate services.

