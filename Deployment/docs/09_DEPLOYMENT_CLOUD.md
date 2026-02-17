# 09 — Cloud Deployment & Docker

> Deploy your backend to Railway and your web frontend to Vercel.

---

## Architecture in the Cloud

```
Users
  │
  ├── Web Browser ──────► Vercel (static React app)
  │                         │
  │                         ▼ WebSocket / REST
  │                    Railway (Python FastAPI + models)
  │
  └── Mobile App ──────► On-device TFLite (no server needed)
```

- **Vercel**: Hosts the React web app (free, fast, global CDN)
- **Railway**: Hosts the Python backend + ML models (free tier: 500 hours/month)
- **Mobile**: Runs entirely on-device, no cloud needed

---

## Part 1: Docker — Containerize the Backend

### Why Docker?

Docker packages your entire backend (Python + models + dependencies) into a single "container" that runs identically everywhere — your laptop, Railway, AWS, etc.

### `backend/Dockerfile`

```dockerfile
# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for TensorFlow + MediaPipe)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Copy model files
COPY model_files/ ./model_files/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `backend/.dockerignore`

```
venv/
__pycache__/
*.pyc
.env
.env.local
tests/
*.md
.git/
```

### Build & Test Locally

```powershell
cd "m:\Term 10\Grad\Deployment\backend"

# Build the Docker image
docker build -t eshara-api .

# Run it
docker run -p 8000:8000 eshara-api

# Test
# Open browser: http://localhost:8000/docs
# Try: http://localhost:8000/health
```

---

## Part 2: Docker Compose — Local Development

### `Deployment/docker-compose.yml`

```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - BACKEND_PORT=8000
      - CORS_ORIGINS=http://localhost:5173,http://localhost:3000
    volumes:
      - ./backend/app:/app/app    # Hot reload during development
    restart: unless-stopped

  # Optional: Local PostgreSQL (instead of Supabase)
  # db:
  #   image: postgres:15
  #   ports:
  #     - "5432:5432"
  #   environment:
  #     POSTGRES_DB: eshara
  #     POSTGRES_USER: eshara
  #     POSTGRES_PASSWORD: eshara_password
  #   volumes:
  #     - pgdata:/var/lib/postgresql/data

# volumes:
#   pgdata:
```

Run:
```powershell
cd "m:\Term 10\Grad\Deployment"
docker-compose up --build
```

---

## Part 3: Deploy Backend to Railway

### Step-by-Step

1. **Push to GitHub** (if not already):
   ```powershell
   cd "m:\Term 10\Grad\Deployment"
   git init
   git add .
   git commit -m "Initial deployment setup"
   git remote add origin https://github.com/YOUR_USERNAME/eshara-deployment.git
   git push -u origin main
   ```

2. **Go to Railway**: https://railway.app/dashboard

3. **Create New Project**:
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your `eshara-deployment` repository

4. **Configure the Service**:
   - Railway will auto-detect the Dockerfile
   - If not, go to Settings → Build:
     - Root Directory: `backend`
     - Builder: Dockerfile

5. **Set Environment Variables** (in Railway dashboard):
   ```
   BACKEND_PORT=8000
   CORS_ORIGINS=https://your-app.vercel.app
   ```

6. **Add a `railway.toml`** file:

### `backend/railway.toml`

```toml
[build]
builder = "DOCKERFILE"
dockerfilePath = "Dockerfile"

[deploy]
startCommand = "uvicorn app.main:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/health"
healthcheckTimeout = 300
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 3
```

7. **Deploy**: Railway will auto-build and deploy.

8. **Get Your URL**: Railway gives you a URL like:
   ```
   https://eshara-api-production.up.railway.app
   ```

9. **Test it**: Open `https://YOUR-URL.up.railway.app/docs`

### Railway Important Notes

- **Free tier**: 500 hours/month (~20 days continuous). Enough for testing/demos.
- **Memory**: Models need ~1-2 GB RAM. Free tier allows 1 GB. Consider:
  - Only loading letter models initially
  - Loading word model on demand
  - Or upgrading to Hobby tier ($5/month, 8 GB RAM)
- **Cold starts**: First request after idle takes 10-30 seconds (model loading).

---

## Part 4: Deploy Web Frontend to Vercel

### Step-by-Step

1. **Update environment variables**: 
   
   ### `web/.env.production`
   ```env
   VITE_API_URL=https://eshara-api-production.up.railway.app
   VITE_WS_URL=wss://eshara-api-production.up.railway.app/ws/recognize
   ```
   
   > Note: Use `wss://` (with SSL) not `ws://` for production.

2. **Add Vercel config**:

### `web/vercel.json`

```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "framework": "vite",
  "rewrites": [
    { "source": "/(.*)", "destination": "/index.html" }
  ],
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        { "key": "X-Frame-Options", "value": "DENY" },
        { "key": "X-Content-Type-Options", "value": "nosniff" }
      ]
    }
  ]
}
```

3. **Go to Vercel**: https://vercel.com/dashboard

4. **Import Project**:
   - Click "Add New" → "Project"
   - Import from GitHub → Select your repo
   - Root Directory: `web`
   - Framework Preset: Vite
   - Build Command: `npm run build`
   - Output Directory: `dist`

5. **Set Environment Variables** (in Vercel dashboard):
   ```
   VITE_API_URL=https://eshara-api-production.up.railway.app
   VITE_WS_URL=wss://eshara-api-production.up.railway.app/ws/recognize
   ```

6. **Deploy**: Click "Deploy". Takes ~1-2 minutes.

7. **Get Your URL**: Vercel gives you:
   ```
   https://eshara.vercel.app
   ```

8. **Custom Domain** (optional): In Vercel settings, add your own domain.

---

## Part 5: Deploy Mobile App

### Build with EAS

```powershell
cd "m:\Term 10\Grad\Deployment\mobile"

# Login to Expo
eas login

# Configure build
eas build:configure

# Build for Android (APK for testing)
eas build --platform android --profile preview

# Build for iOS (need Apple Developer account)
eas build --platform ios --profile preview
```

### `mobile/eas.json`

```json
{
  "build": {
    "preview": {
      "android": {
        "buildType": "apk"
      },
      "ios": {
        "simulator": true
      }
    },
    "production": {
      "android": {
        "buildType": "app-bundle"
      },
      "ios": {
        "autoIncrement": true
      }
    }
  },
  "submit": {
    "production": {
      "android": {
        "serviceAccountKeyPath": "./google-services.json"
      }
    }
  }
}
```

### Submit to Google Play Store

```powershell
# Build production bundle
eas build --platform android --profile production

# Submit to Play Store (need Google Play Console account)
eas submit --platform android
```

---

## Part 6: Update CORS for Production

After deploying, update your backend's CORS to allow your Vercel URL:

### In Railway Environment Variables:
```
CORS_ORIGINS=https://eshara.vercel.app,http://localhost:5173
```

---

## Deployment Checklist

| # | Task | Status |
|---|------|--------|
| 1 | Backend Dockerfile builds locally | ☐ |
| 2 | `docker run` → API works at localhost:8000 | ☐ |
| 3 | `localhost:8000/health` shows all models loaded | ☐ |
| 4 | GitHub repo created and pushed | ☐ |
| 5 | Railway project created from GitHub | ☐ |
| 6 | Railway env vars set (CORS_ORIGINS) | ☐ |
| 7 | Railway deploy successful → URL works | ☐ |
| 8 | `YOUR-URL.up.railway.app/health` responds | ☐ |
| 9 | Web `.env.production` updated with Railway URL | ☐ |
| 10 | Vercel project created from GitHub | ☐ |
| 11 | Vercel env vars set (VITE_API_URL, VITE_WS_URL) | ☐ |
| 12 | Vercel deploy successful → URL works | ☐ |
| 13 | Web app connects to API and WebSocket | ☐ |
| 14 | Camera works and hand detection runs | ☐ |
| 15 | Letter prediction works end-to-end | ☐ |
| 16 | Word prediction works end-to-end | ☐ |
| 17 | Mobile APK builds successfully | ☐ |
| 18 | Mobile TFLite inference works on phone | ☐ |

---

## Cost Summary

| Service | Free Tier | Paid Tier (if needed) |
|---------|-----------|----------------------|
| Railway | 500 hrs/month, 1 GB RAM | $5/month (Hobby) |
| Vercel | Unlimited, 100 GB bandwidth | $20/month (Pro) |
| Supabase | 500 MB DB, 50K users | $25/month (Pro) |
| Google Play | — | $25 one-time |
| Apple Developer | — | $99/year |
| **Total (dev/testing)** | **$0** | — |
| **Total (production)** | — | **$5-50/month** |
