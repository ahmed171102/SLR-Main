# 03 — Accounts, Databases & Services Setup

> Every online account and service you need to create before and during deployment.

---

## Phase 0: Accounts Needed Before You Start

### 1. GitHub Account (FREE) — Version Control

**URL**: https://github.com/signup

**Why?** Store your code, track changes, deploy automatically.

**Steps:**
1. Go to https://github.com/signup
2. Sign up with your email
3. Create a new repository called `eshara-deployment`
4. Set it to **Private** (your code, your rules)
5. Clone it to your local machine:
   ```bash
   git clone https://github.com/YOUR_USERNAME/eshara-deployment.git
   ```

**Tip**: If you already have a GitHub account, just create the new repo.

---

### 2. Node.js Installation (FREE) — Required for Web + Mobile

**URL**: https://nodejs.org

**Steps:**
1. Download **Node.js 20 LTS** (not Current)
2. Install with default options
3. Verify in terminal:
   ```bash
   node --version    # Should show v20.x.x
   npm --version     # Should show 10.x.x
   ```

---

## Phase 2: Accounts for Backend Deployment

### 3. Railway Account (FREE tier) — Backend Hosting

**URL**: https://railway.app

**Why?** Hosts your Python backend API in the cloud for free.

**Free tier includes:**
- 500 hours/month execution
- 1 GB RAM
- 1 GB disk
- Custom domain (yourapp.up.railway.app)

**Steps:**
1. Go to https://railway.app
2. Click "Login" → Sign in with GitHub
3. Allow Railway to access your GitHub repos
4. That's it! You'll deploy later in Phase 5.

**Alternative**: [Render.com](https://render.com) (also free tier, slightly slower cold starts)

---

### 4. Docker Hub Account (FREE) — Container Registry

**URL**: https://hub.docker.com/signup

**Why?** Store your Docker container images. Needed for Railway/cloud deployment.

**Steps:**
1. Go to https://hub.docker.com/signup
2. Sign up with email and username
3. Install Docker Desktop: https://www.docker.com/products/docker-desktop/
4. Login in terminal:
   ```bash
   docker login
   ```

**Note**: Docker Hub is OPTIONAL if you deploy directly from GitHub to Railway. Railway can build from your Dockerfile without Docker Hub.

---

## Phase 3: Accounts for Web Frontend Deployment

### 5. Vercel Account (FREE) — Web Frontend Hosting

**URL**: https://vercel.com/signup

**Why?** Hosts your React web app for free. Automatic deploys from GitHub.

**Free tier includes:**
- Unlimited static sites
- 100 GB bandwidth/month
- Custom domain support
- Automatic HTTPS

**Steps:**
1. Go to https://vercel.com/signup
2. Sign in with GitHub
3. Allow Vercel to access your repo
4. Done! You'll connect your `web/` folder later.

---

## Phase 4: Accounts for Mobile App

### 6. Expo Account (FREE) — Mobile Development

**URL**: https://expo.dev/signup

**Why?** Build and distribute your React Native mobile app.

**Steps:**
1. Go to https://expo.dev/signup
2. Sign up with email
3. Install Expo CLI:
   ```bash
   npm install -g eas-cli
   ```
4. Login:
   ```bash
   eas login
   ```

### 7. Google Play Developer Account (PAID: $25 one-time) — Android Distribution

**URL**: https://play.google.com/console/signup

**Why?** Publish your app on Google Play Store.

**When?** Only when your app is ready for public release. NOT needed during development.

**Steps:**
1. Go to https://play.google.com/console/signup
2. Pay $25 one-time fee
3. Fill in developer details
4. Wait for verification (can take 48 hours)

### 8. Apple Developer Account (PAID: $99/year) — iOS Distribution

**URL**: https://developer.apple.com/programs/

**Why?** Publish on Apple App Store. Requires a Mac for building.

**When?** Only if you want to release on iOS. Skip this for now if Android-only.

---

## Phase 6: Optional Accounts for Extra Features

### 9. Supabase Account (FREE tier) — Database + Auth

**URL**: https://supabase.com

**Why?** Provides PostgreSQL database + user authentication for free.

**Free tier includes:**
- 500 MB database
- 50,000 monthly active users
- 5 GB bandwidth
- 1 GB file storage

**Steps:**
1. Go to https://supabase.com
2. Sign in with GitHub
3. Click "New Project"
4. Name: `eshara`
5. Set a strong database password (SAVE IT!)
6. Region: Choose closest to you
7. Click "Create new project"
8. Wait ~2 minutes for setup
9. Go to Project Settings → API
10. Copy these values:
    ```
    SUPABASE_URL=https://xxxxx.supabase.co
    SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIs...
    SUPABASE_SERVICE_KEY=eyJhbGciOiJIUzI1NiIs...
    ```

**Database Schema (create in Supabase SQL Editor):**

```sql
-- Users table (auto-created by Supabase Auth)

-- Translation history
CREATE TABLE IF NOT EXISTS translations (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id),
    mode TEXT NOT NULL CHECK (mode IN ('letter', 'word')),
    language TEXT NOT NULL CHECK (language IN ('en', 'ar')),
    input_type TEXT NOT NULL CHECK (input_type IN ('camera', 'upload')),
    prediction TEXT NOT NULL,
    confidence FLOAT,
    sentence TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- User preferences
CREATE TABLE IF NOT EXISTS user_preferences (
    user_id UUID REFERENCES auth.users(id) PRIMARY KEY,
    language TEXT DEFAULT 'en',
    mode TEXT DEFAULT 'letter',
    theme TEXT DEFAULT 'light',
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE translations ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_preferences ENABLE ROW LEVEL SECURITY;

-- Users can only see their own data
CREATE POLICY "Users can read own translations"
    ON translations FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own translations"
    ON translations FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can manage own preferences"
    ON user_preferences FOR ALL
    USING (auth.uid() = user_id);
```

---

## Summary: All Accounts

| # | Service | Cost | When Needed | Required? |
|---|---------|------|-------------|-----------|
| 1 | GitHub | FREE | Phase 0 (start) | YES |
| 2 | Node.js | FREE | Phase 0 (start) | YES |
| 3 | Railway | FREE | Phase 5 (deploy) | YES |
| 4 | Docker Hub | FREE | Phase 5 (deploy) | OPTIONAL |
| 5 | Vercel | FREE | Phase 5 (deploy) | YES |
| 6 | Expo | FREE | Phase 4 (mobile) | YES (for mobile) |
| 7 | Google Play | $25 once | Release | ONLY for Play Store |
| 8 | Apple Dev | $99/year | Release | ONLY for App Store |
| 9 | Supabase | FREE | Phase 6 (auth/db) | OPTIONAL |

### Total Cost to Start: $0

Only pay $25 when ready to publish on Google Play. Everything else is free during development.

---

## Environment Variables (.env file)

After creating accounts, you'll have these values. Create a `.env` file:

```env
# Backend
BACKEND_PORT=8000
CORS_ORIGINS=http://localhost:5173,https://your-app.vercel.app
MODEL_DIR=./model_files

# Supabase (optional - only if using auth/database)
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_ANON_KEY=your-anon-key-here
SUPABASE_SERVICE_KEY=your-service-key-here

# Web Frontend
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws
VITE_SUPABASE_URL=https://xxxxx.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key-here
```

> **NEVER commit .env files to GitHub!** Add `.env` to your `.gitignore` file.
