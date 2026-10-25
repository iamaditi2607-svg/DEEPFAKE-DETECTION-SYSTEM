# 🚀 DeepShield — Railway Deployment Guide

## Railway pe Deploy Karne ke Steps (Free Permanent URL)

---

### Step 1: GitHub pe Upload Karo

1. **github.com** pe jaao → Sign in karo / account banao
2. **New Repository** click karo → Name: `deepshield` → Create
3. Apne computer mein ye commands run karo:

```bash
cd "c:\Users\ADITI VISHWAKARMA\Downloads\DEEPFAKE DETECTION SYSTEM"
git add .
git commit -m "Initial: DeepShield deepfake detection system"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/deepshield.git
git push -u origin main
```
*(YOUR_USERNAME ki jagah apna GitHub username daalo)*

---

### Step 2: Railway pe Deploy Ka

1. **railway.app** pe jaao → **"Start a New Project"** click karo
2. **"Deploy from GitHub repo"** select karo
3. Apna `deepshield` repo select karo
4. Railway automatically detect karega:
   - Python project hai
   - `Procfile` se `gunicorn` command milegi
   - `requirements.txt` se packages install honge
5. Deploy hone mein **3-5 minute** lagte hain

---

### Step 3: Public URL Lo

1. Deploy hone ke baad Railway dashboard mein **"Settings"** tab pe jaao
2. **"Domains"** section mein **"Generate Domain"** click karo
3. Aapko milega: `https://deepshield-production.up.railway.app`
4. **Yahi URL share karo!** ✅

---

### Step 4: Environment Variables (Optional)

Railway dashboard → Variables → Add:
```
TRANSFORMERS_CACHE = /tmp/hf_cache
```

---

## ⚠️ Important Notes

- **First load slow hoga** (~1-2 min) kyunki AI model download hota hai (~400MB)
- Railway free tier: **500 hours/month** (enough for demo)
- Model cache hota hai, baad mein fast hoga

---

## 🧪 Local Test karne ke liye

```bash
pip install -r requirements.txt
python app.py
# Browser mein: http://localhost:5000
```

---

## 📋 Files Created

| File | Purpose |
|------|---------|
| `app.py` | Flask backend + AI model |
| `index.html` | Responsive frontend UI |
| `requirements.txt` | Python packages |
| `Procfile` | Railway deployment command |
| `runtime.txt` | Python 3.11 |
| `README.md` | Project docs |

---

*Team Idea Ignitors | AITM Kanpur*
