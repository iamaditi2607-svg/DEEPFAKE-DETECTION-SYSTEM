# DeepShield — Deepfake Detection System

A full-stack AI-powered deepfake detection web app.

**Team:** Idea Ignitors | **College:** AITM Kanpur

## 🌐 Live Demo

**👉 Try it now:** https://deepfake-detection-system-3woz.onrender.com

> Hosted on Render free tier. First load may take ~30–60s (service wakes up from sleep). After that it's instant.

## 🚀 One-Click Live Deploy

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/iamaditi2607-svg/DEEPFAKE-DETECTION-SYSTEM)

Click the button above → sign in with GitHub → Render reads `render.yaml` and builds automatically → you get a public URL like `https://deepshield-xxxx.onrender.com`.

> (Optional) After deploy, go to the service's **Environment** tab and add `HF_API_KEY` with a free token from https://huggingface.co/settings/tokens for accurate AI detection. Without it the app still runs using a built-in fallback.

## 🧠 How It Works
1. Upload image or video
2. Backend extracts frames (video)
3. ViT model (`dima806/deepfake_vs_real_image_detection`) scores each frame
4. Average score → **REAL** or **FAKE** verdict

## 📁 Project Structure
```
├── app.py              # Flask backend (AI model + API)
├── index.html          # Frontend (responsive UI)
├── requirements.txt    # Python dependencies
├── Procfile            # Railway/Render deployment config
├── runtime.txt         # Python 3.11
└── README.md
```

## 🛠️ Local Setup
```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

## ☁️ Deploy to Railway
1. Push to GitHub
2. Go to railway.app → New Project → Deploy from GitHub
3. Done! Public URL auto-generated.
