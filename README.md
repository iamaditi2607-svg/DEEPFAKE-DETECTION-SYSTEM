# DeepShield — Deepfake Detection System

A full-stack AI-powered deepfake detection web app.

**Team:** Idea Ignitors | **College:** AITM Kanpur

## 🚀 Live Demo
Deployed on Railway — see deployment steps below.

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
