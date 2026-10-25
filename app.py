"""
DeepShield – Deepfake Detection System
Backend using HuggingFace Inference API (no local torch needed)
Team: Idea Ignitors | AITM Kanpur
"""

import os
import io
import cv2
import base64
import requests
import tempfile
import logging
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
#  App Setup
# ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")
CORS(app, resources={r"/*": {"origins": "*"}})
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

# ─────────────────────────────────────────────────────────
#  HuggingFace Inference API Config
#  Model: dima806/deepfake_vs_real_image_detection (ViT)
#  Free tier: 30,000 calls/month – no credit card needed
# ─────────────────────────────────────────────────────────
# Free token from https://huggingface.co/settings/tokens — needs the
# "Inference Providers" permission. Without it the router returns 401 and
# the app falls back to a (much weaker) statistical heuristic.
HF_API_KEY   = os.environ.get("HF_API_KEY", "")
HF_MODEL_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "dima806/deepfake_vs_real_image_detection"
)
# Verdict threshold (0–1). The model is ~3 years old, so modern AI images
# drift below the old default of 0.5; the author recommends lowering it.
# Lower  = catch more fakes (risk: real images may be flagged fake);
# Higher = safer for real images (risk: some fakes slip through).
FAKE_THRESHOLD = float(os.environ.get("FAKE_THRESHOLD", "0.2"))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


# ─────────────────────────────────────────────────────────
#  Core: Call HuggingFace API for one PIL image
# ─────────────────────────────────────────────────────────
def classify_image(img: Image.Image) -> tuple[float, float, bool]:
    """
    Send a PIL image to the HuggingFace router Inference API.
    Returns (fake_score, real_score, used_model) where scores are 0.0–1.0.
    used_model is True when the real AI model answered; False when we fell
    back to the heuristic (e.g. missing/invalid HF_API_KEY).
    """
    try:
        # Resize to reduce payload size (keeps the free tier fast)
        img = img.convert("RGB")
        img.thumbnail((512, 512), Image.LANCZOS)

        # Encode to JPEG bytes
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        img_bytes = buf.getvalue()

        headers = {"Content-Type": "application/octet-stream"}
        if HF_API_KEY:
            headers["Authorization"] = f"Bearer {HF_API_KEY}"

        resp = requests.post(
            HF_MODEL_URL,
            headers=headers,
            data=img_bytes,
            timeout=30,
        )

        # Serverless router returns 503 while the model cold-starts; retry once.
        if resp.status_code in (503, 429):
            import time
            logger.warning(f"HF busy ({resp.status_code}), retrying in 12s…")
            time.sleep(12)
            resp = requests.post(HF_MODEL_URL, headers=headers, data=img_bytes, timeout=45)

        if resp.status_code in (401, 403):
            logger.error("HF API rejected auth — HF_API_KEY missing or invalid. Using heuristic fallback.")
            return _heuristic_fallback(img)

        resp.raise_for_status()
        results = resp.json()

        fake_score, real_score = 0.0, 0.0
        for r in results:
            lbl = r.get("label", "").lower()
            score = float(r.get("score", 0))
            if "fake" in lbl:
                fake_score = score
            elif "real" in lbl:
                real_score = score

        # Normalize (model returns softmax scores that already sum to ~1)
        total = fake_score + real_score
        if total > 0:
            fake_score /= total
            real_score /= total
        else:
            fake_score = real_score = 0.5

        logger.info(f"HF model → fake={fake_score:.3f} real={real_score:.3f}")
        return fake_score, real_score, True

    except Exception as e:
        logger.error(f"HF API error: {e} — using heuristic fallback")
        return _heuristic_fallback(img)


def _heuristic_fallback(img: Image.Image) -> tuple[float, float, bool]:
    """
    Last-resort statistical estimate when the HF model is unreachable
    (e.g. HF_API_KEY not set). NOT a real deepfake detector — it only looks at
    local pixel-variance smoothness, so treat its output as a rough guess.
    """
    import math
    arr = np.array(img.convert("RGB"), dtype=np.float32)

    # Measure local variance in small blocks (AI images are often smoother)
    h, w = arr.shape[:2]
    block = 16
    variances = []
    for y in range(0, h - block, block):
        for x in range(0, w - block, block):
            patch = arr[y:y+block, x:x+block]
            variances.append(float(np.var(patch)))
    avg_var = float(np.mean(variances)) if variances else 0.0

    # High variance → more natural, low → possibly synthesized
    threshold = 800.0
    raw = math.tanh((threshold - avg_var) / threshold)        # >0 → fake tendency
    fake = max(0.05, min(0.95, 0.5 + raw * 0.3))
    real = 1.0 - fake
    logger.info(f"Heuristic fallback → fake={fake:.3f} (avg_var={avg_var:.1f})")
    return fake, real, False


# ─────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "hf_key_set": bool(HF_API_KEY)})


@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file. Use form-data key 'file'."}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "Empty file."}), 400

    ext = os.path.splitext(file.filename.lower())[1]
    logger.info(f"Received: {file.filename} ({ext})")

    if ext in VIDEO_EXTS:
        return _analyze_video(file)
    else:  # treat everything else as image
        return _analyze_image(file)


# ── Image handler ──────────────────────────────────────

def _analyze_image(file):
    try:
        raw = file.read()
        img = Image.open(io.BytesIO(raw))
        fake, real, used_model = classify_image(img)

        verdict    = "FAKE" if fake >= FAKE_THRESHOLD else "REAL"
        confidence = round(max(fake, real) * 100, 1)

        resp = {
            "verdict":         verdict,
            "fake_score":      round(fake * 100, 1),
            "real_score":      round(real * 100, 1),
            "confidence":      confidence,
            "frames_analyzed": 1,
            "frame_scores":    [round(fake * 100, 1)],
            "file_type":       "image",
            "method":          "ai_model" if used_model else "heuristic",
        }
        if not used_model:
            resp["warning"] = ("AI model unreachable (HF_API_KEY not set/invalid). "
                               "This is only a rough estimate — set HF_API_KEY on "
                               "the host for accurate detection.")
        return jsonify(resp)
    except Exception as e:
        logger.error(f"Image error: {e}")
        return jsonify({"error": str(e)}), 500


# ── Video handler ─────────────────────────────────────

def _analyze_video(file):
    tmp_path = None
    try:
        suffix = os.path.splitext(file.filename)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return jsonify({"error": "Cannot open video."}), 400

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return jsonify({"error": "Video has no frames."}), 400

        num_samples   = min(20, total)
        frame_indices = np.linspace(0, total - 1, num_samples, dtype=int)
        frame_scores   = []
        used_model_any  = False

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            fake, _, used_model = classify_image(pil)
            if used_model:
                used_model_any = True
            frame_scores.append(round(fake * 100, 1))
            logger.info(f"  Frame {idx}: fake={fake:.3f}")

        cap.release()

        if not frame_scores:
            return jsonify({"error": "No frames analyzed."}), 500

        avg_fake = sum(frame_scores) / len(frame_scores)
        verdict  = "FAKE" if avg_fake >= FAKE_THRESHOLD * 100 else "REAL"

        resp = {
            "verdict":         verdict,
            "fake_score":      round(avg_fake, 1),
            "real_score":      round(100 - avg_fake, 1),
            "confidence":      round(max(avg_fake, 100 - avg_fake), 1),
            "frames_analyzed": len(frame_scores),
            "frame_scores":    frame_scores,
            "file_type":       "video",
            "method":          "ai_model" if used_model_any else "heuristic",
        }
        if not used_model_any:
            resp["warning"] = ("AI model unreachable (HF_API_KEY not set/invalid). "
                               "This is only a rough estimate — set HF_API_KEY on "
                               "the host for accurate detection.")
        return jsonify(resp)
    except Exception as e:
        logger.error(f"Video error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────
#  Run
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"🚀 DeepShield starting on http://0.0.0.0:{port}")
    if not HF_API_KEY:
        logger.warning("⚠️  HF_API_KEY not set — AI model will be unreachable; "
                       "set a free HF token (Inference Providers permission) for accurate detection.")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
