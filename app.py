"""
DeepShield - Deepfake Detection System
Flask Backend with AI Model (dima806/deepfake_vs_real_image_detection)
Team: Idea Ignitors | AITM Kanpur
"""

import os
import io
import cv2
import numpy as np
import tempfile
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  App Setup
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")
CORS(app, resources={r"/*": {"origins": "*"}})

app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB max upload

# ─────────────────────────────────────────────
#  Model Loading (Lazy — loads once on first request)
# ─────────────────────────────────────────────
detector = None

def get_detector():
    global detector
    if detector is None:
        logger.info("⏳ Loading deepfake detection model (first time may take a minute)…")
        try:
            from transformers import pipeline as hf_pipeline
            import torch
            device = 0 if torch.cuda.is_available() else -1
            detector = hf_pipeline(
                "image-classification",
                model="dima806/deepfake_vs_real_image_detection",
                device=device,
            )
            logger.info("✅ Model loaded successfully!")
        except Exception as e:
            logger.error(f"❌ Model load failed: {e}")
            raise
    return detector


# ─────────────────────────────────────────────
#  Helper: Analyze a single PIL image
# ─────────────────────────────────────────────
def analyze_pil_image(img: Image.Image):
    """Run model inference on a PIL image. Returns (fake_score, real_score) as floats 0-1."""
    pipe = get_detector()
    try:
        # Resize to avoid OOM on large images
        img = img.convert("RGB")
        img.thumbnail((512, 512), Image.LANCZOS)
        results = pipe(img, top_k=2)

        fake_score = 0.0
        real_score = 0.0
        for r in results:
            lbl = r["label"].lower()
            if "fake" in lbl:
                fake_score = float(r["score"])
            else:
                real_score = float(r["score"])

        # Normalize in case only one label returned
        total = fake_score + real_score
        if total > 0:
            fake_score /= total
            real_score /= total
        else:
            fake_score = 0.5
            real_score = 0.5

        return fake_score, real_score
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return 0.5, 0.5


# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the frontend."""
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_ready": detector is not None})


@app.route("/warmup", methods=["GET"])
def warmup():
    """Pre-load model so first real request is fast."""
    try:
        get_detector()
        return jsonify({"status": "model_loaded"})
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    POST /analyze
    Form-data key: 'file'  (image or video)

    Returns JSON:
    {
      "verdict":         "REAL" | "FAKE",
      "fake_score":      float  (0-100),
      "real_score":      float  (0-100),
      "confidence":      float  (0-100),
      "frames_analyzed": int,
      "frame_scores":    [float, ...],
      "file_type":       "image" | "video"
    }
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Use form-data key 'file'."}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "Empty file or filename."}), 400

    filename = file.filename.lower()
    ext = os.path.splitext(filename)[1]

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff"}
    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}

    logger.info(f"📥 Received file: {filename} ({ext})")

    if ext in IMAGE_EXTS:
        return _handle_image(file)
    elif ext in VIDEO_EXTS:
        return _handle_video(file)
    else:
        # Try treating as image
        logger.warning(f"Unknown extension '{ext}', trying as image…")
        return _handle_image(file)


def _handle_image(file):
    """Deepfake check on a single image."""
    try:
        raw = file.read()
        img = Image.open(io.BytesIO(raw))
        fake, real = analyze_pil_image(img)

        verdict = "FAKE" if fake >= 0.5 else "REAL"
        confidence = max(fake, real) * 100

        logger.info(f"🖼  Image → {verdict} | fake={fake:.3f} real={real:.3f}")

        return jsonify({
            "verdict":         verdict,
            "fake_score":      round(fake * 100, 1),
            "real_score":      round(real * 100, 1),
            "confidence":      round(confidence, 1),
            "frames_analyzed": 1,
            "frame_scores":    [round(fake * 100, 1)],
            "file_type":       "image",
        })
    except Exception as e:
        logger.error(f"Image handling error: {e}")
        return jsonify({"error": f"Could not process image: {str(e)}"}), 500


def _handle_video(file):
    """Extract frames from video and deepfake-check each frame."""
    tmp_path = None
    try:
        # Save to temp file
        suffix = os.path.splitext(file.filename)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return jsonify({"error": "Cannot open video file."}), 400

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return jsonify({"error": "Video has no frames."}), 400

        # Sample up to 20 frames evenly
        num_samples = min(20, total_frames)
        frame_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)

        frame_scores = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            fake, _ = analyze_pil_image(pil)
            frame_scores.append(round(fake * 100, 1))
            logger.info(f"  Frame {idx}: fake={fake:.3f}")

        cap.release()

        if not frame_scores:
            return jsonify({"error": "No frames could be analyzed."}), 500

        avg_fake = sum(frame_scores) / len(frame_scores)
        verdict = "FAKE" if avg_fake >= 50 else "REAL"
        confidence = abs(avg_fake - 50) * 2  # 0-100 scale

        logger.info(f"🎬 Video → {verdict} | avg_fake={avg_fake:.1f}%")

        return jsonify({
            "verdict":         verdict,
            "fake_score":      round(avg_fake, 1),
            "real_score":      round(100 - avg_fake, 1),
            "confidence":      round(confidence, 1),
            "frames_analyzed": len(frame_scores),
            "frame_scores":    frame_scores,
            "file_type":       "video",
        })
    except Exception as e:
        logger.error(f"Video handling error: {e}")
        return jsonify({"error": f"Video processing failed: {str(e)}"}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"🚀 Starting DeepShield server on port {port}")
    logger.info("📡 Pre-loading model…")
    try:
        get_detector()
    except Exception as e:
        logger.warning(f"Model pre-load skipped: {e}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
