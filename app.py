"""
DeepShield – Deepfake Detection System
Backend running a LOCAL ONNX deepfake-detection model (no API key needed).
Team: Idea Ignitors | AITM Kanpur

Model: onnx-community/Deep-Fake-Detector-v2-Model-ONNX (ViT, ~92% precision)
  - Detects DEEPFAKES (fake / manipulated human faces) → FAKE
  - Real photos → REAL
  - AI-generated images (Stable Diffusion etc.) → REAL (they aren't deepfakes)
"""

import os
import io
import threading
import tempfile
import logging
import requests
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
#  Local ONNX deepfake-detection model
#  Downloaded once on first use, then cached in memory. No API key.
# ─────────────────────────────────────────────────────────
HF_API_KEY = os.environ.get("HF_API_KEY", "")  # kept only for /health compat

ONNX_MODEL_URL = (
    "https://huggingface.co/onnx-community/Deep-Fake-Detector-v2-Model-ONNX/"
    "resolve/main/onnx/model_int8.onnx"
)
ONNX_MODEL_PATH = os.path.join(tempfile.gettempdir(), "deepfake_detector_int8.onnx")

# Verdict threshold (0-1): how high the model's "Deepfake" probability must be
# to call an image FAKE. 0.5 = the model's natural decision boundary.
# Lower it to catch more fakes; raise it to avoid flagging real images.
FAKE_THRESHOLD = float(os.environ.get("FAKE_THRESHOLD", "0.5"))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


# ─────────────────────────────────────────────────────────
#  ONNX model loader (lazy + thread-safe + cached)
# ─────────────────────────────────────────────────────────
_ONNX_SESSION = None        # None = not loaded, "FAILED" = gave up, else session
_ONNX_LOCK = threading.Lock()


def _get_onnx_session():
    """Download (once) and load the ONNX model. Returns session or None."""
    global _ONNX_SESSION
    if _ONNX_SESSION is not None:
        return None if _ONNX_SESSION == "FAILED" else _ONNX_SESSION

    with _ONNX_LOCK:
        if _ONNX_SESSION is not None:  # another thread loaded it already
            return None if _ONNX_SESSION == "FAILED" else _ONNX_SESSION
        try:
            import onnxruntime as ort

            if (not os.path.exists(ONNX_MODEL_PATH)
                    or os.path.getsize(ONNX_MODEL_PATH) < 10_000_000):
                logger.info("Downloading ONNX deepfake model (~86MB, one-time)…")
                with requests.get(ONNX_MODEL_URL, stream=True, timeout=180) as r:
                    r.raise_for_status()
                    with open(ONNX_MODEL_PATH, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            if chunk:
                                f.write(chunk)
                logger.info("Model download complete.")

            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _ONNX_SESSION = ort.InferenceSession(
                ONNX_MODEL_PATH, sess_options=so, providers=["CPUExecutionProvider"]
            )
            logger.info("ONNX deepfake model loaded.")
            return _ONNX_SESSION
        except Exception as e:
            logger.error(f"ONNX model load failed: {e} — using heuristic fallback")
            _ONNX_SESSION = "FAILED"
            return None


def _classify_onnx(img: Image.Image) -> tuple[float, float]:
    """Run the ONNX deepfake model on one PIL image. Returns (fake, real)."""
    # Preprocess: RGB → resize 224x224 (bilinear) → rescale 1/255 → normalize to [-1,1]
    img = img.convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5                       # mean=std=0.5 → range [-1, 1]
    arr = np.transpose(arr, (2, 0, 1))[None, ...]  # NCHW

    session = _get_onnx_session()
    if session is None:
        raise RuntimeError("ONNX session unavailable")

    inp_name = session.get_inputs()[0].name
    logits = session.run(None, {inp_name: arr})[0][0]   # [real_logit, fake_logit]
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    # id2label: 0 = Realism (REAL), 1 = Deepfake (FAKE)
    return float(probs[1]), float(probs[0])


# ─────────────────────────────────────────────────────────
#  Core: classify one image
# ─────────────────────────────────────────────────────────
def classify_image(img: Image.Image) -> tuple[float, float, bool]:
    """
    Classify one image as REAL or (deep)FAKE using the local ONNX model.
    Returns (fake_score, real_score, used_model). used_model is True when the
    real model answered; False when we fell back to the heuristic.
    """
    try:
        fake, real = _classify_onnx(img)
        logger.info(f"ONNX model → fake={fake:.3f} real={real:.3f}")
        return fake, real, True
    except Exception as e:
        logger.warning(f"ONNX inference unavailable ({e}); using heuristic fallback")
        return _heuristic_fallback(img)


def _heuristic_fallback(img: Image.Image) -> tuple[float, float, bool]:
    """
    Last-resort statistical estimate when the ONNX model can't load/run.
    NOT a real deepfake detector — it only looks at local pixel-variance
    smoothness, so treat its output as a rough guess (and it leans REAL so
    genuine photos are not wrongly accused).
    """
    import math
    arr = np.array(img.convert("RGB"), dtype=np.float32)

    h, w = arr.shape[:2]
    block = 16
    variances = []
    for y in range(0, h - block, block):
        for x in range(0, w - block, block):
            patch = arr[y:y + block, x:x + block]
            variances.append(float(np.var(patch)))
    avg_var = float(np.mean(variances)) if variances else 0.0

    variance_threshold = 300.0
    raw = math.tanh((variance_threshold - avg_var) / variance_threshold)  # >0 → fake
    fake = max(0.02, min(0.85, 0.3 + raw * 0.4))   # baseline 0.3 → leans REAL
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
    return jsonify({"status": "ok", "model": "onnx_deepfake",
                    "hf_key_set": bool(HF_API_KEY)})


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

        # Model uses its natural 0.5 boundary; heuristic also uses 0.5.
        thresh = FAKE_THRESHOLD if used_model else 0.5
        verdict = "FAKE" if fake >= thresh else "REAL"
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
            resp["warning"] = ("The AI model could not be loaded on the server, "
                               "so this is only a rough estimate — not a reliable "
                               "deepfake check.")
        return jsonify(resp)
    except Exception as e:
        logger.error(f"Image error: {e}")
        return jsonify({"error": str(e)}), 500


# ── Video handler ─────────────────────────────────────

def _analyze_video(file):
    import cv2  # lazy import — keeps image requests light on memory
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

        num_samples = min(20, total)
        frame_indices = np.linspace(0, total - 1, num_samples, dtype=int)
        frame_scores = []
        used_model_any = False

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
        thresh_pct = (FAKE_THRESHOLD if used_model_any else 0.5) * 100
        verdict = "FAKE" if avg_fake >= thresh_pct else "REAL"

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
            resp["warning"] = ("The AI model could not be loaded on the server, "
                               "so this is only a rough estimate — not a reliable "
                               "deepfake check.")
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
    logger.info("Deepfake detection runs on a local ONNX model (no API key needed). "
                "First request may take ~30s while the model downloads/loads.")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
