"""
Aerial Object Detection System
--------------------------------
A Flask backend that serves a YOLOv8 model trained to detect
AirPlane, Drone, and Helicopter objects in uploaded images.

The model is loaded once at server startup and reused across
requests for fast inference.
"""

import os
import time
import uuid

from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
CONFIDENCE_THRESHOLD = 0.4
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB upload limit

# Class names must match the order the model was trained with.
CLASS_NAMES = {0: "AirPlane", 1: "Drone", 2: "Helicopter"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# --------------------------------------------------------------------------
# Load model once at startup
# --------------------------------------------------------------------------

model = None
model_load_error = None

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model weights not found at '{MODEL_PATH}'. "
            "Place your trained 'best.pt' file in the project root."
        )
    model = YOLO(MODEL_PATH)
    print(f"[INFO] YOLO model loaded successfully from {MODEL_PATH}")
except Exception as exc:  # noqa: BLE001 - we want to surface any load error
    model_load_error = str(exc)
    print(f"[ERROR] Failed to load model: {model_load_error}")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    """Check whether the uploaded file has an allowed image extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def validate_image(filepath: str) -> bool:
    """Verify that the saved file is actually a readable image."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def run_inference(image_path: str, output_path: str):
    """
    Run YOLOv8 inference on a single image, save the annotated result,
    and return a list of detections plus the inference time in ms.
    """
    start_time = time.time()
    results = model.predict(
        source=image_path,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
    )
    inference_ms = round((time.time() - start_time) * 1000)

    result = results[0]

    # Save the annotated image (YOLO draws boxes + labels automatically).
    annotated = result.plot()  # returns a BGR numpy array
    import cv2  # local import keeps startup fast if cv2 isn't needed elsewhere

    cv2.imwrite(output_path, annotated)

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = CLASS_NAMES.get(cls_id, model.names.get(cls_id, "Unknown"))
            detections.append(
                {
                    "class": class_name,
                    "confidence": round(confidence, 4),
                }
            )

    # Highest confidence first, for a nicer table on the frontend.
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    return detections, inference_ms


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the main web page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept an uploaded image, run YOLOv8 inference, and return the
    annotated result image path along with detection details.
    """
    if model is None:
        return (
            jsonify(
                {
                    "success": False,
                    "error": (
                        "Model is not loaded on the server. "
                        f"Reason: {model_load_error or 'unknown error'}"
                    ),
                }
            ),
            503,
        )

    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file was uploaded."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"success": False, "error": "No file was selected."}), 400

    if not allowed_file(file.filename):
        return (
            jsonify(
                {
                    "success": False,
                    "error": (
                        "Unsupported file type. Allowed types: "
                        + ", ".join(sorted(ALLOWED_EXTENSIONS))
                    ),
                }
            ),
            400,
        )

    # Generate a unique, safe filename to avoid collisions and overwrites.
    unique_id = uuid.uuid4().hex[:10]
    safe_name = secure_filename(file.filename)
    base_name, ext = os.path.splitext(safe_name)
    upload_filename = f"{base_name}_{unique_id}{ext}"
    output_filename = f"result_{base_name}_{unique_id}{ext}"

    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], upload_filename)
    output_path = os.path.join(app.config["OUTPUT_FOLDER"], output_filename)

    try:
        file.save(upload_path)
    except Exception:
        return jsonify({"success": False, "error": "Failed to save the uploaded file."}), 500

    if not validate_image(upload_path):
        os.remove(upload_path)
        return (
            jsonify({"success": False, "error": "The uploaded file is not a valid image."}),
            400,
        )

    try:
        detections, inference_ms = run_inference(upload_path, output_path)
    except Exception as exc:  # noqa: BLE001
        return (
            jsonify({"success": False, "error": f"Prediction failed: {exc}"}),
            500,
        )

    return jsonify(
        {
            "success": True,
            "original_image": f"/uploads/{upload_filename}",
            "image": f"/outputs/{output_filename}",
            "detections": detections,
            "count": len(detections),
            "inference_time": f"{inference_ms} ms",
        }
    )


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    """Serve original uploaded images."""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    """Serve annotated prediction images."""
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)


@app.errorhandler(413)
def file_too_large(_error):
    return (
        jsonify({"success": False, "error": "File is too large. Maximum size is 16 MB."}),
        413,
    )


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"success": False, "error": "Resource not found."}), 404


@app.errorhandler(500)
def server_error(_error):
    return jsonify({"success": False, "error": "Internal server error."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)
