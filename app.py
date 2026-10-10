import os
import shutil
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from backend.storage import (
    init_storage,
    save_uploaded_file,
    get_image_path,
    cleanup_session_images,
    generate_image_id,
    UPLOAD_DIR,
    ALLOWED_EXTENSIONS
)
from backend.image_processing import process_image, TECHNIQUE_DESCRIPTIONS
from backend.tesseract_service import detect_tesseract, run_tesseract_ocr, get_installed_languages, PSM_MODES
from backend.ai_ocr_service import get_ai_status, run_ai_ocr
from backend.comparison import compare_results

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12 MB limit

# Initialize storage directories
init_storage()

# Pre-populate sample images if available
SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
SAMPLES_DIR.mkdir(exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/health", methods=["GET"])
def health_check():
    """
    Reports backend status, Tesseract availability, and AI Mode configuration.
    """
    tess_avail, tess_info, tess_err = detect_tesseract()
    ai_status = get_ai_status()
    languages = get_installed_languages() if tess_avail else []

    return jsonify({
        "status": "online",
        "tesseract": {
            "available": tess_avail,
            "info": tess_info,
            "error": tess_err,
            "languages": languages,
            "psm_modes": PSM_MODES
        },
        "ai_mode": ai_status,
        "techniques": TECHNIQUE_DESCRIPTIONS
    })

@app.route("/api/upload", methods=["POST"])
def upload_image():
    """
    Handles image upload, validation, and storage.
    """
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided in request."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No selected file."}), 400

    try:
        meta = save_uploaded_file(file)
        meta["raw_url"] = f"/api/image/{meta['image_id']}/raw"
        return jsonify({
            "success": True,
            "data": meta
        })
    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Upload processing failed: {str(e)}"}), 500

@app.route("/api/process", methods=["POST"])
def process():
    """
    Applies modular OpenCV preprocessing on the specified raw image.
    """
    data = request.get_json(silent=True) or {}
    image_id = data.get("image_id")
    options = data.get("options", {})

    if not image_id:
        return jsonify({"success": False, "error": "Missing 'image_id' parameter."}), 400

    try:
        result = process_image(image_id, options)
        return jsonify({
            "success": True,
            "data": result
        })
    except FileNotFoundError as fnf:
        return jsonify({"success": False, "error": str(fnf)}), 404
    except Exception as e:
        return jsonify({"success": False, "error": f"Image processing failed: {str(e)}"}), 500

@app.route("/api/ocr/tesseract", methods=["POST"])
def ocr_tesseract():
    """
    Executes Tesseract OCR on raw or processed image.
    """
    data = request.get_json(silent=True) or {}
    image_id = data.get("image_id")
    version = data.get("version", "raw")
    lang = data.get("lang", "eng")
    psm = data.get("psm", "3")
    oem = data.get("oem", "3")

    if not image_id:
        return jsonify({"success": False, "error": "Missing 'image_id' parameter."}), 400

    if version not in ["raw", "processed"]:
        return jsonify({"success": False, "error": "Version must be either 'raw' or 'processed'."}), 400

    try:
        ocr_result = run_tesseract_ocr(image_id, version=version, lang=lang, psm=psm, oem=oem)
        return jsonify({
            "success": True,
            "data": ocr_result
        })
    except FileNotFoundError as fnf:
        return jsonify({"success": False, "error": str(fnf)}), 404
    except RuntimeError as re:
        return jsonify({"success": False, "error": str(re)}), 503
    except Exception as e:
        return jsonify({"success": False, "error": f"Tesseract execution error: {str(e)}"}), 500

@app.route("/api/ocr/ai", methods=["POST"])
def ocr_ai():
    """
    Executes Vision AI transcription on raw or processed image.
    """
    data = request.get_json(silent=True) or {}
    image_id = data.get("image_id")
    version = data.get("version", "raw")

    if not image_id:
        return jsonify({"success": False, "error": "Missing 'image_id' parameter."}), 400

    if version not in ["raw", "processed"]:
        return jsonify({"success": False, "error": "Version must be either 'raw' or 'processed'."}), 400

    try:
        ai_result = run_ai_ocr(image_id, version=version)
        return jsonify({
            "success": True,
            "data": ai_result
        })
    except FileNotFoundError as fnf:
        return jsonify({"success": False, "error": str(fnf)}), 404
    except RuntimeError as re:
        return jsonify({"success": False, "error": str(re)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"AI OCR error: {str(e)}"}), 500

@app.route("/api/compare", methods=["POST"])
def compare():
    """
    Generates comparison between Tesseract and AI OCR, with WER/CER if ground truth is supplied.
    """
    data = request.get_json(silent=True) or {}
    tesseract_text = data.get("tesseract_text", "")
    ai_text = data.get("ai_text", "")
    ground_truth = data.get("ground_truth", "")
    ignore_case = data.get("ignore_case", True)

    try:
        report = compare_results(tesseract_text, ai_text, ground_truth, ignore_case=ignore_case)
        return jsonify({
            "success": True,
            "data": report
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"Comparison calculation failed: {str(e)}"}), 500

@app.route("/api/reset", methods=["POST"])
def reset():
    """
    Resets temporary files and returns confirmation.
    """
    data = request.get_json(silent=True) or {}
    image_id = data.get("image_id")
    cleanup_session_images(image_id)
    return jsonify({"success": True, "message": "Workspace reset successfully."})

@app.route("/api/image/<image_id>/<version>", methods=["GET"])
def serve_image(image_id, version):
    """
    Serves requested raw or processed image securely.
    """
    if version not in ["raw", "processed"]:
        return jsonify({"error": "Invalid image version requested."}), 400

    img_path = get_image_path(image_id, version)
    if not img_path or not img_path.exists():
        return jsonify({"error": "Image not found."}), 404

    # Determine mime type
    ext = img_path.suffix.lower()
    mimetypes_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp"
    }
    mimetype = mimetypes_map.get(ext, "application/octet-stream")
    return send_file(img_path, mimetype=mimetype)

@app.route("/api/samples", methods=["GET"])
def list_samples():
    """Returns available sample test images."""
    samples = []
    if SAMPLES_DIR.exists():
        for p in SAMPLES_DIR.iterdir():
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS:
                samples.append({
                    "id": p.stem,
                    "filename": p.name,
                    "size_bytes": p.stat().st_size
                })
    return jsonify({"samples": samples})

@app.route("/api/samples/load", methods=["POST"])
def load_sample():
    """Copies a sample image into active uploads as if the user uploaded it."""
    data = request.get_json(silent=True) or {}
    sample_filename = data.get("filename")
    if not sample_filename:
        return jsonify({"success": False, "error": "Missing filename parameter."}), 400

    sample_path = SAMPLES_DIR / sample_filename
    if not sample_path.exists() or not sample_path.is_file():
        return jsonify({"success": False, "error": "Sample image not found."}), 404

    # Copy to uploads under new UUID
    image_id = generate_image_id()
    ext = sample_path.suffix.lower()
    dest_path = UPLOAD_DIR / f"{image_id}_raw{ext}"
    shutil.copyfile(sample_path, dest_path)

    from PIL import Image as PilImage
    with PilImage.open(dest_path) as img:
        w, h = img.size
        fmt = img.format or ext.replace(".", "").upper()

    return jsonify({
        "success": True,
        "data": {
            "image_id": image_id,
            "filename": sample_filename,
            "width": w,
            "height": h,
            "format": fmt,
            "size_bytes": dest_path.stat().st_size,
            "raw_url": f"/api/image/{image_id}/raw"
        }
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"[*] Starting OCR Vision Lab on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=True)
