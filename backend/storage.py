import os
import uuid
import time
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"

# Allowed extensions and MIME types
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_IMAGE_DIMENSION = 4000  # Prevent decompression bombs / extreme memory usage

def init_storage():
    """Ensure upload and processed directories exist."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def is_valid_extension(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS

def generate_image_id() -> str:
    return str(uuid.uuid4())

def save_uploaded_file(file_storage, session_id: str = None) -> dict:
    """
    Validates and saves an uploaded image file securely.
    Returns metadata dict: {image_id, filename, width, height, format, size_bytes, raw_path}
    """
    init_storage()
    original_filename = file_storage.filename or "upload"
    ext = Path(original_filename).suffix.lower()

    if not is_valid_extension(original_filename):
        raise ValueError(f"Unsupported file format '{ext}'. Allowed formats: PNG, JPG, JPEG, WEBP.")

    image_id = generate_image_id()
    safe_raw_filename = f"{image_id}_raw{ext}"
    raw_path = UPLOAD_DIR / safe_raw_filename

    # Save to disk
    file_storage.save(str(raw_path))

    # Verify actual image content using Pillow
    try:
        with Image.open(raw_path) as img:
            img.verify()
        
        # Re-open to read dimensions and format (verify() closes or invalidates the image)
        with Image.open(raw_path) as img:
            width, height = img.size
            img_format = img.format or ext.replace(".", "").upper()

            if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                # Remove file and raise error
                raw_path.unlink(missing_ok=True)
                raise ValueError(f"Image dimensions ({width}x{height}) exceed maximum allowed limit of {MAX_IMAGE_DIMENSION}px.")

    except Exception as e:
        raw_path.unlink(missing_ok=True)
        raise ValueError(f"Invalid or corrupted image file: {str(e)}")

    file_size = raw_path.stat().st_size
    if file_size > MAX_FILE_SIZE_BYTES:
        raw_path.unlink(missing_ok=True)
        raise ValueError(f"File size ({file_size / (1024*1024):.2f} MB) exceeds maximum allowed size of 10 MB.")

    return {
        "image_id": image_id,
        "filename": original_filename,
        "width": width,
        "height": height,
        "format": img_format,
        "size_bytes": file_size,
        "raw_path": str(raw_path),
        "raw_ext": ext
    }

def get_image_path(image_id: str, version: str) -> Path | None:
    """
    Safely resolves image file path for 'raw' or 'processed'.
    Guards against path traversal by enforcing image_id formatting.
    """
    # Strict validation of image_id as UUID
    try:
        uuid.UUID(image_id)
    except ValueError:
        return None

    if version == "raw":
        # Look in uploads
        for ext in ALLOWED_EXTENSIONS:
            candidate = UPLOAD_DIR / f"{image_id}_raw{ext}"
            if candidate.exists() and candidate.is_file():
                return candidate
    elif version == "processed":
        # Look in processed
        for ext in [".png", ".jpg", ".jpeg", ".webp"]:
            candidate = PROCESSED_DIR / f"{image_id}_processed{ext}"
            if candidate.exists() and candidate.is_file():
                return candidate
    return None

def get_processed_output_path(image_id: str, ext: str = ".png") -> Path:
    init_storage()
    return PROCESSED_DIR / f"{image_id}_processed{ext}"

def cleanup_session_images(image_id: str = None):
    """
    Removes files associated with image_id, or cleans temporary files older than 1 hour.
    """
    if image_id:
        try:
            uuid.UUID(image_id)
            for f in UPLOAD_DIR.glob(f"{image_id}_*"):
                try:
                    f.unlink()
                except Exception:
                    pass
            for f in PROCESSED_DIR.glob(f"{image_id}_*"):
                try:
                    f.unlink()
                except Exception:
                    pass
        except ValueError:
            pass
    else:
        # Periodic cleanup of files older than 3600 seconds
        now = time.time()
        for folder in [UPLOAD_DIR, PROCESSED_DIR]:
            if folder.exists():
                for f in folder.iterdir():
                    if f.is_file() and (now - f.stat().st_mtime > 3600):
                        try:
                            f.unlink()
                        except Exception:
                            pass
