import os
import shutil
import time
from pathlib import Path
from PIL import Image
import pytesseract
from backend.storage import get_image_path

# Standard Windows paths to probe if Tesseract is not in system PATH
COMMON_TESSERACT_PATHS = [
    os.getenv("TESSERACT_CMD"),
    r"D:\Tesseract\tesseract.exe",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\User\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
]

PSM_MODES = {
    "1": "Automatic page segmentation with OSD",
    "3": "Fully automatic page segmentation (Default)",
    "4": "Assume a single column of text of variable sizes",
    "6": "Assume a single uniform block of text",
    "7": "Treat the image as a single text line",
    "8": "Treat the image as a single word",
    "11": "Sparse text: Find as much text as possible in no particular order",
    "12": "Sparse text with OSD"
}

def detect_tesseract():
    """
    Detects and configures the Tesseract OCR executable.
    Returns (is_available, configured_path_or_version_info, error_message)
    """
    # 1. Check custom or common paths
    for path in COMMON_TESSERACT_PATHS:
        if path and Path(path).exists() and Path(path).is_file():
            pytesseract.pytesseract.tesseract_cmd = str(path)
            try:
                ver = pytesseract.get_tesseract_version()
                return True, f"v{ver} ({path})", None
            except Exception as e:
                pass

    # 2. Check if tesseract is available in system PATH
    system_tesseract = shutil.which("tesseract")
    if system_tesseract:
        pytesseract.pytesseract.tesseract_cmd = system_tesseract
        try:
            ver = pytesseract.get_tesseract_version()
            return True, f"v{ver} ({system_tesseract})", None
        except Exception as e:
            pass

    # 3. Try default pytesseract configuration
    try:
        ver = pytesseract.get_tesseract_version()
        return True, f"v{ver}", None
    except pytesseract.TesseractNotFoundError:
        return False, None, (
            "Tesseract OCR executable was not found on your system. "
            "Please install Tesseract (e.g. run tesseract_installer.exe or download from "
            "https://github.com/UB-Mannheim/tesseract/wiki) and set TESSERACT_CMD in your .env file."
        )
    except Exception as e:
        return False, None, f"Tesseract error: {str(e)}"

def get_installed_languages() -> list:
    """Returns list of installed languages, or ['eng'] default."""
    try:
        langs = pytesseract.get_languages()
        if langs and "osd" in langs:
            langs = [l for l in langs if l != "osd"]
        return sorted(langs) if langs else ["eng"]
    except Exception:
        return ["eng"]

def run_tesseract_ocr(image_id: str, version: str = "raw", lang: str = "eng", psm: str = "3", oem: str = "3") -> dict:
    """
    Runs Tesseract OCR on the specified image ('raw' or 'processed').
    Returns {
        "text": str,
        "duration_seconds": float,
        "char_count": int,
        "word_count": int,
        "engine": str,
        "version_used": str,
        "config_used": str
    }
    """
    is_avail, info, err = detect_tesseract()
    if not is_avail:
        raise RuntimeError(err or "Tesseract is not installed or configured.")

    img_path = get_image_path(image_id, version)
    if not img_path or not img_path.exists():
        raise FileNotFoundError(f"{version.capitalize()} image for ID '{image_id}' not found.")

    # Validate psm and oem
    psm_clean = str(psm).strip() if str(psm).strip() in PSM_MODES else "3"
    oem_clean = str(oem).strip() if str(oem).strip() in ["0", "1", "2", "3"] else "3"
    custom_config = f"--oem {oem_clean} --psm {psm_clean}"

    start_time = time.perf_counter()
    with Image.open(img_path) as pil_img:
        extracted_text = pytesseract.image_to_string(pil_img, lang=lang, config=custom_config)
    elapsed = time.perf_counter() - start_time

    # Normalize extracted text
    text_clean = extracted_text.strip()
    words = text_clean.split()
    char_count = len(text_clean)
    word_count = len(words)

    return {
        "text": text_clean,
        "duration_seconds": round(elapsed, 3),
        "char_count": char_count,
        "word_count": word_count,
        "engine": f"Tesseract OCR {info}",
        "version_used": version,
        "config_used": custom_config,
        "lang_used": lang
    }
