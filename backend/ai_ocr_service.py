import os
import base64
import time
import mimetypes
from pathlib import Path
import requests
from backend.storage import get_image_path

TRANSCRIPTION_PROMPT = (
    "Transcribe all visible text faithfully from this image. "
    "Preserve original line breaks, formatting, and natural reading order as closely as possible. "
    "Do not invent, hallucinate, or extrapolate missing, obscured, or illegible words. "
    "If text is cut off or illegible, transcribe only what is visibly identifiable. "
    "Return exclusively the extracted transcription without any greetings, commentary, markdown code fences, or explanations."
)

def get_ai_status() -> dict:
    """
    Checks environment for AI provider credentials.
    Returns status dict indicating availability and configuration details.
    """
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    provider = os.getenv("AI_PROVIDER", "").lower().strip()
    if not provider:
        if gemini_key:
            provider = "gemini"
        elif openai_key:
            provider = "openai"
        else:
            provider = "none"

    is_configured = False
    active_provider = None
    model_name = None

    if provider == "gemini" and gemini_key:
        is_configured = True
        active_provider = "Google Gemini"
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    elif provider == "openai" and openai_key:
        is_configured = True
        active_provider = "OpenAI Vision"
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    instructions = (
        "To enable AI OCR, create a .env file with your API key:\n"
        "- For Google Gemini: GEMINI_API_KEY=your_gemini_key (or GOOGLE_API_KEY)\n"
        "- For OpenAI: OPENAI_API_KEY=your_openai_key"
    )

    return {
        "configured": is_configured,
        "provider": active_provider,
        "model": model_name,
        "setup_instructions": instructions if not is_configured else None
    }

def encode_image(image_path: Path) -> tuple[str, str]:
    """Encodes image file to base64 string and determines MIME type."""
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type or not mime_type.startswith("image/"):
        ext = image_path.suffix.lower()
        if ext in [".jpg", ".jpeg"]:
            mime_type = "image/jpeg"
        elif ext == ".png":
            mime_type = "image/png"
        elif ext == ".webp":
            mime_type = "image/webp"
        else:
            mime_type = "image/jpeg"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return encoded, mime_type

def run_gemini_ocr(image_path: Path, api_key: str, model_name: str) -> str:
    """Calls Google Gemini GenerateContent API with vision payload."""
    encoded_img, mime_type = encode_image(image_path)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": TRANSCRIPTION_PROMPT},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": encoded_img
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 4096
        }
    }

    try:
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
    except requests.exceptions.Timeout:
        raise RuntimeError("Google Gemini API request timed out after 30 seconds.")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error connecting to Gemini API: {str(e)}")

    if response.status_code != 200:
        error_msg = f"Gemini API error (HTTP {response.status_code})"
        try:
            err_json = response.json()
            if "error" in err_json and "message" in err_json["error"]:
                error_msg += f": {err_json['error']['message']}"
        except Exception:
            error_msg += f": {response.text[:200]}"
        raise RuntimeError(error_msg)

    data = response.json()
    try:
        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        text_pieces = [p.get("text", "") for p in parts if "text" in p]
        return "\n".join(text_pieces).strip()
    except Exception as e:
        raise RuntimeError(f"Unexpected response format from Gemini: {str(e)}")

def run_openai_ocr(image_path: Path, api_key: str, model_name: str) -> str:
    """Calls OpenAI chat completions API with vision payload."""
    encoded_img, mime_type = encode_image(image_path)
    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_name,
        "temperature": 0.0,
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": TRANSCRIPTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{encoded_img}"
                        }
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=35)
    except requests.exceptions.Timeout:
        raise RuntimeError("OpenAI API request timed out after 35 seconds.")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error connecting to OpenAI API: {str(e)}")

    if response.status_code != 200:
        error_msg = f"OpenAI API error (HTTP {response.status_code})"
        try:
            err_json = response.json()
            if "error" in err_json and "message" in err_json["error"]:
                error_msg += f": {err_json['error']['message']}"
        except Exception:
            error_msg += f": {response.text[:200]}"
        raise RuntimeError(error_msg)

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(f"Unexpected response format from OpenAI: {str(e)}")

def run_ai_ocr(image_id: str, version: str = "raw") -> dict:
    """
    Executes Vision AI transcription using configured provider.
    Returns:
    {
        "text": str,
        "duration_seconds": float,
        "char_count": int,
        "word_count": int,
        "provider": str,
        "model": str,
        "version_used": str
    }
    """
    status = get_ai_status()
    if not status["configured"]:
        raise RuntimeError(
            "AI Mode is not configured. Please supply an API key (GEMINI_API_KEY or OPENAI_API_KEY) in the .env file."
        )

    img_path = get_image_path(image_id, version)
    if not img_path or not img_path.exists():
        raise FileNotFoundError(f"{version.capitalize()} image for ID '{image_id}' not found.")

    provider_name = status["provider"]
    model_name = status["model"]
    start_time = time.perf_counter()

    if "Gemini" in provider_name:
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        extracted_text = run_gemini_ocr(img_path, key, model_name)
    elif "OpenAI" in provider_name:
        key = os.getenv("OPENAI_API_KEY")
        extracted_text = run_openai_ocr(img_path, key, model_name)
    else:
        raise RuntimeError(f"Unsupported AI provider: {provider_name}")

    elapsed = time.perf_counter() - start_time
    text_clean = extracted_text.strip()
    words = text_clean.split()

    return {
        "text": text_clean,
        "duration_seconds": round(elapsed, 3),
        "char_count": len(text_clean),
        "word_count": len(words),
        "provider": provider_name,
        "model": model_name,
        "version_used": version
    }
