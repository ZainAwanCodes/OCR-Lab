import cv2
import numpy as np
from pathlib import Path
from backend.storage import get_image_path, get_processed_output_path

TECHNIQUE_DESCRIPTIONS = {
    "grayscale": (
        "Grayscale conversion transforms a 3-channel BGR color image into a single 8-bit luminance channel. "
        "This eliminates chromatic noise and reduces data complexity by 66% while preserving spatial text details."
    ),
    "gaussian_blur": (
        "Gaussian denoising convolves the image with a bell-shaped Gaussian kernel to suppress high-frequency speckle "
        "and sensor noise. Note: Overly aggressive kernel sizes can blur fine font serifs and thin character strokes."
    ),
    "adaptive_threshold": (
        "Adaptive thresholding calculates distinct local threshold cutoffs for small pixel neighborhoods. "
        "This is highly effective for uneven lighting, shadows, camera glare, and gradient backgrounds."
    ),
    "otsu_threshold": (
        "Otsu's binarization algorithm calculates an optimal global threshold that minimizes intra-class variance "
        "between foreground text pixels and background paper. Ideal for images with a bimodal histogram."
    ),
    "contrast_enhancement": (
        "Contrast Limited Adaptive Histogram Equalization (CLAHE) enhances local contrast across small contextual tiles "
        "while capping amplification to prevent noise over-saturation. Excellent for faded or low-contrast prints."
    ),
    "resize": (
        "Bicubic interpolation (INTER_CUBIC) upscales the image. Low-resolution text (e.g. font height < 20-30px) "
        "often confuses OCR engines; resizing provides cleaner character contours for feature extraction."
    )
}

def process_image(image_id: str, options: dict) -> dict:
    """
    Executes selected OpenCV preprocessing techniques sequentially on the raw image.
    The raw image is NEVER overwritten; output is saved to the processed directory.

    options dict schema:
    {
        "grayscale": bool,
        "gaussian_blur": bool,
        "blur_kernel": int (odd, 3..15),
        "adaptive_threshold": bool,
        "adaptive_method": "gaussian" | "mean",
        "adaptive_block_size": int (odd, 3..51),
        "otsu_threshold": bool,
        "contrast_enhancement": bool,
        "clahe_clip_limit": float (1.0..5.0),
        "resize": bool,
        "resize_scale": float (1.2..3.0)
    }
    """
    raw_path = get_image_path(image_id, "raw")
    if not raw_path or not raw_path.exists():
        raise FileNotFoundError(f"Raw image for ID '{image_id}' not found.")

    # Read image using OpenCV (using IMREAD_COLOR to ensure 3 channels)
    # Using np.fromfile to handle non-ASCII / Windows paths reliably
    img_data = np.fromfile(str(raw_path), dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Failed to decode image with OpenCV.")

    applied_steps = []
    explanations = []
    
    current_img = img.copy()
    is_grayscale = False

    # 1. Resize/Upscale (if selected early to improve downstream filters)
    if options.get("resize"):
        scale = float(options.get("resize_scale", 2.0))
        scale = max(1.1, min(scale, 3.5))  # Safe bounds
        h, w = current_img.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Enforce maximum safety dimensions
        if new_w <= 4000 and new_h <= 4000:
            current_img = cv2.resize(current_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            applied_steps.append(f"Resized / Upscaled ({scale}x)")
            explanations.append(TECHNIQUE_DESCRIPTIONS["resize"])

    # 2. Grayscale conversion
    # Note: Thresholding and CLAHE require single-channel image, so auto-grayscale if thresholding/clahe selected
    needs_grayscale = (
        options.get("grayscale") or 
        options.get("adaptive_threshold") or 
        options.get("otsu_threshold") or 
        options.get("contrast_enhancement")
    )

    if needs_grayscale and len(current_img.shape) == 3 and current_img.shape[2] == 3:
        current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
        is_grayscale = True
        applied_steps.append("Grayscale Conversion")
        explanations.append(TECHNIQUE_DESCRIPTIONS["grayscale"])
    elif len(current_img.shape) == 2:
        is_grayscale = True

    # 3. Contrast enhancement (CLAHE)
    if options.get("contrast_enhancement"):
        clip_limit = float(options.get("clahe_clip_limit", 2.0))
        clip_limit = max(1.0, min(clip_limit, 5.0))
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        if is_grayscale:
            current_img = clahe.apply(current_img)
        else:
            # LAB color space CLAHE for color images
            lab = cv2.cvtColor(current_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = clahe.apply(l)
            current_img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        applied_steps.append(f"Contrast Enhancement (CLAHE clip={clip_limit})")
        explanations.append(TECHNIQUE_DESCRIPTIONS["contrast_enhancement"])

    # 4. Gaussian Denoising
    if options.get("gaussian_blur"):
        ksize = int(options.get("blur_kernel", 3))
        if ksize % 2 == 0:
            ksize += 1
        ksize = max(3, min(ksize, 15))
        current_img = cv2.GaussianBlur(current_img, (ksize, ksize), 0)
        applied_steps.append(f"Gaussian Denoising (Kernel: {ksize}x{ksize})")
        explanations.append(TECHNIQUE_DESCRIPTIONS["gaussian_blur"])

    # 5. Thresholding (Adaptive or Otsu - Adaptive takes precedence if both requested, or Otsu)
    if options.get("adaptive_threshold"):
        if not is_grayscale:
            current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
            is_grayscale = True

        block_size = int(options.get("adaptive_block_size", 11))
        if block_size % 2 == 0:
            block_size += 1
        block_size = max(3, min(block_size, 51))
        
        c_param = int(options.get("adaptive_c", 2))
        method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if options.get("adaptive_method") != "mean" else cv2.ADAPTIVE_THRESH_MEAN_C
        method_name = "Gaussian" if method == cv2.ADAPTIVE_THRESH_GAUSSIAN_C else "Mean"
        
        current_img = cv2.adaptiveThreshold(
            current_img, 255, method, cv2.THRESH_BINARY, block_size, c_param
        )
        applied_steps.append(f"Adaptive Thresholding ({method_name}, block={block_size})")
        explanations.append(TECHNIQUE_DESCRIPTIONS["adaptive_threshold"])

    elif options.get("otsu_threshold"):
        if not is_grayscale:
            current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
            is_grayscale = True

        # Otsu thresholding
        thresh_val, current_img = cv2.threshold(current_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        applied_steps.append(f"Otsu Thresholding (Optimal threshold={int(thresh_val)})")
        explanations.append(TECHNIQUE_DESCRIPTIONS["otsu_threshold"])

    # If no options selected, apply grayscale by default as minimal enhancement
    if not applied_steps:
        if len(current_img.shape) == 3 and current_img.shape[2] == 3:
            current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
        applied_steps.append("Grayscale (Default)")
        explanations.append(TECHNIQUE_DESCRIPTIONS["grayscale"])

    # Save output image
    output_path = get_processed_output_path(image_id, ext=".png")
    is_success, encoded_img = cv2.imencode(".png", current_img)
    if not is_success:
        raise ValueError("Failed to encode processed image.")

    with open(output_path, "wb") as f:
        f.write(encoded_img.tobytes())

    # Obtain output details
    if len(current_img.shape) == 2:
        out_h, out_w = current_img.shape
        channels = 1
    else:
        out_h, out_w, channels = current_img.shape

    return {
        "image_id": image_id,
        "processed_url": f"/api/image/{image_id}/processed",
        "width": out_w,
        "height": out_h,
        "channels": channels,
        "applied_steps": applied_steps,
        "explanations": explanations
    }
