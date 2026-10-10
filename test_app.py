import os
import sys

# Ensure backend can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app import app
from backend.tesseract_service import detect_tesseract
from backend.comparison import compare_results, calculate_error_rate

def run_tests():
    print("[*] Testing Tesseract detection...")
    avail, info, err = detect_tesseract()
    print(f"    Available: {avail}, Info: {info}, Error: {err}")
    assert avail, "Tesseract should be available at D:\\Tesseract\\tesseract.exe"

    print("[*] Testing comparison metrics (Levenshtein, WER, CER)...")
    ref = "The quick brown fox jumps over the lazy dog."
    hyp = "The quick brown fox jumps over the lazy dog."
    wer_exact = calculate_error_rate(ref, hyp, mode="word")
    cer_exact = calculate_error_rate(ref, hyp, mode="char")
    assert wer_exact["rate"] == 0.0, "WER should be 0.0 for identical text"
    assert cer_exact["rate"] == 0.0, "CER should be 0.0 for identical text"
    print(f"    Exact match WER: {wer_exact['percentage_str']}, CER: {cer_exact['percentage_str']}")

    # One substitution and one insertion
    hyp2 = "The quick brown cat jumps over the very lazy dog."
    wer2 = calculate_error_rate(ref, hyp2, mode="word")
    print(f"    Modified text WER: {wer2['percentage_str']}, S={wer2['substitutions']}, I={wer2['insertions']}")
    assert wer2["rate"] > 0, "WER should detect errors"

    print("[*] Testing Flask API test client...")
    client = app.test_client()

    # 1. Health check
    res_health = client.get("/api/health")
    assert res_health.status_code == 200, f"Health check failed: {res_health.data}"
    health_json = res_health.get_json()
    assert health_json["status"] == "online"
    print(f"    /api/health -> OK (Tesseract: {health_json['tesseract']['available']}, AI configured: {health_json['ai_mode']['configured']})")

    # 2. List samples
    res_samples = client.get("/api/samples")
    assert res_samples.status_code == 200
    samples_json = res_samples.get_json()
    print(f"    /api/samples -> Found {len(samples_json['samples'])} test sample images")
    assert len(samples_json['samples']) > 0

    # 3. Load sample image
    sample_to_load = samples_json['samples'][0]['filename']
    res_load = client.post("/api/samples/load", json={"filename": sample_to_load})
    assert res_load.status_code == 200
    load_json = res_load.get_json()
    assert load_json["success"] is True
    image_id = load_json["data"]["image_id"]
    print(f"    /api/samples/load -> Loaded {sample_to_load} with image_id={image_id} ({load_json['data']['width']}x{load_json['data']['height']} px)")

    # 4. Process image with OpenCV
    options = {
        "grayscale": True,
        "contrast_enhancement": True,
        "clahe_clip_limit": 2.0,
        "gaussian_blur": True,
        "blur_kernel": 3,
        "adaptive_threshold": True,
        "adaptive_method": "gaussian",
        "resize": False
    }
    res_proc = client.post("/api/process", json={"image_id": image_id, "options": options})
    assert res_proc.status_code == 200
    proc_json = res_proc.get_json()
    assert proc_json["success"] is True
    print(f"    /api/process -> OK (Applied steps: {proc_json['data']['applied_steps']})")

    # 5. Run Tesseract OCR on RAW image
    res_ocr_raw = client.post("/api/ocr/tesseract", json={"image_id": image_id, "version": "raw", "psm": "3"})
    assert res_ocr_raw.status_code == 200
    raw_ocr_json = res_ocr_raw.get_json()
    assert raw_ocr_json["success"] is True
    print(f"    /api/ocr/tesseract (RAW) -> Extracted {raw_ocr_json['data']['word_count']} words in {raw_ocr_json['data']['duration_seconds']}s")

    # 6. Run Tesseract OCR on PROCESSED image
    res_ocr_proc = client.post("/api/ocr/tesseract", json={"image_id": image_id, "version": "processed", "psm": "3"})
    assert res_ocr_proc.status_code == 200
    proc_ocr_json = res_ocr_proc.get_json()
    assert proc_ocr_json["success"] is True
    print(f"    /api/ocr/tesseract (PROCESSED) -> Extracted {proc_ocr_json['data']['word_count']} words in {proc_ocr_json['data']['duration_seconds']}s")

    # 7. Comparison API
    tess_text = proc_ocr_json['data']['text']
    gt_text = tess_text # test with self as ground truth
    res_comp = client.post("/api/compare", json={
        "tesseract_text": tess_text,
        "ai_text": "Sample alternative text",
        "ground_truth": gt_text
    })
    assert res_comp.status_code == 200
    comp_json = res_comp.get_json()
    assert comp_json["success"] is True
    assert comp_json["data"]["evaluation"]["tesseract"]["wer"]["rate"] == 0.0
    print(f"    /api/compare -> OK (Tesseract vs Ground Truth WER: {comp_json['data']['evaluation']['tesseract']['wer']['percentage_str']})")

    # 8. Reset API
    res_reset = client.post("/api/reset", json={"image_id": image_id})
    assert res_reset.status_code == 200
    print(f"    /api/reset -> Workspace cleared successfully")

    print("\n[SUCCESS] ALL VERIFICATION TESTS PASSED PERFECTLY!")

if __name__ == "__main__":
    run_tests()
