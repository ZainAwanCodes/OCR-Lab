# OCR Vision Lab

> **Week 5 Assignment: Computer Vision & Intelligent Text Recognition System**  
> *Demonstrating the Impact of Image Preprocessing on Traditional OCR vs. Multimodal Vision AI*

---

## 1. Overview & Purpose

**OCR Vision Lab** is a full-stack educational web application designed for university-level computer vision and artificial intelligence coursework. It allows students and instructors to:
1. Upload and inspect original **RAW document images**.
2. Apply modular, real-time **OpenCV image preprocessing filters** (Grayscale, CLAHE Contrast Enhancement, Gaussian Denoising, Adaptive Thresholding, Otsu Thresholding, Bicubic Upscaling).
3. Execute traditional local optical character recognition via **Tesseract OCR**.
4. Execute modern cloud vision recognition via **Multimodal AI (Google Gemini or OpenAI Vision)**.
5. Benchmark execution duration, character count, word count, and objectively compute **Word Error Rate (WER)** and **Character Error Rate (CER)** against ground-truth transcriptions.

---

## 2. Architecture & Technology Stack

- **Backend**:
  - **Python 3.10+**
  - **Flask**: Lightweight REST API web server
  - **OpenCV (`opencv-python`)**: Computer vision filtering and transformations
  - **Pillow (`PIL`)**: Image validation, dimension verification, and decompression safety
  - **`pytesseract`**: Python wrapper for local Tesseract OCR engine
  - **`requests`**: Secure direct API integration with Gemini/OpenAI vision models
  - **`python-dotenv`**: Environment variable management

- **Frontend**:
  - **HTML5**: Semantic document structure
  - **CSS3 (Vanilla)**: Laboratory dashboard theme, CSS Grid & Flexbox, no frameworks
  - **Vanilla JavaScript (ES6+)**: Asynchronous `fetch()` API calls, drag-and-drop file management, client-side metric calculations

---

## 3. Directory Structure

```
week-5-project/
├── app.py                     # Flask entrypoint and REST API routes
├── requirements.txt           # Python dependencies
├── .env.example               # Template environment configuration
├── .env                       # Local environment configuration (Tesseract & API keys)
├── .gitignore                 # Excluded files and folders
├── README.md                  # System manual and instructions
├── backend/
│   ├── __init__.py            # Python package initializer
│   ├── storage.py             # File validation, UUID generation, cleanup
│   ├── image_processing.py    # OpenCV preprocessing pipeline & explanations
│   ├── tesseract_service.py   # Tesseract execution & PSM/language detection
│   ├── ai_ocr_service.py      # Real Vision AI integration (Gemini & OpenAI)
│   └── comparison.py          # Levenshtein distance WER, CER & analytics
├── templates/
│   └── index.html             # Single-page laboratory interface
├── static/
│   ├── css/
│   │   └── style.css          # Modern laboratory design system & responsive layout
│   └── js/
│       └── app.js             # Pure vanilla JavaScript frontend controller
├── samples/                   # Pre-bundled university test documents
├── uploads/                   # Ephemeral raw image storage (UUID-isolated)
└── processed/                 # Ephemeral processed image storage
```

---

## 4. Setup & Installation (Windows)

### Step 1: Clone or Navigate to the Project Folder
Open PowerShell or Command Prompt:
```powershell
cd C:\Users\User\Desktop\OCR\week-5-project
```

### Step 2: Set Up Python Virtual Environment
Use the existing Python 3 environment or create a dedicated virtual environment:
```powershell
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
Install the required packages using `pip`:
```powershell
pip install -r requirements.txt
```

### Step 4: Tesseract OCR Engine Setup
1. If Tesseract OCR is not yet installed on your Windows machine, run `tesseract_installer.exe` (found in the root `OCR` directory) or download the official Windows installer from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki).
2. Note the installation path (typically `D:\Tesseract\tesseract.exe` or `C:\Program Files\Tesseract-OCR\tesseract.exe`).
3. Verify that Tesseract runs:
   ```powershell
   & "D:\Tesseract\tesseract.exe" --version
   ```

### Step 5: Configure Environment Variables (`.env`)
Copy `.env.example` to `.env` or edit `.env`:
```ini
PORT=5000
FLASK_ENV=development

# Tesseract executable location on Windows
TESSERACT_CMD=D:\Tesseract\tesseract.exe

# AI Vision Configuration (Optional)
AI_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
```

> **Note on AI Mode**: If you do not provide an API key, the application will run in **Tesseract & Image Processing mode** without any errors. The AI button will display clear configuration guidance, and all OpenCV and Tesseract features remain fully operational.

---

## 5. Running the Application

Start the Flask server:
```powershell
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 6. How the Image Preprocessing Pipeline Works

| Preprocessing Technique | OpenCV Implementation | Computer Vision Principle |
| :--- | :--- | :--- |
| **1. Grayscale** | `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` | Discards chrominance channels, reducing 3D color arrays to 1D luminance values and eliminating color cast noise. |
| **2. CLAHE Contrast** | `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))` | Adaptive histogram equalization over local tiles; enhances faint ink and faded characters without over-amplifying background noise. |
| **3. Gaussian Denoising** | `cv2.GaussianBlur(img, (k, k), 0)` | Convolves the image with a bell-shaped kernel to suppress high-frequency speckle and scanner sensor noise. |
| **4. Adaptive Threshold** | `cv2.adaptiveThreshold(..., ADAPTIVE_THRESH_GAUSSIAN_C, ...)` | Calculates local binarization thresholds for neighborhood blocks; crucial for documents with uneven lighting or shadows. |
| **5. Otsu Threshold** | `cv2.threshold(..., THRESH_BINARY + THRESH_OTSU)` | Computes a global threshold that mathematically minimizes intra-class variance between dark foreground ink and bright background paper. |
| **6. Bicubic Upscaling** | `cv2.resize(..., interpolation=cv2.INTER_CUBIC)` | Upscales low-DPI or low-resolution text so character height exceeds Tesseract's optimal feature detection threshold (~30px). |

---

## 7. Accuracy Metrics: WER and CER

The application uses the Levenshtein dynamic programming edit-distance algorithm to calculate:

$$\text{WER} = \frac{\text{Substitutions} + \text{Deletions} + \text{Insertions}}{\text{Number of Reference Words}}$$

$$\text{CER} = \frac{\text{Character Substitutions} + \text{Character Deletions} + \text{Character Insertions}}{\text{Number of Reference Characters}}$$

### Key Academic Concepts:
- **Lower is Better**: A WER/CER of 0% indicates an exact character-for-character reproduction.
- **Can Exceed 100%**: If an OCR engine hallucinates or outputs long strings of false characters (excessive insertions), the error rate mathematically exceeds 100%.
- **Output Volume $\neq$ Accuracy**: A method that produces 500 characters is not necessarily better than one producing 300 characters. Without a ground-truth transcription, comparing only character counts is statistically invalid.

---

## 8. Verification & Test Scenarios

The `samples/` directory includes 4 pre-loaded university test documents:
1. **Clean Invoice (`1_Clean_Invoice.png`)**: High-contrast, clean layout. Tesseract and AI both achieve near 0% WER.
2. **ID Card (`2_ID_Card.png`)**: Mixed field labels and numbers. Tests OCR's ability to handle tabular layouts.
3. **Medical Receipt (`3_Medical_Receipt.png`)**: Dense receipt format with prices, codes, and numerical data.
4. **Noisy Document (`4_Noisy_Document.png`)**: Degraded document demonstrating the dramatic impact of Gaussian blur and adaptive thresholding on OCR readability.

---

## 9. Troubleshooting

- **"Tesseract OCR executable was not found"**:
  - Verify that Tesseract is installed and update `TESSERACT_CMD` in `.env` to point to `tesseract.exe`.
- **"Port 5000 is already in use"**:
  - Change `PORT=5001` in `.env` and restart `python app.py`.
- **"AI Mode: Key Missing"**:
  - Obtain a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey) and set `GEMINI_API_KEY=your_key` in `.env`.
