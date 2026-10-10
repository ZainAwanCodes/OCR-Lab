/**
 * OCR Vision Lab — Frontend Laboratory Controller
 * Week 5 Assignment: Computer Vision & Intelligent Text Recognition
 * Pure Vanilla JavaScript (ES6+)
 */

document.addEventListener("DOMContentLoaded", () => {
  // Global State
  const state = {
    currentImageId: null,
    currentRawMeta: null,
    processedAvailable: false,
    tesseractAvailable: false,
    aiConfigured: false,
    tesseractResult: { text: "", duration: null, charCount: 0, wordCount: 0 },
    aiResult: { text: "", duration: null, charCount: 0, wordCount: 0 }
  };

  // DOM Elements - Status Indicators
  const statusServer = document.getElementById("statusServer");
  const statusTesseract = document.getElementById("statusTesseract");
  const statusAI = document.getElementById("statusAI");
  const btnGlobalReset = document.getElementById("btnGlobalReset");

  // DOM Elements - Notifications
  const notificationBanner = document.getElementById("notificationBanner");
  const notifIcon = document.getElementById("notifIcon");
  const notifMessage = document.getElementById("notifMessage");
  const notifClose = document.getElementById("notifClose");

  // DOM Elements - Upload & Samples
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const browseLink = document.getElementById("browseLink");
  const dropZonePrompt = document.getElementById("dropZonePrompt");
  const uploadActiveInfo = document.getElementById("uploadActiveInfo");
  const activeFileName = document.getElementById("activeFileName");
  const activeFileDimensions = document.getElementById("activeFileDimensions");
  const activeFileSize = document.getElementById("activeFileSize");
  const activeFileFormat = document.getElementById("activeFileFormat");
  const btnRemoveImage = document.getElementById("btnRemoveImage");
  const samplePresetsContainer = document.getElementById("samplePresetsContainer");

  // DOM Elements - Preprocessing Controls
  const chkGrayscale = document.getElementById("chkGrayscale");
  const chkContrast = document.getElementById("chkContrast");
  const rangeClahe = document.getElementById("rangeClahe");
  const valClahe = document.getElementById("valClahe");
  const chkBlur = document.getElementById("chkBlur");
  const selectBlurKernel = document.getElementById("selectBlurKernel");
  const chkAdaptiveThresh = document.getElementById("chkAdaptiveThresh");
  const selectAdaptiveMethod = document.getElementById("selectAdaptiveMethod");
  const chkOtsuThresh = document.getElementById("chkOtsuThresh");
  const chkResize = document.getElementById("chkResize");
  const selectResizeScale = document.getElementById("selectResizeScale");
  const btnProcessImage = document.getElementById("btnProcessImage");
  const techniqueExplainer = document.getElementById("techniqueExplainer");
  const explainerTitle = document.getElementById("explainerTitle");
  const explainerBody = document.getElementById("explainerBody");

  // DOM Elements - Viewports
  const rawViewport = document.getElementById("rawViewport");
  const rawPlaceholder = document.getElementById("rawPlaceholder");
  const rawImagePreview = document.getElementById("rawImagePreview");
  const rawDimensionsLabel = document.getElementById("rawDimensionsLabel");

  const processedViewport = document.getElementById("processedViewport");
  const processedPlaceholder = document.getElementById("processedPlaceholder");
  const processedImagePreview = document.getElementById("processedImagePreview");
  const processedDimensionsLabel = document.getElementById("processedDimensionsLabel");
  const appliedStepsTray = document.getElementById("appliedStepsTray");
  const appliedStepsList = document.getElementById("appliedStepsList");

  // DOM Elements - OCR Execution
  const selectTessInputSource = document.getElementById("selectTessInputSource");
  const selectTessPsm = document.getElementById("selectTessPsm");
  const selectTessLang = document.getElementById("selectTessLang");
  const tesseractEngineDetails = document.getElementById("tesseractEngineDetails");
  const btnRunTesseract = document.getElementById("btnRunTesseract");
  const spinnerTesseract = document.getElementById("spinnerTesseract");

  const selectAiInputSource = document.getElementById("selectAiInputSource");
  const aiConfigStatusBox = document.getElementById("aiConfigStatusBox");
  const aiConfigStatusText = document.getElementById("aiConfigStatusText");
  const btnRunAI = document.getElementById("btnRunAI");
  const spinnerAI = document.getElementById("spinnerAI");

  // DOM Elements - Outputs & Results
  const outputTesseract = document.getElementById("outputTesseract");
  const tessTimeBadge = document.getElementById("tessTimeBadge");
  const tessCharCount = document.getElementById("tessCharCount");
  const tessWordCount = document.getElementById("tessWordCount");
  const btnCopyTess = document.getElementById("btnCopyTess");
  const btnDownloadTess = document.getElementById("btnDownloadTess");
  const btnClearTess = document.getElementById("btnClearTess");

  const outputAI = document.getElementById("outputAI");
  const aiTimeBadge = document.getElementById("aiTimeBadge");
  const aiCharCount = document.getElementById("aiCharCount");
  const aiWordCount = document.getElementById("aiWordCount");
  const btnCopyAI = document.getElementById("btnCopyAI");
  const btnDownloadAI = document.getElementById("btnDownloadAI");
  const btnClearAI = document.getElementById("btnClearAI");

  // DOM Elements - Comparison & Ground Truth
  const inputGroundTruth = document.getElementById("inputGroundTruth");
  const chkIgnoreCase = document.getElementById("chkIgnoreCase");
  const btnCalculateMetrics = document.getElementById("btnCalculateMetrics");
  const metricTessTime = document.getElementById("metricTessTime");
  const metricAiTime = document.getElementById("metricAiTime");
  const metricTessChars = document.getElementById("metricTessChars");
  const metricAiChars = document.getElementById("metricAiChars");
  const metricTessWords = document.getElementById("metricTessWords");
  const metricAiWords = document.getElementById("metricAiWords");
  const metricExactMatch = document.getElementById("metricExactMatch");

  const evalContainer = document.getElementById("evalContainer");
  const evalPlaceholder = document.getElementById("evalPlaceholder");
  const evalTableWrapper = document.getElementById("evalTableWrapper");
  const evalTessWer = document.getElementById("evalTessWer");
  const evalTessCer = document.getElementById("evalTessCer");
  const evalTessSubs = document.getElementById("evalTessSubs");
  const evalTessDels = document.getElementById("evalTessDels");
  const evalTessInss = document.getElementById("evalTessInss");
  const evalTessMatch = document.getElementById("evalTessMatch");

  const evalAiWer = document.getElementById("evalAiWer");
  const evalAiCer = document.getElementById("evalAiCer");
  const evalAiSubs = document.getElementById("evalAiSubs");
  const evalAiDels = document.getElementById("evalAiDels");
  const evalAiInss = document.getElementById("evalAiInss");
  const evalAiMatch = document.getElementById("evalAiMatch");

  // =========================================================================
  // Notification Utility
  // =========================================================================
  function showNotification(type, message) {
    notificationBanner.className = `notification-banner notif-${type}`;
    notifMessage.textContent = message;
    notifIcon.textContent = type === "success" ? "✓" : type === "error" ? "⚠" : "ℹ";
    notificationBanner.classList.remove("hidden");

    if (type === "success" || type === "info") {
      setTimeout(() => {
        notificationBanner.classList.add("hidden");
      }, 5000);
    }
  }

  notifClose.addEventListener("click", () => {
    notificationBanner.classList.add("hidden");
  });

  // Range Slider Value Updates
  rangeClahe.addEventListener("input", (e) => {
    valClahe.textContent = parseFloat(e.target.value).toFixed(1);
  });

  // =========================================================================
  // 1. Backend Connectivity & Health Check
  // =========================================================================
  async function checkBackendHealth() {
    try {
      const resp = await fetch("/api/health");
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();

      // Server Online
      statusServer.innerHTML = `<span class="status-dot dot-online"></span><span class="status-label">Server: Online</span>`;

      // Tesseract Status
      state.tesseractAvailable = data.tesseract?.available || false;
      if (state.tesseractAvailable) {
        statusTesseract.innerHTML = `<span class="status-dot dot-online"></span><span class="status-label">Tesseract: Active</span>`;
        tesseractEngineDetails.innerHTML = `<span>Tesseract: ${data.tesseract.info}</span>`;
        
        // Populate languages dropdown
        if (data.tesseract.languages && data.tesseract.languages.length > 0) {
          selectTessLang.innerHTML = "";
          data.tesseract.languages.forEach(lang => {
            const opt = document.createElement("option");
            opt.value = lang;
            opt.textContent = lang === "eng" ? "English (eng)" : lang;
            if (lang === "eng") opt.selected = true;
            selectTessLang.appendChild(opt);
          });
        }
      } else {
        statusTesseract.innerHTML = `<span class="status-dot dot-warning"></span><span class="status-label">Tesseract: Missing</span>`;
        tesseractEngineDetails.innerHTML = `<span style="color:var(--status-danger)">Engine: Not Detected. (${data.tesseract?.error || "Set TESSERACT_CMD in .env"})</span>`;
      }

      // AI Mode Status
      state.aiConfigured = data.ai_mode?.configured || false;
      if (state.aiConfigured) {
        statusAI.innerHTML = `<span class="status-dot dot-online"></span><span class="status-label">AI: ${data.ai_mode.provider}</span>`;
        aiConfigStatusBox.innerHTML = `<span style="color:var(--status-success); font-weight:600;">✓ Active: ${data.ai_mode.provider} (${data.ai_mode.model})</span>`;
      } else {
        statusAI.innerHTML = `<span class="status-dot dot-warning"></span><span class="status-label">AI: Key Missing</span>`;
        aiConfigStatusBox.innerHTML = `<span>AI Mode requires an API key in <code>.env</code> (GEMINI_API_KEY or OPENAI_API_KEY). Tesseract functions independently.</span>`;
      }

      updateExecutionButtonStates();

    } catch (err) {
      statusServer.innerHTML = `<span class="status-dot dot-offline"></span><span class="status-label">Server: Offline</span>`;
      statusTesseract.innerHTML = `<span class="status-dot dot-offline"></span><span class="status-label">Tesseract: Offline</span>`;
      statusAI.innerHTML = `<span class="status-dot dot-offline"></span><span class="status-label">AI: Offline</span>`;
      showNotification("error", "Unable to connect to Flask server. Please ensure python app.py is running on port 5000.");
    }
  }

  // =========================================================================
  // 2. Upload & Sample Preset Handling
  // =========================================================================
  function validateFile(file) {
    const validTypes = ["image/png", "image/jpeg", "image/jpg", "image/webp"];
    const validExts = [".png", ".jpg", ".jpeg", ".webp"];
    const ext = "." + file.name.split(".").pop().toLowerCase();

    if (!validTypes.includes(file.type) && !validExts.includes(ext)) {
      showNotification("error", `Unsupported file format '${ext}'. Please upload PNG, JPG, JPEG, or WEBP.`);
      return false;
    }

    if (file.size > 10 * 1024 * 1024) {
      showNotification("error", `File size (${(file.size / (1024*1024)).toFixed(2)} MB) exceeds 10 MB limit.`);
      return false;
    }

    return true;
  }

  async function uploadImageFile(file) {
    if (!validateFile(file)) return;

    const formData = new FormData();
    formData.append("image", file);

    try {
      showNotification("info", `Uploading '${file.name}'...`);
      const resp = await fetch("/api/upload", {
        method: "POST",
        body: formData
      });

      const res = await resp.json();
      if (!resp.ok || !res.success) {
        throw new Error(res.error || "Image upload failed.");
      }

      applyUploadedMetadata(res.data);
      showNotification("success", `Image '${res.data.filename}' uploaded successfully.`);
    } catch (err) {
      showNotification("error", err.message);
    }
  }

  function applyUploadedMetadata(meta) {
    state.currentImageId = meta.image_id;
    state.currentRawMeta = meta;
    state.processedAvailable = false;

    // Show metadata pills
    activeFileName.textContent = meta.filename;
    activeFileDimensions.textContent = `${meta.width} × ${meta.height} px`;
    activeFileSize.textContent = `${(meta.size_bytes / 1024).toFixed(1)} KB`;
    activeFileFormat.textContent = meta.format;

    dropZonePrompt.classList.add("hidden");
    uploadActiveInfo.classList.remove("hidden");

    // Display RAW preview
    rawPlaceholder.classList.add("hidden");
    rawImagePreview.src = `${meta.raw_url}?t=${Date.now()}`;
    rawImagePreview.classList.remove("hidden");
    rawDimensionsLabel.textContent = `${meta.width} × ${meta.height} px (${meta.format})`;

    // Reset Processed preview
    processedImagePreview.classList.add("hidden");
    processedPlaceholder.classList.remove("hidden");
    processedDimensionsLabel.textContent = "Awaiting processing";
    appliedStepsTray.classList.add("hidden");

    // Reset results
    resetOutputAreas();

    // Enable process button
    btnProcessImage.disabled = false;
    updateExecutionButtonStates();
  }

  // Drag and Drop Events
  ["dragenter", "dragover"].forEach(evtName => {
    dropZone.addEventListener(evtName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add("drag-active");
    });
  });

  ["dragleave", "drop"].forEach(evtName => {
    dropZone.addEventListener(evtName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove("drag-active");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    if (e.dataTransfer && e.dataTransfer.files.length > 0) {
      uploadImageFile(e.dataTransfer.files[0]);
    }
  });

  browseLink.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      uploadImageFile(e.target.files[0]);
    }
  });

  btnRemoveImage.addEventListener("click", (e) => {
    e.stopPropagation();
    resetActiveWorkspace();
  });

  // Sample Preset Buttons
  samplePresetsContainer.addEventListener("click", async (e) => {
    const btn = e.target.closest("button[data-sample]");
    if (!btn) return;
    const sampleFile = btn.getAttribute("data-sample");

    try {
      showNotification("info", `Loading sample '${sampleFile}'...`);
      const resp = await fetch("/api/samples/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: sampleFile })
      });

      const res = await resp.json();
      if (!resp.ok || !res.success) {
        throw new Error(res.error || "Failed to load sample image.");
      }

      applyUploadedMetadata(res.data);
      showNotification("success", `Sample '${sampleFile}' loaded successfully.`);
    } catch (err) {
      showNotification("error", err.message);
    }
  });

  // =========================================================================
  // 3. Preprocessing Execution
  // =========================================================================
  btnProcessImage.addEventListener("click", async () => {
    if (!state.currentImageId) return;

    const options = {
      grayscale: chkGrayscale.checked,
      contrast_enhancement: chkContrast.checked,
      clahe_clip_limit: parseFloat(rangeClahe.value),
      gaussian_blur: chkBlur.checked,
      blur_kernel: parseInt(selectBlurKernel.value, 10),
      adaptive_threshold: chkAdaptiveThresh.checked,
      adaptive_method: selectAdaptiveMethod.value,
      adaptive_block_size: 11,
      otsu_threshold: chkOtsuThresh.checked,
      resize: chkResize.checked,
      resize_scale: parseFloat(selectResizeScale.value)
    };

    btnProcessImage.disabled = true;
    btnProcessImage.innerHTML = `<span class="btn-spinner"></span> Processing with OpenCV...`;

    try {
      const resp = await fetch("/api/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_id: state.currentImageId,
          options: options
        })
      });

      const res = await resp.json();
      if (!resp.ok || !res.success) {
        throw new Error(res.error || "Image processing failed.");
      }

      const pData = res.data;
      state.processedAvailable = true;

      // Update Processed Viewport
      processedPlaceholder.classList.add("hidden");
      processedImagePreview.src = `${pData.processed_url}?t=${Date.now()}`;
      processedImagePreview.classList.remove("hidden");
      processedDimensionsLabel.textContent = `${pData.width} × ${pData.height} px (${pData.channels}ch)`;

      // Render Applied Steps Pills
      appliedStepsList.innerHTML = "";
      pData.applied_steps.forEach(step => {
        const pill = document.createElement("span");
        pill.className = "step-pill";
        pill.textContent = step;
        appliedStepsList.appendChild(pill);
      });
      appliedStepsTray.classList.remove("hidden");

      // Auto-switch Tesseract input to processed
      selectTessInputSource.value = "processed";

      showNotification("success", `Image processed with ${pData.applied_steps.length} filters applied.`);
      updateExecutionButtonStates();

    } catch (err) {
      showNotification("error", err.message);
    } finally {
      btnProcessImage.disabled = false;
      btnProcessImage.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
        Process Image with OpenCV
      `;
    }
  });

  // =========================================================================
  // 4. Tesseract OCR Execution
  // =========================================================================
  btnRunTesseract.addEventListener("click", async () => {
    if (!state.currentImageId) return;

    const sourceVersion = selectTessInputSource.value;
    if (sourceVersion === "processed" && !state.processedAvailable) {
      showNotification("error", "Processed image is not generated yet. Please process the image first or select RAW image input.");
      return;
    }

    btnRunTesseract.disabled = true;
    spinnerTesseract.classList.remove("hidden");

    try {
      const resp = await fetch("/api/ocr/tesseract", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_id: state.currentImageId,
          version: sourceVersion,
          lang: selectTessLang.value,
          psm: selectTessPsm.value
        })
      });

      const res = await resp.json();
      if (!resp.ok || !res.success) {
        throw new Error(res.error || "Tesseract OCR failed.");
      }

      const ocr = res.data;
      state.tesseractResult = ocr;

      outputTesseract.value = ocr.text;
      tessTimeBadge.textContent = `Time: ${(ocr.duration_seconds * 1000).toFixed(0)} ms`;
      tessCharCount.textContent = `${ocr.char_count} characters`;
      tessWordCount.textContent = `${ocr.word_count} words`;

      metricTessTime.textContent = `${(ocr.duration_seconds * 1000).toFixed(0)} ms`;
      metricTessChars.textContent = `${ocr.char_count} chars`;
      metricTessWords.textContent = `${ocr.word_count} words`;

      showNotification("success", `Tesseract OCR completed in ${(ocr.duration_seconds * 1000).toFixed(0)} ms.`);
      runComparison();

    } catch (err) {
      showNotification("error", err.message);
    } finally {
      btnRunTesseract.disabled = false;
      spinnerTesseract.classList.add("hidden");
    }
  });

  // =========================================================================
  // 5. Vision AI OCR Execution
  // =========================================================================
  btnRunAI.addEventListener("click", async () => {
    if (!state.currentImageId) return;

    const sourceVersion = selectAiInputSource.value;
    if (sourceVersion === "processed" && !state.processedAvailable) {
      showNotification("error", "Processed image is not generated yet. Please process the image first or select RAW image input.");
      return;
    }

    btnRunAI.disabled = true;
    spinnerAI.classList.remove("hidden");

    try {
      const resp = await fetch("/api/ocr/ai", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_id: state.currentImageId,
          version: sourceVersion
        })
      });

      const res = await resp.json();
      if (!resp.ok || !res.success) {
        throw new Error(res.error || "AI OCR failed.");
      }

      const ai = res.data;
      state.aiResult = ai;

      outputAI.value = ai.text;
      aiTimeBadge.textContent = `Time: ${(ai.duration_seconds * 1000).toFixed(0)} ms`;
      aiCharCount.textContent = `${ai.char_count} characters`;
      aiWordCount.textContent = `${ai.word_count} words`;

      metricAiTime.textContent = `${(ai.duration_seconds * 1000).toFixed(0)} ms`;
      metricAiChars.textContent = `${ai.char_count} chars`;
      metricAiWords.textContent = `${ai.word_count} words`;

      showNotification("success", `AI transcription completed in ${(ai.duration_seconds * 1000).toFixed(0)} ms.`);
      runComparison();

    } catch (err) {
      showNotification("error", err.message);
    } finally {
      btnRunAI.disabled = false;
      spinnerAI.classList.add("hidden");
    }
  });

  // =========================================================================
  // 6. Comparison & Ground-Truth Error Rate Calculation
  // =========================================================================
  async function runComparison() {
    const tessText = outputTesseract.value || "";
    const aiText = outputAI.value || "";
    const groundTruth = inputGroundTruth.value.trim();
    const ignoreCase = chkIgnoreCase.checked;

    if (!tessText && !aiText) return;

    try {
      const resp = await fetch("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tesseract_text: tessText,
          ai_text: aiText,
          ground_truth: groundTruth,
          ignore_case: ignoreCase
        })
      });

      const res = await resp.json();
      if (!resp.ok || !res.success) return;

      const comp = res.data;

      // Update Exact Match between Tesseract and AI
      if (tessText && aiText) {
        metricExactMatch.textContent = comp.exact_match_between_ocr ? "Exact 100% Match" : "Differences Detected";
        metricExactMatch.style.color = comp.exact_match_between_ocr ? "var(--status-success)" : "var(--accent-primary)";
      } else {
        metricExactMatch.textContent = "Run both OCR modes to compare";
        metricExactMatch.style.color = "var(--text-muted)";
      }

      // Check ground-truth evaluation
      if (comp.evaluation) {
        evalPlaceholder.classList.add("hidden");
        evalTableWrapper.classList.remove("hidden");

        const tEval = comp.evaluation.tesseract;
        const aEval = comp.evaluation.ai;

        evalTessWer.textContent = tEval.wer.percentage_str;
        evalTessCer.textContent = tEval.cer.percentage_str;
        evalTessSubs.textContent = `${tEval.wer.substitutions} (W) / ${tEval.cer.substitutions} (C)`;
        evalTessDels.textContent = `${tEval.wer.deletions} (W) / ${tEval.cer.deletions} (C)`;
        evalTessInss.textContent = `${tEval.wer.insertions} (W) / ${tEval.cer.insertions} (C)`;
        evalTessMatch.textContent = tEval.exact_match_gt ? "Yes (100%)" : "No";

        evalAiWer.textContent = aEval.wer.percentage_str;
        evalAiCer.textContent = aEval.cer.percentage_str;
        evalAiSubs.textContent = `${aEval.wer.substitutions} (W) / ${aEval.cer.substitutions} (C)`;
        evalAiDels.textContent = `${aEval.wer.deletions} (W) / ${aEval.cer.deletions} (C)`;
        evalAiInss.textContent = `${aEval.wer.insertions} (W) / ${aEval.cer.insertions} (C)`;
        evalAiMatch.textContent = aEval.exact_match_gt ? "Yes (100%)" : "No";

      } else {
        evalPlaceholder.classList.remove("hidden");
        evalTableWrapper.classList.add("hidden");
      }

    } catch (err) {
      console.error("Comparison error:", err);
    }
  }

  btnCalculateMetrics.addEventListener("click", () => {
    runComparison();
    if (!inputGroundTruth.value.trim()) {
      showNotification("info", "Please enter reference ground-truth text to calculate Word Error Rate (WER) and Character Error Rate (CER).");
    }
  });

  // =========================================================================
  // 7. Clipboard, Download & Reset Utilities
  // =========================================================================
  function copyTextToClipboard(text, label) {
    if (!text) {
      showNotification("info", `No ${label} text to copy.`);
      return;
    }
    navigator.clipboard.writeText(text).then(() => {
      showNotification("success", `${label} text copied to clipboard.`);
    }).catch(() => {
      showNotification("error", "Failed to access system clipboard.");
    });
  }

  function downloadTextAsFile(text, filename) {
    if (!text) {
      showNotification("info", `No text available to download.`);
      return;
    }
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showNotification("success", `Downloaded '${filename}'.`);
  }

  btnCopyTess.addEventListener("click", () => copyTextToClipboard(outputTesseract.value, "Tesseract"));
  btnDownloadTess.addEventListener("click", () => downloadTextAsFile(outputTesseract.value, "tesseract_extracted.txt"));
  btnClearTess.addEventListener("click", () => {
    outputTesseract.value = "";
    tessTimeBadge.textContent = "Time: -- ms";
    tessCharCount.textContent = "0 characters";
    tessWordCount.textContent = "0 words";
    metricTessTime.textContent = "--";
    metricTessChars.textContent = "0 chars";
    metricTessWords.textContent = "0 words";
    runComparison();
  });

  btnCopyAI.addEventListener("click", () => copyTextToClipboard(outputAI.value, "AI Vision"));
  btnDownloadAI.addEventListener("click", () => downloadTextAsFile(outputAI.value, "ai_vision_extracted.txt"));
  btnClearAI.addEventListener("click", () => {
    outputAI.value = "";
    aiTimeBadge.textContent = "Time: -- ms";
    aiCharCount.textContent = "0 characters";
    aiWordCount.textContent = "0 words";
    metricAiTime.textContent = "--";
    metricAiChars.textContent = "0 chars";
    metricAiWords.textContent = "0 words";
    runComparison();
  });

  function resetOutputAreas() {
    outputTesseract.value = "";
    tessTimeBadge.textContent = "Time: -- ms";
    tessCharCount.textContent = "0 characters";
    tessWordCount.textContent = "0 words";

    outputAI.value = "";
    aiTimeBadge.textContent = "Time: -- ms";
    aiCharCount.textContent = "0 characters";
    aiWordCount.textContent = "0 words";

    metricTessTime.textContent = "--";
    metricAiTime.textContent = "--";
    metricTessChars.textContent = "0 chars";
    metricAiChars.textContent = "0 chars";
    metricTessWords.textContent = "0 words";
    metricAiWords.textContent = "0 words";
    metricExactMatch.textContent = "Awaiting OCR runs";
    metricExactMatch.style.color = "var(--text-muted)";

    inputGroundTruth.value = "";
    evalPlaceholder.classList.remove("hidden");
    evalTableWrapper.classList.add("hidden");
  }

  function updateExecutionButtonStates() {
    const hasImage = Boolean(state.currentImageId);
    btnRunTesseract.disabled = !hasImage || !state.tesseractAvailable;
    btnRunAI.disabled = !hasImage || !state.aiConfigured;
  }

  async function resetActiveWorkspace() {
    if (state.currentImageId) {
      try {
        await fetch("/api/reset", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image_id: state.currentImageId })
        });
      } catch (e) {
        console.error("Reset error:", e);
      }
    }

    state.currentImageId = null;
    state.currentRawMeta = null;
    state.processedAvailable = false;

    fileInput.value = "";
    dropZonePrompt.classList.remove("hidden");
    uploadActiveInfo.classList.add("hidden");

    rawImagePreview.src = "";
    rawImagePreview.classList.add("hidden");
    rawPlaceholder.classList.remove("hidden");
    rawDimensionsLabel.textContent = "No image uploaded";

    processedImagePreview.src = "";
    processedImagePreview.classList.add("hidden");
    processedPlaceholder.classList.remove("hidden");
    processedDimensionsLabel.textContent = "Awaiting processing";
    appliedStepsTray.classList.add("hidden");

    btnProcessImage.disabled = true;
    resetOutputAreas();
    updateExecutionButtonStates();
    showNotification("info", "Laboratory workspace reset.");
  }

  btnGlobalReset.addEventListener("click", resetActiveWorkspace);

  // Initial Startup
  checkBackendHealth();
});
