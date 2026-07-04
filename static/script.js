/**
 * Aerial Object Detection System — frontend logic.
 * Handles file selection/drag-drop, calls /predict, and renders results.
 */

(() => {
  "use strict";

  // ---- Element references -------------------------------------------------

  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const dropzoneContent = document.getElementById("dropzoneContent");
  const previewContent = document.getElementById("previewContent");
  const previewImage = document.getElementById("previewImage");

  const predictBtn = document.getElementById("predictBtn");
  const clearBtn = document.getElementById("clearBtn");
  const btnSpinner = document.getElementById("btnSpinner");
  const errorMsg = document.getElementById("errorMsg");

  const resultsSection = document.getElementById("resultsSection");
  const originalImage = document.getElementById("originalImage");
  const resultImage = document.getElementById("resultImage");
  const statCount = document.getElementById("statCount");
  const statTime = document.getElementById("statTime");
  const detectionsBody = document.getElementById("detectionsBody");
  const noDetections = document.getElementById("noDetections");
  const downloadBtn = document.getElementById("downloadBtn");
  const newImageBtn = document.getElementById("newImageBtn");

  const modelStatus = document.getElementById("modelStatus");
  const statusDot = modelStatus.querySelector(".status-dot");
  const statusText = modelStatus.querySelector(".status-text");

  const CLASS_COLORS = {
    AirPlane: "#2457ff",
    Drone: "#f79009",
    Helicopter: "#12b76a",
  };

  const MAX_FILE_SIZE = 16 * 1024 * 1024; // 16 MB, matches backend limit
  const ALLOWED_TYPES = ["image/png", "image/jpeg", "image/webp", "image/bmp"];

  let selectedFile = null;

  // ---- Helpers --------------------------------------------------------------

  function showError(message) {
    errorMsg.textContent = message;
    errorMsg.classList.remove("hidden");
  }

  function clearError() {
    errorMsg.textContent = "";
    errorMsg.classList.add("hidden");
  }

  function setLoading(isLoading) {
    predictBtn.disabled = isLoading || !selectedFile;
    clearBtn.disabled = isLoading;
    btnSpinner.classList.toggle("hidden", !isLoading);
    predictBtn.querySelector(".btn-label").textContent = isLoading
      ? "Detecting..."
      : "Run Detection";
  }

  function resetToUploadState() {
    selectedFile = null;
    fileInput.value = "";
    previewImage.src = "";
    previewContent.classList.add("hidden");
    dropzoneContent.classList.remove("hidden");
    predictBtn.disabled = true;
    clearBtn.disabled = true;
    resultsSection.classList.add("hidden");
    clearError();
  }

  function handleFileSelection(file) {
    clearError();

    if (!ALLOWED_TYPES.includes(file.type)) {
      showError("Unsupported file type. Please upload a JPG, PNG, WEBP or BMP image.");
      return;
    }

    if (file.size > MAX_FILE_SIZE) {
      showError("File is too large. Maximum size is 16 MB.");
      return;
    }

    selectedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;
      dropzoneContent.classList.add("hidden");
      previewContent.classList.remove("hidden");
    };
    reader.onerror = () => {
      showError("Could not read the selected file. Please try another image.");
    };
    reader.readAsDataURL(file);

    predictBtn.disabled = false;
    clearBtn.disabled = false;
    resultsSection.classList.add("hidden");
  }

  function renderDetections(detections) {
    detectionsBody.innerHTML = "";

    if (!detections || detections.length === 0) {
      noDetections.classList.remove("hidden");
      return;
    }

    noDetections.classList.add("hidden");

    detections.forEach((det) => {
      const pct = Math.round(det.confidence * 100);
      const color = CLASS_COLORS[det.class] || "#2457ff";

      const row = document.createElement("tr");
      row.innerHTML = `
        <td>
          <span class="class-badge"><i style="background:${color}"></i>${escapeHtml(det.class)}</span>
        </td>
        <td>
          <div class="confidence-cell">
            <div class="confidence-bar">
              <div class="confidence-fill" style="width:${pct}%; background:${color}"></div>
            </div>
            <span class="confidence-value">${pct}%</span>
          </div>
        </td>
      `;
      detectionsBody.appendChild(row);
    });
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  async function checkModelStatus() {
    try {
      // A lightweight way to confirm the server is reachable.
      const response = await fetch("/", { method: "GET" });
      if (response.ok) {
        statusDot.classList.add("online");
        statusText.textContent = "Model ready";
      } else {
        throw new Error("Server responded with an error");
      }
    } catch (err) {
      statusDot.classList.add("offline");
      statusText.textContent = "Server unavailable";
    }
  }

  async function runPrediction() {
    if (!selectedFile) return;

    clearError();
    setLoading(true);

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });

      let data;
      try {
        data = await response.json();
      } catch (parseErr) {
        throw new Error("Unexpected response from the server.");
      }

      if (!response.ok || !data.success) {
        throw new Error(data.error || "Prediction failed. Please try again.");
      }

      originalImage.src = data.original_image + `?t=${Date.now()}`;
      resultImage.src = data.image + `?t=${Date.now()}`;
      statCount.textContent = data.count;
      statTime.textContent = data.inference_time;
      renderDetections(data.detections);

      downloadBtn.href = data.image;
      downloadBtn.setAttribute("download", data.image.split("/").pop());

      resultsSection.classList.remove("hidden");
      resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      if (err.message === "Failed to fetch") {
        showError("Could not reach the server. Please check your connection and try again.");
      } else {
        showError(err.message);
      }
    } finally {
      setLoading(false);
    }
  }

  // ---- Event listeners --------------------------------------------------------

  dropzone.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelection(e.target.files[0]);
    }
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) handleFileSelection(file);
  });

  predictBtn.addEventListener("click", runPrediction);
  clearBtn.addEventListener("click", resetToUploadState);
  newImageBtn.addEventListener("click", resetToUploadState);

  // ---- Init --------------------------------------------------------------

  checkModelStatus();
})();
