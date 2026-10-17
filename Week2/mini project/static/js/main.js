document.addEventListener("DOMContentLoaded", () => {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const previewContainer = document.getElementById("previewContainer");
  const previewImg = document.getElementById("previewImg");
  const removeBtn = document.getElementById("removeBtn");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const resetBtn = document.getElementById("resetBtn");

  const emptyState = document.getElementById("emptyState");
  const loadingState = document.getElementById("loadingState");
  const diagnosisContainer = document.getElementById("diagnosisContainer");

  let currentFile = null;
  let currentSample = null;

  // Drag and drop events
  ["dragenter", "dragover"].forEach(event => {
    dropzone.addEventListener(event, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(event => {
    dropzone.addEventListener(event, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleSelectedFile(files[0]);
    }
  });

  dropzone.addEventListener("click", () => {
    fileInput.click();
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleSelectedFile(e.target.files[0]);
    }
  });

  function handleSelectedFile(file) {
    if (!file.type.startsWith("image/")) {
      alert("Please upload an image file (JPG, PNG, or WEBP).");
      return;
    }
    currentFile = file;
    currentSample = null;

    // Deselect any active sample
    document.querySelectorAll(".sample-item").forEach(el => el.classList.remove("active"));

    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      dropzone.style.display = "none";
      previewContainer.style.display = "block";
      analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  // Remove preview
  removeBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    resetUploader();
  });

  resetBtn.addEventListener("click", () => {
    resetUploader();
    showEmptyState();
  });

  function resetUploader() {
    currentFile = null;
    currentSample = null;
    fileInput.value = "";
    previewImg.src = "";
    previewContainer.style.display = "none";
    dropzone.style.display = "block";
    analyzeBtn.disabled = true;
    document.querySelectorAll(".sample-item").forEach(el => el.classList.remove("active"));
  }

  // Sample Images click handling
  document.querySelectorAll(".sample-item").forEach(item => {
    item.addEventListener("click", () => {
      const filename = item.getAttribute("data-filename");
      currentSample = filename;
      currentFile = null;

      document.querySelectorAll(".sample-item").forEach(el => el.classList.remove("active"));
      item.classList.add("active");

      // Set preview
      const imgSrc = item.querySelector("img").src;
      previewImg.src = imgSrc;
      dropzone.style.display = "none";
      previewContainer.style.display = "block";
      analyzeBtn.disabled = false;

      // Automatically run diagnosis on sample leaf click!
      runSamplePrediction(filename);
    });
  });

  // Analyze button click
  analyzeBtn.addEventListener("click", () => {
    if (currentFile) {
      runFilePrediction(currentFile);
    } else if (currentSample) {
      runSamplePrediction(currentSample);
    }
  });

  function showLoading() {
    emptyState.style.display = "none";
    diagnosisContainer.style.display = "none";
    loadingState.style.display = "flex";
  }

  function showEmptyState() {
    loadingState.style.display = "none";
    diagnosisContainer.style.display = "none";
    emptyState.style.display = "flex";
  }

  function showDiagnosis(data) {
    loadingState.style.display = "none";
    emptyState.style.display = "none";
    diagnosisContainer.style.display = "block";

    // Set Status Badge
    const statusBadge = document.getElementById("statusBadge");
    if (data.is_healthy) {
      statusBadge.className = "status-badge status-healthy";
      statusBadge.innerHTML = "<span>&#10004;</span> Healthy Leaf";
    } else {
      statusBadge.className = "status-badge status-diseased";
      statusBadge.innerHTML = "<span>&#9888;</span> Disease Detected";
    }

    // Disease & Crop Name
    document.getElementById("diseaseName").textContent = data.display_name;
    document.getElementById("cropName").textContent = `Crop: ${data.crop}`;

    // Confidence
    const confVal = data.confidence.toFixed(1);
    document.getElementById("confidenceVal").textContent = `${confVal}%`;
    const progressFill = document.getElementById("progressFill");
    progressFill.style.width = "0%";
    setTimeout(() => {
      progressFill.style.width = `${confVal}%`;
    }, 50);

    // Metadata Badges
    const severityPill = document.getElementById("severityPill");
    const pathogenPill = document.getElementById("pathogenPill");
    const timePill = document.getElementById("timePill");

    const details = data.details || {};
    severityPill.textContent = `Severity: ${details.severity || "N/A"}`;
    pathogenPill.textContent = `Pathogen: ${details.pathogen || "N/A"}`;
    timePill.textContent = `Inference: ${data.inference_time_ms || 0} ms`;

    // Candidates
    const candidatesList = document.getElementById("candidatesList");
    candidatesList.innerHTML = "";
    if (data.top_predictions && data.top_predictions.length > 0) {
      data.top_predictions.forEach(c => {
        const item = document.createElement("div");
        item.className = "candidate-item";
        item.innerHTML = `
          <span class="candidate-name">${c.display_name} (${c.crop})</span>
          <div class="candidate-bar">
            <div class="candidate-fill-track">
              <div class="candidate-fill" style="width: ${c.confidence}%"></div>
            </div>
            <span style="font-size: 0.8rem; font-weight: 600; width: 40px; text-align: right;">${c.confidence}%</span>
          </div>
        `;
        candidatesList.appendChild(item);
      });
    }

    // Tabs content (Symptoms, Treatment, Prevention)
    window.currentDiseaseData = details;
    renderTabContent("symptoms");
  }

  function runFilePrediction(file) {
    showLoading();
    const formData = new FormData();
    formData.append("file", file);

    fetch("/predict", {
      method: "POST",
      body: formData
    })
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          showDiagnosis(data);
        } else {
          alert("Diagnosis error: " + (data.error || "Unknown error"));
          showEmptyState();
        }
      })
      .catch(err => {
        alert("Failed to communicate with prediction server: " + err.message);
        showEmptyState();
      });
  }

  function runSamplePrediction(filename) {
    showLoading();
    fetch(`/api/sample/${encodeURIComponent(filename)}`)
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          showDiagnosis(data);
        } else {
          alert("Diagnosis error: " + (data.error || "Unknown error"));
          showEmptyState();
        }
      })
      .catch(err => {
        alert("Failed to communicate with prediction server: " + err.message);
        showEmptyState();
      });
  }

  // Advice Tabs
  const tabBtns = document.querySelectorAll(".tab-btn");
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const tab = btn.getAttribute("data-tab");
      renderTabContent(tab);
    });
  });

  function renderTabContent(tab) {
    const tabContent = document.getElementById("tabContent");
    const data = window.currentDiseaseData || {};

    if (tab === "symptoms") {
      const items = data.symptoms || ["No symptoms reported."];
      tabContent.innerHTML = `<strong>Observed Symptoms:</strong><ul>${items.map(s => `<li>${s}</li>`).join("")}</ul>`;
    } else if (tab === "treatment") {
      const items = data.treatment || ["No specific chemical treatment required."];
      tabContent.innerHTML = `<strong>Recommended Actions & Remedies:</strong><ul>${items.map(t => `<li>${t}</li>`).join("")}</ul>`;
    } else if (tab === "prevention") {
      const prev = data.prevention || "Maintain good agronomic practices.";
      tabContent.innerHTML = `<strong>Prevention Strategy:</strong><p style="margin-top:0.4rem;">${prev}</p>`;
    }
  }
});
