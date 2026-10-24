/**
 * TwoHearts Clinical Dashboard Controller
 * Pure Vanilla JavaScript — Zero BOM — Ultra Robust Error Handling
 */

let viewer = null;
let currentPatientId = "ARR_01";
let currentWindowIdx = 0;
let totalWindows = 1;
let isPlaying = false;
let playInterval = null;
let currentAnalysisData = null;
let allPatientsData = [];

document.addEventListener("DOMContentLoaded", async () => {
  try {
    viewer = new ECGCanvasViewer("ecgCanvas");
  } catch (err) {
    console.error("Failed to initialize ECG canvas viewer:", err);
  }

  await loadPatientsList();
  setupNavigationAndModals();
  setupStreamControls();
  setupXAIControls();
  setupModalUpload();

  // Load initial analysis for ARR_01 immediately
  await analyzeCurrentWindow();

  // Render Overview Patient Grid
  renderOverviewCohort();

  // Setup print hook
  setupPrintOptimization();
});

// Load patient list from API
async function loadPatientsList() {
  try {
    const res = await fetch("/api/patients");
    if (!res.ok) throw new Error("Failed to load patients");
    const data = await res.json();
    allPatientsData = data.patients || [];

    const select = document.getElementById("patientSelect");
    if (select) {
      select.innerHTML = "";
      allPatientsData.forEach(p => {
        const opt = document.createElement("option");
        opt.value = p.patient_id;
        const tag = p.is_arrhythmia ? "🔴 ARRHYTHMIA" : "🟢 NORMAL";
        opt.textContent = `${p.patient_id} (${tag}) — ${p.duration_sec}s`;
        select.appendChild(opt);
      });
      select.value = currentPatientId;
    }
  } catch (err) {
    console.error("Error loading patients list:", err);
  }
}

// Navigation and Modals Handling
function setupNavigationAndModals() {
  // 1. Overview Modal Controls (Both Sidebar and Top Header)
  const overviewModal = document.getElementById("overviewModal");
  const navOverviewBtn = document.getElementById("navOverviewBtn");
  const topOverviewBtn = document.getElementById("topOverviewBtn");
  const closeOverviewBtn = document.getElementById("closeOverviewBtn");

  function openOverview() {
    renderOverviewCohort();
    if (overviewModal) overviewModal.classList.remove("hidden");
  }

  function closeOverview() {
    if (overviewModal) overviewModal.classList.add("hidden");
  }

  if (navOverviewBtn) navOverviewBtn.addEventListener("click", openOverview);
  if (topOverviewBtn) topOverviewBtn.addEventListener("click", openOverview);
  if (closeOverviewBtn) closeOverviewBtn.addEventListener("click", closeOverview);

  // 2. Validation Matrix Modal Controls (Sidebar and Top Header)
  const benchmarkModal = document.getElementById("benchmarkModal");
  const navBenchmarkBtn = document.getElementById("navBenchmarkBtn");
  const topBenchmarkBtn = document.getElementById("topBenchmarkBtn");
  const closeBenchmarkBtn = document.getElementById("closeBenchmarkBtn");

  function openBenchmark() {
    loadBenchmarkData();
    if (benchmarkModal) benchmarkModal.classList.remove("hidden");
  }

  function closeBenchmark() {
    if (benchmarkModal) benchmarkModal.classList.add("hidden");
  }

  if (navBenchmarkBtn) navBenchmarkBtn.addEventListener("click", openBenchmark);
  if (topBenchmarkBtn) topBenchmarkBtn.addEventListener("click", openBenchmark);
  if (closeBenchmarkBtn) closeBenchmarkBtn.addEventListener("click", closeBenchmark);

  // 3. Screening Box Nav Button
  const navScreeningBtn = document.getElementById("navScreeningBtn");
  if (navScreeningBtn) {
    navScreeningBtn.addEventListener("click", () => {
      closeOverview();
      closeBenchmark();
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  // 4. Print Report Button
  const printReportBtn = document.getElementById("printReportBtn");
  if (printReportBtn) {
    printReportBtn.addEventListener("click", () => {
      updatePrintHeaders();
      window.print();
    });
  }

  // Close modals on overlay backdrop click
  [overviewModal, benchmarkModal].forEach(modal => {
    if (modal) {
      modal.addEventListener("click", (e) => {
        if (e.target === modal) modal.classList.add("hidden");
      });
    }
  });
}

// Render Overview Cohort Directory
function renderOverviewCohort() {
  const grid = document.getElementById("patientCohortGrid");
  if (!grid || !allPatientsData || allPatientsData.length === 0) return;

  grid.innerHTML = "";

  allPatientsData.forEach(p => {
    const card = document.createElement("div");
    card.className = "cohort-card";

    const isArr = p.is_arrhythmia;
    const badgeBg = isArr ? "background:rgba(239,68,68,0.2); color:#FCA5A5; border:1px solid rgba(239,68,68,0.5);" : "background:rgba(16,185,129,0.2); color:#6EE7B7; border:1px solid rgba(16,185,129,0.5);";
    const badgeText = isArr ? "🔴 Arrhythmia" : "🟢 Normal";

    card.innerHTML = `
      <div>
        <div class="cohort-card-top">
          <span class="cohort-card-id">${p.patient_id}</span>
          <span style="font-size:10px; font-weight:800; padding:2px 8px; border-radius:9999px; ${badgeBg}">${badgeText}</span>
        </div>
        <div style="font-size:11px; color:#9CA3AF; margin-top:8px; line-height:1.6;">
          <div style="display:flex; justify-content:space-between;"><span>Duration:</span><strong style="color:#FFFFFF;">${p.duration_sec}s</strong></div>
          <div style="display:flex; justify-content:space-between;"><span>Maternal HR:</span><strong style="color:#FFFFFF;">${p.maternal_hr} bpm</strong></div>
          <div style="display:flex; justify-content:space-between;"><span>Stress Index:</span><strong style="color:#06B6D4;">${p.stress_index} SI</strong></div>
        </div>
      </div>
      <button class="btn btn-secondary select-patient-btn" style="width:100%; justify-content:center; margin-top:8px; font-size:11px; padding:6px;" data-id="${p.patient_id}">
        Inspect in Screening Box →
      </button>
    `;

    grid.appendChild(card);
  });

  // Attach click listener to each inspect button
  const btns = grid.querySelectorAll(".select-patient-btn");
  btns.forEach(btn => {
    btn.addEventListener("click", () => {
      const pid = btn.dataset.id;
      selectPatientDirect(pid);
      document.getElementById("overviewModal").classList.add("hidden");
    });
  });
}

function selectPatientDirect(patientId) {
  currentPatientId = patientId;
  const select = document.getElementById("patientSelect");
  if (select) select.value = patientId;

  currentWindowIdx = 0;
  if (isPlaying) togglePlay();

  analyzeCurrentWindow();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// Stream Scrubber & Playback Controls
function setupStreamControls() {
  const patientSelect = document.getElementById("patientSelect");
  const slider = document.getElementById("windowSlider");
  const prevBtn = document.getElementById("prevWindowBtn");
  const nextBtn = document.getElementById("nextWindowBtn");
  const playBtn = document.getElementById("playBtn");

  if (patientSelect) {
    patientSelect.addEventListener("change", async (e) => {
      currentPatientId = e.target.value;
      currentWindowIdx = 0;
      if (isPlaying) togglePlay();
      await analyzeCurrentWindow();
    });
  }

  if (slider) {
    slider.addEventListener("input", async (e) => {
      currentWindowIdx = parseInt(e.target.value, 10);
      await analyzeCurrentWindow();
    });
  }

  if (prevBtn) {
    prevBtn.addEventListener("click", async () => {
      if (currentWindowIdx > 0) {
        currentWindowIdx--;
        if (slider) slider.value = currentWindowIdx;
        await analyzeCurrentWindow();
      }
    });
  }

  if (nextBtn) {
    nextBtn.addEventListener("click", async () => {
      if (currentWindowIdx < totalWindows - 1) {
        currentWindowIdx++;
        if (slider) slider.value = currentWindowIdx;
        await analyzeCurrentWindow();
      }
    });
  }

  if (playBtn) {
    playBtn.addEventListener("click", togglePlay);
  }
}

function togglePlay() {
  isPlaying = !isPlaying;
  const playBtn = document.getElementById("playBtn");
  if (!playBtn) return;

  if (isPlaying) {
    playBtn.innerHTML = `
      <svg width="14" height="14" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zM7 8a1 1 0 012 0v4a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v4a1 1 0 102 0V8a1 1 0 00-1-1z" clip-rule="evenodd"/></svg>
      Pause
    `;
    playBtn.className = "btn btn-amber";

    playInterval = setInterval(async () => {
      if (currentWindowIdx < totalWindows - 1) {
        currentWindowIdx++;
      } else {
        currentWindowIdx = 0;
      }
      const slider = document.getElementById("windowSlider");
      if (slider) slider.value = currentWindowIdx;
      await analyzeCurrentWindow();
    }, 2200);
  } else {
    clearInterval(playInterval);
    playBtn.innerHTML = `
      <svg width="14" height="14" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clip-rule="evenodd"/></svg>
      Play Stream
    `;
    playBtn.className = "btn btn-cyan";
  }
}

// 1D Grad-CAM Explainable AI Controls
function setupXAIControls() {
  const xaiBtns = document.querySelectorAll(".xai-toggle-btn");
  xaiBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      xaiBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const mode = btn.dataset.mode;
      if (viewer) viewer.setXAIMode(mode);
    });
  });

  const opacitySlider = document.getElementById("xaiOpacitySlider");
  if (opacitySlider) {
    opacitySlider.addEventListener("input", (e) => {
      if (viewer) viewer.setXAIOpacity(parseFloat(e.target.value));
    });
  }
}

// File Ingestion Upload Modal
function setupModalUpload() {
  const uploadModal = document.getElementById("uploadModal");
  const openBtn = document.getElementById("openUploadModalBtn");
  const closeBtn = document.getElementById("closeUploadModalBtn");
  const browseBtn = document.getElementById("modalBrowseBtn");
  const fileInput = document.getElementById("modalFileInput");
  const dropzone = document.getElementById("modalDropzone");
  const loadDemoBtn = document.getElementById("modalLoadDemoBtn");

  if (openBtn && uploadModal) {
    openBtn.addEventListener("click", () => uploadModal.classList.remove("hidden"));
  }

  if (closeBtn && uploadModal) {
    closeBtn.addEventListener("click", () => uploadModal.classList.add("hidden"));
  }

  if (browseBtn && fileInput) {
    browseBtn.addEventListener("click", () => fileInput.click());
  }

  if (loadDemoBtn && uploadModal) {
    loadDemoBtn.addEventListener("click", () => {
      uploadModal.classList.add("hidden");
      selectPatientDirect("ARR_01");
    });
  }

  if (fileInput) {
    fileInput.addEventListener("change", (e) => handleFileUpload(e.target.files));
  }

  if (dropzone) {
    ["dragenter", "dragover"].forEach(evt => {
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach(evt => {
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
      });
    });

    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      handleFileUpload(e.dataTransfer.files);
    });
  }
}

async function handleFileUpload(files) {
  if (!files || files.length === 0) return;

  const status = document.getElementById("modalUploadStatus");
  if (status) {
    status.textContent = `Uploading ${files.length} file(s) into TwoHearts DL engine...`;
    status.style.color = "#06B6D4";
  }

  const formData = new FormData();
  for (let i = 0; i < files.length; i++) {
    formData.append("files", files[i]);
  }

  try {
    const res = await fetch("/api/upload", {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Upload analysis failed");
    }

    const data = await res.json();
    currentAnalysisData = data;
    currentPatientId = data.patient_id;
    totalWindows = data.total_windows;
    currentWindowIdx = 0;

    if (status) {
      status.textContent = `Successfully processed: ${data.patient_id}`;
      status.style.color = "#10B981";
    }

    setTimeout(() => {
      const uploadModal = document.getElementById("uploadModal");
      if (uploadModal) uploadModal.classList.add("hidden");

      const select = document.getElementById("patientSelect");
      if (select) {
        let opt = Array.from(select.options).find(o => o.value === data.patient_id);
        if (!opt) {
          opt = document.createElement("option");
          opt.value = data.patient_id;
          opt.textContent = `${data.patient_id} (CUSTOM UPLOAD)`;
          select.prepend(opt);
        }
        select.value = data.patient_id;
      }

      if (viewer) viewer.setData(data.waveforms);
      updateDiagnosticCards(data);
    }, 600);

  } catch (err) {
    console.error("Upload error:", err);
    if (status) {
      status.textContent = `Error: ${err.message}`;
      status.style.color = "#EF4444";
    }
  }
}

// Window Analysis via API
async function analyzeCurrentWindow() {
  try {
    const res = await fetch(`/api/analyze?patient_id=${currentPatientId}&window_idx=${currentWindowIdx}`);
    if (!res.ok) throw new Error("Failed to analyze window");

    const data = await res.json();
    currentAnalysisData = data;
    totalWindows = data.total_windows;

    const slider = document.getElementById("windowSlider");
    if (slider) {
      slider.max = totalWindows - 1;
      slider.value = currentWindowIdx;
    }

    const timeLabel = document.getElementById("windowTimeLabel");
    if (timeLabel) timeLabel.textContent = data.time_range;

    const counterLabel = document.getElementById("windowCounterLabel");
    if (counterLabel) counterLabel.textContent = `Window ${currentWindowIdx + 1} / ${totalWindows}`;

    if (viewer) {
      viewer.setData(data.waveforms);
    }

    updateDiagnosticCards(data);

  } catch (err) {
    console.error("Error analyzing window:", err);
  }
}

// Update 3 Clinical Triage Cards
function updateDiagnosticCards(data) {
  const ft = data.fetal_triage;
  const mw = data.maternal_wellness;

  // 1. Fetal Cardiac Status Card
  const fetalBadge = document.getElementById("fetalStatusBadge");
  const fetalProbBar = document.getElementById("fetalProbBar");
  const fetalProbVal = document.getElementById("fetalProbValue");
  const fetalAction = document.getElementById("fetalActionText");

  if (fetalProbVal) fetalProbVal.textContent = `${(ft.probability * 100).toFixed(1)}%`;
  if (fetalProbBar) {
    fetalProbBar.style.width = `${Math.min(100, ft.probability * 100)}%`;
    fetalProbBar.className = ft.is_arrhythmia ? "progress-bar-fill red" : "progress-bar-fill green";
  }
  if (fetalAction) fetalAction.textContent = ft.clinical_action;

  if (fetalBadge) {
    if (ft.is_arrhythmia) {
      fetalBadge.textContent = "🔴 " + ft.status;
      fetalBadge.className = "status-badge arrhythmia";
    } else {
      fetalBadge.textContent = "🟢 " + ft.status;
      fetalBadge.className = "status-badge normal";
    }
  }

  // 2. Maternal Wellness Card
  const matStatus = document.getElementById("maternalStatusBadge");
  const matHR = document.getElementById("maternalHRVal");
  const matRMSSD = document.getElementById("maternalRMSSDVal");
  const matSI = document.getElementById("maternalSIVal");

  if (matHR) matHR.textContent = `${mw.heart_rate_bpm} bpm`;
  if (matRMSSD) matRMSSD.textContent = `${mw.rmssd_ms} ms`;
  if (matSI) matSI.textContent = `${mw.baevsky_stress_index}`;

  if (matStatus) {
    if (mw.stress_badge === "warning") {
      matStatus.textContent = "⚠️ " + mw.status;
      matStatus.className = "status-badge stress-warning";
    } else {
      matStatus.textContent = "🟢 " + mw.status;
      matStatus.className = "status-badge normal";
    }
  }

  updatePrintHeaders();
}

// Populate Print Headers
function updatePrintHeaders() {
  if (!currentAnalysisData) return;

  const data = currentAnalysisData;
  const ft = data.fetal_triage;
  const mw = data.maternal_wellness;

  const setT = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  };

  setT("printPatientId", data.patient_id);
  setT("printCondition", data.ground_truth_condition);
  setT("printTimeRange", data.time_range);
  setT("printMaternalHR", `${mw.heart_rate_bpm} bpm`);
  setT("printMaternalRMSSD", `${mw.rmssd_ms} ms`);
  setT("printStressIndex", `${mw.baevsky_stress_index} SI`);
  setT("printDiagnosis", `${ft.status} (Prob: ${(ft.probability * 100).toFixed(1)}%)`);
  setT("printAction", ft.clinical_action);

  const now = new Date();
  setT("printDate", now.toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  }));
}

function setupPrintOptimization() {
  window.addEventListener("beforeprint", () => {
    updatePrintHeaders();
  });
}

// Load Benchmark Validation Data
async function loadBenchmarkData() {
  try {
    const res = await fetch("/api/benchmarks");
    if (!res.ok) throw new Error("Failed to load benchmarks");
    const data = await res.json();
    const tableBody = document.getElementById("cvTableBody");
    if (!tableBody) return;

    tableBody.innerHTML = "";
    data.cv_fold_table.forEach(r => {
      const tr = document.createElement("tr");
      tr.style.borderBottom = "1px solid #1F2937";
      tr.innerHTML = `
        <td style="padding:10px 14px; font-weight:700; color:#FFFFFF;">${r.fold}</td>
        <td style="padding:10px 14px; color:#06B6D4; font-family:'JetBrains Mono', monospace; font-weight:700;">${(r.fetal_auc * 100).toFixed(2)}%</td>
        <td style="padding:10px 14px; color:#10B981; font-family:'JetBrains Mono', monospace;">${(r.fetal_f1 * 100).toFixed(1)}%</td>
        <td style="padding:10px 14px; color:#D1D5DB; font-family:'JetBrains Mono', monospace;">${(r.fetal_acc * 100).toFixed(1)}%</td>
        <td style="padding:10px 14px; color:#D1D5DB; font-family:'JetBrains Mono', monospace;">${(r.fetal_specificity * 100).toFixed(1)}%</td>
        <td style="padding:10px 14px; color:#A855F7; font-family:'JetBrains Mono', monospace;">${r.maternal_mae}</td>
      `;
      tableBody.appendChild(tr);
    });
  } catch (err) {
    console.error("Failed to load benchmarks:", err);
  }
}
