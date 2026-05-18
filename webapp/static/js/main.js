(() => {
  const $ = (id) => document.getElementById(id);

  // Elements
  const dropzone     = $("dropzone");
  const fileInput    = $("fileInput");
  const dzEmpty      = $("dz-empty");
  const dzPreview    = $("dz-preview");
  const previewImg   = $("previewImg");
  const changeBtn    = $("changeBtn");
  const predictBtn   = $("predictBtn");
  const resetBtn     = $("resetBtn");

  const emptyState   = $("empty-state");
  const loadingState = $("loading-state");
  const resultsState = $("results-state");
  const errorState   = $("error-state");
  const errorMsg     = $("error-msg");

  const knnLabel = $("knn-label");
  const knnConf  = $("knn-conf");
  const knnK     = $("knn-k");
  const knnProba = $("knn-proba");
  const rfLabel  = $("rf-label");
  const rfConf   = $("rf-conf");
  const rfProba  = $("rf-proba");
  const agreement = $("agreement");

  let currentFile = null;

  // ── Init ──────────────────────────────────────────────────────────────
  knnK.textContent = `(K=${window.APP_CONFIG.bestK})`;

  // ── Dropzone interactions ─────────────────────────────────────────────
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  ["dragenter", "dragover"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.add("drag-over");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.remove("drag-over");
    })
  );

  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  });

  fileInput.addEventListener("change", (e) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  });

  changeBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  resetBtn.addEventListener("click", reset);
  predictBtn.addEventListener("click", predict);

  // ── File handling ─────────────────────────────────────────────────────
  function handleFile(file) {
    if (!file.type.startsWith("image/")) {
      showError("Veuillez sélectionner une image (JPG ou PNG).");
      return;
    }
    if (file.size > 8 * 1024 * 1024) {
      showError("L'image dépasse 8 MB.");
      return;
    }

    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      dzEmpty.classList.add("hidden");
      dzPreview.classList.remove("hidden");
      predictBtn.disabled = false;
      resetBtn.disabled   = false;
      hideAllResultStates();
      emptyState.classList.remove("hidden");
    };
    reader.readAsDataURL(file);
  }

  function reset() {
    currentFile = null;
    fileInput.value = "";
    previewImg.src = "";
    dzPreview.classList.add("hidden");
    dzEmpty.classList.remove("hidden");
    predictBtn.disabled = true;
    resetBtn.disabled   = true;
    hideAllResultStates();
    emptyState.classList.remove("hidden");
  }

  // ── Predict ───────────────────────────────────────────────────────────
  async function predict() {
    if (!currentFile) return;

    hideAllResultStates();
    loadingState.classList.remove("hidden");
    predictBtn.disabled = true;

    const formData = new FormData();
    formData.append("image", currentFile);

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Erreur serveur");
      renderResults(data);
    } catch (err) {
      showError(err.message || "Une erreur est survenue.");
    } finally {
      predictBtn.disabled = false;
    }
  }

  // ── Render ────────────────────────────────────────────────────────────
  function renderResults(data) {
    hideAllResultStates();

    renderBlock(data.knn, knnLabel, knnConf, knnProba);
    renderBlock(data.rf,  rfLabel,  rfConf,  rfProba);

    // Agreement banner
    if (data.knn.label === data.rf.label) {
      agreement.className = "agreement agree";
      agreement.innerHTML = `<span>✓</span> <span>Les deux modèles sont d'accord : <strong>${data.knn.label}</strong></span>`;
    } else {
      agreement.className = "agreement disagree";
      agreement.innerHTML = `<span>!</span> <span>Désaccord entre les modèles — résultat à interpréter avec prudence.</span>`;
    }

    resultsState.classList.remove("hidden");
  }

  function renderBlock(modelData, labelEl, confEl, probaEl) {
    labelEl.textContent = modelData.label;
    confEl.textContent  = `${(modelData.confidence * 100).toFixed(1)}% de confiance`;

    probaEl.innerHTML = "";
    const max = modelData.label;
    modelData.probabilities
      .slice()
      .sort((a, b) => b.p - a.p)
      .forEach(({ class: cls, p }) => {
        const row = document.createElement("div");
        row.className = "proba-row" + (cls === max ? " is-winner" : "");
        row.innerHTML = `
          <span class="proba-class">${cls}</span>
          <div class="proba-bar"><div class="proba-bar-fill" style="width: ${p * 100}%"></div></div>
          <span class="proba-value">${(p * 100).toFixed(1)}%</span>
        `;
        probaEl.appendChild(row);
      });
  }

  // ── State helpers ─────────────────────────────────────────────────────
  function hideAllResultStates() {
    emptyState.classList.add("hidden");
    loadingState.classList.add("hidden");
    resultsState.classList.add("hidden");
    errorState.classList.add("hidden");
  }

  function showError(msg) {
    hideAllResultStates();
    errorMsg.textContent = msg;
    errorState.classList.remove("hidden");
  }
})();
