/* easy-vLLM front-end logic.
 * Vanilla JS, no framework. Talks to /api/* endpoints.
 */

(function () {
  "use strict";

  const $  = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  // -------------------------------------------------------------------------
  // Theme toggle
  // -------------------------------------------------------------------------
  const themeToggle = $("#theme-toggle");
  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      const root = document.documentElement;
      const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("easy-vllm-theme", next); } catch (e) {}
    });
  }

  // -------------------------------------------------------------------------
  // GPU preset dropdown population
  // -------------------------------------------------------------------------
  const presetsRaw = $("#gpu-presets-data")?.textContent || "[]";
  let presets = [];
  try { presets = JSON.parse(presetsRaw); } catch (e) { presets = []; }

  const presetSelect = $("#gpu_preset");
  const vramInput    = $("#gpu_memory_gb");
  if (presetSelect) {
    presets.forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = p.label + (p.notes ? " - " + p.notes : "");
      if (p.id === "rtx_4090_24gb") opt.selected = true;
      presetSelect.appendChild(opt);
    });
    presetSelect.addEventListener("change", () => {
      const p = presets.find((x) => x.id === presetSelect.value);
      if (p && p.id !== "custom" && vramInput) {
        vramInput.value = p.vram_gb;
        vramInput.dispatchEvent(new Event("input", { bubbles: true }));
      }
    });
  }

  // -------------------------------------------------------------------------
  // Wizard navigation
  // -------------------------------------------------------------------------
  const form    = $("#wizard-form");
  const steps   = $$(".step", form);
  const stepperItems = $$(".stepper__item");
  const btnPrev = $("#btn-prev");
  const btnNext = $("#btn-next");
  const btnGenerate = $("#btn-generate");
  let currentStep = 1;
  const TOTAL_STEPS = steps.length;

  function showStep(n) {
    currentStep = Math.max(1, Math.min(TOTAL_STEPS, n));
    steps.forEach((s) => {
      const idx = Number(s.dataset.step);
      s.hidden = idx !== currentStep;
      if (idx === currentStep) s.classList.add("is-active");
      else s.classList.remove("is-active");
    });
    stepperItems.forEach((el) => {
      const idx = Number(el.dataset.step);
      el.classList.toggle("is-active", idx === currentStep);
      el.classList.toggle("is-complete", idx < currentStep);
    });
    btnPrev.disabled = currentStep === 1;
    if (currentStep === TOTAL_STEPS) {
      btnNext.hidden = true;
      btnGenerate.hidden = false;
    } else {
      btnNext.hidden = false;
      btnGenerate.hidden = true;
    }
    form.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  btnPrev?.addEventListener("click", () => showStep(currentStep - 1));
  btnNext?.addEventListener("click", () => showStep(currentStep + 1));

  // -------------------------------------------------------------------------
  // Mode (simple / advanced)
  // -------------------------------------------------------------------------
  $$("input[name='ui_mode']").forEach((r) => {
    r.addEventListener("change", () => {
      form.classList.toggle("is-advanced", r.value === "advanced" && r.checked);
    });
  });

  // -------------------------------------------------------------------------
  // Model source toggle
  // -------------------------------------------------------------------------
  $$("input[name='model_source']").forEach((r) => {
    r.addEventListener("change", () => {
      const src = r.value;
      $$("[data-source]").forEach((el) => {
        el.hidden = el.dataset.source !== src;
      });
      requestEstimate();
    });
  });

  // Auto-derive served_model_name from model_id
  const modelIdInput = $("#model_id");
  const localPathInput = $("#local_model_path");
  const servedNameInput = $("#served_model_name");
  let servedManuallyEdited = false;
  servedNameInput?.addEventListener("input", () => { servedManuallyEdited = true; });

  function maybeDeriveServedName() {
    if (servedManuallyEdited) return;
    const src = ($$("input[name='model_source']").find((r) => r.checked) || {}).value || "huggingface";
    let raw = src === "local" ? localPathInput.value : modelIdInput.value;
    if (!raw) return;
    const base = raw.replace(/\/$/, "").split("/").pop() || "";
    if (base) servedNameInput.value = base.toLowerCase().replace(/\s+/g, "-");
  }
  modelIdInput?.addEventListener("input", maybeDeriveServedName);
  localPathInput?.addEventListener("input", maybeDeriveServedName);

  // -------------------------------------------------------------------------
  // Slider live output + fill background
  // -------------------------------------------------------------------------
  const slider = $("#gpu_memory_utilization");
  const sliderOut = $("#gpu_memory_utilization_out");
  function updateSliderUi() {
    if (!slider) return;
    const v = parseFloat(slider.value);
    if (sliderOut) sliderOut.textContent = v.toFixed(2);
    const min = parseFloat(slider.min);
    const max = parseFloat(slider.max);
    const pct = ((v - min) / (max - min)) * 100;
    slider.style.setProperty("--slider-fill", pct + "%");
    slider.style.backgroundSize = pct + "% 100%";
  }
  slider?.addEventListener("input", updateSliderUi);
  updateSliderUi();

  // Auto-sync TP with GPU count when TP wasn't manually changed
  const gpuCountInput = $("#gpu_count");
  const tpInput = $("#tensor_parallel_size");
  let tpManuallyEdited = false;
  tpInput?.addEventListener("input", () => { tpManuallyEdited = true; });
  gpuCountInput?.addEventListener("input", () => {
    if (!tpManuallyEdited && tpInput) {
      tpInput.value = gpuCountInput.value;
    }
  });

  // -------------------------------------------------------------------------
  // Drag-and-drop config.json upload
  // -------------------------------------------------------------------------
  const dropzone = $("#config-dropzone");
  const fileInput = $("#config-file-input");
  const configCard = $("#config-card");
  const configCardGrid = $("#config-card-grid");
  const configCardModel = $("#config-model-name");
  const configCardNotes = $("#config-card-notes");
  const configClear = $("#config-clear");
  const configInfoJsonInput = $("#config_info_json");
  const manualParamsField = $("#manual-params-field");

  let parsedConfig = null;

  function setParsedConfig(info) {
    parsedConfig = info || null;
    configInfoJsonInput.value = info ? JSON.stringify(info) : "";
    if (!info) {
      configCard.hidden = true;
      manualParamsField.hidden = false;
      requestEstimate();
      return;
    }
    configCard.hidden = false;
    manualParamsField.hidden = !info.is_uncertain;
    configCardModel.textContent = info.model_type ? info.model_type : "(no model_type)";
    configCardGrid.innerHTML = "";
    const fields = [
      ["layers", info.num_hidden_layers],
      ["hidden_size", info.hidden_size],
      ["attn heads", info.num_attention_heads],
      ["kv heads", info.num_key_value_heads],
      ["head_dim", info.head_dim],
      ["vocab", info.vocab_size],
      ["dtype", info.torch_dtype],
      ["max ctx", info.max_position_embeddings],
    ];
    fields.forEach(([k, v]) => {
      if (v == null || v === "") return;
      const div = document.createElement("div");
      const dt  = document.createElement("dt");
      dt.textContent = k;
      const dd  = document.createElement("dd");
      dd.textContent = String(v);
      div.appendChild(dt); div.appendChild(dd);
      configCardGrid.appendChild(div);
    });
    configCardNotes.innerHTML = "";
    (info.notes || []).forEach((n) => {
      const p = document.createElement("p");
      p.textContent = "- " + n;
      configCardNotes.appendChild(p);
    });
    requestEstimate();
  }

  configClear?.addEventListener("click", () => {
    setParsedConfig(null);
    if (fileInput) fileInput.value = "";
  });

  if (dropzone && fileInput) {
    fileInput.addEventListener("change", () => {
      if (fileInput.files && fileInput.files[0]) handleConfigFile(fileInput.files[0]);
    });
    ["dragenter", "dragover"].forEach((ev) =>
      dropzone.addEventListener(ev, (e) => {
        e.preventDefault();
        dropzone.classList.add("is-dragover");
      })
    );
    ["dragleave", "drop"].forEach((ev) =>
      dropzone.addEventListener(ev, (e) => {
        e.preventDefault();
        dropzone.classList.remove("is-dragover");
      })
    );
    dropzone.addEventListener("drop", (e) => {
      const f = e.dataTransfer?.files?.[0];
      if (f) handleConfigFile(f);
    });
  }

  async function handleConfigFile(file) {
    if (!file.name.toLowerCase().endsWith(".json")) {
      toast("Please drop a .json file (Hugging Face config.json).", "error");
      return;
    }
    const fd = new FormData();
    fd.append("config", file);
    try {
      const res = await fetch("/api/parse-config", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) {
        toast(data.error || "Failed to parse config.json", "error");
        return;
      }
      dropzone.classList.add("is-dropped");
      setTimeout(() => dropzone.classList.remove("is-dropped"), 600);
      setParsedConfig(data);
      toast("config.json parsed successfully", "success");
    } catch (err) {
      toast("Network error while uploading config.json", "error");
    }
  }

  // -------------------------------------------------------------------------
  // Form -> JSON payload
  // -------------------------------------------------------------------------
  function readPayload() {
    const fd = new FormData(form);
    const obj = {};
    for (const [k, v] of fd.entries()) {
      if (k === "ui_mode" || k === "config_info_json") continue;
      obj[k] = v;
    }
    obj.is_private_hf_model = $("#is_private_hf_model").checked;
    obj.trust_remote_code   = $("#trust_remote_code").checked;
    obj.enable_prefix_caching = $("#enable_prefix_caching").checked;
    obj.generation_config_vllm = $("#generation_config_vllm").checked;
    obj.api_key_required    = $("#api_key_required").checked;

    ["gpu_memory_gb","gpu_memory_utilization","manual_param_count_b","cpu_offload_gb"].forEach((k) => {
      if (obj[k] === "" || obj[k] === undefined) delete obj[k];
      else obj[k] = parseFloat(obj[k]);
    });
    ["gpu_count","tensor_parallel_size","pipeline_parallel_size",
     "input_tokens","output_tokens","max_num_seqs","max_num_batched_tokens"].forEach((k) => {
      if (obj[k] === "" || obj[k] === undefined) delete obj[k];
      else obj[k] = parseInt(obj[k], 10);
    });

    if (parsedConfig) obj.config_info = parsedConfig;
    return obj;
  }

  // -------------------------------------------------------------------------
  // Live estimator (debounced)
  // -------------------------------------------------------------------------
  const livePulse = $("#live-pulse");
  const cmdText = $("#cmd-text");
  const gauge = $("#gauge");
  const gaugeFill = $("#gauge-fill");
  const gaugePercent = $("#gauge-percent");
  const gaugeLabel = $("#gauge-label");
  const mWeight = $("#m-weight");
  const mKv = $("#m-kv");
  const mRuntime = $("#m-runtime");
  const mTotal = $("#m-total");
  const mUsable = $("#m-usable");
  const warningsEl = $("#warnings");
  const warningsEmpty = $("#warnings-empty");
  const suggestionsEl = $("#suggestions");
  const suggestionsList = $("#suggestions-list");

  const GAUGE_CIRC = 2 * Math.PI * 52;

  let lastTimer = null;
  let inFlight = null;

  function requestEstimate() {
    if (lastTimer) clearTimeout(lastTimer);
    lastTimer = setTimeout(async () => {
      const payload = readPayload();
      if (inFlight) inFlight.abort?.();
      const ctrl = ("AbortController" in window) ? new AbortController() : null;
      inFlight = ctrl;
      try {
        const res = await fetch("/api/estimate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
          signal: ctrl?.signal,
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          renderEstimateError(err);
          return;
        }
        const data = await res.json();
        renderEstimate(data);
      } catch (e) {
        if (e.name !== "AbortError") {
          renderEstimateError({ error: "network" });
        }
      }
    }, 280);
  }

  function pulseLive() {
    livePulse?.classList.remove("is-pulsing");
    void livePulse?.offsetWidth;
    livePulse?.classList.add("is-pulsing");
  }

  // count-up animation for a number element
  function countUp(el, target) {
    if (!el) return;
    const from = parseFloat(el.textContent) || 0;
    const dur = 360;
    const start = performance.now();
    function step(t) {
      const k = Math.min(1, (t - start) / dur);
      const eased = 1 - Math.pow(1 - k, 3);
      const val = from + (target - from) * eased;
      el.textContent = val.toFixed(1);
      if (k < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  function renderEstimate(data) {
    pulseLive();
    const m = data.memory;

    countUp(mWeight, m.weight_gb);
    countUp(mKv, m.kv_cache_gb);
    countUp(mRuntime, m.runtime_gb);
    countUp(mTotal, m.total_required_gb);
    countUp(mUsable, m.usable_gb);

    gauge.classList.remove("is-good","is-risky","is-oom","is-unknown");
    gauge.classList.add("is-" + (m.fit_status || "unknown"));
    const pct = Math.min(110, Math.max(0, m.percent_used || 0));
    const offset = GAUGE_CIRC * (1 - Math.min(1, pct / 100));
    gaugeFill.style.strokeDashoffset = offset;
    gaugePercent.textContent = (m.fit_status === "unknown") ? "--%" : Math.round(pct) + "%";
    gaugeLabel.textContent = ({
      good: "good fit",
      risky: "risky",
      oom: "likely OOM",
      unknown: "awaiting input",
    })[m.fit_status || "unknown"];

    cmdText.textContent = "vllm serve " + (data.vllm_command_oneline || "");

    renderWarnings(data.warnings || []);
    renderSuggestions(data.suggestions || []);
  }

  function renderEstimateError() {
    gauge.classList.remove("is-good","is-risky","is-oom");
    gauge.classList.add("is-unknown");
    gaugePercent.textContent = "--%";
    gaugeLabel.textContent = "validation error";
  }

  function renderWarnings(warnings) {
    while (warningsEl.lastChild && warningsEl.lastChild !== warningsEmpty) {
      warningsEl.removeChild(warningsEl.lastChild);
    }
    if (!warnings.length) {
      warningsEmpty.hidden = false;
      return;
    }
    warningsEmpty.hidden = true;
    warnings.forEach((w) => {
      const el = document.createElement("div");
      el.className = "warning-chip warning-chip--" + w.severity;
      const dot = document.createElement("span");
      dot.className = "warning-chip__dot";
      const text = document.createElement("span");
      text.textContent = w.message;
      el.appendChild(dot); el.appendChild(text);
      warningsEl.appendChild(el);
    });
  }

  function renderSuggestions(suggestions) {
    suggestionsList.innerHTML = "";
    if (!suggestions.length) { suggestionsEl.hidden = true; return; }
    suggestionsEl.hidden = false;
    suggestions.forEach((s) => {
      const li = document.createElement("li");
      const strong = document.createElement("strong");
      strong.textContent = s.title;
      const span = document.createElement("span");
      span.textContent = s.detail;
      li.appendChild(strong); li.appendChild(span);
      suggestionsList.appendChild(li);
    });
  }

  // Wire up form-wide change listeners
  form?.addEventListener("input", requestEstimate);
  form?.addEventListener("change", requestEstimate);

  // -------------------------------------------------------------------------
  // Copy-to-clipboard buttons
  // -------------------------------------------------------------------------
  document.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-copy]");
    if (!btn) return;
    e.preventDefault();
    const id = btn.getAttribute("data-copy");
    const node = document.getElementById(id);
    if (!node) return;
    const text = node.innerText || node.textContent || "";
    navigator.clipboard?.writeText(text).then(
      () => toast("Copied to clipboard", "success"),
      () => toast("Copy failed", "error")
    );
  });

  // -------------------------------------------------------------------------
  // Generate -> zip download
  // -------------------------------------------------------------------------
  btnGenerate?.addEventListener("click", async () => {
    btnGenerate.disabled = true;
    const orig = btnGenerate.innerHTML;
    btnGenerate.innerHTML = '<span class="spinner" style="width:14px;height:14px;border:2px solid currentColor;border-right-color:transparent;border-radius:50%;display:inline-block;animation:spin 700ms linear infinite;"></span> Generating...';
    try {
      const payload = readPayload();
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: "generate_failed" }));
        if (err.warnings && err.warnings.length) {
          renderWarnings(err.warnings);
          toast("Fix the highlighted errors before generating.", "error");
        } else {
          toast(err.error || "Failed to generate deployment", "error");
        }
        return;
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = $("#download-zip");
      a.href = url;
      const cd = res.headers.get("Content-Disposition") || "";
      const m = /filename="?([^"]+)"?/.exec(cd);
      a.download = m ? m[1] : "easy-vllm-output.zip";
      openModal();
      a.click();
    } catch (e) {
      toast("Network error during generation", "error");
    } finally {
      btnGenerate.disabled = false;
      btnGenerate.innerHTML = orig;
    }
  });

  // -------------------------------------------------------------------------
  // Modal open/close
  // -------------------------------------------------------------------------
  const modal = $("#result-modal");
  function openModal() { modal.hidden = false; document.body.style.overflow = "hidden"; }
  function closeModal() { modal.hidden = true; document.body.style.overflow = ""; }
  document.addEventListener("click", (e) => {
    if (e.target.matches("[data-modal-close]")) closeModal();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !modal.hidden) closeModal();
  });

  // -------------------------------------------------------------------------
  // Toasts
  // -------------------------------------------------------------------------
  const toastStack = $("#toast-stack");
  function toast(msg, kind) {
    if (!toastStack) return;
    const el = document.createElement("div");
    el.className = "toast " + (kind ? "toast--" + kind : "");
    el.textContent = msg;
    toastStack.appendChild(el);
    setTimeout(() => {
      el.style.transition = "opacity 240ms, transform 240ms";
      el.style.opacity = "0";
      el.style.transform = "translateY(8px)";
      setTimeout(() => el.remove(), 260);
    }, 2600);
  }

  // -------------------------------------------------------------------------
  // Quantization hint
  // -------------------------------------------------------------------------
  const QUANT_HINTS = {
    none: "Use full precision. Best quality; needs the most VRAM.",
    awq:  "AWQ - 4-bit, NVIDIA-friendly. Use with a pre-quantized AWQ checkpoint.",
    gptq: "GPTQ - 4-bit, broad support. Use with a pre-quantized GPTQ checkpoint.",
    fp8:  "FP8 - excellent on Hopper / Ada / MI300X (~2x memory savings).",
    bitsandbytes: "BitsAndBytes - simplest 4-bit. Image needs bitsandbytes installed (see README).",
    gguf: "GGUF - llama.cpp-style files. Advanced; usually needs a tokenizer model id.",
    marlin: "Marlin - optimized AWQ/GPTQ/FP8/FP4 kernels for newer NVIDIA GPUs.",
  };
  const quantSelect = $("#quantization");
  const quantHint = $("#quantization-hint");
  quantSelect?.addEventListener("change", () => {
    quantHint.textContent = QUANT_HINTS[quantSelect.value] || "";
  });

  // Spin animation keyframes (injected so we don't rely on @keyframes for inline use)
  const styleTag = document.createElement("style");
  styleTag.textContent = "@keyframes spin { to { transform: rotate(360deg); } }";
  document.head.appendChild(styleTag);

  // First estimate on load
  showStep(1);
  requestEstimate();
})();
