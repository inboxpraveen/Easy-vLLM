/* easy-vLLM front-end logic.
 * Vanilla JS, no framework. Talks to /api/* endpoints.
 *
 * Architecture: a single page with three views (home, new, deployment),
 * routed by URL hash. Wizard state lives in the form; deployment artifacts
 * are fetched on demand and cached in `lastDeployment` for tab rendering.
 */

(function () {
  "use strict";

  const $  = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  // -------------------------------------------------------------------------
  // Theme toggle
  // -------------------------------------------------------------------------
  $("#theme-toggle")?.addEventListener("click", () => {
    const root = document.documentElement;
    const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("easy-vllm-theme", next); } catch (e) {}
  });

  // -------------------------------------------------------------------------
  // Hash router
  // -------------------------------------------------------------------------
  const VIEWS = ["home", "new", "deployment"];

  function parseHash() {
    const raw = (location.hash || "#home").replace(/^#/, "");
    const [route, id] = raw.split("/");
    return { route: VIEWS.includes(route) ? route : "home", id: id || null };
  }

  function setHash(route, id) {
    const next = id ? `#${route}/${id}` : `#${route}`;
    if (location.hash !== next) location.hash = next;
  }

  function applyRoute() {
    const { route, id } = parseHash();
    $$(".view").forEach((el) => {
      const isActive = el.dataset.view === route;
      el.hidden = !isActive;
      el.classList.toggle("is-active", isActive);
    });
    if (route === "deployment" && id) {
      loadDeployment(id);
    } else if (route === "home") {
      loadHistory();
    } else if (route === "new") {
      // Make sure the form is in a useful state on every visit
      requestEstimate();
    }
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  window.addEventListener("hashchange", applyRoute);

  document.addEventListener("click", (e) => {
    const a = e.target.closest("[data-route]");
    if (!a) return;
    e.preventDefault();
    setHash(a.getAttribute("data-route"));
  });

  // -------------------------------------------------------------------------
  // GPU presets
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
  // Wizard navigation (single dynamic action button)
  // -------------------------------------------------------------------------
  const form    = $("#wizard-form");
  const steps   = $$(".step", form);
  const stepperItems = $$(".stepper__item");
  const btnPrev = $("#btn-prev");
  const btnAction = $("#btn-action");
  const actionLabel = $(".btn-action__label", btnAction);
  const iconNext = $(".btn-action__icon-next", btnAction);
  const iconGen  = $(".btn-action__icon-gen", btnAction);
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

    // Single-button design: one dynamic primary action.
    const isLast = currentStep === TOTAL_STEPS;
    btnAction.dataset.mode = isLast ? "generate" : "next";
    btnAction.classList.toggle("btn--generate", isLast);
    actionLabel.textContent = isLast ? "Generate deployment" : "Next";
    iconNext.hidden = isLast;
    iconGen.hidden = !isLast;
  }

  btnPrev?.addEventListener("click", () => showStep(currentStep - 1));
  btnAction?.addEventListener("click", () => {
    if (btnAction.dataset.mode === "generate") {
      generateDeployment();
    } else {
      showStep(currentStep + 1);
    }
  });

  // Mode (simple / advanced)
  $$("input[name='ui_mode']").forEach((r) => {
    r.addEventListener("change", () => {
      form.classList.toggle("is-advanced", r.value === "advanced" && r.checked);
    });
  });

  // Model source toggle
  $$("input[name='model_source']").forEach((r) => {
    r.addEventListener("change", () => {
      const src = r.value;
      $$("[data-source]").forEach((el) => {
        el.hidden = el.dataset.source !== src;
      });
      requestEstimate();
    });
  });

  // Auto-derive served_model_name
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

  // Slider live output
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

  // TP autosync with GPU count
  const gpuCountInput = $("#gpu_count");
  const tpInput = $("#tensor_parallel_size");
  let tpManuallyEdited = false;
  tpInput?.addEventListener("input", () => { tpManuallyEdited = true; });
  gpuCountInput?.addEventListener("input", () => {
    if (!tpManuallyEdited && tpInput) tpInput.value = gpuCountInput.value;
  });

  // Speculative model field visibility
  const specMethod = $("#speculative_method");
  const specModelField = $("#speculative-model-field");
  function updateSpecModelVisibility() {
    if (!specMethod || !specModelField) return;
    const v = specMethod.value;
    specModelField.hidden = !["draft_model", "mtp", "eagle3"].includes(v);
  }
  specMethod?.addEventListener("change", updateSpecModelVisibility);
  updateSpecModelVisibility();

  // -------------------------------------------------------------------------
  // Drag-and-drop config.json
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
      const dt = document.createElement("dt"); dt.textContent = k;
      const dd = document.createElement("dd"); dd.textContent = String(v);
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
  const FLOAT_FIELDS = [
    "gpu_memory_gb", "gpu_memory_utilization", "manual_param_count_b",
    "cpu_offload_gb", "swap_space_gb",
  ];
  const INT_FIELDS = [
    "gpu_count", "tensor_parallel_size", "pipeline_parallel_size",
    "input_tokens", "output_tokens", "max_num_seqs", "max_num_batched_tokens",
    "seed", "max_num_partial_prefills", "long_prefill_token_threshold",
    "max_loras", "max_lora_rank", "num_speculative_tokens",
    "max_log_len", "data_parallel_size",
  ];
  const CHECKBOX_FIELDS = [
    "is_private_hf_model", "trust_remote_code",
    "enable_prefix_caching", "enable_chunked_prefill", "enforce_eager",
    "disable_sliding_window", "disable_cascade_attn",
    "async_scheduling", "enable_lora", "enable_auto_tool_choice",
    "api_key_required", "enable_log_requests",
    "generation_config_vllm",
  ];

  function readPayload() {
    const fd = new FormData(form);
    const obj = {};
    for (const [k, v] of fd.entries()) {
      if (k === "ui_mode" || k === "config_info_json") continue;
      obj[k] = v;
    }
    CHECKBOX_FIELDS.forEach((k) => {
      const el = document.getElementById(k);
      if (el) obj[k] = !!el.checked;
    });
    FLOAT_FIELDS.forEach((k) => {
      if (obj[k] === "" || obj[k] === undefined) delete obj[k];
      else obj[k] = parseFloat(obj[k]);
    });
    INT_FIELDS.forEach((k) => {
      if (obj[k] === "" || obj[k] === undefined) delete obj[k];
      else obj[k] = parseInt(obj[k], 10);
    });
    if (obj.tool_call_parser === "") delete obj.tool_call_parser;
    if (obj.reasoning_parser === "") delete obj.reasoning_parser;
    if (parsedConfig) obj.config_info = parsedConfig;
    return obj;
  }

  function applyPayloadToForm(payload) {
    if (!payload) return;
    Object.keys(payload).forEach((k) => {
      const el = document.getElementById(k);
      if (!el) return;
      if (el.type === "checkbox") {
        el.checked = !!payload[k];
      } else if (el.tagName === "SELECT" || el.tagName === "INPUT" || el.tagName === "TEXTAREA") {
        if (payload[k] != null) el.value = payload[k];
      }
    });
    // model source radio
    if (payload.model_source) {
      const radio = document.getElementById(payload.model_source === "local" ? "src-local" : "src-hf");
      if (radio) {
        radio.checked = true;
        radio.dispatchEvent(new Event("change", { bubbles: true }));
      }
    }
    if (payload.config_info) setParsedConfig(payload.config_info);
    if (payload.served_model_name) servedManuallyEdited = true;
    updateSliderUi();
    updateSpecModelVisibility();
    requestEstimate();
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
          renderEstimateError();
          return;
        }
        renderEstimate(await res.json());
      } catch (e) {
        if (e.name !== "AbortError") renderEstimateError();
      }
    }, 280);
  }

  function pulseLive() {
    livePulse?.classList.remove("is-pulsing");
    void livePulse?.offsetWidth;
    livePulse?.classList.add("is-pulsing");
  }

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
      good: "good fit", risky: "risky", oom: "likely OOM", unknown: "awaiting input",
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
      const strong = document.createElement("strong"); strong.textContent = s.title;
      const span = document.createElement("span"); span.textContent = s.detail;
      li.appendChild(strong); li.appendChild(span);
      suggestionsList.appendChild(li);
    });
  }

  form?.addEventListener("input", requestEstimate);
  form?.addEventListener("change", requestEstimate);

  // -------------------------------------------------------------------------
  // Generate -> save & switch to artifacts view
  // -------------------------------------------------------------------------
  let lastDeployment = null;

  async function generateDeployment() {
    if (btnAction.dataset.mode !== "generate") return;
    const orig = btnAction.innerHTML;
    btnAction.disabled = true;
    btnAction.innerHTML = '<span class="spinner"></span><span>Generating...</span>';
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
      const data = await res.json();
      lastDeployment = data;
      renderDeployment(data);
      setHash("deployment", data.id);
      toast("Saved to history", "success");
    } catch (e) {
      toast("Network error during generation", "error");
    } finally {
      btnAction.disabled = false;
      btnAction.innerHTML = orig;
    }
  }

  // -------------------------------------------------------------------------
  // Deployment view: tabs + render
  // -------------------------------------------------------------------------
  const artName = $("#artifacts-name");
  const artFit  = $("#artifacts-fit");
  const artMeta = $("#artifacts-meta");
  const cliMulti = $("#art-cli-multi");
  const cliOne   = $("#art-cli-one");
  const compose  = $("#art-compose");
  const envCode  = $("#art-env");
  const client   = $("#art-client");
  const curl     = $("#art-curl");
  const readme   = $("#art-readme");

  function renderDeployment(data) {
    if (!data) return;
    artName.textContent = data.name || "deployment";
    artFit.textContent = (data.fit_status || data.memory?.fit_status || "unknown").toUpperCase();
    artFit.className = "artifacts__badge artifacts__badge--" + (data.fit_status || data.memory?.fit_status || "unknown");

    artMeta.innerHTML = "";
    const metaRows = [
      ["Model", data.model_id || data.request?.model_id || data.request?.local_model_path || "-"],
      ["GPU", `${data.gpu_preset || "-"} x${data.gpu_count || 1}`],
      ["Quant", data.quantization || "none"],
      ["VRAM used", data.memory?.percent_used != null ? `${Math.round(data.memory.percent_used)}%` : "-"],
      ["Created", formatTime(data.created_at)],
    ];
    metaRows.forEach(([k, v]) => {
      const span = document.createElement("span");
      span.className = "artifacts__meta-pill";
      span.innerHTML = `<em>${k}</em><strong>${escapeHtml(String(v))}</strong>`;
      artMeta.appendChild(span);
    });

    const a = data.artifacts || {};
    cliMulti.textContent = "vllm serve " + (data.command_multiline || "");
    cliOne.textContent = "vllm serve " + (data.command_oneline || "");
    compose.textContent = a["docker-compose.yml"] || "";
    envCode.textContent = a[".env"] || "";
    client.textContent = a["test_client.py"] || "";
    curl.textContent = a["test_curl.sh"] || "";
    readme.textContent = a["README.md"] || "";

    $("#zip-download")?.setAttribute("data-id", data.id);
    $("#artifacts-download")?.setAttribute("data-id", data.id);
    $("#artifacts-duplicate")?.setAttribute("data-id", data.id);
    $("#artifacts-delete")?.setAttribute("data-id", data.id);
  }

  $$(".tabs .tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      const target = tab.dataset.tab;
      $$(".tabs .tab").forEach((t) => {
        const active = t === tab;
        t.classList.toggle("is-active", active);
        t.setAttribute("aria-selected", String(active));
      });
      $$(".tab-panel").forEach((p) => {
        const active = p.dataset.panel === target;
        p.classList.toggle("is-active", active);
        p.hidden = !active;
      });
    });
  });

  async function loadDeployment(id) {
    if (lastDeployment && lastDeployment.id === id) {
      renderDeployment(lastDeployment);
      return;
    }
    try {
      const res = await fetch(`/api/deployments/${encodeURIComponent(id)}`);
      if (!res.ok) {
        toast("Deployment not found", "error");
        setHash("home");
        return;
      }
      const data = await res.json();
      lastDeployment = {
        ...data,
        download_url: `/api/deployments/${data.id}/zip`,
      };
      renderDeployment(lastDeployment);
    } catch (e) {
      toast("Network error loading deployment", "error");
      setHash("home");
    }
  }

  function downloadZip(id) {
    if (!id) return;
    window.location.href = `/api/deployments/${encodeURIComponent(id)}/zip`;
  }

  $("#zip-download")?.addEventListener("click", (e) => {
    downloadZip(e.currentTarget.getAttribute("data-id") || lastDeployment?.id);
  });
  $("#artifacts-download")?.addEventListener("click", (e) => {
    downloadZip(e.currentTarget.getAttribute("data-id") || lastDeployment?.id);
  });
  $("#artifacts-duplicate")?.addEventListener("click", () => {
    if (!lastDeployment?.request) return;
    applyPayloadToForm(lastDeployment.request);
    setHash("new");
    showStep(1);
    toast("Loaded into wizard. Edit anything, then Generate.", "success");
  });
  $("#artifacts-delete")?.addEventListener("click", async () => {
    if (!lastDeployment?.id) return;
    if (!confirm(`Delete "${lastDeployment.name}" from history?`)) return;
    try {
      const res = await fetch(`/api/deployments/${encodeURIComponent(lastDeployment.id)}`, { method: "DELETE" });
      if (res.ok) {
        toast("Deleted", "success");
        lastDeployment = null;
        setHash("home");
      } else {
        toast("Delete failed", "error");
      }
    } catch (e) {
      toast("Network error", "error");
    }
  });

  // -------------------------------------------------------------------------
  // History grid
  // -------------------------------------------------------------------------
  const historyGrid = $("#history-grid");
  const historyEmpty = $("#history-empty");

  async function loadHistory() {
    if (!historyGrid) return;
    try {
      const res = await fetch("/api/deployments");
      if (!res.ok) return;
      const rows = await res.json();
      historyGrid.innerHTML = "";
      if (!rows.length) {
        historyEmpty.hidden = false;
        return;
      }
      historyEmpty.hidden = true;
      rows.forEach((row) => historyGrid.appendChild(historyCard(row)));
    } catch (e) {
      // silent
    }
  }

  function historyCard(row) {
    const a = document.createElement("a");
    a.className = "h-card h-card--" + (row.fit_status || "unknown");
    a.href = `#deployment/${row.id}`;
    a.innerHTML = `
      <header class="h-card__head">
        <span class="h-card__name">${escapeHtml(row.name || "deployment")}</span>
        <span class="h-card__badge h-card__badge--${row.fit_status || "unknown"}">${(row.fit_status || "?").toUpperCase()}</span>
      </header>
      <p class="h-card__model"><code>${escapeHtml(row.model_id || "(no model id)")}</code></p>
      <dl class="h-card__meta">
        <div><dt>GPU</dt><dd>${escapeHtml(row.gpu_preset || "-")} x${row.gpu_count || 1}</dd></div>
        <div><dt>Quant</dt><dd>${escapeHtml(row.quantization || "none")}</dd></div>
        <div><dt>VRAM</dt><dd>${row.percent_used != null ? Math.round(row.percent_used) + "%" : "-"}</dd></div>
        <div><dt>When</dt><dd>${escapeHtml(formatTime(row.created_at))}</dd></div>
      </dl>
    `;
    return a;
  }

  // -------------------------------------------------------------------------
  // Copy / toast / utilities
  // -------------------------------------------------------------------------
  document.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-copy], [data-copy-target]");
    if (!btn) return;
    e.preventDefault();
    const id = btn.getAttribute("data-copy-target") || btn.getAttribute("data-copy");
    const node = document.getElementById(id);
    if (!node) return;
    const text = node.innerText || node.textContent || "";
    navigator.clipboard?.writeText(text).then(
      () => toast("Copied to clipboard", "success"),
      () => toast("Copy failed", "error")
    );
  });

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

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatTime(iso) {
    if (!iso) return "-";
    try {
      const d = new Date(iso);
      const diff = (Date.now() - d.getTime()) / 1000;
      if (diff < 60) return "just now";
      if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
      if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
      if (diff < 86400 * 7) return `${Math.floor(diff / 86400)}d ago`;
      return d.toLocaleDateString();
    } catch (e) { return iso; }
  }

  // -------------------------------------------------------------------------
  // Quantization hint
  // -------------------------------------------------------------------------
  const QUANT_HINTS = {
    none: "Use full precision. Best quality; needs the most VRAM.",
    awq:  "AWQ - 4-bit, NVIDIA-friendly. Use with a pre-quantized AWQ checkpoint.",
    gptq: "GPTQ - 4-bit, broad support. Use with a pre-quantized GPTQ checkpoint.",
    fp8:  "FP8 - excellent on Hopper / Ada / MI300X (~2x memory savings).",
    bitsandbytes: "BitsAndBytes - simplest 4-bit. Image needs bitsandbytes installed.",
    gguf: "GGUF - llama.cpp-style files. Usually needs a tokenizer model id.",
    marlin: "Marlin - optimized AWQ/GPTQ/FP8/FP4 kernels for newer NVIDIA GPUs.",
  };
  const quantSelect = $("#quantization");
  const quantHint = $("#quantization-hint");
  quantSelect?.addEventListener("change", () => {
    quantHint.textContent = QUANT_HINTS[quantSelect.value] || "";
  });

  // -------------------------------------------------------------------------
  // Boot
  // -------------------------------------------------------------------------
  showStep(1);
  applyRoute();
  requestEstimate();
})();
