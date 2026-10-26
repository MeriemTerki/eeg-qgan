const state = {
  sampleId: null,
  meta: null,
  overview: null,
  traces: [],
  seconds: 1.28,
};

const $ = (id) => document.getElementById(id);

const ROLE = {
  quantum: { label: "Rematch", cls: "rematch" },
  classical_equiv: { label: "Rematch peer", cls: "rematch" },
  gmm: { label: "Reference", cls: "ref" },
};

const FOCUS_CH = [0, 1, 2, 4, 5, 6, 10, 11]; // temporal / central subset

const TRACE_COLORS = {
  real: "#f4f7fb",
  quantum: "#5ad1c4",
  classical_equiv: "#f0a35e",
  gmm: "#9aa7b5",
};

const fmt = (v, d = 4) =>
  v === null || v === undefined || Number.isNaN(Number(v)) ? "—" : Number(v).toFixed(d);

const signed = (v, d = 4) => {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "—";
  const n = Number(v);
  return (n >= 0 ? "+" : "") + n.toFixed(d);
};

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText || "Request failed");
  return data;
}

function setBusy(busy, message) {
  $("btn-live").disabled = busy;
  $("run-status").textContent = message;
}

function visibleArms() {
  const out = [];
  if ($("show-real").checked) out.push("real");
  if ($("show-quantum").checked) out.push("quantum");
  if ($("show-twin").checked) out.push("classical_equiv");
  if ($("show-gmm").checked) out.push("gmm");
  return out;
}

function seriesRms(series) {
  if (!series || !series.length) return 1;
  let s = 0;
  for (let i = 0; i < series.length; i += 1) s += series[i] * series[i];
  return Math.sqrt(s / series.length) || 1;
}

function drawTraces() {
  const canvas = $("eeg");
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  const gain = Number($("eeg-gain").value) || 3;
  const allCh = $("eeg-all").checked;
  const overlay = $("eeg-overlay") && $("eeg-overlay").checked;
  const want = new Set(visibleArms());

  ctx.fillStyle = "#0b1219";
  ctx.fillRect(0, 0, w, h);

  const traces = state.traces.filter((t) => want.has(t.key || "real"));
  if (!traces.length) {
    ctx.fillStyle = "#8a98a8";
    ctx.font = "500 13px Manrope, sans-serif";
    ctx.fillText("Select a holdout case", 24, 36);
    return;
  }

  const channels = traces[0].channels || [];
  const idxs = allCh
    ? channels.map((_, i) => i)
    : FOCUS_CH.filter((i) => i < channels.length);
  const nCh = idxs.length;
  const top = 28;
  const bottom = 28;
  const plotH = h - top - bottom;
  const row = plotH / nCh;
  const span = state.seconds || traces[0].duration_s || 1.28;
  const nPanels = overlay ? 1 : traces.length;
  const gap = 10;
  const leftMargin = 72;
  const usable = w - leftMargin - 12;
  const panelW = (usable - gap * (nPanels - 1)) / Math.max(1, nPanels);

  const finishLegend = () => {
    $("eeg-legend").innerHTML = traces
      .map((t) => {
        const c = TRACE_COLORS[t.key] || "#fff";
        return `<span><i style="background:${c}"></i>${t.label}</span>`;
      })
      .join("");
    $("eeg-caption").textContent = overlay
      ? `Qualitative · shared scale · ${allCh ? "22 ch" : "8 ch"} · generated traces are unconditional`
      : `Qualitative · per-column scale · ${allCh ? "22 ch" : "8 ch"} · generated traces are unconditional`;
  };

  if (overlay) {
    const x0 = leftMargin;
    ctx.fillStyle = "#8a98a8";
    ctx.font = "500 11px Manrope, sans-serif";
    ctx.fillText("Overlay · qualitative", x0, 16);
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    for (let i = 0; i <= nCh; i += 1) {
      const y = top + i * row;
      ctx.beginPath();
      ctx.moveTo(x0, y);
      ctx.lineTo(x0 + usable, y);
      ctx.stroke();
    }
    traces.forEach((trace, ti) => {
      const color = TRACE_COLORS[trace.key] || "#5ad1c4";
      ctx.strokeStyle = color;
      ctx.lineWidth = trace.key === "real" ? 1.7 : 1.3;
      idxs.forEach((c, rowIdx) => {
        const series = trace.values[c];
        if (!series || !series.length) return;
        const mid = top + rowIdx * row + row / 2;
        const amp = row * 0.38 * gain;
        ctx.beginPath();
        series.forEach((v, i) => {
          const x = x0 + (i / (series.length - 1)) * usable;
          const y = mid - v * amp;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
        if (ti === 0) {
          ctx.fillStyle = "#b7c4d2";
          ctx.font = "500 10px IBM Plex Mono, monospace";
          ctx.fillText(channels[c] || `ch${c + 1}`, 4, mid + 3);
        }
      });
    });
    ctx.fillStyle = "#8a98a8";
    ctx.font = "500 10px IBM Plex Mono, monospace";
    ctx.fillText("0 s", leftMargin, h - 10);
    ctx.fillText(`${span.toFixed(2)} s`, w - 56, h - 10);
    finishLegend();
    return;
  }

  traces.forEach((trace, panelIndex) => {
    const color = TRACE_COLORS[trace.key] || "#5ad1c4";
    const x0 = leftMargin + panelIndex * (panelW + gap);
    // subtle panel separation
    ctx.fillStyle = "rgba(255,255,255,0.02)";
    ctx.fillRect(x0 - 4, top - 8, panelW + 8, plotH + 16);
    ctx.fillStyle = "#8a98a8";
    ctx.font = "600 11px Manrope, sans-serif";
    ctx.fillText(trace.label || trace.key, x0, 16);
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    for (let i = 0; i <= nCh; i += 1) {
      const y = top + i * row;
      ctx.beginPath();
      ctx.moveTo(x0, y);
      ctx.lineTo(x0 + panelW, y);
      ctx.stroke();
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = trace.key === "real" ? 1.7 : 1.4;
    idxs.forEach((c, rowIdx) => {
      const series = trace.values[c];
      if (!series || !series.length) return;
      const mid = top + rowIdx * row + row / 2;
      const scale = (row * 0.38 * gain) / Math.max(0.05, seriesRms(series));
      ctx.beginPath();
      series.forEach((v, i) => {
        const x = x0 + (i / (series.length - 1)) * panelW;
        const y = mid - v * scale;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      if (panelIndex === 0) {
        ctx.fillStyle = "#b7c4d2";
        ctx.font = "500 10px IBM Plex Mono, monospace";
        ctx.fillText(channels[c] || `ch${c + 1}`, 4, mid + 3);
      }
    });
    ctx.fillStyle = "#8a98a8";
    ctx.font = "500 10px IBM Plex Mono, monospace";
    ctx.fillText("0 s", x0, h - 10);
    ctx.fillText(`${span.toFixed(2)} s`, x0 + panelW - 40, h - 10);
  });
  finishLegend();
}

function renderCall(detection) {
  const base = detection?.baseline || {};
  const el = $("call-label");
  const p = base.probability;
  const thr = base.threshold;
  const call = base.call || "unavailable";
  const seconds = detection?.clip?.seconds || 10.24;

  el.textContent =
    call === "seizure" ? "Seizure" :
    call === "not_seizure" ? "No seizure" : "—";
  el.className = "decision " + call;

  $("truth-label").textContent = "Seizure";
  $("truth-label").className = "decision muted-dec";

  const stats = $("detect-stats");
  if (p == null) {
    $("call-meta").textContent = "Select a window";
    if (stats) stats.hidden = true;
  } else {
    $("call-meta").textContent =
      `Decision rule · seize if p ≥ threshold (${fmt(p, 3)} ${p >= thr ? "≥" : "<"} ${fmt(thr, 3)})`;
    if (stats) {
      stats.hidden = false;
      $("stat-p").textContent = fmt(p, 3);
      $("stat-thr").textContent = fmt(thr, 3);
      $("stat-len").textContent = `${seconds} s`;
    }
  }

  if (call === "seizure") {
    $("real-call-note").textContent = "Call matches ictal ground truth on this clip";
  } else if (call === "not_seizure") {
    $("real-call-note").textContent = "Probability below decision threshold on this clip";
  } else {
    $("real-call-note").textContent = "";
  }
}

function selectSample(id) {
  state.sampleId = id;
  document.querySelectorAll(".sample").forEach((el) => {
    el.classList.toggle("on", el.dataset.id === id);
  });
}

async function showSample() {
  if (!state.sampleId) return;
  setBusy(true, "Loading case…");
  try {
    const data = await api("/api/samples/" + state.sampleId);
    state.traces = [{
      key: "real",
      label: "Real",
      ...data.window,
    }];
    state.seconds = 1.28;
    drawTraces();
    renderCall(data.detection);
    setBusy(false, "Case loaded");
  } catch (err) {
    setBusy(false, String(err));
  }
}

function renderSamples(samples) {
  const list = $("sample-list");
  list.innerHTML = "";
  samples.forEach((s, i) => {
    const b = document.createElement("button");
    b.className = "sample";
    b.dataset.id = s.id;
    const title = `Case ${String(i + 1).padStart(2, "0")}`;
    b.innerHTML = `<span class="badge">${s.label === "seizure" ? "Ictal" : s.label}</span><b>${title}</b>`;
    b.onclick = () => {
      selectSample(s.id);
      showSample();
    };
    list.appendChild(b);
  });
  if (samples[0]) {
    selectSample(samples[0].id);
    showSample();
  }
}

function genCells(r, { highlightMmd = false } = {}) {
  const g = r.generation || {};
  const d = r.decoded || {};
  return `
    <td>${r.n_params || "—"}</td>
    <td>${fmt(r.generation_ms, 0)}</td>
    <td class="${highlightMmd ? "win primary-col" : "primary-col"}">${fmt(g.mmd2, 5)}</td>
    <td>${fmt(d.sliced_wasserstein)}</td>
    <td>${fmt(d.autocorrelation_distance)}</td>
    <td>${fmt(d.spectral_band_error)}</td>`;
}

function renderGeneration(rows, meta = {}) {
  const gmm = rows.find((r) => r.arm === "gmm");
  const others = rows.filter((r) => r.arm !== "gmm");
  const hybrid = others.find((r) => r.arm === "quantum");
  const twin = others.find((r) => r.arm === "classical_equiv");
  const hybridWins =
    hybrid && twin && Number(hybrid.generation?.mmd2) < Number(twin.generation?.mmd2);

  const bestMmd = Math.min(
    ...others.map((r) => Number(r.generation?.mmd2)).filter((x) => !Number.isNaN(x))
  );

  const runMeta = $("run-meta");
  if (runMeta) {
    const hasRematch = Boolean(hybrid && twin);
    runMeta.hidden = !hasRematch;
    if (hasRematch) {
      $("run-seed").textContent = meta.seed != null ? String(meta.seed) : String($("seed").value);
      $("run-codes").textContent = meta.n_codes != null
        ? String(meta.n_codes)
        : String(hybrid.n_codes || $("n-codes").value);
      const hm = Number(hybrid.generation?.mmd2);
      const tm = Number(twin.generation?.mmd2);
      $("run-ratio").textContent = hm > 0 ? `${fmt(tm / hm, 2)}×` : "—";
    }
  }

  const cards = $("gen-simple-cards");
  if (hybrid && twin) {
    const hm = Number(hybrid.generation?.mmd2);
    const tm = Number(twin.generation?.mmd2);
    cards.innerHTML = `
      <div class="simple-card ${hybridWins ? "win" : ""}">
        <b>Hybrid</b>
        <strong>${fmt(hm, 5)}</strong>
        <span>MMD² · ${hybrid.n_params} params</span>
      </div>
      <div class="simple-card ${!hybridWins ? "win" : ""}">
        <b>Twin</b>
        <strong>${fmt(tm, 5)}</strong>
        <span>MMD² · ${twin.n_params} params</span>
      </div>`;
    $("gen-verdict").textContent = hybridWins
      ? `Lower latent MMD² · Hybrid ${fmt(hm, 5)} vs Twin ${fmt(tm, 5)}`
      : `Lower latent MMD² · Twin ${fmt(tm, 5)} vs Hybrid ${fmt(hm, 5)}`;
    $("gen-callout").textContent =
      `Generation latency · Hybrid ${fmt(hybrid.generation_ms, 0)} ms · Twin ${fmt(twin.generation_ms, 0)} ms`;
  } else {
    cards.innerHTML = `<p class="empty">Run rematch</p>`;
    $("gen-verdict").textContent = hybrid
      ? "Incomplete rematch · Twin required"
      : "";
    $("gen-callout").textContent = "";
  }

  const note = $("decoded-note");
  if (note && hybrid && twin) {
    const ha = Number(hybrid.decoded?.autocorrelation_distance);
    const ta = Number(twin.decoded?.autocorrelation_distance);
    if (!Number.isNaN(ha) && !Number.isNaN(ta) && ta < ha) {
      note.textContent =
        "Rematch ranking uses latent MMD² only. Twin can be better on Autocorr here; that does not overturn the rematch.";
    } else {
      note.textContent =
        "Rematch ranking uses latent MMD² only. W₁ / Autocorr / Spectral are supporting distances and may disagree.";
    }
  }

  $("gen-table").querySelector("tbody").innerHTML =
    others.map((r) => {
      const mmd = Number(r.generation?.mmd2);
      const win = mmd === bestMmd;
      return `<tr class="${win ? "rematch-win" : ""}">
        <td class="arm">${r.label}</td>${genCells(r, { highlightMmd: win })}
      </tr>`;
    }).join("") || `<tr><td colspan="7">Run rematch</td></tr>`;

  if (gmm) {
    $("gmm-table").querySelector("tbody").innerHTML =
      `<tr class="ref"><td class="arm">${gmm.label}</td>${genCells(gmm)}</tr>`;
    $("gmm-verdict").textContent =
      `MMD² ${fmt(gmm.generation?.mmd2, 5)} · ${gmm.n_params} params`;
    $("gmm-callout").textContent =
      "Density reference only · not capacity-matched · excluded from rematch ranking";
  } else {
    $("gmm-verdict").textContent = "—";
    $("gmm-callout").textContent = "";
    $("gmm-table").querySelector("tbody").innerHTML =
      `<tr><td colspan="7">—</td></tr>`;
  }
}

function renderWaveforms(payload) {
  const w = payload.waveforms || {};
  const traces = [];
  if (w.real) traces.push({ key: "real", label: "Real", ...w.real });
  [
    ["quantum", "Hybrid"],
    ["classical_equiv", "Twin"],
    ["gmm", "GMM"],
  ].forEach(([arm, label]) => {
    const row = (payload.rows || []).find((r) => r.arm === arm);
    const src = row?.window || w[arm];
    if (src) traces.push({ key: arm, label, ...src });
  });
  state.traces = traces;
  state.seconds = 1.28;
  drawTraces();
}

const TAB_SCOPE = {
  gen: "Live capacity rematch · latent MMD² primary",
  detect: "Single real clip call · not holdout AUROC",
  eeg: "Qualitative waveforms · not used for rematch ranking",
};

function bindTabs() {
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.onclick = () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("on", b === btn));
      document.querySelectorAll(".panel").forEach((p) => {
        p.classList.toggle("on", p.id === "panel-" + btn.dataset.tab);
      });
      const scope = $("tab-scope");
      if (scope) scope.textContent = TAB_SCOPE[btn.dataset.tab] || "";
      if (btn.dataset.tab === "eeg") drawTraces();
    };
  });
}

async function runLiveCompare() {
  setBusy(true, "Running rematch…");
  try {
    const payload = await api("/api/pipeline/live-compare", {
      method: "POST",
      body: JSON.stringify({
        sample_id: state.sampleId,
        seed: Number($("seed").value),
        n_codes: Number($("n-codes").value),
      }),
    });
    renderWaveforms(payload);
    renderGeneration(payload.rows || [], {
      seed: payload.seed,
      n_codes: payload.n_codes,
    });
    if (payload.selected_detection) renderCall(payload.selected_detection);
    const hybrid = (payload.rows || []).find((r) => r.arm === "quantum");
    const twin = (payload.rows || []).find((r) => r.arm === "classical_equiv");
    setBusy(
      false,
      hybrid && twin
        ? `Live rematch · Hybrid ${fmt(hybrid.generation?.mmd2, 5)} · Twin ${fmt(twin.generation?.mmd2, 5)}`
        : "Done"
    );
    document.querySelector('.tab[data-tab="gen"]')?.click();
  } catch (err) {
    setBusy(false, String(err));
  }
}

async function boot() {
  bindTabs();
  $("btn-live").onclick = runLiveCompare;
  ["eeg-gain", "eeg-all", "eeg-overlay", "show-real", "show-quantum", "show-twin", "show-gmm"].forEach((id) => {
    const el = $(id);
    if (!el) return;
    el.addEventListener("change", drawTraces);
    el.addEventListener("input", drawTraces);
  });

  const [meta, samples] = await Promise.all([
    api("/api/meta"),
    api("/api/samples"),
  ]);
  state.meta = meta;
  $("weight-source").textContent = meta.weight_source;
  $("data-source").textContent = meta.data_source;
  renderSamples(samples.samples || []);
  $("gen-table").querySelector("tbody").innerHTML =
    `<tr><td colspan="7">Run rematch</td></tr>`;
}

boot().catch((err) => {
  $("run-status").textContent = String(err);
});
