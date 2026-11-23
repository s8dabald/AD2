const API_BASE = window.AD2_API_BASE || "/api";

const TAG_CHOICES = [
  ["delta_0.05", "High criticality (±5%)"],
  ["delta_0.10", "Medium criticality (±10%)"],
  ["delta_0.25", "Low criticality (±25%)"],
  ["ignore", "Not critical (ignore SHAP)"]
];

const state = {
  phase: "loading",
  reviewCount: 0,
  reviewTarget: 25,
  totalReviewCount: 0,
  totalReviewTarget: 500,
  training: false,
  submitting: false,
  selectedLabel: null,
  featureChoices: {},
  shapSearch: "",
  shapOrder: "abs_desc",
  shapMin: 0,
  case: null,
  reviewHistory: [],
  error: null
};

const savedTheme = window.localStorage.getItem("ad2-theme");
document.documentElement.dataset.theme = savedTheme === "dark" ? "dark" : "light";

function render() {
  document.getElementById("app").innerHTML = `
    <div class="shell">
      ${topbar()}
      <div class="layout">
        ${sidebar()}
        <main class="main">
          ${workspaceHead()}
          ${phaseView()}
        </main>
      </div>
    </div>
    <div id="toast" class="toast"></div>
  `;
  bindEvents();
}

function topbar() {
  const status = state.phase === "initial_training" ? "Initial training"
    : state.phase === "retrain" ? "Retraining" : state.phase === "idle"
    ? "Run not started" : state.phase === "loading" ? "Connecting"
    : state.phase === "complete" ? "Run complete" : state.phase === "error"
    ? "Run failed" : "Reviewer ready";
  return `
    <header class="topbar">
      <div class="brand">AD2 <span>Review Cockpit</span></div>
      <div class="run-meta">
        <span>Active run</span>
        <span class="run-pill">shap_box_selection</span>
        <span>retrain @ ${state.reviewTarget} reviews</span>
        <span class="run-status ${state.training ? "training" : "ready"}">${status}</span>
      </div>
      <div class="top-actions">
        <button class="btn primary" id="initiateRunBtn" ${state.phase !== "idle" ? "disabled" : ""}>Initiate Run</button>
        <button class="icon-btn theme-toggle" id="themeToggle" title="Switch theme" aria-label="Switch theme">${document.documentElement.dataset.theme === "dark" ? "☀" : "☾"}</button>
        <button class="icon-btn" id="refreshBtn" title="Refresh">↻</button>
      </div>
    </header>
  `;
}

function sidebar() {
  return `
    <aside class="sidebar">
      <button class="nav-item active"><span class="nav-mark">✓</span>Review</button>
      <button class="nav-item"><span class="nav-mark">↻</span>Run</button>
    </aside>
  `;
}

function workspaceHead() {
  const phases = [
    ["initial_training", "Initial model"],
    ["review", "Review"],
    ["retrain", "Retrain"]
  ];
  return `
    <div class="workspace-head">
      <div class="title-wrap">
        <h1>Journal Entry Review</h1>
        <div class="sub">Human-in-the-loop anomaly classification · Reviewer workspace</div>
      </div>
      <div class="phase-strip">
        ${phases.map(([id, label]) => `<div class="phase ${state.phase === id ? "active" : ""}">${label}</div>`).join("")}
      </div>
    </div>
  `;
}

function phaseView() {
  if (state.phase === "loading") return trainingView("Connecting to AD2", "Loading run state.");
  if (state.phase === "idle") return startView();
  if (state.phase === "initial_training") return trainingView("Initial training", "Preparing data, Isolation Forest, CatBoost, and SHAP values.");
  if (state.phase === "review") return state.case ? reviewView() : trainingView("Preparing next review", "Selecting next case using canonical query order.");
  if (state.phase === "retrain") return retrainView();
  if (state.phase === "complete") return completeView();
  if (state.phase === "error") return trainingView("Pipeline error", state.error || "The run stopped unexpectedly.");
  return trainingView("Working", "The canonical pipeline is running.");
}

function startView() {
  return `<section class="phase-view active"><div class="panel empty-state">
    <h2>Start the canonical AD2 run</h2>
    <p>Use <strong>Initiate Run</strong> in the top bar. The real run starts; reviewer input pauses it at each selected case.</p>
  </div></section>`;
}

function trainingView(title, text) {
  return `<section class="phase-view active"><div class="panel empty-state">
    <h2>${escapeHtml(title)}</h2><p>${escapeHtml(text)}</p><div class="spinner"></div>
  </div></section>`;
}

function reviewView() {
  const c = state.case;
  const rows = filteredShapRows();
  const totalProgress = `${state.totalReviewCount} / ${state.totalReviewTarget}`;
  const blockProgress = `${state.reviewCount} / ${state.reviewTarget}`;
  return `
    <section class="phase-view active">
      <div class="toolbar">
        <div class="field"><span class="field-label">Queue</span><span class="mono">Margin uncertainty</span></div>
        <div class="field"><span class="field-label">Batch</span><span class="mono">${blockProgress}</span></div>
        <div class="field"><span class="field-label">Total reviews</span><span class="mono">${totalProgress}</span></div>
      </div>

      <section class="panel current-case-panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">Current case</div>
            <div class="panel-sub">Selected by canonical margin-uncertainty order</div>
          </div>
          <div class="badge">review ${state.totalReviewCount + 1} of ${state.totalReviewTarget}</div>
        </div>
        <div class="case-banner">
          <div>
            <div class="case-id">Posting ID ${escapeHtml(c.posting_id)}</div>
            <div class="panel-sub">Source values below are original journal entries, not model-encoded columns.</div>
          </div>
          <div class="score"><small>uncertainty</small><strong>${Number(c.uncertainty ?? 0).toFixed(3)}</strong></div>
          <div class="label-box ${c.predicted_label === "Anomaly" ? "anomaly" : "normal"}">${escapeHtml(c.predicted_label)}</div>
        </div>

        ${journalEntries(c)}
        ${shapPolicyTable(c, rows)}
      </section>

      <div class="actionbar">
        <div class="action-info"><strong>Review decision required.</strong> Label case and choose per-feature SHAP constraints.</div>
        <div class="action-buttons">
          <button class="btn label-choice ${state.selectedLabel === "Normal" ? "selected" : ""}" data-case-label="Normal" ${state.submitting ? "disabled" : ""}>Normal</button>
          <button class="btn label-choice anomaly-choice ${state.selectedLabel === "Anomaly" ? "selected" : ""}" data-case-label="Anomaly" ${state.submitting ? "disabled" : ""}>Anomaly</button>
          <button class="btn primary" id="submitReviewBtn" ${!state.selectedLabel || state.submitting ? "disabled" : ""}>${state.submitting ? "Submitting…" : "Submit review"}</button>
        </div>
      </div>
    </section>
  `;
}

function journalEntries(caseData) {
  const entries = caseData.journal_entries || [];
  const derived = caseData.derived_features || [];
  return `
    <section class="journal-section">
      <div class="section-bar">
        <h3>Original journal values</h3>
        <div class="section-tools"><span class="panel-sub">${entries.length} source ${entries.length === 1 ? "entry" : "entries"} · raw CSV fields</span></div>
      </div>
      ${derived.length ? `<div class="derived-feature-list">${derived.map(item => `
        <div class="derived-feature-card">
          <span><strong>${escapeHtml(humanize(item.feature))}</strong><small>${escapeHtml(item.description)}</small></span>
          <strong class="mono">${escapeHtml(formatRawValue(item.value))} days</strong>
        </div>`).join("")}</div>` : ""}
      ${entries.length ? entries.map((entry, index) => `
        <article class="journal-entry-card">
          <div class="journal-entry-head">
            <span>Line item ${index + 1}</span>
            <span class="mono">${escapeHtml(caseData.posting_id)}</span>
          </div>
          <dl class="journal-field-grid">
            ${Object.entries(entry).map(([name, value]) => `
              <div class="journal-field ${name === "text" ? "journal-field-wide" : ""}">
                <dt>${escapeHtml(humanize(name))}</dt>
                <dd>${escapeHtml(formatRawValue(value))}</dd>
              </div>`).join("")}
          </dl>
        </article>`).join("")
        : `<div class="empty-state">No source journal rows found for this posting.</div>`}
    </section>
  `;
}

function shapPolicyTable(caseData, rows) {
  const adaptive = new Set(caseData.adaptive_features || []);
  return `
    <section class="shap-policy-section">
      <div class="section-bar">
        <h3>SHAP feature policy</h3>
        <div class="section-tools">
          <span class="badge">adaptive top-k: ${Number(caseData.adaptive_k || 0)}</span>
          <span class="panel-sub">Policy applies after batch retrain using refreshed SHAP values.</span>
        </div>
      </div>
      <div class="policy-explainer">
        Choose each feature’s SHAP-box constraint. Highlight shows current model’s adaptive top-k;
        top-k and numeric bands are recomputed from post-retrain SHAP before propagation.
      </div>
      <div class="shap-toolbar policy-toolbar">
        <div class="inline-field"><input id="shapSearch" placeholder="Search SHAP feature" value="${escapeAttr(state.shapSearch)}" /></div>
        <div class="inline-field"><span class="field-label">Min |SHAP|</span><input id="shapMin" type="number" step="0.01" min="0" value="${state.shapMin}" /></div>
        <div class="inline-field"><select id="shapOrder">
          <option value="abs_desc" ${state.shapOrder === "abs_desc" ? "selected" : ""}>|SHAP| high → low</option>
          <option value="value_desc" ${state.shapOrder === "value_desc" ? "selected" : ""}>SHAP high → low</option>
          <option value="value_asc" ${state.shapOrder === "value_asc" ? "selected" : ""}>SHAP low → high</option>
          <option value="name" ${state.shapOrder === "name" ? "selected" : ""}>Feature A → Z</option>
        </select></div>
        <button class="sort-button" id="clearShapFilter" title="Clear SHAP filters">×</button>
      </div>
      <div class="table-wrap shap-policy-wrap">
        <table>
          <thead><tr><th>Model feature</th><th>Current SHAP</th><th>Adaptive top-k</th><th>Feature criticality</th><th>Band preview</th></tr></thead>
          <tbody>${rows.map(row => {
            const choice = state.featureChoices[row.name] || "ignore";
            return `<tr class="${adaptive.has(row.name) ? "adaptive-row" : ""}">
              <td class="shap-name">${escapeHtml(row.name)}</td>
              <td class="shap-value ${row.value >= 0 ? "pos" : "neg"}">${formatSigned(row.value)}</td>
              <td>${adaptive.has(row.name) ? `<span class="badge adaptive-badge">Current top-k</span>` : `<span class="panel-sub">Outside current top-k</span>`}</td>
              <td><select class="tag-select" data-policy-feature="${escapeAttr(row.name)}" ${state.submitting ? "disabled" : ""}>
                ${TAG_CHOICES.map(([value, label]) => `<option value="${value}" ${choice === value ? "selected" : ""}>${label}</option>`).join("")}
              </select></td>
              <td class="mono band-preview">${escapeHtml(formatPolicyBand(row.value, choice))}</td>
            </tr>`;
          }).join("")}</tbody>
        </table>
        ${rows.length ? "" : `<div class="empty-state">No SHAP features match current filters.</div>`}
      </div>
    </section>
  `;
}

function retrainView() {
  return `
    <section class="phase-view active">
      <div class="retrain-box training-card">
        <div class="spinner"></div>
        <h2>Retraining CatBoost</h2>
        <p>Batch complete (${state.reviewTarget} direct reviews). Applying labels and SHAP-box policies, then fitting next model.</p>
        <div class="retrain-summary">
          <div><small>Batch reviews</small><strong>${state.reviewCount}/${state.reviewTarget}</strong></div>
          <div><small>Total reviews</small><strong>${state.totalReviewCount}/${state.totalReviewTarget}</strong></div>
          <div><small>Method</small><strong>shap_box_selection</strong></div>
        </div>
        <div class="training-lock">Review is paused until canonical retrain and propagation finish.</div>
      </div>
    </section>`;
}

function completeView() {
  return `<section class="phase-view active"><div class="panel empty-state">
    <h2>Run complete</h2>
    <p>Reviewed ${state.totalReviewCount} cases. Canonical training and propagation finished.</p>
  </div></section>`;
}

function detail(label, value) {
  return `<div class="detail"><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(String(value ?? "—"))}</dd></div>`;
}

function filteredShapRows() {
  let rows = (state.case?.shap || []).map(row => ({
    name: Array.isArray(row) ? row[0] : row.feature,
    value: Number(Array.isArray(row) ? row[1] : row.value)
  }));
  const query = state.shapSearch.toLowerCase().trim();
  rows = rows.filter(row =>
    (!query || row.name.toLowerCase().includes(query))
    && Math.abs(row.value) >= Number(state.shapMin || 0)
  );
  if (state.shapOrder === "abs_desc") rows.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  if (state.shapOrder === "value_desc") rows.sort((a, b) => b.value - a.value);
  if (state.shapOrder === "value_asc") rows.sort((a, b) => a.value - b.value);
  if (state.shapOrder === "name") rows.sort((a, b) => a.name.localeCompare(b.name));
  return rows;
}

function formatSigned(value) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
}

function formatPolicyBand(value, choice) {
  if (!choice || choice === "ignore") return "Unrestricted";
  const delta = Number(choice.slice("delta_".length));
  if (value === 0) return "[-0.000, +0.000]";
  const bounds = [value * (1 - delta), value * (1 + delta)].sort((a, b) => a - b);
  return `[${bounds[0].toFixed(3)}, ${bounds[1].toFixed(3)}]`;
}

function formatRawValue(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toLocaleString(undefined, {maximumFractionDigits: 6});
  return String(value);
}

function humanize(name) {
  const labels = {
    document_date: "Document date", posting_date: "Posting date",
    entered_date: "Entered date", gl_account: "GL account",
    gl_account_name: "GL account name", cd_flag: "CD flag",
    ref_id: "Reference ID", tax_rate: "Tax rate", high_cash: "High cash",
    top_n: "Top-N", nwh: "NWH"
  };
  return labels[name] || name.replaceAll("_", " ").replace(/\b\w/g, char => char.toUpperCase());
}

function showToast(message) {
  const el = document.getElementById("toast");
  el.textContent = message;
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 2600);
}

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {"Content-Type": "application/json"},
    ...options
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function applyApiState(payload) {
  if (!payload) return;
  const previousId = state.case?.posting_id;
  state.phase = payload.phase || "loading";
  state.reviewCount = payload.review_count ?? state.reviewCount;
  state.reviewTarget = payload.review_target ?? state.reviewTarget;
  state.totalReviewCount = payload.total_review_count ?? state.totalReviewCount;
  state.totalReviewTarget = payload.total_review_target ?? state.totalReviewTarget;
  state.training = Boolean(payload.training);
  state.error = payload.error || null;
  state.reviewHistory = payload.review_history || [];
  if (payload.case) {
    state.case = {...payload.case};
    state.case.shap = (payload.case.shap || []).map(row =>
      Array.isArray(row) ? row : [row.feature, row.value]
    );
    if (previousId !== state.case.posting_id) {
      state.selectedLabel = null;
      state.featureChoices = Object.fromEntries(
        state.case.shap.map(([feature]) => [feature, "ignore"])
      );
      state.shapSearch = "";
      state.shapMin = 0;
      state.shapOrder = "abs_desc";
    }
  } else {
    state.case = null;
  }
}

async function loadApiState() {
  try {
    applyApiState(await api("/run/state"));
  } catch (error) {
    showToast(`Backend unavailable: ${error.message}`);
  }
}

async function waitForStateChange(previousPostingId) {
  for (let attempt = 0; attempt < 240; attempt += 1) {
    const payload = await api("/run/state");
    if (payload.phase !== "review" || (payload.case && payload.case.posting_id !== previousPostingId)) {
      return payload;
    }
    await new Promise(resolve => window.setTimeout(resolve, 250));
  }
  throw new Error("The backend did not advance from this review.");
}

async function submitReviewDecision() {
  if (!state.case || !state.selectedLabel || state.submitting) return;
  const postingId = state.case.posting_id;
  state.submitting = true;
  render();
  try {
    const choices = Object.fromEntries(
      state.case.shap.map(([feature]) => [feature, state.featureChoices[feature] || "ignore"])
    );
    await api("/review", {
      method: "POST",
      body: JSON.stringify({
        posting_id: postingId,
        label: state.selectedLabel,
        feature_choices: choices
      })
    });
    applyApiState(await waitForStateChange(postingId));
    state.selectedLabel = null;
    state.submitting = false;
    render();
    scheduleStatePoll();
  } catch (error) {
    state.submitting = false;
    render();
    showToast(`Review failed: ${error.message}`);
  }
}

function bindEvents() {
  const byId = id => document.getElementById(id);
  byId("themeToggle")?.addEventListener("click", () => {
    const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem("ad2-theme", theme);
    render();
  });
  byId("refreshBtn")?.addEventListener("click", async () => {
    await loadApiState();
    render();
  });
  byId("initiateRunBtn")?.addEventListener("click", async () => {
    try {
      applyApiState(await api("/run/start", {
        method: "POST", body: JSON.stringify({run_id: "active"})
      }));
      render();
      scheduleStatePoll();
    } catch (error) {
      showToast(`Run start failed: ${error.message}`);
    }
  });
  document.querySelectorAll("[data-case-label]").forEach(button => {
    button.addEventListener("click", () => {
      state.selectedLabel = button.dataset.caseLabel;
      render();
    });
  });
  document.querySelectorAll("[data-policy-feature]").forEach(select => {
    select.addEventListener("change", event => {
      const feature = event.target.dataset.policyFeature;
      state.featureChoices[feature] = event.target.value;
      render();
      [...document.querySelectorAll("[data-policy-feature]")]
        .find(input => input.dataset.policyFeature === feature)
        ?.focus();
    });
  });
  byId("submitReviewBtn")?.addEventListener("click", submitReviewDecision);
  byId("shapSearch")?.addEventListener("input", event => {
    state.shapSearch = event.target.value;
    render();
    focusEnd("shapSearch");
  });
  byId("shapMin")?.addEventListener("input", event => {
    state.shapMin = event.target.value;
    render();
    focusEnd("shapMin");
  });
  byId("shapOrder")?.addEventListener("change", event => {
    state.shapOrder = event.target.value;
    render();
  });
  byId("clearShapFilter")?.addEventListener("click", () => {
    state.shapSearch = "";
    state.shapMin = 0;
    state.shapOrder = "abs_desc";
    render();
  });
}

function focusEnd(id) {
  const element = document.getElementById(id);
  if (element) {
    element.focus();
    element.selectionStart = element.selectionEnd = element.value.length;
  }
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[char]));
}

function escapeAttr(value) { return escapeHtml(value); }

let statePollTimer = null;
function shouldPollState() {
  return state.phase === "loading"
    || state.phase === "initial_training"
    || state.phase === "retrain"
    || (state.phase === "review" && !state.case);
}

function scheduleStatePoll() {
  if (statePollTimer !== null || !shouldPollState()) return;
  statePollTimer = window.setTimeout(async () => {
    statePollTimer = null;
    await loadApiState();
    render();
    scheduleStatePoll();
  }, 1200);
}

render();
loadApiState().then(() => {
  render();
  scheduleStatePoll();
});
