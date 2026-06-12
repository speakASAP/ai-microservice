const state = {
  agents: [],
  allAgents: [],
  models: [],
  logs: [],
  selectedId: null,
  view: "agents",
  filters: {
    status: "",
    modelTier: "",
  },
};

const els = {
  navItems: Array.from(document.querySelectorAll(".nav-item")),
  viewPanels: Array.from(document.querySelectorAll("[data-view-panel]")),
  viewTitle: document.getElementById("view-title"),
  viewSubtitle: document.getElementById("view-subtitle"),
  quickFilters: document.getElementById("quick-filters"),
  refreshButton: document.getElementById("refresh-button"),
  newAgentButton: document.getElementById("new-agent-button"),
  searchInput: document.getElementById("search-input"),
  statusFilter: document.getElementById("status-filter"),
  modelFilter: document.getElementById("model-filter"),
  agentList: document.getElementById("agent-list"),
  emptyState: document.getElementById("empty-state"),
  form: document.getElementById("agent-form"),
  editorTitle: document.getElementById("editor-title"),
  editorSubtitle: document.getElementById("editor-subtitle"),
  duplicateButton: document.getElementById("duplicate-button"),
  deleteButton: document.getElementById("delete-button"),
  message: document.getElementById("message"),
  listSummary: document.getElementById("list-summary"),
  providerModelSelect: document.getElementById("provider-model-select"),
  providerModelNote: document.getElementById("provider-model-note"),
  promptList: document.getElementById("prompt-list"),
  modelList: document.getElementById("model-list"),
  modelCatalog: document.getElementById("model-catalog"),
  logList: document.getElementById("log-list"),
  logRangeFilter: document.getElementById("log-range-filter"),
  logLevelFilter: document.getElementById("log-level-filter"),
  logServiceFilter: document.getElementById("log-service-filter"),
  logSearchInput: document.getElementById("log-search-input"),
  logRefreshButton: document.getElementById("log-refresh-button"),
  agentModal: document.getElementById("agent-modal"),
  agentModalTitle: document.getElementById("agent-modal-title"),
  agentModalSubtitle: document.getElementById("agent-modal-subtitle"),
  agentModalBody: document.getElementById("agent-modal-body"),
  agentModalClose: document.getElementById("agent-modal-close"),
};

const viewCopy = {
  agents: {
    title: "Agent Registry",
    subtitle: "Manage prompts, model routing, execution scope, and JSON configuration for AI agents.",
  },
  prompts: {
    title: "Prompt Library",
    subtitle: "Review the system prompts and user templates used by registered agents.",
  },
  models: {
    title: "Model Routing",
    subtitle: "Compare model tiers, provider overrides, token limits, and temperatures.",
  },
  logs: {
    title: "Registry Activity",
    subtitle: "Review recently updated agent definitions and admin registry changes.",
  },
};

function setMessage(text, type = "") {
  els.message.textContent = text;
  els.message.className = `message ${type}`.trim();
}

function authHeaders() {
  return {
    "Content-Type": "application/json",
  };
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    method: options.method || "GET",
    headers: authHeaders(),
    body: options.body ? JSON.stringify(options.body) : undefined,
    credentials: "same-origin",
  });

  if (response.status === 401) {
    window.location.assign("/admin/logout");
    throw new Error("Admin authentication required.");
  }

  if (!response.ok) {
    let details = `Request failed (${response.status})`;
    try {
      const payload = await response.json();
      details = payload.message || details;
    } catch (_err) {
      // Empty or non-JSON error body.
    }
    throw new Error(Array.isArray(details) ? details.join(", ") : details);
  }

  if (response.status === 204) return null;
  return response.json();
}

function parseJsonField(value, fieldName) {
  const trimmed = String(value || "").trim();
  if (!trimmed) return null;
  try {
    const parsed = JSON.parse(trimmed);
    if (parsed === null || Array.isArray(parsed) || typeof parsed !== "object") {
      throw new Error(`${fieldName} must be a JSON object.`);
    }
    return parsed;
  } catch (error) {
    throw new Error(`${fieldName}: ${error.message}`);
  }
}

function formatDate(value) {
  if (!value) return "—";
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function summarize() {
  els.listSummary.textContent = state.agents.length === 1 ? "1 agent loaded" : `${state.agents.length} agents loaded`;
  renderQuickFilters();
}

function renderQuickFilters() {
  const statuses = countBy(state.allAgents, "status");
  const tiers = countBy(state.allAgents, "modelTier");
  const buttons = [
    { label: "All", count: state.allAgents.length, kind: "all", value: "" },
    ...["active", "draft", "disabled"].filter((status) => statuses[status]).map((status) => ({
      label: status,
      count: statuses[status],
      kind: "status",
      value: status,
    })),
    ...["free", "cheap", "smart", "premium"].filter((tier) => tiers[tier]).map((tier) => ({
      label: tier,
      count: tiers[tier],
      kind: "modelTier",
      value: tier,
    })),
  ];

  els.quickFilters.innerHTML = buttons.map((button) => `
    <button class="metric-chip ${isQuickFilterActive(button) ? "active" : ""}" type="button" data-kind="${button.kind}" data-value="${button.value}">
      <span>${escapeHtml(button.label)}</span>
      <strong>${button.count}</strong>
    </button>
  `).join("");

  els.quickFilters.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => applyQuickFilter(button.dataset.kind, button.dataset.value || ""));
  });
}

function countBy(items, field) {
  return items.reduce((acc, item) => {
    const key = item[field] || "";
    if (!key) return acc;
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});
}

function isQuickFilterActive(button) {
  if (button.kind === "all") return !state.filters.status && !state.filters.modelTier;
  return state.filters[button.kind] === button.value;
}

function applyQuickFilter(kind, value) {
  if (kind === "all") {
    state.filters.status = "";
    state.filters.modelTier = "";
  } else {
    state.filters[kind] = state.filters[kind] === value ? "" : value;
    if (kind === "status") state.filters.modelTier = "";
    if (kind === "modelTier") state.filters.status = "";
  }
  els.statusFilter.value = state.filters.status;
  els.modelFilter.value = state.filters.modelTier;
  loadAgents().catch((error) => setMessage(error.message, "error"));
}

function renderAgents() {
  els.agentList.innerHTML = "";
  els.emptyState.classList.toggle("visible", state.agents.length === 0);

  for (const agent of state.agents) {
    const row = document.createElement("tr");
    row.className = agent.id === state.selectedId ? "selected" : "";
    row.dataset.id = agent.id;
    row.innerHTML = `
      <td>
        <div class="agent-name">
          <strong>${escapeHtml(agent.name)}</strong>
          <span>${escapeHtml(agent.slug)}</span>
        </div>
      </td>
      <td>${escapeHtml(agent.serviceScope || "—")}</td>
      <td>${renderChips(agent.metadata?.usedBy || agent.metadata?.applications || [])}</td>
      <td><span class="pill tier">${escapeHtml(agent.modelTier || "free")}</span></td>
      <td><span class="pill ${escapeHtml(agent.status || "draft")}">${escapeHtml(agent.status || "draft")}</span></td>
      <td class="muted">${formatDate(agent.updatedAt)}</td>
    `;
    row.addEventListener("click", () => selectAgent(agent.id));
    row.addEventListener("dblclick", () => openAgentDetail(agent.id));
    els.agentList.appendChild(row);
  }
}

function renderPromptView() {
  els.promptList.innerHTML = "";
  for (const agent of state.agents) {
    const item = document.createElement("article");
    item.className = "detail-item";
    item.innerHTML = `
      <div class="detail-item-head">
        <div>
          <h3>${escapeHtml(agent.name)}</h3>
          <span>${escapeHtml(agent.serviceScope || "unscoped")} ${agent.routePath ? "· " + escapeHtml(agent.routePath) : ""}</span>
        </div>
        <span class="pill ${escapeHtml(agent.status || "draft")}">${escapeHtml(agent.status || "draft")}</span>
      </div>
      <div class="prompt-grid" data-agent-id="${escapeHtml(agent.id)}">
        <label>System prompt<textarea name="systemPrompt" rows="4">${escapeHtml(agent.systemPrompt || "")}</textarea></label>
        <label>User template<textarea name="userPromptTemplate" rows="4">${escapeHtml(agent.userPromptTemplate || "")}</textarea></label>
      </div>
      <div class="row-actions">
        <button class="ghost" type="button" data-action="details" data-id="${escapeHtml(agent.id)}">Details</button>
        <button class="primary" type="button" data-action="save-prompts" data-id="${escapeHtml(agent.id)}">Save prompts</button>
      </div>
    `;
    els.promptList.appendChild(item);
  }
  if (!state.agents.length) renderEmptyDetail(els.promptList, "No prompts are registered yet.");
  els.promptList.querySelectorAll("[data-action='save-prompts']").forEach((button) => {
    button.addEventListener("click", () => savePromptEdits(button.dataset.id));
  });
  els.promptList.querySelectorAll("[data-action='details']").forEach((button) => {
    button.addEventListener("click", () => openAgentDetail(button.dataset.id));
  });
}

function renderModelView() {
  els.modelList.innerHTML = "";
  els.modelCatalog.innerHTML = "";
  const tiers = ["free", "cheap", "smart", "premium"];
  for (const tier of tiers) {
    const agents = state.agents.filter((agent) => (agent.modelTier || "free") === tier);
    if (!agents.length) continue;
    const item = document.createElement("article");
    item.className = "detail-item";
    item.innerHTML = `
      <div class="detail-item-head">
        <div>
          <h3>${tier[0].toUpperCase() + tier.slice(1)}</h3>
          <span>${agents.length === 1 ? "1 agent" : `${agents.length} agents`}</span>
        </div>
        <span class="pill tier">${tier}</span>
      </div>
      <div class="route-list">
        ${agents.map((agent) => `
          <button class="route-row" type="button" data-id="${escapeHtml(agent.id)}">
            <strong>${escapeHtml(agent.name)}</strong>
            <span>${escapeHtml(modelLabel(agent.providerModel))} · temp ${escapeHtml(agent.temperature || "0.20")} · ${escapeHtml(agent.maxTokens || 1000)} tokens</span>
            <span>${renderChips(agent.metadata?.usedBy || [])}</span>
          </button>
        `).join("") || '<div class="empty-detail">No agents use this tier.</div>'}
      </div>
    `;
    els.modelList.appendChild(item);
  }
  els.modelList.querySelectorAll(".route-row").forEach((button) => {
    button.addEventListener("click", () => openAgentDetail(button.dataset.id));
  });
  for (const model of state.models) {
    const item = document.createElement("article");
    item.className = "model-card";
    const users = state.allAgents.filter((agent) => (agent.providerModel || "") === model.id || (!agent.providerModel && model.id === ""));
    item.innerHTML = `
      <div class="detail-item-head">
        <div>
          <h3>${escapeHtml(model.label)}</h3>
          <span>${escapeHtml(model.provider)} · ${escapeHtml(model.tier)}</span>
        </div>
        <span class="pill tier">${escapeHtml(model.price)}</span>
      </div>
      <p>${escapeHtml(model.bestFor)}</p>
      <p class="muted">${escapeHtml(model.caution)}</p>
      <div class="chip-row">${renderChips(users.map((agent) => agent.name)) || '<span class="mini-chip">No direct assignments</span>'}</div>
    `;
    els.modelCatalog.appendChild(item);
  }
}

async function renderLogView() {
  if (!state.logs.length) await loadLogs();
  els.logList.innerHTML = "";
  for (const [index, log] of state.logs.entries()) {
    const item = document.createElement("article");
    item.className = "activity-row";
    item.innerHTML = `
      <span class="activity-time">${formatDate(log.timestamp)}</span>
      <strong>${escapeHtml(log.service || "unknown")}</strong>
      <span>${escapeHtml(log.level || "info")} · ${escapeHtml(log.message || log.msg || log.event || "log entry")}</span>
      <button class="ghost" type="button" data-log-index="${index}">Open</button>
    `;
    els.logList.appendChild(item);
  }
  if (!state.logs.length) renderEmptyDetail(els.logList, "No logs match the current filters.");
  els.logList.querySelectorAll("[data-log-index]").forEach((button) => {
    button.addEventListener("click", () => openLogDetail(Number(button.dataset.logIndex)));
  });
}

function renderEmptyDetail(container, text) {
  const item = document.createElement("div");
  item.className = "empty-detail";
  item.textContent = text;
  container.appendChild(item);
}

function renderCurrentView() {
  if (state.view === "prompts") renderPromptView();
  if (state.view === "models") renderModelView();
  if (state.view === "logs") renderLogView();
}

function renderChips(values) {
  const list = Array.isArray(values) ? values : [values].filter(Boolean);
  return `<span class="chip-row">${list.map((value) => `<span class="mini-chip">${escapeHtml(value)}</span>`).join("")}</span>`;
}

function modelLabel(id) {
  return state.models.find((model) => model.id === (id || ""))?.label || id || "Default route";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function loadAgents() {
  const params = new URLSearchParams();
  if (els.searchInput.value.trim()) params.set("q", els.searchInput.value.trim());
  if (state.filters.status || els.statusFilter.value) params.set("status", state.filters.status || els.statusFilter.value);
  if (state.filters.modelTier || els.modelFilter.value) params.set("modelTier", state.filters.modelTier || els.modelFilter.value);

  const payload = await api(`/admin/api/agents?${params.toString()}`);
  state.agents = payload.items || [];
  if (!params.toString()) state.allAgents = state.agents;
  if (!state.allAgents.length) {
    const allPayload = await api("/admin/api/agents");
    state.allAgents = allPayload.items || [];
  }
  if (!state.selectedId || !state.agents.some((agent) => agent.id === state.selectedId)) {
    state.selectedId = state.agents[0]?.id || null;
  }
  summarize();
  renderAgents();
  renderCurrentView();
  if (state.selectedId) fillForm(state.agents.find((agent) => agent.id === state.selectedId));
  else resetForm();
}

function setView(view) {
  state.view = view;
  els.navItems.forEach((item) => item.classList.toggle("active", item.dataset.view === view));
  els.viewPanels.forEach((panel) => {
    panel.hidden = panel.dataset.viewPanel !== view;
  });
  const copy = viewCopy[view] || viewCopy.agents;
  els.viewTitle.textContent = copy.title;
  els.viewSubtitle.textContent = copy.subtitle;
  renderCurrentView();
  if (view !== "agents") setMessage(`${copy.title} loaded.`, "ok");
}

async function loadModels() {
  const payload = await api("/admin/api/models");
  state.models = payload.items || [];
  renderProviderModelOptions();
}

async function loadLogs() {
  const params = new URLSearchParams();
  params.set("range", els.logRangeFilter.value || "24h");
  params.set("limit", "150");
  if (els.logLevelFilter.value) params.set("level", els.logLevelFilter.value);
  if (els.logServiceFilter.value.trim()) params.set("service", els.logServiceFilter.value.trim());
  if (els.logSearchInput.value.trim()) params.set("q", els.logSearchInput.value.trim());
  const payload = await api(`/admin/api/logs?${params.toString()}`);
  state.logs = payload.items || [];
}

function renderProviderModelOptions() {
  els.providerModelSelect.innerHTML = state.models.map((model) => `
    <option value="${escapeHtml(model.id)}">${escapeHtml(model.label)} · ${escapeHtml(model.price)}</option>
  `).join("");
}

function selectAgent(id) {
  state.selectedId = id;
  renderAgents();
  fillForm(state.agents.find((agent) => agent.id === id));
}

function resetForm() {
  els.form.reset();
  els.form.elements.id.value = "";
  els.form.elements.status.value = "draft";
  els.form.elements.modelTier.value = "free";
  els.form.elements.providerModel.value = "";
  updateProviderModelNote();
  els.form.elements.temperature.value = "0.20";
  els.form.elements.maxTokens.value = "1000";
  els.editorTitle.textContent = "Create agent";
  els.editorSubtitle.textContent = "Define prompt, routing, and runtime configuration.";
  state.selectedId = null;
  renderAgents();
}

function fillForm(agent) {
  if (!agent) {
    resetForm();
    return;
  }

  els.form.elements.id.value = agent.id || "";
  els.form.elements.name.value = agent.name || "";
  els.form.elements.slug.value = agent.slug || "";
  els.form.elements.status.value = agent.status || "draft";
  els.form.elements.serviceScope.value = agent.serviceScope || "";
  els.form.elements.routePath.value = agent.routePath || "";
  els.form.elements.description.value = agent.description || "";
  els.form.elements.modelTier.value = agent.modelTier || "free";
  els.form.elements.providerModel.value = agent.providerModel || "";
  updateProviderModelNote();
  els.form.elements.temperature.value = agent.temperature || "0.20";
  els.form.elements.maxTokens.value = agent.maxTokens || 1000;
  els.form.elements.tags.value = (agent.tags || []).join(", ");
  els.form.elements.systemPrompt.value = agent.systemPrompt || "";
  els.form.elements.userPromptTemplate.value = agent.userPromptTemplate || "";
  els.form.elements.outputSchema.value = agent.outputSchema ? JSON.stringify(agent.outputSchema, null, 2) : "";
  els.form.elements.metadata.value = agent.metadata ? JSON.stringify(agent.metadata, null, 2) : "";
  els.editorTitle.textContent = agent.name || "Edit agent";
  els.editorSubtitle.textContent = `${agent.serviceScope || "unscoped"} ${agent.routePath ? "· " + agent.routePath : ""}`;
}

function updateProviderModelNote() {
  const selected = state.models.find((model) => model.id === els.form.elements.providerModel.value);
  if (!selected) {
    els.providerModelNote.textContent = "";
    return;
  }
  els.providerModelNote.textContent = `${selected.tier} · ${selected.provider} · ${selected.bestFor} Price: ${selected.price}`;
}

function formPayload() {
  const form = els.form.elements;
  return {
    name: form.name.value,
    slug: form.slug.value,
    status: form.status.value,
    serviceScope: form.serviceScope.value,
    routePath: form.routePath.value,
    description: form.description.value,
    modelTier: form.modelTier.value,
    providerModel: form.providerModel.value,
    temperature: Number(form.temperature.value),
    maxTokens: Number(form.maxTokens.value),
    tags: form.tags.value,
    systemPrompt: form.systemPrompt.value,
    userPromptTemplate: form.userPromptTemplate.value,
    outputSchema: parseJsonField(form.outputSchema.value, "Output schema JSON"),
    metadata: parseJsonField(form.metadata.value, "Metadata JSON"),
  };
}

function slugFromName(name) {
  return String(name || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 160);
}

els.refreshButton.addEventListener("click", async () => {
  try {
    await loadAgents();
    setMessage("Agent registry refreshed.", "ok");
  } catch (error) {
    setMessage(error.message, "error");
  }
});

els.newAgentButton.addEventListener("click", () => {
  setView("agents");
  resetForm();
  setMessage("Ready to create a new agent.", "");
});

els.navItems.forEach((item) => {
  item.addEventListener("click", () => setView(item.dataset.view || "agents"));
});

els.searchInput.addEventListener("input", debounce(() => loadAgents().catch((error) => setMessage(error.message, "error")), 250));
els.statusFilter.addEventListener("change", () => {
  state.filters.status = els.statusFilter.value;
  state.filters.modelTier = "";
  els.modelFilter.value = "";
  loadAgents().catch((error) => setMessage(error.message, "error"));
});
els.modelFilter.addEventListener("change", () => {
  state.filters.modelTier = els.modelFilter.value;
  state.filters.status = "";
  els.statusFilter.value = "";
  loadAgents().catch((error) => setMessage(error.message, "error"));
});
els.form.elements.providerModel.addEventListener("change", updateProviderModelNote);

async function savePromptEdits(id) {
  const agent = state.allAgents.find((item) => item.id === id) || state.agents.find((item) => item.id === id);
  const container = els.promptList.querySelector(`[data-agent-id="${CSS.escape(id)}"]`);
  if (!agent || !container) return;
  try {
    await api(`/admin/api/agents/${id}`, {
      method: "PUT",
      body: {
        systemPrompt: container.querySelector("[name='systemPrompt']").value,
        userPromptTemplate: container.querySelector("[name='userPromptTemplate']").value,
      },
    });
    await loadAgents();
    setMessage("Prompts saved.", "ok");
  } catch (error) {
    setMessage(error.message, "error");
  }
}

function openAgentDetail(id) {
  const agent = state.allAgents.find((item) => item.id === id) || state.agents.find((item) => item.id === id);
  if (!agent) return;
  const metadata = agent.metadata || {};
  const model = metadata.modelInfo || state.models.find((item) => item.id === (agent.providerModel || ""));
  els.agentModalTitle.textContent = agent.name || "Agent details";
  els.agentModalSubtitle.textContent = `${agent.serviceScope || "unscoped"} ${agent.routePath ? "· " + agent.routePath : ""}`;
  els.agentModalBody.innerHTML = `
    <div class="detail-columns">
      <section>
        <h3>Assignment</h3>
        <dl>
          <dt>Status</dt><dd>${escapeHtml(agent.status)}</dd>
          <dt>Applications</dt><dd>${renderChips(metadata.applications || [])}</dd>
          <dt>Used by</dt><dd>${renderChips(metadata.usedBy || [])}</dd>
          <dt>Source</dt><dd>${escapeHtml(metadata.source || "unknown")}</dd>
        </dl>
      </section>
      <section>
        <h3>Model</h3>
        <dl>
          <dt>Tier</dt><dd>${escapeHtml(agent.modelTier)}</dd>
          <dt>Provider model</dt><dd>${escapeHtml(model?.label || agent.providerModel || "Default route")}</dd>
          <dt>Price</dt><dd>${escapeHtml(model?.price || "Unknown")}</dd>
          <dt>Best for</dt><dd>${escapeHtml(model?.bestFor || "")}</dd>
          <dt>Temperature</dt><dd>${escapeHtml(agent.temperature)}</dd>
          <dt>Max tokens</dt><dd>${escapeHtml(agent.maxTokens)}</dd>
        </dl>
      </section>
    </div>
    <section>
      <h3>Description</h3>
      <p>${escapeHtml(agent.description || "")}</p>
    </section>
    <section>
      <h3>Prompts</h3>
      <pre>${escapeHtml(agent.systemPrompt || "")}</pre>
      <pre>${escapeHtml(agent.userPromptTemplate || "")}</pre>
    </section>
    <section>
      <h3>Metadata</h3>
      <pre>${escapeHtml(JSON.stringify(metadata, null, 2))}</pre>
    </section>
  `;
  els.agentModal.hidden = false;
  window.location.hash = `agent=${encodeURIComponent(id)}`;
}

function openLogDetail(index) {
  const log = state.logs[index];
  if (!log) return;
  els.agentModalTitle.textContent = "Log details";
  els.agentModalSubtitle.textContent = `${log.service || "unknown"} · ${log.level || "info"} · ${formatDate(log.timestamp)}`;
  els.agentModalBody.innerHTML = `<pre>${escapeHtml(JSON.stringify(log, null, 2))}</pre>`;
  els.agentModal.hidden = false;
  window.location.hash = `log=${index}`;
}

function closeModal() {
  els.agentModal.hidden = true;
  if (window.location.hash.startsWith("#agent=") || window.location.hash.startsWith("#log=")) {
    history.replaceState(null, "", window.location.pathname);
  }
}

els.agentModalClose.addEventListener("click", closeModal);
els.agentModal.addEventListener("click", (event) => {
  if (event.target === els.agentModal) closeModal();
});

els.logRefreshButton.addEventListener("click", async () => {
  try {
    await loadLogs();
    await renderLogView();
    setMessage("Logs refreshed.", "ok");
  } catch (error) {
    setMessage(error.message, "error");
  }
});
for (const input of [els.logRangeFilter, els.logLevelFilter, els.logServiceFilter, els.logSearchInput]) {
  input.addEventListener("change", () => {
    state.logs = [];
    if (state.view === "logs") renderLogView().catch((error) => setMessage(error.message, "error"));
  });
}

els.form.elements.name.addEventListener("input", () => {
  if (!els.form.elements.id.value && !els.form.elements.slug.value) {
    els.form.elements.slug.value = slugFromName(els.form.elements.name.value);
  }
});

els.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const id = els.form.elements.id.value;
    const saved = await api(id ? `/admin/api/agents/${id}` : "/admin/api/agents", {
      method: id ? "PUT" : "POST",
      body: formPayload(),
    });
    state.selectedId = saved.id;
    await loadAgents();
    setMessage("Agent saved.", "ok");
  } catch (error) {
    setMessage(error.message, "error");
  }
});

els.duplicateButton.addEventListener("click", () => {
  const current = state.agents.find((agent) => agent.id === state.selectedId);
  if (!current) return;
  fillForm({
    ...current,
    id: "",
    name: `${current.name} copy`,
    slug: `${current.slug}-copy`,
    status: "draft",
  });
  els.form.elements.id.value = "";
  state.selectedId = null;
  renderAgents();
  setMessage("Duplicated into an unsaved draft.", "");
});

els.deleteButton.addEventListener("click", async () => {
  const id = els.form.elements.id.value;
  if (!id) {
    resetForm();
    return;
  }
  const agent = state.agents.find((item) => item.id === id);
  if (!window.confirm(`Delete ${agent?.name || "this agent"}?`)) return;
  try {
    await api(`/admin/api/agents/${id}`, { method: "DELETE" });
    state.selectedId = null;
    await loadAgents();
    setMessage("Agent deleted.", "ok");
  } catch (error) {
    setMessage(error.message, "error");
  }
});

function debounce(fn, delay) {
  let timeout;
  return (...args) => {
    window.clearTimeout(timeout);
    timeout = window.setTimeout(() => fn(...args), delay);
  };
}

resetForm();
(async () => {
  await loadModels();
  await loadAgents();
})()
  .then(() => setMessage("Connected. Agent registry loaded.", "ok"))
  .catch((error) => setMessage(error.message, "error"));
