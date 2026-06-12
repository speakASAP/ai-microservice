const state = {
  agents: [],
  selectedId: null,
  view: "agents",
};

const els = {
  navItems: Array.from(document.querySelectorAll(".nav-item")),
  viewPanels: Array.from(document.querySelectorAll("[data-view-panel]")),
  viewTitle: document.getElementById("view-title"),
  viewSubtitle: document.getElementById("view-subtitle"),
  authPanel: document.getElementById("auth-panel"),
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
  metricTotal: document.getElementById("metric-total"),
  metricActive: document.getElementById("metric-active"),
  metricTiers: document.getElementById("metric-tiers"),
  promptList: document.getElementById("prompt-list"),
  modelList: document.getElementById("model-list"),
  logList: document.getElementById("log-list"),
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
  const active = state.agents.filter((agent) => agent.status === "active").length;
  const tiers = new Set(state.agents.map((agent) => agent.modelTier).filter(Boolean)).size;
  els.metricTotal.textContent = String(state.agents.length);
  els.metricActive.textContent = String(active);
  els.metricTiers.textContent = String(tiers);
  els.listSummary.textContent = state.agents.length === 1 ? "1 agent loaded" : `${state.agents.length} agents loaded`;
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
      <td><span class="pill tier">${escapeHtml(agent.modelTier || "free")}</span></td>
      <td><span class="pill ${escapeHtml(agent.status || "draft")}">${escapeHtml(agent.status || "draft")}</span></td>
      <td class="muted">${formatDate(agent.updatedAt)}</td>
    `;
    row.addEventListener("click", () => selectAgent(agent.id));
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
      <div class="prompt-grid">
        <label>System prompt<textarea readonly rows="4">${escapeHtml(agent.systemPrompt || "")}</textarea></label>
        <label>User template<textarea readonly rows="4">${escapeHtml(agent.userPromptTemplate || "")}</textarea></label>
      </div>
    `;
    els.promptList.appendChild(item);
  }
  if (!state.agents.length) renderEmptyDetail(els.promptList, "No prompts are registered yet.");
}

function renderModelView() {
  els.modelList.innerHTML = "";
  const tiers = ["free", "cheap", "smart", "premium"];
  for (const tier of tiers) {
    const agents = state.agents.filter((agent) => (agent.modelTier || "free") === tier);
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
          <div class="route-row">
            <strong>${escapeHtml(agent.name)}</strong>
            <span>${escapeHtml(agent.providerModel || "default route")} · temp ${escapeHtml(agent.temperature || "0.20")} · ${escapeHtml(agent.maxTokens || 1000)} tokens</span>
          </div>
        `).join("") || '<div class="empty-detail">No agents use this tier.</div>'}
      </div>
    `;
    els.modelList.appendChild(item);
  }
}

function renderLogView() {
  els.logList.innerHTML = "";
  const recent = [...state.agents].sort((a, b) => new Date(b.updatedAt || 0) - new Date(a.updatedAt || 0)).slice(0, 25);
  for (const agent of recent) {
    const item = document.createElement("article");
    item.className = "activity-row";
    item.innerHTML = `
      <span class="activity-time">${formatDate(agent.updatedAt)}</span>
      <strong>${escapeHtml(agent.name)}</strong>
      <span>${escapeHtml(agent.serviceScope || "unscoped")} · ${escapeHtml(agent.slug || "")}</span>
    `;
    els.logList.appendChild(item);
  }
  if (!recent.length) renderEmptyDetail(els.logList, "No registry activity is available yet.");
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
  if (els.statusFilter.value) params.set("status", els.statusFilter.value);
  if (els.modelFilter.value) params.set("modelTier", els.modelFilter.value);

  const payload = await api(`/admin/api/agents?${params.toString()}`);
  state.agents = payload.items || [];
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
els.statusFilter.addEventListener("change", () => loadAgents().catch((error) => setMessage(error.message, "error")));
els.modelFilter.addEventListener("change", () => loadAgents().catch((error) => setMessage(error.message, "error")));

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
loadAgents()
  .then(() => setMessage("Connected. Agent registry loaded.", "ok"))
  .catch((error) => setMessage(error.message, "error"));
