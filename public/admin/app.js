const state = {
  agents: [],
  selectedId: null,
};

const els = {
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
  if (state.selectedId) fillForm(state.agents.find((agent) => agent.id === state.selectedId));
  else resetForm();
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
  resetForm();
  setMessage("Ready to create a new agent.", "");
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
