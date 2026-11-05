const apiPrefix = "/api";
const tokenInput = document.querySelector("#access-token");
const statusText = document.querySelector("#status-text");
const notice = document.querySelector("#notice");
const documentsNode = document.querySelector("#documents");
const messagesNode = document.querySelector("#messages");
const welcomeNode = document.querySelector("#welcome");
const questionInput = document.querySelector("#question");
const sendButton = document.querySelector("#send-question");
const deleteButton = document.querySelector("#delete-documents");
const selectedDocuments = new Set();
let modelConfig;
let documents = [];
let history = JSON.parse(localStorage.getItem("papertrail-history") || "[]");
let sessionId = localStorage.getItem("papertrail-session") || crypto.randomUUID();
localStorage.setItem("papertrail-session", sessionId);
tokenInput.value = sessionStorage.getItem("papertrail-token") || "";

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  const token = sessionStorage.getItem("papertrail-token");
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const response = await fetch(`${apiPrefix}${path}`, { ...options, headers });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
  return payload;
}

function showNotice(message = "") {
  notice.textContent = message;
}

function updateChatControls() {
  const hasSelection = selectedDocuments.size > 0;
  sendButton.disabled = !hasSelection;
  deleteButton.disabled = !hasSelection;
  document.querySelector("#selection-hint").textContent = hasSelection
    ? `${selectedDocuments.size} document${selectedDocuments.size === 1 ? "" : "s"} selected`
    : "Select at least one document";
}

function renderDocuments() {
  document.querySelector("#doc-count").textContent = String(documents.length);
  documentsNode.replaceChildren();
  if (!documents.length) {
    const empty = document.createElement("p");
    empty.className = "empty-library";
    empty.textContent = "Your processed PDFs will appear here.";
    documentsNode.append(empty);
  }
  for (const doc of documents) {
    const row = document.createElement("label");
    row.className = "doc-row";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = selectedDocuments.has(doc.doc_id);
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) selectedDocuments.add(doc.doc_id);
      else selectedDocuments.delete(doc.doc_id);
      updateChatControls();
    });
    const text = document.createElement("span");
    text.className = "doc-name";
    text.append(document.createTextNode(doc.filename));
    const meta = document.createElement("small");
    meta.className = "doc-meta";
    meta.textContent = `${doc.pages} pages / ${doc.chunks} chunks`;
    text.append(meta);
    row.append(checkbox, text);
    documentsNode.append(row);
  }
  updateChatControls();
}

function renderHistory() {
  messagesNode.replaceChildren();
  welcomeNode.hidden = history.length > 0;
  for (const item of history) {
    const article = document.createElement("article");
    article.className = `message ${item.role}`;
    const role = document.createElement("div");
    role.className = "message-role";
    role.textContent = item.role === "user" ? "You" : "Papertrail / Groq";
    const body = document.createElement("div");
    body.className = "message-body";
    body.textContent = item.content;
    article.append(role, body);
    if (item.sources?.length) {
      const details = document.createElement("details");
      details.className = "sources";
      const summary = document.createElement("summary");
      summary.textContent = `Sources (${item.sources.length})`;
      details.append(summary);
      for (const source of item.sources) {
        const sourceNode = document.createElement("div");
        sourceNode.className = "source-item";
        const title = document.createElement("strong");
        title.textContent = `${source.filename} - page ${source.page}`;
        sourceNode.append(title, document.createElement("br"), document.createTextNode(source.snippet));
        details.append(sourceNode);
      }
      article.append(details);
    }
    messagesNode.append(article);
  }
  document.querySelector("#conversation").scrollTop = document.querySelector("#conversation").scrollHeight;
}

async function loadDocuments() {
  documents = await api("/documents");
  for (const id of [...selectedDocuments]) {
    if (!documents.some((doc) => doc.doc_id === id)) selectedDocuments.delete(id);
  }
  if (!selectedDocuments.size && documents.length) selectedDocuments.add(documents[documents.length - 1].doc_id);
  renderDocuments();
}

async function initialize() {
  try {
    modelConfig = await api("/models");
    const embeddingSelect = document.querySelector("#embedding-model");
    for (const model of modelConfig.embedding) {
      const option = document.createElement("option");
      option.value = model;
      option.textContent = model;
      option.selected = model === modelConfig.default_embedding;
      embeddingSelect.append(option);
    }
    await loadDocuments();
    statusText.textContent = "Ready";
    renderHistory();
  } catch (error) {
    statusText.textContent = "Needs setup";
    showNotice(error.message.includes("token") ? "Enter the deployment access token in the sidebar." : error.message);
  }
}

document.querySelector("#save-token").addEventListener("click", async () => {
  sessionStorage.setItem("papertrail-token", tokenInput.value.trim());
  showNotice("Access token saved for this tab.");
  await initialize();
});

document.querySelector("#new-chat").addEventListener("click", () => {
  history = [];
  sessionId = crypto.randomUUID();
  localStorage.setItem("papertrail-history", "[]");
  localStorage.setItem("papertrail-session", sessionId);
  renderHistory();
  showNotice("New conversation started.");
});

document.querySelector("#upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = document.querySelector("#pdf-file").files[0];
  if (!file) return;
  if (file.size > 4 * 1024 * 1024) {
    showNotice("This Vercel deployment accepts PDFs up to 4 MB.");
    return;
  }
  const button = event.currentTarget.querySelector("button[type=submit]");
  button.disabled = true;
  button.textContent = "Processing...";
  showNotice("");
  try {
    const form = new FormData();
    form.append("file", file);
    form.append("embedding_model", document.querySelector("#embedding-model").value);
    const result = await api("/documents", { method: "POST", body: form });
    await loadDocuments();
    selectedDocuments.add(result.doc_id);
    renderDocuments();
    showNotice(`${result.filename} processed into ${result.chunks} chunks.`);
    event.currentTarget.reset();
  } catch (error) {
    showNotice(error.message);
  } finally {
    button.disabled = false;
    button.textContent = "Process document";
  }
});

deleteButton.addEventListener("click", async () => {
  if (!selectedDocuments.size || !confirm("Remove the selected documents and their vectors?")) return;
  deleteButton.disabled = true;
  try {
    for (const id of selectedDocuments) await api(`/documents/${encodeURIComponent(id)}`, { method: "DELETE" });
    selectedDocuments.clear();
    await loadDocuments();
    showNotice("Selected documents removed.");
  } catch (error) {
    showNotice(error.message);
  }
});

document.querySelector("#chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question || !selectedDocuments.size) return;
  const priorHistory = history.filter((item) => ["user", "assistant"].includes(item.role)).slice(-8);
  history.push({ role: "user", content: question });
  questionInput.value = "";
  sendButton.disabled = true;
  renderHistory();
  showNotice("Searching the selected PDFs...");
  try {
    const result = await api("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        question,
        doc_ids: [...selectedDocuments],
        generation_model: modelConfig.default_generation,
        history: priorHistory,
      }),
    });
    history.push({ role: "assistant", content: result.answer, sources: result.sources });
  } catch (error) {
    history.push({ role: "assistant", content: `Request failed: ${error.message}` });
  } finally {
    localStorage.setItem("papertrail-history", JSON.stringify(history));
    renderHistory();
    updateChatControls();
    showNotice("");
  }
});

questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto";
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 160)}px`;
});

initialize();
