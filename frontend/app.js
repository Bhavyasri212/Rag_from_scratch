const API = "http://localhost:8000";

// ── DOM refs ──────────────────────────────────────────────────────────────────
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const uploadStatus = document.getElementById("uploadStatus");
const uploadProg = document.getElementById("uploadProgress");
const progressBar = document.getElementById("progressBar");
const pipeline = document.getElementById("pipeline");
const docList = document.getElementById("docList");
const docCount = document.getElementById("docCount");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const chatMessages = document.getElementById("chatMessages");
const statusText = document.getElementById("statusText");
const newChatBtn = document.getElementById("newChatBtn");
const workspaceSources = document.getElementById("workspaceSources");
const workspaceChunks = document.getElementById("workspaceChunks");

let selectedFile = null;
let progressTimer = null;

document.querySelectorAll(".starter").forEach((button) => {
  button.addEventListener("click", () => {
    chatInput.value = button.dataset.question;
    chatInput.focus();
  });
});

newChatBtn.addEventListener("click", () => {
  chatMessages.innerHTML = `
    <div class="ambient-orbit" aria-hidden="true">
      <div class="orbit orbit-one"></div>
      <div class="orbit orbit-two"></div>
      <div class="core"><span>✦</span></div>
    </div>
    <div class="welcome-msg">
      <p class="eyebrow">Retrieval workspace</p>
      <h3>Ask QueryNest.</h3>
      <p>Upload a source to build a private, searchable knowledge base.</p>
      <div class="starter-grid">
        <button class="starter" type="button" data-question="Summarize the main ideas in this document">Summarize the main ideas <span>↗</span></button>
        <button class="starter" type="button" data-question="What are the key concepts I should understand?">Find the key concepts <span>↗</span></button>
        <button class="starter" type="button" data-question="Explain the most important process step by step">Explain a process <span>↗</span></button>
      </div>
    </div>`;
  bindStarterButtons();
  chatInput.value = "";
  chatInput.focus();
});

function bindStarterButtons() {
  document.querySelectorAll(".starter").forEach((button) => {
    button.addEventListener("click", () => {
      chatInput.value = button.dataset.question;
      chatInput.focus();
    });
  });
}

// ── File selection ────────────────────────────────────────────────────────────
dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("drag-over");
});
dropzone.addEventListener("dragleave", () =>
  dropzone.classList.remove("drag-over"),
);
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag-over");
  const f = e.dataTransfer.files[0];
  if (f) selectFile(f);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) selectFile(fileInput.files[0]);
});

function selectFile(file) {
  selectedFile = file;
  dropzone.querySelector(".dropzone-label").textContent = file.name;
  dropzone.querySelector(".dropzone-hint").textContent =
    `${(file.size / 1024).toFixed(1)} KB`;
  uploadBtn.disabled = false;
  uploadStatus.textContent = "";
}

// ── Upload ────────────────────────────────────────────────────────────────────
uploadBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  uploadBtn.disabled = true;
  uploadStatus.textContent = "Uploading source…";
  uploadProg.hidden = false;
  pipeline.hidden = false;
  progressBar.style.width = "8%";

  const form = new FormData();
  form.append("file", selectedFile);

  try {
    const res = await fetch(`${API}/ingest/`, { method: "POST", body: form });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Upload failed");
    }

    const job = await res.json();
    await watchIngestion(job.job_id);
  } catch (err) {
    uploadStatus.textContent = `✗ ${err.message}`;
  } finally {
    if (!progressTimer) {
      setTimeout(() => {
        uploadProg.hidden = true;
        pipeline.hidden = true;
        progressBar.style.width = "0%";
      }, 600);
    }
    uploadBtn.disabled = false;
  }
});

async function watchIngestion(jobId) {
  clearInterval(progressTimer);
  return new Promise((resolve) => {
    progressTimer = setInterval(async () => {
      try {
        const res = await fetch(`${API}/ingest/status/${jobId}`);
        if (!res.ok) throw new Error("Could not read indexing status");
        const job = await res.json();
        progressBar.style.width = `${job.progress}%`;
        uploadStatus.textContent = job.message;
        document.querySelectorAll(".pipeline-step").forEach((step) => {
          const stages = ["reading", "chunking", "embedding", "indexing"];
          const current = stages.indexOf(job.stage);
          const position = stages.indexOf(step.dataset.stage);
          step.classList.toggle(
            "complete",
            job.status === "complete" || position < current,
          );
          step.classList.toggle(
            "active",
            position === current && job.status !== "complete",
          );
        });

        if (job.status === "complete") {
          clearInterval(progressTimer);
          progressTimer = null;
          uploadStatus.textContent = `Ready · ${job.chunks_added} chunks indexed`;
          addDocItem(job.filename, job.chunks_added);
          refreshStatus();
          resolve();
        } else if (job.status === "error") {
          throw new Error(job.error || "Indexing failed");
        }
      } catch (error) {
        clearInterval(progressTimer);
        progressTimer = null;
        uploadStatus.textContent = `✗ ${error.message}`;
        resolve();
      }
    }, 500);
  });
}

function addDocItem(filename, chunks) {
  const empty = docList.querySelector(".doc-empty");
  if (empty) empty.remove();

  const ext = filename.split(".").pop().toUpperCase();
  const icons = { TXT: "📄", MD: "📝", PDF: "📕" };

  const el = document.createElement("div");
  el.className = "doc-item";
  el.innerHTML = `
    <span class="doc-item-icon">${icons[ext] || "📄"}</span>
    <span class="doc-item-name">${filename}</span>
    <span class="doc-item-chunks">${chunks} chunks</span>
  `;
  docList.appendChild(el);
  const indexedFiles = docList.querySelectorAll(".doc-item").length;
  docCount.textContent = indexedFiles;
  workspaceSources.textContent = indexedFiles;
  workspaceChunks.textContent = [
    ...docList.querySelectorAll(".doc-item-chunks"),
  ].reduce((total, item) => total + Number.parseInt(item.textContent, 10), 0);
}

async function refreshStatus() {
  try {
    const res = await fetch(`${API}/documents`);
    const data = await res.json();
    statusText.textContent = `${data.count} document${data.count !== 1 ? "s" : ""}`;
    docCount.textContent = data.count;
    workspaceSources.textContent = data.count;
  } catch (_) {}
}

// ── Chat ──────────────────────────────────────────────────────────────────────
chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = chatInput.value.trim();
  if (!q) return;

  chatInput.value = "";

  // Remove welcome message on first question
  const welcome = chatMessages.querySelector(".welcome-msg");
  if (welcome) welcome.remove();

  appendMessage("user", q);
  const thinkingEl = appendThinking();

  try {
    const res = await fetch(`${API}/query/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, k: 5 }),
    });

    thinkingEl.remove();

    if (!res.ok) {
      const err = await res.json();
      appendError(err.detail || "Something went wrong.");
      return;
    }

    const data = await res.json();
    appendMessage("assistant", data.answer, data.sources);
  } catch (err) {
    thinkingEl.remove();
    appendError("Could not reach the API. Is the server running?");
  }
});

function appendMessage(role, text, sources = []) {
  const el = document.createElement("div");
  el.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  el.appendChild(bubble);

  chatMessages.appendChild(el);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return el;
}

function appendThinking() {
  const el = document.createElement("div");
  el.className = "message assistant";
  el.innerHTML = `<div class="thinking">Thinking <div class="dots"><span></span><span></span><span></span></div></div>`;
  chatMessages.appendChild(el);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return el;
}

function appendError(msg) {
  const el = document.createElement("div");
  el.className = "message assistant";
  el.innerHTML = `<div class="error-msg">⚠ ${msg}</div>`;
  chatMessages.appendChild(el);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Load doc count on start
refreshStatus();
