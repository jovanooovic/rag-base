// rag-base admin console -- token-gated upload against POST /upload.
// Not linked from index.html; reachable only if you know the URL, and
// useless without the token the server was started with (APP_ADMIN_TOKEN).

const themeToggle = document.getElementById("theme-toggle");
const tokenInput = document.getElementById("token-input");
const tokenSave = document.getElementById("token-save");
const tokenState = document.getElementById("token-state");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const fileList = document.getElementById("file-list");
const ingestBtn = document.getElementById("ingest-btn");
const ingestResult = document.getElementById("ingest-result");
const docTable = document.getElementById("doc-table");

let selected = []; // File[]

// ---------- theme (shared behaviour with the chat console) ----------

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  document.getElementById("icon-sun").style.display = theme === "light" ? "block" : "none";
  document.getElementById("icon-moon").style.display = theme === "light" ? "none" : "block";
  localStorage.setItem("rag-theme", theme);
}
applyTheme(
  localStorage.getItem("rag-theme") ||
  (window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark")
);
themeToggle.addEventListener("click", () => {
  applyTheme(document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light");
});

// ---------- token ----------

function getToken() {
  return localStorage.getItem("rag-admin-token") || "";
}

function renderTokenState() {
  const t = getToken();
  if (t) {
    tokenState.textContent = `saved (${t.length} chars) -- cleared only if you clear it here`;
    tokenState.classList.add("is-set");
  } else {
    tokenState.textContent = "no token saved -- /upload and /ingest will 401 without one";
    tokenState.classList.remove("is-set");
  }
}
tokenInput.value = getToken();
renderTokenState();

tokenSave.addEventListener("click", () => {
  localStorage.setItem("rag-admin-token", tokenInput.value.trim());
  renderTokenState();
});

// ---------- file selection ----------

const ACCEPTED = [".txt", ".md", ".markdown", ".html", ".htm", ".csv", ".json", ".pdf"];
const MAX_BYTES = 15 * 1024 * 1024;
const MAX_FILES = 10;

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("is-dragover"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("is-dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("is-dragover");
  addFiles(e.dataTransfer.files);
});
fileInput.addEventListener("change", () => addFiles(fileInput.files));

function addFiles(fileListLike) {
  for (const f of fileListLike) {
    const ext = "." + (f.name.split(".").pop() || "").toLowerCase();
    if (!ACCEPTED.includes(ext)) {
      flashResult(`Skipped ${f.name}: unsupported type (${ext})`, true);
      continue;
    }
    if (f.size > MAX_BYTES) {
      flashResult(`Skipped ${f.name}: exceeds 15MB`, true);
      continue;
    }
    if (selected.length >= MAX_FILES) {
      flashResult(`Skipped ${f.name}: ${MAX_FILES}-file limit per batch`, true);
      continue;
    }
    selected.push(f);
  }
  renderFileList();
  fileInput.value = "";
}

function renderFileList() {
  fileList.innerHTML = "";
  selected.forEach((f, i) => {
    const li = document.createElement("li");
    li.className = "file-row";
    li.innerHTML = `<span class="file-row-name">${escapeHtml(f.name)}</span>
      <span class="file-row-size">${(f.size / 1024).toFixed(0)}KB</span>
      <button class="file-row-remove" aria-label="Remove" data-i="${i}">&times;</button>`;
    fileList.appendChild(li);
  });
  fileList.querySelectorAll(".file-row-remove").forEach((btn) => {
    btn.addEventListener("click", () => {
      selected.splice(Number(btn.dataset.i), 1);
      renderFileList();
    });
  });
  ingestBtn.disabled = selected.length === 0;
}

// ---------- ingest ----------

ingestBtn.addEventListener("click", async () => {
  if (selected.length === 0) return;
  ingestBtn.disabled = true;
  ingestBtn.textContent = "Ingesting…";
  ingestResult.classList.remove("is-error");
  ingestResult.textContent = "";

  const form = new FormData();
  selected.forEach((f) => form.append("files", f));

  try {
    const res = await fetch("/upload", {
      method: "POST",
      headers: { "X-Admin-Token": getToken() },
      body: form,
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      flashResult(`HTTP ${res.status}: ${body.detail || "upload failed"}`, true);
    } else {
      flashResult(
        `documents: ${body.documents}\n` +
        `chunks seen: ${body.chunks_seen}\n` +
        `chunks embedded: ${body.chunks_embedded}\n` +
        `chunks skipped (unchanged): ${body.chunks_skipped_unchanged}\n` +
        `total chunks in index: ${body.total_chunks_in_index}\n` +
        `saved: ${(body.saved_files || []).join(", ")}`,
        false
      );
      selected = [];
      renderFileList();
      loadDocuments();
    }
  } catch (err) {
    flashResult(`Request failed: ${err.message}`, true);
  } finally {
    ingestBtn.disabled = selected.length === 0;
    ingestBtn.textContent = "Ingest";
  }
});

function flashResult(text, isError) {
  ingestResult.textContent = text;
  ingestResult.classList.toggle("is-error", isError);
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

// ---------- current index ----------

async function loadDocuments() {
  try {
    const res = await fetch("/documents");
    const body = await res.json();
    if (!body.by_source || body.by_source.length === 0) {
      docTable.innerHTML = `<p class="admin-sub">Nothing indexed yet.</p>`;
      return;
    }
    const rows = body.by_source
      .map((d) => `<tr><td>${escapeHtml(d.source)}</td><td>${d.chunks} chunks</td></tr>`)
      .join("");
    docTable.innerHTML = `<table><tbody>${rows}</tbody></table>
      <p class="admin-sub" style="margin-top:10px">${body.documents} documents &middot; ${body.chunks} chunks total</p>`;
  } catch {
    docTable.innerHTML = `<p class="admin-sub">Could not reach the API.</p>`;
  }
}
loadDocuments();
