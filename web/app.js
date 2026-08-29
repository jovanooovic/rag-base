// rag-base chat console — vanilla JS, no build step, same-origin fetch
// against /health, /ask, /documents.

const chat = document.getElementById("chat");
const emptyState = document.getElementById("empty-state");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const themeToggle = document.getElementById("theme-toggle");
const docModal = document.getElementById("doc-modal");
const docModalBackdrop = document.getElementById("doc-modal-backdrop");
const docModalClose = document.getElementById("doc-modal-close");
const docModalTitle = document.getElementById("doc-modal-title");
const docModalBody = document.getElementById("doc-modal-body");

const history = []; // [{role, content}]
let indexed = 0;
let busy = false;

// ---------- theme ----------

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
  const current = document.documentElement.getAttribute("data-theme");
  applyTheme(current === "light" ? "dark" : "light");
});

// ---------- health ----------

function applyBrand(body) {
  if (body.project) {
    document.title = body.project;
    document.getElementById("brand-name").textContent = body.project;
  }
  if (body.brand_accent) {
    document.documentElement.style.setProperty("--accent", body.brand_accent);
  }
  if (body.brand_description) {
    document.getElementById("empty-sub").textContent = body.brand_description;
  }
  if (body.show_source_link === false) {
    document.getElementById("source-link").remove();
  }
}

function topicLabel(source) {
  const base = source.split("/").pop().replace(/\.[^.]+$/, "");
  return base.split(/[-_]+/).map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
}

async function loadTopics() {
  const container = document.getElementById("topics");
  const label = document.getElementById("topics-label");
  try {
    const res = await fetch("/documents");
    const body = await res.json();
    const sources = (body.by_source || []).map((d) => d.source);
    if (sources.length === 0) return;
    const labels = [...new Set(sources.map(topicLabel))].sort((a, b) => a.localeCompare(b));
    container.innerHTML = labels.map((l) => `<span class="topic-pill">${escapeHtml(l)}</span>`).join("");
    label.hidden = false;
  } catch {
    // Empty-state still works without this -- suggested questions are enough
    // to get started even if /documents is unreachable for some reason.
  }
}
loadTopics();

async function checkHealth() {
  try {
    const res = await fetch("/health");
    const body = await res.json();
    applyBrand(body);
    indexed = body.chunks_indexed || 0;
    if (indexed > 0) {
      statusDot.className = "dot dot--ok";
      statusText.textContent = `${indexed} chunks indexed`;
    } else {
      statusDot.className = "dot dot--warn";
      statusText.textContent = "index empty";
    }
  } catch {
    statusDot.className = "dot dot--off";
    statusText.textContent = "offline";
  }
}
checkHealth();

// ---------- composer ----------

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    submit();
  }
});

sendBtn.addEventListener("click", submit);

document.getElementById("suggestions").addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  // data-question, not textContent -- a chip can carry a decorative child
  // (see .chip-tag on the refusal demo) that textContent would fold into
  // the submitted question otherwise.
  input.value = chip.dataset.question || chip.textContent.trim();
  submit();
});

function submit() {
  const question = input.value.trim();
  if (!question || busy) return;
  input.value = "";
  input.style.height = "auto";
  ask(question);
}

// ---------- message rendering ----------

function scrollToEnd() {
  requestAnimationFrame(() => chat.scrollTo({ top: chat.scrollHeight, behavior: "smooth" }));
}

function addUserMessage(text) {
  emptyState.style.display = "none";
  const node = document.getElementById("tpl-user").content.cloneNode(true);
  node.querySelector(".bubble").textContent = text;
  chat.appendChild(node);
  scrollToEnd();
}

function addTyping() {
  const node = document.getElementById("tpl-typing").content.cloneNode(true);
  const el = node.querySelector(".msg");
  chat.appendChild(node);
  scrollToEnd();
  return chat.lastElementChild;
}

function revealWords(el, text) {
  const words = text.split(/(\s+)/);
  const perWord = Math.max(6, Math.min(28, 700 / Math.max(words.length, 1)));
  el.innerHTML = "";
  words.forEach((w, i) => {
    const span = document.createElement("span");
    span.className = "word";
    span.textContent = w;
    span.style.animationDelay = `${Math.round(i * perWord)}ms`;
    el.appendChild(span);
  });
  const total = words.length * perWord + 400;
  return total;
}

function formatMeta(trace, sourceCount) {
  const ms = trace && trace.latency_ms != null ? Math.round(trace.latency_ms) : null;
  const cost = trace && typeof trace.cost_usd === "number" ? trace.cost_usd : 0;
  const parts = [];
  if (ms != null) parts.push(`${ms}ms`);
  parts.push(`$${cost.toFixed(4)}`);
  parts.push(`${sourceCount} source${sourceCount === 1 ? "" : "s"}`);
  return parts.join(" · ");
}

// ---------- document preview ----------
// Click-to-view for a citation's original file. PDF renders in the browser's
// own viewer; docx/xlsx render client-side via CDN libraries loaded on first
// use, so a visitor who never opens a document never pays for them; every
// other supported format is plain text, shown as-is.

function closeDocModal() {
  docModal.hidden = true;
  docModalBody.innerHTML = "";
}
docModalBackdrop.addEventListener("click", closeDocModal);
docModalClose.addEventListener("click", closeDocModal);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !docModal.hidden) closeDocModal();
});

const loadedScripts = new Set();
function loadScriptOnce(src) {
  if (loadedScripts.has(src)) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const el = document.createElement("script");
    el.src = src;
    el.onload = () => { loadedScripts.add(src); resolve(); };
    el.onerror = () => reject(new Error(`could not load ${src}`));
    document.head.appendChild(el);
  });
}

function assertOk(res) {
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res;
}

// Jump to the cited passage inside the opened document, instead of leaving
// the reader to scroll a multi-page doc looking for one paragraph. Matching
// strategy differs by rendering: flat text search for pre/docx, row-content
// match for xlsx (its cells never literally contain "Field: value" -- that
// shape only exists in the excerpt load_xlsx synthesized for the chunk).

function highlightPreText(pre, fullText, excerpt) {
  const probe = (excerpt || "").trim().slice(0, 60);
  const idx = probe.length >= 8 ? fullText.indexOf(probe) : -1;
  if (idx === -1) {
    pre.textContent = fullText;
    return false;
  }
  const before = fullText.slice(0, idx);
  const match = fullText.slice(idx, idx + probe.length);
  const after = fullText.slice(idx + probe.length);
  pre.innerHTML = `${escapeHtml(before)}<mark class="doc-highlight">${escapeHtml(match)}</mark>${escapeHtml(after)}`;
  return true;
}

function highlightRenderedText(container, excerpt) {
  const probe = (excerpt || "").trim().replace(/\s+/g, " ").slice(0, 60);
  if (probe.length < 8) return false;

  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  const textNodes = [];
  let combined = "";
  let n;
  while ((n = walker.nextNode())) {
    textNodes.push(n);
    combined += n.nodeValue;
  }
  const idx = combined.replace(/\s+/g, " ").indexOf(probe);
  if (idx === -1) return false;

  // idx is an offset into the whitespace-collapsed string; walk the real
  // (uncollapsed) text to find the matching node/offset pair for start/end.
  let collapsedPos = 0, rawPos = 0, startNode = null, startOffset = 0;
  let endNode = null, endOffset = 0;
  outer: for (const tn of textNodes) {
    const raw = tn.nodeValue;
    for (let i = 0; i < raw.length; i++) {
      const isSpace = /\s/.test(raw[i]);
      if (!isSpace || (i > 0 && !/\s/.test(raw[i - 1]))) {
        if (startNode === null && collapsedPos === idx) { startNode = tn; startOffset = i; }
        if (collapsedPos === idx + probe.length) { endNode = tn; endOffset = i; break outer; }
        collapsedPos++;
      }
      rawPos++;
    }
  }
  if (!startNode) return false;
  if (!endNode) { endNode = textNodes[textNodes.length - 1]; endOffset = endNode.nodeValue.length; }

  const range = document.createRange();
  range.setStart(startNode, startOffset);
  range.setEnd(endNode, endOffset);
  const mark = document.createElement("mark");
  mark.className = "doc-highlight";
  try {
    range.surroundContents(mark);
  } catch {
    const frag = range.extractContents();
    mark.appendChild(frag);
    range.insertNode(mark);
  }
  mark.scrollIntoView({ behavior: "smooth", block: "center" });
  return true;
}

function highlightTableRow(container, excerpt) {
  const values = (excerpt || "").split("\n")
    .map((line) => { const i = line.indexOf(": "); return i === -1 ? line.trim() : line.slice(i + 2).trim(); })
    .filter((v) => v.length > 0);
  if (values.length === 0) return false;
  for (const row of container.querySelectorAll("tr")) {
    const rowText = row.textContent;
    if (values.every((v) => rowText.includes(v))) {
      row.classList.add("doc-highlight-row");
      row.scrollIntoView({ behavior: "smooth", block: "center" });
      return true;
    }
  }
  return false;
}

async function openDocument(source, heading, excerpt) {
  docModalTitle.textContent = heading ? `${shortSource(source)} ‹ ${heading}` : shortSource(source);
  docModalBody.innerHTML = `<p class="doc-status">Loading&hellip;</p>`;
  docModal.hidden = false;

  const ext = (source.split(".").pop() || "").toLowerCase();
  const url = `/source?path=${encodeURIComponent(source)}`;

  try {
    if (ext === "pdf") {
      // No in-page anchor for a plain iframe'd PDF -- the browser's own
      // viewer owns that surface and doesn't expose a "scroll to text" hook
      // without PDF.js integration, which is more than this earns right now.
      docModalBody.innerHTML = `<iframe class="doc-frame" src="${url}"></iframe>`;
    } else if (ext === "docx") {
      await loadScriptOnce("https://unpkg.com/jszip@3.10.1/dist/jszip.min.js");
      await loadScriptOnce("https://unpkg.com/docx-preview@0.4.0/dist/docx-preview.min.js");
      const buf = await fetch(url).then(assertOk).then((r) => r.arrayBuffer());
      docModalBody.innerHTML = "";
      await window.docx.renderAsync(buf, docModalBody);
      highlightRenderedText(docModalBody, excerpt);
    } else if (ext === "xlsx") {
      await loadScriptOnce("https://cdn.sheetjs.com/xlsx-0.20.3/package/dist/xlsx.full.min.js");
      const buf = await fetch(url).then(assertOk).then((r) => r.arrayBuffer());
      const wb = window.XLSX.read(buf, { type: "array" });
      docModalBody.innerHTML = "";
      const wrap = document.createElement("div");
      wrap.className = "doc-sheets";
      wb.SheetNames.forEach((name) => {
        const section = document.createElement("section");
        section.className = "doc-sheet";
        section.innerHTML = `<h4>${escapeHtml(name)}</h4>${window.XLSX.utils.sheet_to_html(wb.Sheets[name])}`;
        wrap.appendChild(section);
      });
      docModalBody.appendChild(wrap);
      highlightTableRow(wrap, excerpt);
    } else {
      const text = await fetch(url).then(assertOk).then((r) => r.text());
      docModalBody.innerHTML = "";
      const pre = document.createElement("pre");
      pre.className = "doc-text";
      docModalBody.appendChild(pre);
      highlightPreText(pre, text, excerpt);
      docModalBody.querySelector(".doc-highlight")?.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  } catch (err) {
    docModalBody.innerHTML = `<p class="doc-status doc-status--error">Could not load this document: ${escapeHtml(err.message)}</p>`;
  }
}

function excerptChip(labelHtml, source, heading, excerpt, { quote = false } = {}) {
  const chip = document.createElement("button");
  chip.type = "button";
  chip.className = "citation-chip";
  chip.setAttribute("aria-expanded", "false");
  const text = (excerpt || "").trim();
  const shown = quote && text ? `&hellip;${escapeHtml(text)}&hellip;` : escapeHtml(text);
  chip.innerHTML = `${labelHtml}
    <div class="citation-pop" role="tooltip">
      <div class="citation-pop-excerpt${quote ? " citation-pop-excerpt--quote" : ""}">${shown}</div>
      <div class="citation-pop-source">${escapeHtml(source)}${heading ? " &rsaquo; " + escapeHtml(heading) : ""}</div>
      <div class="citation-pop-open" tabindex="0" role="button">Open document &rarr;</div>
    </div>`;
  chip.addEventListener("click", (e) => {
    e.stopPropagation();
    const isOpen = chip.classList.contains("is-open");
    document.querySelectorAll(".citation-chip.is-open").forEach((el) => {
      el.classList.remove("is-open");
      el.setAttribute("aria-expanded", "false");
    });
    if (!isOpen) {
      chip.classList.add("is-open");
      chip.setAttribute("aria-expanded", "true");
    }
  });
  const openTrigger = chip.querySelector(".citation-pop-open");
  const activateOpen = (e) => {
    e.stopPropagation();
    openDocument(source, heading, excerpt);
  };
  openTrigger.addEventListener("click", activateOpen);
  openTrigger.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      activateOpen(e);
    }
  });
  return chip;
}

function addAssistantMessage(result) {
  const node = document.getElementById("tpl-assistant").content.cloneNode(true);
  const answerEl = node.querySelector(".answer-text");
  const citationsEl = node.querySelector(".citations");
  const bestMatchEl = node.querySelector(".best-match");
  const metaEl = node.querySelector(".meta");
  const details = node.querySelector(".retrieval-details");

  const citations = result.citations || [];
  const retrieved = result.retrieved || [];

  citations.forEach((c) => {
    citationsEl.appendChild(excerptChip(
      `[${c.n}] ${escapeHtml(shortSource(c.source))}`, c.source, c.heading, c.excerpt));
  });
  if (citations.length === 0) citationsEl.remove();

  if (retrieved.length === 0) {
    details.remove();
  } else {
    const top = retrieved[0];
    bestMatchEl.appendChild(excerptChip(
      escapeHtml(shortSource(top.source)), top.source, top.heading, top.excerpt, { quote: true }));
  }

  chat.appendChild(node);
  const msgEl = chat.lastElementChild;
  metaEl.textContent = formatMeta(result.trace, citations.length);
  scrollToEnd();
  const duration = revealWords(answerEl, result.answer);
  setTimeout(scrollToEnd, Math.min(duration, 300));
}

function addRefusalMessage(result) {
  const node = document.getElementById("tpl-refusal").content.cloneNode(true);
  node.querySelector(".answer-text").textContent = result.refusal_reason || result.answer;
  const metaEl = node.querySelector(".meta");
  chat.appendChild(node);
  metaEl.textContent = formatMeta(result.trace, 0);
  scrollToEnd();
}

function addClarificationMessage(result) {
  const node = document.getElementById("tpl-clarify").content.cloneNode(true);
  node.querySelector(".answer-text").textContent = result.answer;
  const metaEl = node.querySelector(".meta");
  chat.appendChild(node);
  metaEl.textContent = formatMeta(result.trace, 0);
  scrollToEnd();
}

function shortSource(source) {
  return source.split("/").pop();
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

// ---------- ask ----------

async function ask(question) {
  addUserMessage(question);
  busy = true;
  sendBtn.disabled = true;
  const typingEl = addTyping();

  const startedAt = performance.now();
  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        history: history.slice(-10),
      }),
    });
    const elapsedMs = performance.now() - startedAt;

    typingEl.remove();

    if (res.status === 409) {
      addRefusalMessage({
        answer: "The index is empty — run ingestion on the server before asking questions.",
        refusal_reason: "The index is empty — run ingestion on the server before asking questions.",
        trace: {},
      });
      busy = false;
      sendBtn.disabled = false;
      return;
    }
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const result = await res.json();
    result.trace = { ...result.trace, latency_ms: result.trace?.latency_ms ?? elapsedMs };
    history.push({ role: "user", content: question });
    history.push({ role: "assistant", content: result.answer });

    if (result.needs_clarification) {
      addClarificationMessage(result);
    } else if (result.refused) {
      addRefusalMessage(result);
    } else {
      addAssistantMessage(result);
    }
  } catch (err) {
    typingEl.remove();
    addRefusalMessage({
      answer: "Request failed — the API may be unreachable.",
      refusal_reason: `Request failed: ${err.message}`,
      trace: {},
    });
  } finally {
    busy = false;
    sendBtn.disabled = false;
  }
}

document.addEventListener("click", () => {
  document.querySelectorAll(".citation-chip.is-open").forEach((el) => el.classList.remove("is-open"));
});
