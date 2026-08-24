// rag-base chat console — vanilla JS, no build step, same-origin fetch
// against /health, /ask, /documents.

const chat = document.getElementById("chat");
const emptyState = document.getElementById("empty-state");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const themeToggle = document.getElementById("theme-toggle");

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
  if (body.show_source_link === false) {
    document.getElementById("source-link").remove();
  }
}

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
  input.value = chip.textContent;
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

function addAssistantMessage(result) {
  const node = document.getElementById("tpl-assistant").content.cloneNode(true);
  const answerEl = node.querySelector(".answer-text");
  const citationsEl = node.querySelector(".citations");
  const table = node.querySelector(".retrieval-table tbody");
  const metaEl = node.querySelector(".meta");
  const details = node.querySelector(".retrieval-details");

  const citations = result.citations || [];
  const retrieved = result.retrieved || [];

  citations.forEach((c) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "citation-chip";
    chip.setAttribute("aria-expanded", "false");
    chip.innerHTML = `[${c.n}] ${escapeHtml(shortSource(c.source))}
      <div class="citation-pop" role="tooltip">
        <div class="citation-pop-source">${escapeHtml(c.source)}${c.heading ? " &rsaquo; " + escapeHtml(c.heading) : ""}</div>
        <div class="citation-pop-excerpt">${escapeHtml(c.excerpt || "")}</div>
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
    citationsEl.appendChild(chip);
  });
  if (citations.length === 0) citationsEl.remove();

  if (retrieved.length === 0) {
    details.remove();
  } else {
    retrieved.forEach((h) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${escapeHtml(shortSource(h.source))}</td><td>${h.score}</td>`;
      table.appendChild(tr);
    });
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

    if (result.refused) {
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
