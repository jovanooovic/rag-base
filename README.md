# rag-base

A production-shaped starting point for retrieval-augmented generation work, with an
evaluation harness rigorous enough to answer "is this actually good?" instead of "it
feels better."

## Scorecard

Real-provider baseline: **`mistral` (Ollama, local) + `qwen3-embedding:8b`**, 135 cases,
run on consumer hardware (12GB GPU). No cloud key, no spend.

| Metric | Value | 95% CI | n |
|---|---|---|---|
| **Retrieval — pre-rerank** (raw hybrid fusion candidate set) | | | |
| recall@5 | 0.7685 | [0.6944, 0.8380] | 108 |
| recall@10 | 0.9907 | [0.9722, 1.0000] | 108 |
| mrr | 0.6006 | [0.5326, 0.6688] | 108 |
| hit_rate@1 | 0.4167 | [0.3241, 0.5093] | 108 |
| **Retrieval — post-rerank** (what the pipeline actually answers from) | | | |
| recall@5_reranked | 0.9537 | [0.9213, 0.9815] | 108 |
| mrr_reranked | 0.9614 | [0.9306, 0.9861] | 108 |
| hit_rate@1_reranked | 0.9352 | [0.8889, 0.9815] | 108 |
| **Answer quality** | | | |
| citation_precision | 0.8927 | [0.8429, 0.9368] | 87 |
| citation_recall | 0.7546 | [0.6759, 0.8287] | 108 |
| refusal_accuracy | **1.0000** | [1.0000, 1.0000] | 27 |
| clarification_rate | 0.0000 | [0.0000, 0.0000] | 13 |
| answer_correctness | 0.7269 | [0.6574, 0.7963] | 108 |
| faithfulness | 0.8078 | [0.7546, 0.8572] | 87 |
| **Operational** | | | |
| cost_per_task_usd | 0.0000 | [0.0000, 0.0000] | 135 |
| latency_total_ms | 10337.8 | [9855.5, 10834.7] | 135 |
| latency_retrieve_ms | 8501.6 | [8042.8, 8983.3] | 135 |

Full 40-metric list in `eval/report/latest.md` after `make eval`.

Judge-vs-human kappa: **pending** — the calibration file
(`eval/calibration/judge_calibration.jsonl`) ships with draft scores only; see
[CONTRIBUTING.md](CONTRIBUTING.md#judge-calibration) to make it real.

**Read the retrieval rows as a before/after, not two unrelated numbers.**
`recall@k`/`mrr`/`hit_rate@k` score the wide hybrid-search candidate set *before* any
reranking — deliberately, so the ablation matrix below can isolate chunking and
retrieval-mode choices from the reranker (see `EVAL_RETRIEVAL_K` in
`app/evaluation/rag_adapter.py`). The `*_reranked` rows score the same cases after the
pipeline's actual reranking step — i.e. what a real user's answer was generated from.
The gap between them (hit_rate@1 goes from **0.42 to 0.94**) is the reranker's entire
job, measured rather than assumed: raw hybrid fusion alone gets the right document into
the top slot less than half the time on this corpus; reranking fixes that for 9 cases
in 10. That is the answer to "is the reranker worth the extra model call" for this
corpus — measure it again on yours before trusting it there.

**`refusal_accuracy = 1.0000`** is the number worth sitting with: a real model, tested
against all 27 out-of-scope questions in the golden set, never once fabricated an
answer. `clarification_rate = 0.0000` is not a companion success — the pipeline has no
clarification-seeking behaviour at all today, so every ambiguous-slice case gets a
guess instead of a question back (see Known Limitations). Publishing the number that
shows what's unfinished, next to the one that shows what works, is the point of this
whole exercise.

## How these numbers were produced

- **Corpus**: 13 synthetic Acme support docs (`data/sample/`), 51 chunks, structure-first
  chunking. Fully synthetic — no client data in this repo, ever.
- **Golden set**: `eval/data/golden.jsonl`, 135 hand-authored cases, stratified:

  | Type | Count | Share |
  |---|---|---|
  | factoid | 54 | 40% |
  | multi_hop | 27 | 20% |
  | aggregation | 14 | 10% |
  | unanswerable | 27 | 20% |
  | ambiguous | 13 | 10% |

  The unanswerable and ambiguous slices are the ones worth having. Most public RAG
  evals only test the happy path; a refusal-rate number is unusual, and it's exactly
  what an enterprise client is nervous about.
- **LLM / embedding model**: `mistral` / `qwen3-embedding:8b`, both local via Ollama,
  pinned by exact model string — never a floating `-latest` alias (see
  `core/eval/judge.py`). The judge is the same `mistral` instance; a self-judging model
  is a real caveat, noted again in Known Limitations.
- **Judge**: the same model, temperature 0, majority vote of 3 independent calls per
  case; disagreement rate recorded, not discarded. Structured JSON output only — a
  judge reply that doesn't parse is recorded as a hard failure, not silently coerced.
- **Seed**: bootstrap confidence intervals use a fixed seed (0), so re-running against
  the same predictions reproduces the identical interval.
- **Reproduce it**:

  ```bash
  export APP_LLM_PROVIDER=ollama        # or openai / anthropic
  python -m app.cli ingest data/sample
  make eval
  ```

  CI itself runs on `llm_provider: "mock"` (free, deterministic, zero spend) so every
  code path — including the judge — is exercised on every PR with no API key. The mock
  numbers are a plumbing check, not a quality estimate; don't confuse the two. See
  `.github/workflows/eval.yml`.

## Ablation matrix

Chunking × retrieval mode × reranker, scored on retrieval quality only (no LLM calls —
see `eval/ablations.py` for why that's the right scope). Reproduce with `make ablations`.

| chunking | retrieval | reranker | recall@5 | recall@10 | mrr | ndcg@5 |
|---|---|---|---|---|---|---|
| structure-first | dense-only | off | 0.9583 | 0.9815 | 0.8466 | 0.8473 |
| structure-first | bm25-only | off | 0.9630 | 0.9769 | 0.9105 | 0.9032 |
| structure-first | hybrid-rrf | off | 0.9630 | 0.9722 | 0.9290 | 0.9161 |
| fixed-512 | dense-only | off | 0.9259 | 0.9444 | 0.8630 | 0.8470 |
| fixed-512 | bm25-only | off | 0.9630 | 0.9815 | 0.8907 | 0.8905 |
| fixed-512 | hybrid-rrf | off | 0.9583 | 0.9907 | 0.9168 | 0.9016 |
| fixed-1024 | dense-only | off | 0.9583 | 0.9861 | 0.8745 | 0.8727 |
| fixed-1024 | bm25-only | off | 0.9444 | 0.9722 | 0.8629 | 0.8717 |
| fixed-1024 | hybrid-rrf | off | 0.9676 | 0.9861 | 0.9095 | 0.9062 |
| recursive-overlap | dense-only | off | 0.9491 | 0.9861 | 0.8276 | 0.8303 |
| recursive-overlap | bm25-only | off | 0.9583 | 0.9769 | 0.8691 | 0.8766 |
| recursive-overlap | hybrid-rrf | off | 0.9583 | 0.9907 | 0.8993 | 0.8919 |
| semantic | dense-only | off | 0.8565 | 0.9259 | 0.7822 | 0.7661 |
| semantic | bm25-only | off | 0.9491 | 0.9676 | 0.9213 | 0.9021 |
| semantic | hybrid-rrf | off | 0.9352 | 0.9537 | 0.8748 | 0.8611 |

At this corpus size (13 docs, 51 chunks) none of the differences are large enough to
call a clear winner — `structure-first + hybrid-rrf` (the shipped default) sits in the
top cluster on every column, and `semantic + dense-only` is the clearest loser (worst
mrr and ndcg@5 by a real margin). That is the honest reading, not "structure-first
wins": a 13-document corpus does not have enough signal to separate the top four rows,
and this table is exactly what tells you that rather than letting you guess.

`reranker: cross-encoder` rows report `skipped` unless `sentence-transformers` is
installed — that's an optional dependency, not a missing feature.

## Quickstart

```bash
make setup
make ingest                       # indexes data/sample
make ask Q="how long is the warranty on batteries?"
make eval                         # scores against eval/data/golden.jsonl
```

`make serve` starts the API on `:8000` and also serves `web/` — a small chat
console (no framework, no build step) at `http://localhost:8000` for demoing
the pipeline to someone who isn't going to read `curl` output. Same citations,
refusals and cost/latency numbers the API returns, just legible.

`http://localhost:8000/admin.html` uploads and indexes files directly (not
linked from the chat page, not in the sitemap). It calls `POST /upload`, which
-- like `POST /ingest` -- is a no-op-gated behind an `X-Admin-Token` header:
open with no `APP_ADMIN_TOKEN` set, enforced the moment you set one. Set it
before putting the API anywhere reachable by someone other than you.

No API key needed for the first three — `llm_provider: "mock"` runs the entire pipeline
offline with a deterministic fake model, so every code path (including the eval judge)
runs in CI with no spend. Point it at a real model when you're ready:

```bash
export OPENAI_API_KEY=sk-...
# project.config.json: "llm_provider": "openai", "embedding_provider": "openai", "embedding_dim": 1536
```

or Ollama, fully local:

```bash
ollama pull mistral && ollama pull qwen3-embedding:8b
# project.config.json: "llm_provider": "ollama", "embedding_provider": "ollama",
#                      "llm_model": "mistral", "embedding_model": "qwen3-embedding:8b"
```

Pick a model that fits fully in your GPU's VRAM before reaching for a bigger one — see
the first item in Known Limitations for what happens when it doesn't. `mistral` (7B,
~4.4GB) and `llama3.1` (8B, ~4.9GB) were both reliably fast on a 12GB card; a 27B model
was not.

## Architecture

```
question
  │
  ├─ rewrite (multi-turn only) ──► 1-3 standalone queries
  │
  ├─ vector search  ─┐
  ├─ BM25 keyword    ─┤► reciprocal rank fusion ──► rerank ──► top_k chunks
  │                   ┘
  ├─ answer with numbered sources, citations required
  │
  └─ guardrails: refuse if uncited / below score floor / model said not-in-sources
```

Decisions worth defending in a client call:

| Choice | Why |
|---|---|
| Hybrid, not pure vector | Embeddings miss exact identifiers — order numbers, SKUs, error codes. That is the #1 "it can't find the right doc" complaint. |
| RRF fusion | Cosine and BM25 live on incomparable scales; RRF reads positions only, so one sparse leg can't destabilise it. |
| Structure-first chunking (default) | Chunking on headings before size. Measured against 4 alternatives in the ablation matrix above, not assumed. |
| SQLite default | Zero infrastructure on day one. `PgVectorStore` is a one-line swap when scale justifies it. |
| Citations enforced | An uncited answer is blocked, not returned. Fail closed. |
| Per-run cost budget | Refuses past `max_cost_usd_per_run`. Cheap insurance against the overnight-loop invoice. |
| No LangChain | Nothing here is hard enough to justify the abstraction, and provider changes stay in one 400-line file. |
| Shared `core/eval/` | The same metric ABC, judge, baseline, and CI machinery are meant to be reused by an agent runtime too — the reuse is the story, not an implementation detail. |

## Known limitations — what this eval does and does not cover

Written at length on purpose: a candidate who states the boundaries of their own
evidence is signalling something no benchmark table can.

- **A 27B model was tried first and failed — that's why the scorecard is `mistral`, not
  something bigger.** The full 135-case suite against a local 27B model on a 12GB GPU
  hit repeated request timeouts: the model didn't fully fit in VRAM (Ollama reported a
  44%/56% CPU/GPU split), and CPU fallback for the remainder was too slow to reliably
  complete even a single reranker call within a 300-second timeout. Swapping to a 7B
  model that fits fully in VRAM fixed it completely — the run above finished in 15
  minutes with zero failures. If you're reproducing this locally: check that your model
  fits in VRAM before reaching for a bigger one, or budget real wall-clock time (hours,
  not minutes) and a longer `request_timeout_s` if you don't.
- **The judge is the same model that generated the answers.** `mistral` grades
  `mistral`'s own output for `answer_correctness` and `faithfulness`. That's a real
  self-judging bias, not a neutral third-party score — treat these two numbers as
  optimistic until a different (ideally stronger) model is used as judge, or until the
  calibration file below gives an independent human check.
- **Judge calibration is unfilled.** The kappa row reads "pending" because
  `eval/calibration/judge_calibration.jsonl` ships with draft (machine-guessed) scores,
  not real human verdicts. Until someone labels at least 30 of the 40 rows, treat the
  judge-based metrics (`answer_correctness`, `faithfulness`) as *plausible*, not
  *validated*.
- **`citation_precision`/`citation_recall` are a simplification.** They check whether a
  cited source is one of the gold-relevant documents, not whether that specific citation
  supports that specific claim. A full judge-verified per-claim check is the textbook
  version of this metric and costs a judge call per citation; this repo's version is the
  cheap, deterministic approximation. Documented in `core/eval/metrics/generation.py`.
- **`clarification_rate` reads 0.0 by design, not by bug.** The pipeline has no
  clarification-seeking behaviour today — it only ever answers or refuses. The ambiguous
  slice exists so this gap is measured and visible rather than silently absent from the
  eval.
- **Operational numbers (latency, cost) are hardware- and provider-dependent.** They are
  reported, not gated, in CI (see `core/eval/types.py`'s `regression_gated` flag) —
  gating an unbounded-scale metric with the same "N percentage points" threshold used
  for recall/precision would fail on ordinary run-to-run timing noise, not a real
  regression.
- **Don't confuse a CI run with the scorecard above.** CI runs on `llm_provider: "mock"`
  for speed and zero spend, and `MockLLM`'s answer synthesis is a near-verbatim excerpt
  of the retrieved passages, so lexical-overlap faithfulness is inflated by
  construction on mock (it measured 0.95 on mock vs the real 0.81 above). Mock is a
  plumbing check, not a quality estimate — that's why `eval/check_thresholds.py`'s
  absolute-floor check is informational on mock, not a hard CI gate (see
  `.github/workflows/eval.yml`).
- **The corpus is small and fully synthetic (13 docs, 51 chunks).** Retrieval numbers
  at this scale are easy to make look good; the ablation matrix and the golden set's
  multi-hop/aggregation slices exist to make that harder, but a client's real corpus
  will surface failure modes this sample corpus cannot.
- **The ablation matrix is retrieval-only.** It does not measure whether a chunking
  strategy or reranker choice changes final answer quality, faithfulness, or citation
  correctness — only whether the gold document gets retrieved. Run `make eval` with a
  chunking override to check the generation-side effect of a specific change.
- **SQLite cosine search is a linear scan.** Fine to roughly 100k chunks, then move to
  `PgVectorStore`.
- **The LLM reranker costs a call per query batch, and on this corpus it's earning it**
  — hit_rate@1 goes from 0.42 pre-rerank to 0.94 post-rerank (see the scorecard). That's
  this corpus, this model, this query mix; measure it again before trusting it
  elsewhere. `CrossEncoderReranker` is the slot for a local cross-encoder at volume once
  you have.
- **PDF support needs `pypdf`**, and scanned PDFs need OCR — a separate scoping
  conversation.

## Using this for a client engagement

1. `git clone` this repo into a **new, private repo** for the client (e.g.
   `<you>/rag-<client>`) — never a branch pushed back to this public one. Their
   config, corpus, and eval set are theirs; this repo stays a clean, generic base.
2. Send them `intake/intake.html`. It is a self-contained page; they fill it in and it
   downloads `project.config.json`. Drop that in the repo root. Set `"brand_accent"`
   and `"show_source_link": false` in there too — see [Branding](#branding-per-client)
   below.
3. `make ingest PATH_=/their/corpus` then draft a real golden set for their domain
   (`python -m app.cli bootstrap-eval` scaffolds one from the indexed corpus — have
   their expert correct it, an unreviewed golden set will flatter you and then fail in
   production) and run `make eval` for a **day-one baseline number**.
4. Change one thing at a time. Re-run `make eval LABEL=<what-you-changed>`.
   `make eval-accept` updates the baseline deliberately when a change is a real,
   reviewed improvement — see [CONTRIBUTING.md](CONTRIBUTING.md).
5. Anything you build that is not client-specific gets backported to `main` **before**
   you start the next client — this base only stays reusable if fixes don't have to be
   re-applied by hand to every client fork.

### Branding per client

The chat console (`web/`) reads its identity from the API at runtime — nothing to
hand-edit per client:

- `project_name` in `project.config.json` sets the page title and topbar mark.
- `"brand_accent": "#2f6f4f"` (any hex) recolors the whole UI — every derived shade
  (hover states, soft backgrounds, focus rings) is computed from that one value via
  CSS `color-mix()`, so one hex is enough for a full client palette.
- `"show_source_link": false` hides the "view source on GitHub" icon — you don't want
  a client-branded deployment linking back to the public portfolio repo.

Step 3 is the differentiator. Walking into the first check-in with "your current
pipeline answers 61% of your own 40 questions correctly, here are the 16 failures
grouped by cause" is a different conversation from "I've set up LangChain."

## Layout

```
core/eval/        shared, system-agnostic eval framework: metrics, judge, baseline, CLI
app/core/         config, providers (openai/anthropic/ollama/mock), retry, tracing, cost budget
app/ingest/       loaders (md/html/csv/json/pdf), 5 chunking strategies, idempotent pipeline
app/store/        SQLiteStore (default), PgVectorStore (scale), shared interface
app/retrieve/     BM25, RRF hybrid, LLM + cross-encoder reranker, multi-turn query rewriting
app/answer/       cited generation, refusal guardrails, PII redaction
app/evaluation/   the RAG-specific adapter into core/eval, and the golden-set bootstrapper
app/api.py        FastAPI: /ask /ingest /upload /documents /health, serves web/ at "/"
app/cli.py        ingest, ask, chat, stats, bootstrap-eval
web/              chat console + admin.html upload panel (vanilla JS, no build step)
eval/             golden set, calibration template, ablations, HTML/PNG report tooling
intake/           the client questionnaire
```

## Where to extend, per client

- **Access control** — `where=` filters flow to both retrieval legs. Tag chunks with
  tenant/role at ingest and pass the filter from your auth layer. Do this before
  multi-tenant go-live.
- **Freshness** — ingestion is content-hashed and idempotent; wire it to cron or a
  webhook.
- **Better reranking** — `CrossEncoderReranker` is the slot. Prove it with the ablation
  matrix first; sometimes it wins 15 nDCG points, sometimes nothing, and then you've
  saved them a GPU.
- **Streaming** — add SSE to `/ask` once the guardrail decision is made. Do not stream
  before the citation check, or you stream text you were going to block.
- **Observability** — `app/core/trace.py` is ~80 lines and intentionally vendor-free.
  Swap for OTel when a client names their stack.
