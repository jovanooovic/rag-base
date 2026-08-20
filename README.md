# rag-base

A production-shaped starting point for retrieval-augmented generation work, with an
evaluation harness rigorous enough to answer "is this actually good?" instead of "it
feels better."

## Scorecard

> **This is the `mock` provider baseline** — a plumbing check, not a quality estimate.
> `MockLLM`'s answer synthesis is lexical, not semantic, and its guardrail heuristic
> is a coverage-ratio approximation, not real judgment. It exists so every code path
> (including the judge) is exercised for free, offline, deterministically, in CI. The
> real-model number (this same table, run against a real provider) is the one that
> belongs in a client conversation — see "How these numbers were produced" below for
> the one-line command to produce it yourself.

| Metric | Value | 95% CI | n | vs. baseline |
|---|---|---|---|---|
| recall@1 | 0.7222 | [0.6528, 0.7870] | 108 | – |
| recall@3 | 0.9306 | [0.8935, 0.9630] | 108 | – |
| recall@5 | 0.9630 | [0.9352, 0.9861] | 108 | – |
| recall@10 | 1.0000 | [1.0000, 1.0000] | 108 | – |
| mrr | 0.9302 | [0.8931, 0.9630] | 108 | – |
| ndcg@5 | 0.9161 | [0.8826, 0.9464] | 108 | – |
| precision@5 | 0.2556 | [0.2370, 0.2741] | 108 | – |
| citation_precision | 0.7234 | [0.6436, 0.7979] | 94 | – |
| citation_recall | 0.6528 | [0.5694, 0.7361] | 108 | – |
| refusal_accuracy | 0.7037 | [0.5185, 0.8519] | 27 | – |
| clarification_rate | 0.0000 | [0.0000, 0.0000] | 13 | – |
| answer_correctness | 0.5255 | [0.4468, 0.6042] | 108 | – |
| faithfulness | 0.9484 | [0.9432, 0.9534] | 94 | – |
| cost_per_task_usd | 0.0000 | [0.0000, 0.0000] | 135 | – |
| latency_total_ms | 149.2 | [143.0, 155.3] | 135 | – |

Full metric list (all k values, both latency splits) in `eval/baseline.json` and
`eval/report/latest.md` after running `make eval`.

Judge-vs-human kappa: **pending** — the calibration file
(`eval/calibration/judge_calibration.jsonl`) ships with draft scores only; see
[CONTRIBUTING.md](CONTRIBUTING.md#judge-calibration) to make it real.

Two numbers worth reading carefully rather than skimming past: **refusal_accuracy is
0.70** and **clarification_rate is 0.0**, both against the mock provider. Neither is a
bug — `refusal_accuracy` reflects `MockLLM`'s crude lexical coverage-floor heuristic for
deciding when to refuse, and `clarification_rate` is 0 because the pipeline has no
clarification-seeking behaviour at all today (see Known Limitations). Publishing the
number that makes the system look unfinished, in the section a client reads first, is
the entire point of this repo.

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
- **LLM / embedding model**: `mock` for the table above (see the callout). Any real
  provider is pinned by its exact model string, never a floating `-latest` alias — see
  `core/eval/judge.py`.
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

No API key needed for the first three — `llm_provider: "mock"` runs the entire pipeline
offline with a deterministic fake model, so every code path (including the eval judge)
runs in CI with no spend. Point it at a real model when you're ready:

```bash
export OPENAI_API_KEY=sk-...
# project.config.json: "llm_provider": "openai", "embedding_provider": "openai", "embedding_dim": 1536
```

or Ollama, fully local:

```bash
ollama pull llama3.1 && ollama pull nomic-embed-text
# project.config.json: "llm_provider": "ollama", "embedding_provider": "ollama",
#                      "llm_model": "llama3.1", "embedding_model": "nomic-embed-text"
```

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

- **No real-provider scorecard is published yet.** An attempt to run the full 135-case
  suite against a local 27B model on a 12GB GPU hit repeated request timeouts — the
  model doesn't fully fit in VRAM (Ollama reported a 44%/56% CPU/GPU split) and CPU
  fallback for the remainder was too slow to reliably complete a single reranker call
  within a 300-second timeout, let alone ~135 pipeline runs plus ~600 judge calls. If
  you're reproducing this locally: pick a model that fits fully in VRAM (an 8B model
  was reliably fast on the same hardware) before reaching for a bigger one, or budget
  real wall-clock time (several hours) and a longer `request_timeout_s` if you don't.
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
- **`faithfulness` runs near ceiling on the `mock` provider specifically.**
  `MockLLM`'s answer synthesis is a near-verbatim excerpt of the retrieved passages, so
  lexical-overlap faithfulness is inflated by construction. Meaningful on a real
  provider; a plumbing check on mock. Same caveat applies to `answer_correctness` and
  `refusal_accuracy` — CI's absolute-floor check (`eval/check_thresholds.py`) is
  informational on mock for exactly this reason (see `.github/workflows/eval.yml`).
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
- **The LLM reranker costs a call per query batch.** `CrossEncoderReranker` is the slot
  for a local cross-encoder at volume; prove it with the ablation matrix first.
- **PDF support needs `pypdf`**, and scanned PDFs need OCR — a separate scoping
  conversation.

## Using this for a client engagement

1. `git clone` → `git checkout -b client/<name>`
2. Send them `intake/intake.html`. It is a self-contained page; they fill it in and it
   downloads `project.config.json`. Drop that in the repo root.
3. `make ingest PATH_=/their/corpus` then draft a real golden set for their domain
   (`python -m app.cli bootstrap-eval` scaffolds one from the indexed corpus — have
   their expert correct it, an unreviewed golden set will flatter you and then fail in
   production) and run `make eval` for a **day-one baseline number**.
4. Change one thing at a time. Re-run `make eval LABEL=<what-you-changed>`.
   `make eval-accept` updates the baseline deliberately when a change is a real,
   reviewed improvement — see [CONTRIBUTING.md](CONTRIBUTING.md).
5. Anything you build that is not client-specific gets backported to `main`.

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
app/api.py        FastAPI: /ask /ingest /documents /health
app/cli.py        ingest, ask, chat, stats, bootstrap-eval
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
