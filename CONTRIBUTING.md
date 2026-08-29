# Contributing

## Before you touch retrieval, generation, or chunking

Run `make eval` first and read the scorecard. Every change to
`app/ingest`, `app/retrieve`, or `app/answer` should be followed by another `make eval`
so you can see what actually moved. "It feels better" is not a finding; a delta on the
scorecard is.

## Changing the baseline on purpose

`eval/baseline.json` is committed and CI (`.github/workflows/eval.yml`) fails a PR if
any regression-gated metric drops by more than 2 percentage points against it. That is
the point — it should be annoying to regress retrieval or citation quality by accident.

When a change is a deliberate, reviewed improvement (or an accepted tradeoff — e.g. a
chunking change that trades some precision for a big recall win), update the baseline
as its own commit:

```bash
make eval-accept
git add eval/baseline.json
git commit -m "accept new eval baseline: <what changed and why>"
```

Do this in a separate commit from the change itself, so the diff to `eval/baseline.json`
is reviewable on its own — a baseline bump hidden inside an unrelated refactor is exactly
the kind of thing this gate exists to catch.

**Run `eval-accept` with `project.config.json` pointed at `mock`, always.** CI only ever
runs mock (see `.github/workflows/eval.yml`: `cp project.config.example.json
project.config.json` before it evals), so `eval/baseline.json` only means something as a
mock-vs-mock comparison. Running `eval-accept` against a real provider silently replaces
it with real-model numbers — CI keeps running mock against that real-model baseline
afterward, mock can never clear it, and *every subsequent PR* fails with a wall of
false "regressions." This isn't hypothetical: it happened in this repo (see the commit
that reverts it). A real-provider quality snapshot belongs in the README's scorecard
section as its own manual update — never in `eval/baseline.json`.

`make eval` runs on the `mock` provider (free, deterministic, matches CI). The absolute
quality floors in `eval/check_thresholds.py` (`faithfulness >= 0.85`,
`refusal_accuracy >= 0.90`) are a *real-model* bar — `MockLLM`'s answer synthesis and
guardrail behaviour are lexical, not semantic, and cannot be expected to clear them.
CI reports that check for visibility on every PR but does not gate on it, for exactly
that reason. Run it for real before it should block anything:

```bash
export APP_LLM_PROVIDER=openai   # or anthropic, or ollama
python -m app.cli ingest data/sample
make eval
python -m eval.check_thresholds eval/report/latest.json
```

## Judge calibration

`eval/calibration/judge_calibration.jsonl` ships as a **template**: 40 cases sampled
from the golden set, each with a draft `candidate_answer` and a draft judge-style
`draft_score`, both marked `"draft_by": "claude"`. Every `human_verdict` field is `null`.
The README's kappa row stays "pending" until that changes — a kappa computed against
absent human labels would not be measuring anything, and it is not.

To make it real:

1. Open `eval/calibration/judge_calibration.jsonl`. For each row, read the `question`,
   `gold_answer`, and `candidate_answer`, and decide independently what score (0-4,
   same rubric as `core/eval/judge.py`'s `CORRECTNESS_SYSTEM`) you would give the
   candidate answer.
2. Set `human_verdict` to that score as a string (`"0"`-`"4"`). Leave `draft_score` and
   `draft_by` alone — they record what the machine guessed, which is useful context for
   whoever reviews the calibration later.
3. Label at least 30 rows (the full 40 is better) before trusting the kappa.
4. Run:

   ```bash
   python -m core.eval calibrate --subset eval/calibration/judge_calibration.jsonl \
       --judge-model <the exact model string you're using as judge>
   ```

5. Put the reported kappa in the README's scorecard, replacing "pending". If kappa is
   below ~0.6, that is a real finding — it means the automated judge and a human
   reader disagree too often for the judge's score to be trusted on its own, and the
   honest move is to say so, not to hide the number.

## Ablations

`eval/ablations.py` is retrieval-only by design (no LLM calls) — see the module
docstring for why. `--reranker cross-encoder` rows report `skipped` unless
`sentence-transformers` is installed; that's expected, not a bug, on a machine that
hasn't opted into the optional dependency.

## Tests, types, lint

```bash
make test   # pytest + mypy --strict on core/eval + ruff on core/eval
```

`core/eval` is held to `mypy --strict`; the rest of the codebase is not (yet). If you
add a metric, a judge behaviour, or anything else under `core/eval`, add a unit test
with a hand-computed expected value next to it — every existing metric has one, and a
metric without one is the kind of thing that silently drifts wrong.
