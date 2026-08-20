"""Render a self-contained HTML eval report: scorecard, ablation table, per-slice
breakdown, and the ten worst-scoring cases with their retrieved context.

No JS framework, inline CSS, one file -- matches this repo's "no LangChain, no extra
vendor" stance. The failure cases are what a technical client actually reads.

    python -m core.eval.html_report --scorecard eval/report/latest.json \\
        --ablations eval/report/ablations.json --cases eval/report/cases.json \\
        --out eval/report/report.html
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def _case_score(entry: dict[str, Any]) -> float | None:
    """A cheap, deterministic 0-1 signal for ranking "worst cases" in the report.

    Not one of core.eval's Metric classes on purpose: a metric's `per_case` tuple is
    already filtered (a metric can skip cases), so it does not align 1:1 back to a
    case id. This works directly off the raw case/prediction dump instead. It is a
    RAG-shaped heuristic (gold_doc_ids, retrieved_doc_ids, citations, refused) --
    cases whose fields don't match that shape are skipped, not mis-scored.
    """
    case, pred = entry.get("case") or {}, entry.get("prediction")
    if pred is None:
        return 0.0
    if pred.get("error"):
        return 0.0
    output = pred.get("output") or {}
    metadata = case.get("metadata") or {}
    case_type = metadata.get("type")

    if case_type == "unanswerable":
        return 1.0 if output.get("refused") else 0.0
    if case_type == "ambiguous":
        return 1.0 if output.get("asked_clarification") else 0.0

    gold = set((case.get("expected") or {}).get("gold_doc_ids", []))
    if not gold:
        return None

    signals = [1.0 if gold & set(output.get("retrieved_doc_ids", [])[:5]) else 0.0]
    cited = set(output.get("citations", []))
    if cited:
        signals.append(len(gold & cited) / len(cited))
    if output.get("refused"):
        signals.append(0.0)
    return sum(signals) / len(signals)


def _slice_breakdown(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_type: dict[str, list[float]] = {}
    for entry in cases:
        case_type = ((entry.get("case") or {}).get("metadata") or {}).get("type") or "(untyped)"
        score = _case_score(entry)
        if score is not None:
            by_type.setdefault(case_type, []).append(score)
    return [
        {"type": t, "n": len(scores), "avg_score": round(sum(scores) / len(scores), 4)}
        for t, scores in sorted(by_type.items())
    ]


def _worst_cases(cases: list[dict[str, Any]], n: int = 10) -> list[dict[str, Any]]:
    scored = []
    for entry in cases:
        score = _case_score(entry)
        if score is not None:
            scored.append((score, entry))
    scored.sort(key=lambda t: t[0])
    return [{"score": s, **e} for s, e in scored[:n]]


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _scorecard_table(scorecard: dict[str, Any]) -> str:
    rows = []
    for m in scorecard.get("metrics", []):
        delta = m.get("delta")
        delta_cell = "-" if delta is None else f"{delta:+.4f}"
        rows.append(f"<tr><td>{_esc(m['name'])}</td><td>{m['value']:.4f}</td>"
                   f"<td>[{m['ci'][0]:.4f}, {m['ci'][1]:.4f}]</td><td>{m['n']}</td>"
                   f"<td>{delta_cell}</td></tr>")
    kappa = scorecard.get("meta", {}).get("kappa")
    kappa_str = "pending" if kappa is None else f"{kappa:.3f}"
    return (f"<h2>Scorecard</h2><p>n = {scorecard.get('n_cases', '?')} &middot; "
           f"judge-vs-human kappa: <strong>{kappa_str}</strong></p>"
           "<table><thead><tr><th>Metric</th><th>Value</th><th>95% CI</th><th>n</th>"
           f"<th>vs. baseline</th></tr></thead><tbody>{''.join(rows)}</tbody></table>")


def _ablation_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    body = []
    for r in rows:
        if r.get("skipped"):
            body.append(f"<tr><td>{_esc(r['chunking'])}</td><td>{_esc(r['retrieval'])}</td>"
                       f"<td>{_esc(r['reranker'])}</td>"
                       f"<td colspan='4'><em>skipped: {_esc(r['skipped'])}</em></td></tr>")
        else:
            body.append(f"<tr><td>{_esc(r['chunking'])}</td><td>{_esc(r['retrieval'])}</td>"
                       f"<td>{_esc(r['reranker'])}</td>"
                       f"<td>{r.get('recall@5', 0):.4f}</td><td>{r.get('recall@10', 0):.4f}</td>"
                       f"<td>{r.get('mrr', 0):.4f}</td><td>{r.get('ndcg@5', 0):.4f}</td></tr>")
    return ("<h2>Ablation matrix</h2>"
           "<table><thead><tr><th>Chunking</th><th>Retrieval</th><th>Reranker</th>"
           "<th>recall@5</th><th>recall@10</th><th>mrr</th><th>ndcg@5</th></tr></thead>"
           f"<tbody>{''.join(body)}</tbody></table>")


def _slice_table(slices: list[dict[str, Any]]) -> str:
    if not slices:
        return ""
    rows = "".join(f"<tr><td>{_esc(s['type'])}</td><td>{s['n']}</td><td>{s['avg_score']:.4f}</td></tr>"
                   for s in slices)
    return ("<h2>Per-slice breakdown</h2>"
           "<table><thead><tr><th>Type</th><th>n</th><th>avg score</th></tr></thead>"
           f"<tbody>{rows}</tbody></table>")


def _worst_cases_panel(worst: list[dict[str, Any]]) -> str:
    if not worst:
        return ""
    blocks = []
    for w in worst:
        case, pred = w.get("case") or {}, w.get("prediction") or {}
        question = (case.get("input") or {}).get("question", "")
        output = pred.get("output") or {}
        answer = output.get("answer_text", "")
        context = output.get("retrieved_context", "")
        blocks.append(
            f"<div class='case'><h3>{_esc(case.get('id', '?'))} "
            f"<span class='score'>score {w['score']:.2f}</span></h3>"
            f"<p><strong>Q:</strong> {_esc(question)}</p>"
            f"<p><strong>A:</strong> {_esc(answer)}</p>"
            f"<details><summary>Retrieved context</summary>"
            f"<pre>{_esc(context)[:3000]}</pre></details></div>"
        )
    return "<h2>Ten worst-scoring cases</h2>" + "".join(blocks)


_STYLE = """
body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; max-width: 960px;
       margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; line-height: 1.5; }
h1 { border-bottom: 3px solid #1a1a1a; padding-bottom: .5rem; }
h2 { margin-top: 2.5rem; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
th, td { border: 1px solid #ccc; padding: .4rem .6rem; text-align: left; font-size: .9rem; }
th { background: #f2f2f2; }
.case { border: 1px solid #ddd; border-radius: 6px; padding: 1rem; margin: 1rem 0; }
.score { color: #a00; font-weight: normal; font-size: .85rem; }
pre { white-space: pre-wrap; background: #f7f7f7; padding: .75rem; border-radius: 4px;
      font-size: .8rem; max-height: 300px; overflow-y: auto; }
"""


def render(scorecard: dict[str, Any], ablations: list[dict[str, Any]],
          cases: list[dict[str, Any]]) -> str:
    slices = _slice_breakdown(cases)
    worst = _worst_cases(cases)
    return (f"<!doctype html><html><head><meta charset='utf-8'>"
           f"<title>{_esc(scorecard.get('label', 'eval report'))}</title>"
           f"<style>{_STYLE}</style></head><body>"
           f"<h1>{_esc(scorecard.get('label', 'eval report'))}</h1>"
           f"{_scorecard_table(scorecard)}{_ablation_table(ablations)}"
           f"{_slice_table(slices)}{_worst_cases_panel(worst)}"
           f"</body></html>")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scorecard", required=True)
    ap.add_argument("--ablations", default=None)
    ap.add_argument("--cases", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    scorecard = json.loads(Path(args.scorecard).read_text())
    ablations = json.loads(Path(args.ablations).read_text()) if args.ablations and Path(args.ablations).is_file() else []
    cases = json.loads(Path(args.cases).read_text()) if args.cases and Path(args.cases).is_file() else []

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(scorecard, ablations, cases))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
