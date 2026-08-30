from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core.config import Settings
from .core.providers import Message
from .pipeline import RAGPipeline


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rag", description="RAG base CLI")
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="index a file or directory")
    p_ing.add_argument("path")

    p_ask = sub.add_parser("ask", help="ask one question")
    p_ask.add_argument("question")
    p_ask.add_argument("--json", action="store_true")
    p_ask.add_argument("--top-k", type=int, default=None)

    sub.add_parser("chat", help="interactive multi-turn session")
    sub.add_parser("stats", help="show what is indexed")

    p_boot = sub.add_parser("bootstrap-eval", help="draft a golden set from the indexed corpus")
    p_boot.add_argument("--out", default="eval/golden.generated.jsonl")
    p_boot.add_argument("--limit", type=int, default=25)

    p_fb = sub.add_parser("feedback-export",
                          help="turn thumbs-down feedback into draft golden cases")
    p_fb.add_argument("--out", default="eval/golden.from-feedback.jsonl")
    p_fb.add_argument("--limit", type=int, default=100)

    args = ap.parse_args(argv)
    settings = Settings.load(args.config)

    # Before the pipeline is built: this reads a local JSONL file and nothing
    # else, and constructing RAGPipeline would demand a provider API key to do
    # it -- which would make triaging feedback impossible offline.
    if args.cmd == "feedback-export":
        return _feedback_export(settings, out=Path(args.out), limit=args.limit)

    pipeline = RAGPipeline(settings)

    if args.cmd == "ingest":
        report = pipeline.ingest(args.path)
        print(json.dumps(report.as_dict(), indent=2))
        return 0

    if args.cmd == "stats":
        chunks = pipeline.store.all_chunks()
        sources = sorted({c.source for c in chunks})
        print(f"{len(chunks)} chunks across {len(sources)} documents")
        for s in sources[:50]:
            print("  ", s)
        return 0

    if args.cmd == "ask":
        result = pipeline.ask(args.question, top_k=args.top_k)
        if args.json:
            print(json.dumps(result.as_dict(), indent=2))
        else:
            _print(result)
        return 0

    if args.cmd == "chat":
        history: list[Message] = []
        print("multi-turn chat. ctrl-d to exit.\n")
        while True:
            try:
                q = input("you> ").strip()
            except EOFError:
                return 0
            if not q:
                continue
            result = pipeline.ask(q, history=history or None)
            _print(result)
            history += [Message.user(q), Message.assistant(result.answer.text)]

    if args.cmd == "bootstrap-eval":
        from .evaluation.dataset import bootstrap_from_corpus
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        settings.max_llm_calls_per_run = 10_000
        rows = list(bootstrap_from_corpus(pipeline.store.all_chunks(), pipeline.llm,
                                          limit=args.limit))
        out.write_text("\n".join(json.dumps(r) for r in rows))
        print(f"wrote {len(rows)} draft cases to {out}")
        print("These are a scaffold. Have the client's expert review them before you quote a number.")
        return 0

    return 1


def _feedback_export(settings, *, out: Path, limit: int) -> int:
    """Turn thumbs-down feedback into draft golden cases.

    Drafts, not cases. Every row lands with gold_answer empty and a note saying
    it needs review, in the same shape bootstrap-eval produces -- because a
    reader clicking thumbs-down tells you they were unhappy, not what the right
    answer was. Promoting these unreviewed would let user disappointment quietly
    redefine correctness, which is how a golden set stops meaning anything.

    Deduplicated by question: the same broken answer found by ten people is one
    case to fix, not ten.
    """
    from .api import read_feedback

    rows = [r for r in read_feedback(settings) if r.get("verdict") == "down"]
    if not rows:
        print("no negative feedback recorded yet")
        return 0

    seen: dict[str, dict] = {}
    for r in rows:
        q = (r.get("question") or "").strip()
        if q and q not in seen:
            seen[q] = r

    drafts = []
    for i, (question, row) in enumerate(list(seen.items())[:limit], start=1):
        note = "from user feedback, needs review"
        if row.get("note"):
            note += f" -- reader said: {row['note']}"
        drafts.append({
            "id": f"fb-{i:02d}",
            "question": question,
            "gold_answer": "",
            "gold_doc_ids": [],
            "gold_spans": [],
            "type": "factoid",
            "difficulty": "unknown",
            "notes": note,
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in drafts) + "\n",
                   encoding="utf-8")
    print(f"wrote {len(drafts)} draft cases to {out} "
          f"(from {len(rows)} thumbs-down, {len(seen)} distinct questions)")
    print("These are a scaffold. Fill in gold_answer and gold_doc_ids, and have someone "
          "who knows the domain confirm them, before any of it reaches a scorecard.")
    return 0


def _print(result) -> None:
    print("\n" + result.answer.text + "\n")
    if result.needs_clarification:
        print("  [asked for clarification instead of guessing]")
    elif result.refused:
        print(f"  [refused: {result.refusal_reason}]")
    for c in result.answer.citations:
        loc = f" > {c.heading_path}" if c.heading_path else ""
        print(f"  [{c.number}] {c.source}{loc}")
    t = result.trace
    print(f"  ({t.get('llm_calls', 0)} llm calls, ${t.get('cost_usd', 0):.4f})\n")


if __name__ == "__main__":
    sys.exit(main())
