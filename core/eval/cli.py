from __future__ import annotations

import argparse
import asyncio
import importlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .baseline import compare, load_baseline, write_baseline
from .report import to_json, to_markdown, to_terminal
from .runner import content_hash, git_sha, run_suite, write_results
from .types import Case, Prediction, Scorecard


def _resolve_adapter(spec: str) -> Any:
    """`module.path:function_name` -> the function, imported lazily so core.eval has no
    hard dependency on any specific system under test.

    The resolved function must take no arguments and return
    `(build_system, metrics_factory, load_cases)`: case loading is schema-specific (a
    RAG golden set and an agent scenario suite don't share a JSONL shape), so it lives
    with the adapter rather than a one-size-fits-all loader in this module.
    """
    module_name, _, func_name = spec.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def _dump_cases(cases: Sequence[Case], predictions: Sequence[Prediction]) -> list[dict[str, Any]]:
    """Raw case/prediction pairs, keyed by case id -- not per-metric scores (a metric
    can skip cases, e.g. recall@k on an unanswerable case, so its `per_case` tuple
    does not align 1:1 with the full case list). A report that wants a "worst cases"
    panel computes its own ranking from this raw data; core.eval does not assume one
    system's definition of "worst" applies to another's.
    """
    preds_by_id = {p.case_id: p for p in predictions}
    out = []
    for c in cases:
        p = preds_by_id.get(c.id)
        out.append({"case": asdict(c), "prediction": asdict(p) if p else None})
    return out


def run(args: argparse.Namespace) -> int:
    build_system, metrics_factory, load_cases = _resolve_adapter(args.adapter)()
    cases = load_cases(args.suite)
    system = build_system()
    predictions = asyncio.run(run_suite(cases, system, concurrency=args.concurrency))
    metrics = metrics_factory()
    results = tuple(m.compute(cases, predictions) for m in metrics)

    baseline = load_baseline(args.baseline) if args.baseline else None
    scorecard = Scorecard(label=args.label, metrics=results, n_cases=len(cases),
                          meta={"git_sha": git_sha(), "dataset": str(args.suite)}, baseline=baseline)

    print(to_terminal(scorecard))

    dataset_bytes = Path(args.suite).read_bytes()
    # `label` is part of the hash input deliberately: two runs of the same dataset
    # against different providers/models (e.g. "mock-baseline" vs "real-mistral") are
    # not the same result and must not silently overwrite each other's <hash>.json --
    # only `content_hash`'s other inputs (dataset, adapter, git SHA) are visible to
    # this generic CLI, and none of them capture "which model actually ran."
    run_hash = content_hash(dataset_bytes, {"adapter": args.adapter, "label": args.label},
                            scorecard.meta["git_sha"])
    write_results(args.out, run_hash, to_json(scorecard))
    markdown = to_markdown(scorecard)
    Path(args.out, f"{run_hash}.md").write_text(markdown)
    Path(args.out, "latest.md").write_text(markdown)
    Path(args.out, "cases.json").write_text(json.dumps(_dump_cases(cases, predictions), indent=2))

    if baseline is not None:
        cmp = compare(scorecard, baseline, regression_threshold=args.regression_threshold)
        if cmp["regressions"]:
            print(f"\nREGRESSIONS: {cmp['regressions']}")
            if args.fail_on_regression:
                return 1
    return 0


def accept(args: argparse.Namespace) -> int:
    _guard_baseline_provider(allow_real=args.allow_real_baseline)
    build_system, metrics_factory, load_cases = _resolve_adapter(args.adapter)()
    cases = load_cases(args.suite)
    system = build_system()
    predictions = asyncio.run(run_suite(cases, system, concurrency=args.concurrency))
    metrics = metrics_factory()
    results = tuple(m.compute(cases, predictions) for m in metrics)
    scorecard = Scorecard(label=args.label, metrics=results, n_cases=len(cases),
                          meta={"git_sha": git_sha()})
    write_baseline(args.baseline, scorecard)
    print(f"baseline written to {args.baseline}")
    return 0


def _guard_baseline_provider(*, allow_real: bool) -> None:
    """Refuse to write a real-provider baseline over the mock-vs-mock one.

    CI runs on llm_provider "mock" and gates every PR against this file, so a
    baseline captured from a real model turns the next PR into a wall of false
    regressions -- every mock number sits below the real one. That has already
    happened here once. CONTRIBUTING documented the rule; nothing enforced it,
    and `accept` silently uses whatever project.config.json currently says,
    which during real-model work is exactly the wrong thing.

    This is the enforcement. `--allow-real-baseline` is the deliberate override
    for a repo whose CI actually runs a real provider.
    """
    if allow_real:
        return
    try:
        from app.core.config import Settings
        provider = Settings.load().llm_provider
    except Exception:  # pragma: no cover - no config is not this check's problem
        return
    if provider != "mock":
        raise SystemExit(
            f'refusing to write a baseline captured with llm_provider "{provider}".\n'
            "eval/baseline.json is the mock-vs-mock reference CI compares every PR "
            "against; a real-provider baseline makes every subsequent CI run fail with "
            "regressions that are really just mock-vs-real gaps.\n\n"
            "Either set llm_provider to \"mock\" in project.config.json first, or pass "
            "--allow-real-baseline if your CI genuinely runs this provider."
        )


def calibrate(args: argparse.Namespace) -> int:
    from app.core.config import Settings

    from .judge import Judge
    from .judge import calibrate as run_calibration
    settings = Settings.load(args.config)
    judge = Judge(settings, model=args.judge_model)
    result = run_calibration(judge, args.subset)
    print(json.dumps(result, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m core.eval")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run a suite and score it")
    p_run.add_argument("--suite", required=True)
    p_run.add_argument("--adapter", required=True,
                       help="module:function returning (build_system, metrics_factory, load_cases)")
    p_run.add_argument("--out", default="results/")
    p_run.add_argument("--label", default="run")
    p_run.add_argument("--concurrency", type=int, default=4)
    p_run.add_argument("--baseline", default=None)
    p_run.add_argument("--regression-threshold", type=float, default=0.02)
    p_run.add_argument("--fail-on-regression", action="store_true")
    p_run.set_defaults(func=run)

    p_accept = sub.add_parser("accept", help="write the current run as the new baseline")
    p_accept.add_argument("--suite", required=True)
    p_accept.add_argument("--adapter", required=True)
    p_accept.add_argument("--label", default="baseline")
    p_accept.add_argument("--concurrency", type=int, default=4)
    p_accept.add_argument("--baseline", required=True)
    p_accept.add_argument("--allow-real-baseline", action="store_true",
                          help="write the baseline even though llm_provider is not 'mock' "
                               "(only correct if CI runs that same provider)")
    p_accept.set_defaults(func=accept)

    p_cal = sub.add_parser("calibrate", help="judge-vs-human kappa on a labelled subset")
    p_cal.add_argument("--subset", required=True)
    p_cal.add_argument("--judge-model", required=True)
    p_cal.add_argument("--config", default=None)
    p_cal.set_defaults(func=calibrate)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
