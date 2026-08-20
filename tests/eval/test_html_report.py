from core.eval.html_report import _case_score, _slice_breakdown, _worst_cases, render


def _entry(case_id, case_type, gold=("a.md",), retrieved=("a.md",), citations=("a.md",),
          refused=False, error=None):
    return {
        "case": {"id": case_id, "input": {"question": f"q {case_id}"},
                 "expected": {"gold_doc_ids": list(gold)}, "metadata": {"type": case_type}},
        "prediction": {"error": error, "output": {
            "retrieved_doc_ids": list(retrieved), "citations": list(citations),
            "refused": refused, "answer_text": "an answer", "retrieved_context": "some context",
        }},
    }


def test_case_score_full_credit_when_gold_retrieved_and_cited():
    entry = _entry("1", "factoid")
    assert _case_score(entry) == 1.0


def test_case_score_zero_when_prediction_errored():
    entry = _entry("1", "factoid", error="boom")
    assert _case_score(entry) == 0.0


def test_case_score_unanswerable_rewards_refusal():
    good = _entry("1", "unanswerable", gold=(), refused=True)
    bad = _entry("2", "unanswerable", gold=(), refused=False)
    assert _case_score(good) == 1.0
    assert _case_score(bad) == 0.0


def test_case_score_none_when_no_gold_and_not_unanswerable_or_ambiguous():
    entry = _entry("1", "factoid", gold=())
    assert _case_score(entry) is None


def test_worst_cases_sorts_ascending_and_caps_at_n():
    entries = [_entry(str(i), "factoid", retrieved=() if i < 3 else ("a.md",)) for i in range(15)]
    worst = _worst_cases(entries, n=5)
    assert len(worst) == 5
    assert all(w["score"] <= worst[-1]["score"] for w in worst)


def test_slice_breakdown_groups_by_type():
    entries = [_entry("1", "factoid"), _entry("2", "factoid", retrieved=()),
              _entry("3", "unanswerable", gold=(), refused=True)]
    slices = _slice_breakdown(entries)
    by_type = {s["type"]: s for s in slices}
    assert by_type["factoid"]["n"] == 2
    assert by_type["unanswerable"]["n"] == 1
    assert by_type["unanswerable"]["avg_score"] == 1.0


def test_render_produces_self_contained_html_with_no_external_requests():
    scorecard = {"label": "test-run", "n_cases": 2, "meta": {"kappa": None},
                "metrics": [{"name": "recall@5", "value": 0.9, "ci": [0.8, 1.0], "n": 2,
                            "delta": None}]}
    ablations = [{"chunking": "structure-first", "retrieval": "hybrid-rrf", "reranker": "off",
                 "recall@5": 0.9, "recall@10": 1.0, "mrr": 0.8, "ndcg@5": 0.85}]
    cases = [_entry("1", "factoid")]

    out = render(scorecard, ablations, cases)

    assert "<!doctype html>" in out.lower()
    assert "http://" not in out and "https://" not in out
    assert "test-run" in out
    assert "structure-first" in out
    assert "judge-vs-human kappa" in out and "pending" in out
