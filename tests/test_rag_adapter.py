from app.evaluation.rag_adapter import build_metrics, build_rag_system, load_golden_cases


def test_golden_set_loads_and_is_stratified():
    cases = load_golden_cases("eval/data/golden.jsonl")
    assert len(cases) >= 120
    types = {c.metadata["type"] for c in cases}
    assert {"factoid", "multi_hop", "aggregation", "unanswerable", "ambiguous"} <= types
    assert any(c.metadata["type"] == "unanswerable" for c in cases), \
        "a golden set without unanswerable cases measures nothing about refusal"


def test_unanswerable_cases_carry_no_gold_answer_to_grade_against():
    cases = load_golden_cases("eval/data/golden.jsonl")
    unanswerable = [c for c in cases if c.metadata["type"] == "unanswerable"]
    assert unanswerable
    assert all("gold_answer" not in c.expected for c in unanswerable)


def test_answerable_cases_carry_gold_doc_ids_and_gold_answer():
    cases = load_golden_cases("eval/data/golden.jsonl")
    factoid = [c for c in cases if c.metadata["type"] == "factoid"]
    assert factoid
    assert all(c.expected.get("gold_doc_ids") for c in factoid)
    assert all(c.expected.get("gold_answer") for c in factoid)


def test_run_case_against_the_mock_pipeline_produces_a_well_formed_prediction(pipeline):
    cases = load_golden_cases("eval/data/golden.jsonl")
    case = next(c for c in cases if c.id == "f-01")
    run_case = build_rag_system(settings=pipeline.settings)

    pred = run_case(case)

    assert pred.case_id == case.id
    assert pred.error is None
    assert isinstance(pred.output["retrieved_doc_ids"], list)
    assert isinstance(pred.output["refused"], bool)
    assert pred.latency_ms >= 0
    assert "retrieve_ms" in pred.trace and "answer_ms" in pred.trace


def test_run_case_refuses_an_unanswerable_question(pipeline):
    cases = load_golden_cases("eval/data/golden.jsonl")
    case = next(c for c in cases if c.metadata["type"] == "unanswerable")
    run_case = build_rag_system(settings=pipeline.settings)

    pred = run_case(case)

    assert pred.output["refused"] is True


def test_build_metrics_returns_the_full_battery(settings):
    metrics = build_metrics(settings=settings, judge_model="mock")
    names = {m.name for m in metrics}
    assert "mrr" in names
    assert "recall@10" in names
    assert "citation_precision" in names
    assert "refusal_accuracy" in names
    assert "answer_correctness" in names
    assert "faithfulness" in names
    assert "cost_per_task_usd" in names
    assert "latency_total_ms" in names
