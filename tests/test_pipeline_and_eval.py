import json

from app.core.providers import MockLLM


def test_ingestion_is_idempotent_and_skips_unchanged_chunks(pipeline):
    first = pipeline.store.count()
    report = pipeline.ingest("data/sample")
    assert first > 0
    assert report.chunks_embedded == 0
    assert report.chunks_skipped_unchanged == report.chunks_seen


def test_pipeline_retrieves_the_right_document(pipeline):
    result = pipeline.ask("how long is the warranty on batteries")
    assert any("warranty" in h.chunk.source for h in result.hits)


def test_pipeline_result_serialises_for_the_api(pipeline):
    payload = pipeline.ask("what is the return window").as_dict()
    json.dumps(payload)
    assert {"question", "answer", "citations", "retrieved", "trace"} <= set(payload)


def test_a_clarifying_result_serialises_as_not_refused():
    from app.answer.generate import Answer
    from app.pipeline import RAGResult

    result = RAGResult(question="q", answer=Answer("Which tier are you on?", answered=False,
                                                    needs_clarification=True),
                       needs_clarification=True)
    payload = result.as_dict()
    assert payload["needs_clarification"] is True
    assert payload["refused"] is False
    assert payload["answer"] == "Which tier are you on?"


def test_budget_guard_stops_a_runaway_loop(settings, store):
    from app.core.providers import BudgetedLLM, Message
    from app.core.trace import Trace
    settings.max_llm_calls_per_run = 2
    trace = Trace(enabled=False)
    llm = BudgetedLLM(MockLLM(), settings, trace)
    llm.chat([Message.user("a")])
    llm.chat([Message.user("b")])
    try:
        llm.chat([Message.user("c")])
    except Exception as exc:
        assert "max_llm_calls_per_run" in str(exc)
    else:
        raise AssertionError("expected the budget guard to fire")


def test_bootstrap_from_corpus_drafts_reviewable_cases(pipeline):
    from app.evaluation.dataset import bootstrap_from_corpus
    chunks = pipeline.store.all_chunks()[:2]
    rows = list(bootstrap_from_corpus(chunks, pipeline.llm, limit=2))
    assert len(rows) == 2
    assert all(r["type"] == "factoid" and r["gold_doc_ids"] for r in rows)
