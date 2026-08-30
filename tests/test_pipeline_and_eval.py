import json

from app.core.providers import MockLLM
from app.pipeline import RAGPipeline


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


def test_redaction_covers_excerpts_not_just_the_answer_text(settings, store):
    """The response ships verbatim source excerpts for the citation popovers, so
    redacting only answer text left the PII one field further down the same JSON
    -- and the excerpt is drawn from the document, which is where it lives."""
    import json as _json
    from pathlib import Path as _Path

    corpus = _Path(settings.data_dir) / "pii"
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "contacts.md").write_text(
        "# Escalation contacts\n\nFor billing disputes email ana.petrovic@acme-example.com "
        "or call 060 123 4567 with your account number.\n", encoding="utf-8")

    settings.extra["redact_pii"] = True
    p = RAGPipeline(settings, store=store)
    p.ingest(corpus)

    result = p.ask("who do I email about billing disputes").as_dict()
    payload = _json.dumps(result)

    # Guard against a vacuous pass: if retrieval returned nothing there would be
    # no excerpt to leak and the assertions below would hold trivially.
    assert result["retrieved"], "no excerpt in the response -- test proves nothing"
    assert "ana.petrovic@acme-example.com" not in payload
    assert "<EMAIL>" in payload


def test_redaction_stays_off_by_default(settings, store):
    import json as _json
    from pathlib import Path as _Path

    corpus = _Path(settings.data_dir) / "pii_off"
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "contacts.md").write_text(
        "# Escalation contacts\n\nEmail ana.petrovic@acme-example.com for billing disputes.\n",
        encoding="utf-8")

    p = RAGPipeline(settings, store=store)
    p.ingest(corpus)

    payload = _json.dumps(p.ask("who do I email about billing disputes").as_dict())

    assert "ana.petrovic@acme-example.com" in payload
