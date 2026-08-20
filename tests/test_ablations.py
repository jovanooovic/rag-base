import json

from eval.ablations import _build_reranker, _retrieved_doc_ids, render_markdown, run_matrix


def test_retrieval_weights_cover_every_mode():
    from eval.ablations import _RETRIEVAL_WEIGHTS, RETRIEVAL_MODES
    assert set(_RETRIEVAL_WEIGHTS) == set(RETRIEVAL_MODES)
    assert _RETRIEVAL_WEIGHTS["dense-only"] == (1.0, 0.0)
    assert _RETRIEVAL_WEIGHTS["bm25-only"] == (0.0, 1.0)
    assert _RETRIEVAL_WEIGHTS["hybrid-rrf"] == (1.0, 1.0)


def test_build_reranker_off_returns_none_with_no_skip_reason():
    reranker, skip_reason = _build_reranker("off")
    assert reranker is None
    assert skip_reason is None


def test_build_reranker_cross_encoder_skips_cleanly_without_the_optional_dependency():
    reranker, skip_reason = _build_reranker("cross-encoder")
    # sentence-transformers is not installed in this environment -- exercising the
    # skip path is the point, not the happy path.
    assert reranker is None
    assert skip_reason is not None and "sentence-transformers" in skip_reason


def test_retrieved_doc_ids_dedupes_by_first_occurrence():
    class _Chunk:
        def __init__(self, source):
            self.source = source

    class _Hit:
        def __init__(self, source):
            self.chunk = _Chunk(source)

    hits = [_Hit("a.md"), _Hit("b.md"), _Hit("a.md"), _Hit("c.md")]
    assert _retrieved_doc_ids(hits) == ["a.md", "b.md", "c.md"]


def test_render_markdown_marks_skipped_rows():
    rows = [
        {"chunking": "structure-first", "retrieval": "hybrid-rrf", "reranker": "off",
         "recall@5": 0.9, "recall@10": 1.0, "mrr": 0.8, "ndcg@5": 0.85},
        {"chunking": "structure-first", "retrieval": "hybrid-rrf", "reranker": "cross-encoder",
         "skipped": "sentence-transformers not installed"},
    ]
    md = render_markdown(rows)
    assert "0.9000" in md
    assert "_skipped: sentence-transformers not installed_" in md


def test_run_matrix_end_to_end_on_a_narrow_slice(tmp_path, settings):
    golden = tmp_path / "golden.jsonl"
    golden.write_text(json.dumps({
        "id": "t-1", "question": "how long is the warranty on batteries?",
        "gold_doc_ids": ["data/sample/warranty.md"], "gold_answer": "12 months",
        "type": "factoid", "difficulty": "easy",
    }))

    rows = run_matrix(settings, corpus_path="data/sample", golden_path=str(golden),
                      chunkings=("structure-first",), retrievals=("hybrid-rrf",),
                      rerankers=("off",))

    assert len(rows) == 1
    row = rows[0]
    assert row["chunking"] == "structure-first"
    assert row["retrieval"] == "hybrid-rrf"
    assert row["reranker"] == "off"
    assert "skipped" not in row
    assert row["n_chunks"] > 0
    assert 0.0 <= row["recall@5"] <= 1.0
