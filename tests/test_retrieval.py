import pytest

from app.core.providers import MockEmbeddings
from app.ingest.chunking import Chunk
from app.retrieve.bm25 import BM25
from app.retrieve.hybrid import HybridRetriever, reciprocal_rank_fusion
from app.store.base import ScoredChunk


def _chunk(cid, text):
    return Chunk(chunk_id=cid, doc_id="d", text=text, source="s", ordinal=0)


def test_bm25_finds_exact_identifiers_vectors_would_miss():
    corpus = ["the refund policy lasts thirty days",
              "shipping is handled by our partner",
              "order ACM-4417291 was cancelled by the customer"]
    scores = BM25(corpus).score("ACM-4417291")
    assert scores[2] > 0
    assert scores[0] == scores[1] == 0


def test_bm25_penalises_long_documents_for_the_same_term_count():
    short = "refund refund"
    long = "refund refund " + "filler word here " * 100
    scores = BM25([short, long]).score("refund")
    assert scores[0] > scores[1]


def test_rrf_prefers_a_chunk_ranked_well_by_both_legs():
    a, b, c = _chunk("a", "x"), _chunk("b", "y"), _chunk("c", "z")
    vector_leg = [ScoredChunk(a, 0.9), ScoredChunk(b, 0.8), ScoredChunk(c, 0.1)]
    keyword_leg = [ScoredChunk(c, 9.0), ScoredChunk(a, 4.0)]
    fused = reciprocal_rank_fusion([vector_leg, keyword_leg])
    assert fused[0].chunk.chunk_id == "a", "a is top-2 in both legs; c is top only in one"


def test_rrf_is_immune_to_scale_differences_between_legs():
    a, b = _chunk("a", "x"), _chunk("b", "y")
    tiny = [ScoredChunk(a, 0.0001), ScoredChunk(b, 0.00001)]
    huge = [ScoredChunk(a, 5000.0), ScoredChunk(b, 4000.0)]
    assert [s.chunk.chunk_id for s in reciprocal_rank_fusion([tiny, huge])] == ["a", "b"]


def test_rrf_records_which_leg_contributed():
    a = _chunk("a", "x")
    fused = reciprocal_rank_fusion([[ScoredChunk(a, 1.0, {"vector": 1.0})],
                                    [ScoredChunk(a, 2.0, {"bm25": 2.0})]])
    assert {"vector", "bm25"} <= set(fused[0].signals)


def test_hybrid_retriever_returns_the_matching_chunk(store):
    chunks = [_chunk("c1", "Refunds are issued within five business days."),
              _chunk("c2", "Batteries are covered for twelve months only."),
              _chunk("c3", "Express delivery costs an extra twelve euros.")]
    emb = MockEmbeddings(256)
    store.upsert(chunks, emb.embed([c.text for c in chunks]))
    r = HybridRetriever(store, emb, top_k=1, fetch_k=10)
    assert r.retrieve("how long do batteries stay covered")[0].chunk.chunk_id == "c2"


def test_store_upsert_is_idempotent(store):
    c = _chunk("c1", "hello world")
    emb = MockEmbeddings(256)
    store.upsert([c], emb.embed(["hello world"]))
    store.upsert([c], emb.embed(["hello world"]))
    assert store.count() == 1


def test_store_metadata_filter(store):
    emb = MockEmbeddings(256)
    a, b = _chunk("a", "public policy text"), _chunk("b", "public policy text")
    a.metadata = {"tenant": "acme"}
    b.metadata = {"tenant": "other"}
    store.upsert([a, b], emb.embed([a.text, b.text]))
    hits = store.search(emb.embed(["policy"])[0], k=10, where={"tenant": "acme"})
    assert [h.chunk.chunk_id for h in hits] == ["a"]
