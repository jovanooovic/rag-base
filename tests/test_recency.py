from datetime import date

import pytest

from app.ingest.chunking import Chunk
from app.ingest.loaders import load_path, parse_effective_date, split_front_matter
from app.retrieve.hybrid import apply_recency
from app.store.base import ScoredChunk


def _hit(source: str, score: float, effective_date: str | None,
         confidence: float | None = None) -> ScoredChunk:
    """A hit as a reranker hands it over: a raw score plus the normalised
    confidence every scorer publishes."""
    meta = {"effective_date": effective_date} if effective_date else {}
    sc = ScoredChunk(Chunk(f"{source}#0", source, "body", source, 0, metadata=meta), score)
    sc.signals["confidence"] = score / 10.0 if confidence is None else confidence
    return sc


# ---------- front matter ----------

def test_front_matter_is_parsed_and_removed_from_the_text():
    """Left in the body, _sections() turns it into a junk chunk with an empty
    heading that then competes for retrieval against real content."""
    meta, body = split_front_matter("---\neffective_date: 2026-06-01\n---\n# Policy\n\nBody.\n")

    assert meta["effective_date"] == "2026-06-01"
    assert body.startswith("# Policy")
    assert "effective_date" not in body


def test_text_without_front_matter_is_untouched():
    meta, body = split_front_matter("# Policy\n\nBody.\n")
    assert meta == {} and body == "# Policy\n\nBody.\n"


def test_a_horizontal_rule_is_not_mistaken_for_front_matter():
    """Markdown uses --- for rules too; only a block at the very start counts."""
    text = "# Policy\n\nBody.\n\n---\n\nMore body.\n"
    meta, body = split_front_matter(text)
    assert meta == {} and body == text


@pytest.mark.parametrize("raw", ["2026-13-01", "01/06/2026", "yesterday", "2026-06-31"])
def test_a_malformed_date_is_rejected_rather_than_guessed(raw):
    assert parse_effective_date(raw) is None


def test_ingest_reads_effective_date_from_front_matter(tmp_path):
    doc = tmp_path / "policy.md"
    doc.write_text("---\neffective_date: 2026-06-01\n---\n# Policy\n\nReturns take 30 days.\n",
                   encoding="utf-8")

    loaded = next(iter(load_path(doc)))

    assert loaded.metadata["effective_date"] == "2026-06-01"
    assert loaded.metadata["date_source"] == "front_matter"


def test_ingest_falls_back_to_mtime_without_front_matter(tmp_path):
    doc = tmp_path / "policy.md"
    doc.write_text("# Policy\n\nReturns take 30 days.\n", encoding="utf-8")

    loaded = next(iter(load_path(doc)))

    assert loaded.metadata["date_source"] == "mtime"
    assert loaded.metadata["effective_date"]


def test_a_bad_effective_date_fails_loudly_rather_than_looking_undated(tmp_path):
    doc = tmp_path / "policy.md"
    doc.write_text("---\neffective_date: last Tuesday\n---\n# P\n\nBody text here.\n",
                   encoding="utf-8")

    with pytest.raises(ValueError, match="not an ISO date"):
        list(load_path(doc))


# ---------- ranking ----------

def test_recency_weight_zero_is_exactly_a_no_op():
    """The proof that turning the feature off restores the previous behaviour
    byte for byte -- without it, the A/B comparison measures nothing."""
    hits = [_hit("a.md", 9.0, "2020-01-01"), _hit("b.md", 8.0, "2026-06-01")]
    before = [(h.chunk.source, h.score) for h in hits]

    after = apply_recency(hits, weight=0.0, now=date(2026, 8, 30))

    assert [(h.chunk.source, h.score) for h in after] == before


def test_a_fresher_document_outranks_a_stale_one_of_similar_relevance():
    """The superseded-policy case: both documents answer the question well, and
    only the newer one is still true."""
    stale = _hit("returns-2024.md", 9.0, "2024-01-15")
    fresh = _hit("returns-2026.md", 8.5, "2026-06-01")

    ranked = apply_recency([stale, fresh], weight=0.5, now=date(2026, 8, 30))

    assert ranked[0].chunk.source == "returns-2026.md"


def test_relevance_comes_from_confidence_not_the_spread_of_the_result_set():
    """Regression on the first implementation: min-max normalising .score
    across the hits maps any two candidates to exactly 0.0 and 1.0, so a
    trivial rubric gap became the maximum possible difference and recency could
    never move anything."""
    stale = _hit("old.md", 9.0, "2024-01-15", confidence=1.0)
    fresh = _hit("new.md", 8.9, "2026-06-01", confidence=1.0)

    ranked = apply_recency([stale, fresh], weight=0.4, now=date(2026, 8, 30))

    assert ranked[0].chunk.source == "new.md", (
        "equally confident documents must be separated by date, not by a "
        "hundredth of a point of raw score")


def test_recency_cannot_promote_an_irrelevant_document():
    """Freshness is a tiebreaker among plausible answers, not a way for an
    unrelated but recent page to reach the top."""
    relevant = _hit("returns.md", 10.0, "2019-01-01")
    irrelevant = _hit("careers.md", 0.0, "2026-08-01")

    ranked = apply_recency([relevant, irrelevant], weight=0.3, now=date(2026, 8, 30))

    assert ranked[0].chunk.source == "returns.md"


def test_an_undated_document_is_not_treated_as_infinitely_old():
    """Sinking documents for missing metadata would quietly bury a corpus that
    simply has no dates anywhere."""
    undated = _hit("a.md", 9.0, None)
    ancient = _hit("b.md", 9.0, "2005-01-01")

    ranked = apply_recency([undated, ancient], weight=0.5, now=date(2026, 8, 30))

    assert ranked[0].chunk.source == "a.md"


def test_recency_leaves_confidence_alone():
    """Confidence answers 'does this passage answer the question'. A document
    does not become a worse answer by being older -- only its ordering changes."""
    hit = _hit("a.md", 9.0, "2019-01-01")
    hit.signals["confidence"] = 0.9

    apply_recency([hit], weight=0.9, now=date(2026, 8, 30))

    assert hit.signals["confidence"] == 0.9
