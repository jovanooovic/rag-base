import pytest

from app.answer.generate import Answer, answer_question, build_context
from app.answer.guardrails import check_answer, redact
from app.core.providers import LLMResponse, MockLLM, Usage
from app.ingest.chunking import Chunk
from app.store.base import ScoredChunk


def _hits(n=2):
    return [ScoredChunk(Chunk(f"c{i}", "d", f"body text {i}", f"src{i}.md", i), 0.9 - i * 0.1)
            for i in range(n)]


def test_context_is_numbered_from_one():
    ctx, cites = build_context(_hits(3))
    assert ctx.startswith("[1]")
    assert [c.number for c in cites] == [1, 2, 3]


def test_context_respects_the_char_budget():
    hits = [ScoredChunk(Chunk(f"c{i}", "d", "x" * 900, "s.md", i), 1.0) for i in range(10)]
    ctx, cites = build_context(hits, max_chars=2000)
    assert len(ctx) < 3000 and 0 < len(cites) < 10


def test_no_hits_yields_a_refusal_without_calling_the_model():
    llm = MockLLM()
    ans = answer_question(llm, "anything?", [])
    assert not ans.answered and llm.calls == []


def test_cited_numbers_are_parsed_and_out_of_range_ones_dropped():
    llm = MockLLM(scripted=[LLMResponse("Refunds take five days [1]. Also [2]. Bogus [9].", usage=Usage())])
    ans = answer_question(llm, "q", _hits(2))
    assert ans.used_citations == [1, 2]


def test_answer_with_no_citations_is_flagged_unsupported():
    llm = MockLLM(scripted=[LLMResponse("Refunds take five days.", usage=Usage())])
    ans = answer_question(llm, "q", _hits(2))
    assert ans.unsupported


def test_guardrail_blocks_uncited_answers():
    result = check_answer(Answer("Confident claim with no citation.", used_citations=[]), _hits())
    assert not result.ok and "citation" in result.reason


def test_guardrail_blocks_when_retrieval_score_is_below_the_floor():
    weak = [ScoredChunk(Chunk("c", "d", "t", "s", 0), 0.05)]
    result = check_answer(Answer("Answer [1].", used_citations=[1]), weak, min_top_score=0.3)
    assert not result.ok and "below floor" in result.reason


def test_guardrail_passes_a_well_cited_answer():
    result = check_answer(Answer("Answer [1].", used_citations=[1]), _hits(), min_top_score=0.3)
    assert result.ok and result.answer.text == "Answer [1]."


def test_model_refusal_is_converted_to_a_human_message():
    result = check_answer(Answer("NOT_IN_SOURCES need the pricing doc", answered=False), _hits())
    assert not result.ok and "NOT_IN_SOURCES" not in (result.answer.text)


@pytest.mark.parametrize("raw,label", [
    ("mail me at ana@example.com", "EMAIL"),
    ("card 4111 1111 1111 1111 please", "CARD"),
    ("IBAN DE89370400440532013000 here", "IBAN"),
    ("ssn 123-45-6789", "SSN"),
])
def test_redaction_catches_common_pii(raw, label):
    out = redact(raw)
    assert f"<{label}>" in out
