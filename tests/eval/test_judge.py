from app.core.providers import LLMResponse, MockLLM, Usage
from core.eval.judge import Judge, JudgeCache, calibrate


def _scripted(*texts):
    return MockLLM(scripted=[LLMResponse(text=t, usage=Usage()) for t in texts])


def test_judge_caches_identical_calls(tmp_path):
    llm = _scripted('{"score": 3}', '{"score": 3}')  # second should never be consumed
    cache = JudgeCache(tmp_path / "cache.sqlite")
    judge = Judge(model="test-model", llm=llm, votes=1, cache=cache)
    judge.score_correctness(question="q", gold_answer="g", candidate="c")
    calls_after_first = len(llm.calls)
    judge.score_correctness(question="q", gold_answer="g", candidate="c")
    assert len(llm.calls) == calls_after_first


def test_judge_majority_vote_and_disagreement_rate(tmp_path):
    llm = _scripted('{"score": 4}', '{"score": 4}', '{"score": 1}')
    judge = Judge(model="test-model", llm=llm, votes=3, cache=JudgeCache(tmp_path / "c.sqlite"))
    verdict = judge.score_correctness(question="q", gold_answer="g", candidate="c")
    assert verdict.value["score"] == 4  # 2-of-3 majority
    assert round(verdict.disagreement_rate, 4) == round(1 / 3, 4)


def test_judge_records_the_pinned_model_string(tmp_path):
    llm = _scripted('{"score": 2}')
    judge = Judge(model="pinned-model-v1", llm=llm, votes=1, cache=JudgeCache(tmp_path / "c.sqlite"))
    verdict = judge.score_correctness(question="q", gold_answer="g", candidate="c")
    assert verdict.model == "pinned-model-v1"


def test_judge_tolerates_unparseable_output(tmp_path):
    llm = _scripted("not json at all")
    judge = Judge(model="test-model", llm=llm, votes=1, cache=JudgeCache(tmp_path / "c.sqlite"))
    verdict = judge.score_correctness(question="q", gold_answer="g", candidate="c")
    assert verdict.value.get("_unparseable") is True


def test_calibrate_reports_pending_with_no_human_verdicts(tmp_path):
    subset = tmp_path / "subset.jsonl"
    subset.write_text('{"case_id": "1", "question": "q", "gold_answer": "g", '
                      '"candidate_answer": "c", "human_verdict": null}')
    llm = _scripted('{"score": 3}')
    judge = Judge(model="test-model", llm=llm, votes=1, cache=JudgeCache(tmp_path / "c.sqlite"))
    result = calibrate(judge, subset)
    assert result["kappa"] is None
    assert result["n_labelled"] == 0


def test_calibrate_computes_kappa_when_human_verdicts_are_present(tmp_path):
    subset = tmp_path / "subset.jsonl"
    subset.write_text(
        '{"case_id": "1", "question": "q", "gold_answer": "g", "candidate_answer": "c", "human_verdict": "4"}\n'
        '{"case_id": "2", "question": "q2", "gold_answer": "g2", "candidate_answer": "c2", "human_verdict": "4"}'
    )
    llm = _scripted('{"score": 4}', '{"score": 4}')
    judge = Judge(model="test-model", llm=llm, votes=1, cache=JudgeCache(tmp_path / "c.sqlite"))
    result = calibrate(judge, subset)
    assert result["kappa"] == 1.0
    assert result["n_labelled"] == 2


def test_mock_provider_gives_correctness_judge_meaningful_signal(tmp_path):
    """Without the MockLLM judge fallback, this would always score 0 -- the real MockLLM
    (unscripted) is used here to exercise that fallback path end to end."""
    judge = Judge(model="mock", llm=MockLLM("mock"), votes=1, cache=JudgeCache(tmp_path / "c.sqlite"))
    close = judge.score_correctness(question="q", gold_answer="the warranty is 24 months",
                                    candidate="the warranty lasts 24 months")
    unrelated = judge.score_correctness(question="q", gold_answer="the warranty is 24 months",
                                        candidate="bananas are yellow fruit")
    assert close.value.get("score", 0) > unrelated.value.get("score", 0)


def test_mock_provider_gives_faithfulness_judge_meaningful_signal(tmp_path):
    judge = Judge(model="mock", llm=MockLLM("mock"), votes=1, cache=JudgeCache(tmp_path / "c.sqlite"))
    grounded = judge.faithfulness(answer_text="the warranty is 24 months",
                                  retrieved_context="Acme hardware carries a 24 month warranty")
    ungrounded = judge.faithfulness(answer_text="the warranty is 24 months",
                                    retrieved_context="bananas are a good source of potassium")
    assert grounded.value.get("supported", 0) > ungrounded.value.get("supported", 0)
