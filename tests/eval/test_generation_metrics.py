from core.eval.judge import JudgeVerdict
from core.eval.metrics.generation import (AnswerCorrectness, CitationPrecision, CitationRecall,
                                          ClarificationRate, Faithfulness, RefusalAccuracy)
from core.eval.types import Case, Prediction


def test_citation_precision_hand_computed():
    case = Case(id="1", input={}, expected={"gold_doc_ids": ["a"]})
    pred = Prediction(case_id="1", output={"citations": ["a", "b"]})
    assert CitationPrecision().compute([case], [pred]).value == 0.5


def test_citation_recall_hand_computed():
    case = Case(id="1", input={}, expected={"gold_doc_ids": ["a", "b"]})
    pred = Prediction(case_id="1", output={"citations": ["a"]})
    assert CitationRecall().compute([case], [pred]).value == 0.5


def test_refusal_accuracy_only_scores_unanswerable_slice():
    cases = [Case(id="1", input={}, metadata={"type": "unanswerable"}),
             Case(id="2", input={}, metadata={"type": "factoid"})]
    preds = [Prediction(case_id="1", output={"refused": True}),
             Prediction(case_id="2", output={"refused": True})]  # must not count
    result = RefusalAccuracy().compute(cases, preds)
    assert result.n == 1
    assert result.value == 1.0


def test_clarification_rate_only_scores_ambiguous_slice():
    cases = [Case(id="1", input={}, metadata={"type": "ambiguous"})]
    preds = [Prediction(case_id="1", output={"asked_clarification": False})]
    result = ClarificationRate().compute(cases, preds)
    assert result.n == 1
    assert result.value == 0.0


class _FakeJudge:
    def __init__(self, score=4, disagreement=0.0):
        self.score = score
        self.disagreement = disagreement

    def score_correctness(self, **kw):
        return JudgeVerdict(value={"score": self.score}, raw_votes=[],
                            disagreement_rate=self.disagreement, model="fake")

    def faithfulness(self, **kw):
        return JudgeVerdict(value={"supported": 1.0}, raw_votes=[],
                            disagreement_rate=self.disagreement, model="fake")


def test_answer_correctness_normalises_rubric_to_unit_interval():
    case = Case(id="1", input={"question": "q"}, expected={"gold_answer": "gold"})
    pred = Prediction(case_id="1", output={"answer_text": "candidate"})
    result = AnswerCorrectness(_FakeJudge(score=2)).compute([case], [pred])
    assert result.value == 0.5


def test_faithfulness_reads_supported_fraction():
    case = Case(id="1", input={}, expected={})
    pred = Prediction(case_id="1", output={"answer_text": "a", "retrieved_context": "ctx"})
    result = Faithfulness(_FakeJudge()).compute([case], [pred])
    assert result.value == 1.0


def test_faithfulness_skips_refused_cases():
    case = Case(id="1", input={}, expected={})
    pred = Prediction(case_id="1", output={"answer_text": "I could not find this in the sources.",
                                           "retrieved_context": "ctx", "refused": True})
    result = Faithfulness(_FakeJudge()).compute([case], [pred])
    assert result.n == 0
