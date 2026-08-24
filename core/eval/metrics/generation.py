from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from ..types import Case, MetricResult, Prediction
from .base import Metric, pair_by_id, summarize

if TYPE_CHECKING:
    from ..judge import Judge


class CitationPrecision(Metric):
    """Fraction of cited sources that are actually gold-relevant.

    A full judge-verified "does this citation support this specific claim" check is the
    textbook version of this metric; here it is approximated as "does the citation point
    at a source the gold answer actually depends on", which is cheap, deterministic, and
    catches the common failure (citing a plausible but wrong document) without a judge
    call per citation. Documented as a simplification in the README.
    """
    name = "citation_precision"
    higher_is_better = True

    def compute(self, cases: Sequence[Case], predictions: Sequence[Prediction]) -> MetricResult:
        per_case = []
        for case, pred in pair_by_id(cases, predictions):
            cited = list(pred.output.get("citations", []))
            if not cited:
                continue
            gold = set(case.expected.get("gold_doc_ids", []))
            correct = sum(1 for c in cited if c in gold)
            per_case.append(correct / len(cited))
        return summarize(self.name, per_case, higher_is_better=self.higher_is_better)


class CitationRecall(Metric):
    """Fraction of gold docs that appear among the cited sources."""
    name = "citation_recall"
    higher_is_better = True

    def compute(self, cases: Sequence[Case], predictions: Sequence[Prediction]) -> MetricResult:
        per_case = []
        for case, pred in pair_by_id(cases, predictions):
            if pred.output.get("asked_clarification"):
                # Correctly asking instead of guessing cites nothing by design --
                # that isn't a recall failure, it's the case working as intended.
                continue
            gold = set(case.expected.get("gold_doc_ids", []))
            if not gold:
                continue
            cited = set(pred.output.get("citations", []))
            per_case.append(len(gold & cited) / len(gold))
        return summarize(self.name, per_case, higher_is_better=self.higher_is_better)


class RefusalAccuracy(Metric):
    """Scored only on the unanswerable slice: did the system correctly decline instead
    of fabricating an answer?"""
    name = "refusal_accuracy"
    higher_is_better = True

    def compute(self, cases: Sequence[Case], predictions: Sequence[Prediction]) -> MetricResult:
        per_case = []
        for case, pred in pair_by_id(cases, predictions):
            if case.metadata.get("type") != "unanswerable":
                continue
            per_case.append(1.0 if pred.output.get("refused") else 0.0)
        return summarize(self.name, per_case, higher_is_better=self.higher_is_better)


class ClarificationRate(Metric):
    """Scored only on the ambiguous slice: did the system ask instead of guessing?"""
    name = "clarification_rate"
    higher_is_better = True

    def compute(self, cases: Sequence[Case], predictions: Sequence[Prediction]) -> MetricResult:
        per_case = []
        for case, pred in pair_by_id(cases, predictions):
            if case.metadata.get("type") != "ambiguous":
                continue
            per_case.append(1.0 if pred.output.get("asked_clarification") else 0.0)
        return summarize(self.name, per_case, higher_is_better=self.higher_is_better)


class AnswerCorrectness(Metric):
    """0-4 rubric, judged against gold_answer. Requires a Judge (see judge.py); the
    raw 0-4 score is normalised to [0, 1] so it sits on the same scale as every other
    metric in the scorecard."""
    name = "answer_correctness"
    higher_is_better = True

    def __init__(self, judge: Judge) -> None:
        self.judge = judge

    def compute(self, cases: Sequence[Case], predictions: Sequence[Prediction]) -> MetricResult:
        per_case = []
        disagreements = []
        for case, pred in pair_by_id(cases, predictions):
            gold = case.expected.get("gold_answer")
            if not gold:
                continue
            verdict = self.judge.score_correctness(
                question=case.input.get("question", ""), gold_answer=gold,
                candidate=pred.output.get("answer_text", ""))
            score = float(verdict.value.get("score", 0)) / 4.0
            per_case.append(max(0.0, min(1.0, score)))
            disagreements.append(verdict.disagreement_rate)
        extra = ({"mean_judge_disagreement": round(sum(disagreements) / len(disagreements), 4)}
                if disagreements else {})
        return summarize(self.name, per_case, higher_is_better=self.higher_is_better, extra=extra)


class Faithfulness(Metric):
    """Claim-level groundedness against RETRIEVED CONTEXT ONLY, never the gold answer --
    those measure different things (is the answer grounded vs. is the answer correct)."""
    name = "faithfulness"
    higher_is_better = True

    def __init__(self, judge: Judge) -> None:
        self.judge = judge

    def compute(self, cases: Sequence[Case], predictions: Sequence[Prediction]) -> MetricResult:
        per_case = []
        disagreements = []
        for _case, pred in pair_by_id(cases, predictions):
            if pred.output.get("refused") or pred.output.get("asked_clarification"):
                # Grading a refusal or a clarifying question for "groundedness in the
                # sources" isn't a meaningful check -- there's no claim being made to
                # verify -- and folding it in would dilute the signal on the cases
                # that matter.
                continue
            answer = pred.output.get("answer_text", "")
            context = pred.output.get("retrieved_context", "")
            if not answer or not context:
                continue
            verdict = self.judge.faithfulness(answer_text=answer, retrieved_context=context)
            per_case.append(max(0.0, min(1.0, float(verdict.value.get("supported", 0.0)))))
            disagreements.append(verdict.disagreement_rate)
        extra = ({"mean_judge_disagreement": round(sum(disagreements) / len(disagreements), 4)}
                if disagreements else {})
        return summarize(self.name, per_case, higher_is_better=self.higher_is_better, extra=extra)
