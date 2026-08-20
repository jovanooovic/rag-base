import math

from core.eval.metrics.retrieval import HitRateAtK, MRR, NDCGAtK, PrecisionAtK, RecallAtK
from core.eval.types import Case, Prediction


def _case(id_, gold):
    return Case(id=id_, input={}, expected={"gold_doc_ids": gold})


def _pred(id_, retrieved):
    return Prediction(case_id=id_, output={"retrieved_doc_ids": retrieved})


def test_recall_at_k_hand_computed():
    cases = [_case("1", ["a", "b"])]
    preds = [_pred("1", ["a", "x", "y"])]
    assert RecallAtK(3).compute(cases, preds).value == 0.5


def test_precision_at_k_hand_computed():
    cases = [_case("1", ["a"])]
    preds = [_pred("1", ["a", "x", "y"])]
    assert math.isclose(PrecisionAtK(3).compute(cases, preds).value, 1 / 3, abs_tol=1e-4)


def test_hit_rate_is_binary_per_case():
    cases = [_case("1", ["a"]), _case("2", ["z"])]
    preds = [_pred("1", ["a", "b"]), _pred("2", ["a", "b"])]
    assert HitRateAtK(2).compute(cases, preds).value == 0.5


def test_mrr_hand_computed():
    cases = [_case("1", ["b"])]
    preds = [_pred("1", ["a", "b", "c"])]  # gold at rank 2
    assert MRR().compute(cases, preds).value == 0.5


def test_mrr_is_zero_when_gold_not_retrieved():
    cases = [_case("1", ["z"])]
    preds = [_pred("1", ["a", "b"])]
    assert MRR().compute(cases, preds).value == 0.0


def test_ndcg_hand_computed():
    cases = [_case("1", ["a", "b"])]
    preds = [_pred("1", ["x", "a", "b"])]  # gold at ranks 2, 3
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    expected = dcg / idcg
    assert math.isclose(NDCGAtK(3).compute(cases, preds).value, expected, abs_tol=1e-4)


def test_metrics_skip_cases_without_gold_ids():
    cases = [_case("1", [])]
    preds = [_pred("1", ["a"])]
    assert RecallAtK(5).compute(cases, preds).n == 0
