import math

from core.eval.metrics.retrieval import (HitRateAtK, MRR, NDCGAtK, PrecisionAtK, RecallAtK,
                                         retrieval_metrics)
from core.eval.types import Case, Prediction


def _case(id_, gold):
    return Case(id=id_, input={}, expected={"gold_doc_ids": gold})


def _pred(id_, retrieved, field="retrieved_doc_ids"):
    return Prediction(case_id=id_, output={field: retrieved})


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


def test_field_and_suffix_score_a_different_pipeline_stage_without_name_collision():
    # A prediction can carry two different rankings (pre- and post-rerank) under two
    # different output keys; the same metric class should score whichever is named.
    cases = [_case("1", ["a"])]
    preds = [Prediction(case_id="1", output={"retrieved_doc_ids": ["z", "a"],
                                             "reranked_doc_ids": ["a", "z"]})]
    pre = RecallAtK(1, field="retrieved_doc_ids").compute(cases, preds)
    post = RecallAtK(1, field="reranked_doc_ids", name="recall@1_reranked").compute(cases, preds)
    assert pre.value == 0.0    # "a" is not in the top 1 of the pre-rerank list
    assert post.value == 1.0   # but it is in the top 1 of the reranked list
    assert post.name == "recall@1_reranked"


def test_retrieval_metrics_factory_applies_field_and_suffix_to_every_metric():
    metrics = retrieval_metrics(ks=(1,), field="reranked_doc_ids", suffix="_reranked")
    names = {m.name for m in metrics}
    assert names == {"mrr_reranked", "recall@1_reranked", "precision@1_reranked",
                     "hit_rate@1_reranked", "ndcg@1_reranked"}
