import math

from core.eval.stats import bootstrap_ci, cohens_kappa, confusion_matrix


def test_bootstrap_ci_is_deterministic_and_contains_the_mean():
    values = [0.0, 1.0, 1.0, 1.0, 0.0]
    lo, hi = bootstrap_ci(values, seed=0)
    mean = sum(values) / len(values)
    assert lo <= mean <= hi
    lo2, hi2 = bootstrap_ci(values, seed=0)
    assert (lo, hi) == (lo2, hi2)


def test_bootstrap_ci_narrows_with_more_data():
    small = bootstrap_ci([1.0, 0.0], seed=1)
    large = bootstrap_ci([1.0, 0.0] * 100, seed=1)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_bootstrap_ci_single_value_is_a_point():
    assert bootstrap_ci([0.7]) == (0.7, 0.7)


def test_bootstrap_ci_empty_is_zero():
    assert bootstrap_ci([]) == (0.0, 0.0)


def test_cohens_kappa_perfect_agreement_is_one():
    pairs = [("a", "a"), ("b", "b"), ("a", "a"), ("b", "b")]
    assert cohens_kappa(pairs) == 1.0


def test_cohens_kappa_hand_computed():
    # agree 8/10; row totals a=7,b=3; col totals a=7,b=3
    pairs = [("a", "a")] * 6 + [("b", "b")] * 2 + [("a", "b")] * 1 + [("b", "a")] * 1
    p_o = 0.8
    p_e = (7 * 7 + 3 * 3) / 100
    expected = (p_o - p_e) / (1 - p_e)
    assert math.isclose(cohens_kappa(pairs), expected, rel_tol=1e-9)


def test_cohens_kappa_empty_is_zero():
    assert cohens_kappa([]) == 0.0


def test_confusion_matrix_counts_pairs():
    pairs = [("a", "a"), ("a", "b"), ("b", "b")]
    assert confusion_matrix(pairs) == {"a": {"a": 1, "b": 1}, "b": {"b": 1}}
