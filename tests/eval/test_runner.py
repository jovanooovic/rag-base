import asyncio
import time

from core.eval.runner import content_hash, git_sha, run_suite
from core.eval.types import Case, Prediction


def test_run_suite_preserves_input_order_regardless_of_completion_speed():
    cases = [Case(id=str(i), input={"delay": (3 - i) * 0.01}) for i in range(4)]

    def system(case: Case) -> Prediction:
        time.sleep(case.input["delay"])
        return Prediction(case_id=case.id, output={})

    results = asyncio.run(run_suite(cases, system, concurrency=4))
    assert [r.case_id for r in results] == ["0", "1", "2", "3"]


def test_run_suite_respects_concurrency_limit():
    in_flight = {"current": 0, "max": 0}

    def system(case: Case) -> Prediction:
        in_flight["current"] += 1
        in_flight["max"] = max(in_flight["max"], in_flight["current"])
        time.sleep(0.02)
        in_flight["current"] -= 1
        return Prediction(case_id=case.id, output={})

    cases = [Case(id=str(i), input={}) for i in range(6)]
    asyncio.run(run_suite(cases, system, concurrency=2))
    assert in_flight["max"] <= 2


def test_run_suite_captures_a_case_error_without_killing_the_run():
    def system(case: Case) -> Prediction:
        if case.id == "1":
            raise ValueError("boom")
        return Prediction(case_id=case.id, output={})

    cases = [Case(id="0", input={}), Case(id="1", input={})]
    results = asyncio.run(run_suite(cases, system, concurrency=2))
    assert results[0].error is None
    assert results[1].error is not None and "boom" in results[1].error


def test_content_hash_changes_when_dataset_changes():
    h1 = content_hash(b"data1", {"k": "v"}, "sha1")
    h2 = content_hash(b"data2", {"k": "v"}, "sha1")
    assert h1 != h2


def test_content_hash_is_stable():
    h1 = content_hash(b"data", {"k": "v"}, "sha1")
    h2 = content_hash(b"data", {"k": "v"}, "sha1")
    assert h1 == h2


def test_git_sha_never_raises():
    assert isinstance(git_sha(), str)
