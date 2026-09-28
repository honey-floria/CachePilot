from benchmarks.progressive_load import _step_summary, classify_result, summarize_safety


def test_classify_result_distinguishes_protection_and_oom():
    assert classify_result(200) == "success"
    assert classify_result(429) == "overload_protected"
    assert classify_result(500, "CUDA out of memory") == "oom"
    assert classify_result(None, exception="timed out") == "timeout"


def test_step_is_safe_only_when_every_request_succeeds():
    results = [
        {"classification": "success", "request_id": "a"},
        {"classification": "overload_protected", "request_id": "b"},
    ]
    summary = _step_summary(results, 128, 2, 1)
    assert summary["safe"] is False
    assert summary["counts"] == {"success": 1, "overload_protected": 1}


def test_matrix_point_requires_every_repetition_to_succeed():
    steps = [
        _step_summary([{"classification": "success"}], 128, 1, 1),
        _step_summary([{"classification": "failed"}], 128, 1, 2),
        _step_summary([{"classification": "success"}], 128, 1, 3),
    ]
    points = summarize_safety(steps, repetitions=3)
    assert points[0]["safe"] is False
    assert points[0]["completed_repetitions"] == 3


def test_matrix_point_rejects_missing_repetitions():
    steps = [_step_summary([{"classification": "success"}], 32, 1, 1)]
    assert summarize_safety(steps, repetitions=3)[0]["safe"] is False
