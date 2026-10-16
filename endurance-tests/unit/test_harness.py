import threading

import pytest
from endurance.harness import (
    CLOSE_TIMEOUT_S,
    FRAMES_PER_CYCLE,
    RuntimeProcess,
    V1Client,
    assert_answered,
    assert_full_streams,
    assert_released,
)


def test_full_streams_pass_when_every_cycle_got_its_frames() -> None:
    assert assert_full_streams([FRAMES_PER_CYCLE] * 5).status == "ok"


def test_a_cycle_one_frame_short_fails_the_run() -> None:
    result = assert_full_streams([FRAMES_PER_CYCLE, FRAMES_PER_CYCLE - 1, FRAMES_PER_CYCLE])
    assert result.status == "fail"
    assert result.first_bad_cycle == 1
    assert "1/3 cycles" in result.detail


def test_a_cycle_with_no_frames_fails_the_run() -> None:
    assert assert_full_streams([0]).status == "fail"


def test_a_run_whose_requests_were_all_answered_passes() -> None:
    assert assert_answered([[], [], []]).status == "ok"


def test_a_refused_or_unanswered_request_fails_its_cycle() -> None:
    result = assert_answered([[], ["set_effect (command-0): invalid_command"], []])
    assert result.status == "fail"
    assert result.first_bad_cycle == 1
    assert "invalid_command" in result.detail


def test_a_run_whose_connections_were_all_dropped_passes_and_reports_how_fast() -> None:
    result = assert_released([7.0, 7.5, 9.0])
    assert result.status == "ok"
    assert "7.5s median, 9.0s at worst" in result.detail


def test_a_connection_still_registered_after_the_timeout_fails_its_cycle() -> None:
    result = assert_released([7.0, None, 8.0])
    assert result.status == "fail"
    assert result.first_bad_cycle == 1
    assert f"still registered {CLOSE_TIMEOUT_S:g}s after its client left on 1/3 cycles" in (
        result.detail
    )


def test_a_run_whose_connections_all_stayed_reports_no_drop_time() -> None:
    result = assert_released([None])
    assert result.status == "fail"
    assert "the others" not in result.detail


def _client_owed(request_id: str, asked: str, answers: frozenset[str | None]) -> V1Client:
    client = V1Client.__new__(V1Client)
    client._lock = threading.Lock()
    client._pending = {request_id: (asked, answers)}
    client._failed = []
    return client


def test_a_reply_of_the_expected_type_answers_its_request() -> None:
    client = _client_owed("schema", "request_schema", frozenset({"model_schema"}))
    client._answered("schema", "model_schema", None)
    assert client._pending == {}
    assert client._failed == []


def test_a_reply_of_another_type_is_a_failure() -> None:
    client = _client_owed("publish-webcam", "publish_track", frozenset({"publish_track"}))
    client._answered("publish-webcam", "model_schema", None)
    assert client._failed == ["publish_track (publish-webcam): answered with model_schema"]


def test_a_bodyless_ack_answers_a_command() -> None:
    client = _client_owed("command-0", "set_effect", frozenset({None, "message"}))
    client._answered("command-0", None, None)
    assert client._failed == []


def test_a_settled_probe_waits_only_for_the_connections_a_scenario_does_not_keep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readings = iter(
        [
            {"registered_connections": 2, "live": {"participants": 1}},
            {"registered_connections": 1, "live": {"participants": 0}},
        ]
    )
    runtime = object.__new__(RuntimeProcess)
    monkeypatch.setattr(runtime, "probe", lambda: next(readings))

    reading, waited = runtime.settled_probe(connections=1)

    assert reading == {"registered_connections": 1, "live": {"participants": 0}}
    assert waited is not None
    assert waited < 5.0


def test_a_settled_probe_given_a_task_count_waits_for_teardown_tasks_to_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readings = iter(
        [
            {"registered_connections": 1, "live": {}, "tasks": 8},
            {"registered_connections": 1, "live": {}, "tasks": 7},
        ]
    )
    runtime = object.__new__(RuntimeProcess)
    monkeypatch.setattr(runtime, "probe", lambda: next(readings))

    reading, _ = runtime.settled_probe(connections=1, tasks=7)

    assert reading["tasks"] == 7
