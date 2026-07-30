from __future__ import annotations

from prometheus_client import Counter

from reactor_runtime.core import EndReason, SessionEvent, SessionState
from reactor_runtime.metrics import CONTENT_TYPE, MetricsRecorder, RuntimeMetrics
from reactor_runtime.runner.state_machine import SessionStateMachine


def test_publishes_the_identity_of_the_process() -> None:
    metrics = RuntimeMetrics(version="1.4.2", model="pipeline:Brightness")

    rendered = metrics.render().decode()

    assert 'runtime_info{model="pipeline:Brightness",version="1.4.2"} 1.0' in rendered


def test_renders_in_the_prometheus_text_format() -> None:
    metrics = RuntimeMetrics(version="0.0.0", model="pipeline:Echo")

    assert CONTENT_TYPE.startswith("text/plain")
    assert metrics.render().startswith(b"# HELP runtime_info")


def test_each_holder_owns_its_own_registry() -> None:
    # Two holders in one process must not see each other's numbers. The registry
    # is a field of the holder rather than a module global, so a component that
    # observes on one leaves the other empty. Every test in this suite relies on
    # that: they build a holder each and read it back.
    first = RuntimeMetrics(version="0.0.0", model="pipeline:Echo")
    second = RuntimeMetrics(version="0.0.0", model="pipeline:Echo")

    Counter("runtime_test_total", "A counter one holder carries.", registry=first.registry).inc()

    assert "runtime_test_total" in first.render().decode()
    assert "runtime_test_total" not in second.render().decode()


def test_the_registry_carries_only_the_runtime_instruments() -> None:
    # A dedicated registry, not the default one: the process reports what the
    # runtime declares, and the scraper adds what it knows about the machine.
    rendered = RuntimeMetrics(version="0.0.0", model="pipeline:Echo").render().decode()

    assert "python_gc_objects_collected_total" not in rendered
    assert "process_virtual_memory_bytes" not in rendered


class _Clock:
    """A clock the test moves by hand, so a measured duration is exact."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += seconds


class _Session:
    """A state machine with a recorder on it, plus readers for the registry."""

    def __init__(self) -> None:
        self.clock = _Clock()
        self.metrics = RuntimeMetrics(version="0.0.0", model="pipeline:Echo")
        self.machine = SessionStateMachine()
        self.recorder = MetricsRecorder(
            self.metrics, state=self.machine.current_state, clock=self.clock
        )
        self.machine.on_transition(self.recorder.observe)

    def send(self, event: SessionEvent, **detail: object) -> None:
        assert self.machine.send(event, **detail), f"{event.name} was rejected"

    def open_session(self) -> None:
        """Drive the machine from a fresh process to a session waiting for a client."""
        self.send(SessionEvent.INITIALIZATION_SUCCESS)
        self.send(SessionEvent.START_SESSION)

    def value(self, name: str, **labels: str) -> float | None:
        return self.metrics.registry.get_sample_value(name, labels or None)

    def rendered(self) -> str:
        return self.metrics.render().decode()


def test_publishes_the_state_of_a_session_that_has_not_moved() -> None:
    session = _Session()

    # Every state has a series from the first scrape, so a query never has to
    # tell "no session yet" apart from "no metric yet".
    assert session.value("runtime_session_state", state="created") == 1.0
    assert session.value("runtime_session_state", state="ready") == 0.0
    assert session.value("runtime_session_state", state="terminated") == 0.0


def test_raises_one_state_series_at_a_time() -> None:
    session = _Session()

    session.open_session()

    assert session.value("runtime_session_state", state="waiting") == 1.0
    assert session.value("runtime_session_state", state="ready") == 0.0
    assert session.value("runtime_session_state", state="created") == 0.0


def test_a_journal_fact_leaves_the_state_alone() -> None:
    session = _Session()
    session.open_session()

    session.send(SessionEvent.CHUNK_READY, recording_id="r", idx=1)

    assert session.value("runtime_session_state", state="waiting") == 1.0


def test_counts_a_stopped_session_and_how_long_it_ran() -> None:
    session = _Session()
    session.open_session()

    session.clock.tick(90.0)
    session.send(SessionEvent.STOP_SESSION, reason=EndReason.STOPPED)
    session.send(SessionEvent.CLEANUP_COMPLETE, reason=EndReason.STOPPED)

    assert session.value("runtime_sessions_total", reason="stopped") == 1.0
    assert session.value("runtime_session_duration_seconds_count", reason="stopped") == 1.0
    assert session.value("runtime_session_duration_seconds_sum", reason="stopped") == 90.0
    assert session.value("runtime_session_state", state="ready") == 1.0


def test_carries_the_reason_a_session_ended() -> None:
    session = _Session()
    session.open_session()

    session.send(SessionEvent.TIMEOUT, reason=EndReason.TIMED_OUT)
    session.send(SessionEvent.CLEANUP_COMPLETE, reason=EndReason.TIMED_OUT)

    assert session.value("runtime_sessions_total", reason="timed_out") == 1.0
    assert session.value("runtime_sessions_total", reason="stopped") == 0.0


def test_measures_the_wait_for_the_first_client() -> None:
    session = _Session()
    session.open_session()

    session.clock.tick(2.5)
    session.send(SessionEvent.CONNECTION_OPENED, conn_id=1002)
    session.clock.tick(4.0)
    session.send(SessionEvent.CONNECTION_OPENED, conn_id=1003)

    # The second client is not the first one: the measurement is how long the
    # session sat empty, so it happens once per session.
    assert session.value("runtime_session_time_to_first_client_seconds_count") == 1.0
    assert session.value("runtime_session_time_to_first_client_seconds_sum") == 2.5


def test_measures_how_long_a_teardown_took() -> None:
    session = _Session()
    session.open_session()

    session.send(SessionEvent.STOP_SESSION, reason=EndReason.STOPPED)
    session.clock.tick(0.4)
    session.send(SessionEvent.CLEANUP_COMPLETE, reason=EndReason.STOPPED)

    assert session.value("runtime_session_teardown_seconds_count") == 1.0
    assert session.value("runtime_session_teardown_seconds_sum") == 0.4


def test_counts_each_client_that_arrives_and_leaves() -> None:
    session = _Session()
    session.open_session()

    session.send(SessionEvent.CONNECTION_OPENED, conn_id=1002)
    session.send(SessionEvent.CONNECTION_OPENED, conn_id=1003)
    session.send(SessionEvent.CONNECTION_CLOSED, conn_id=1003)

    assert session.value("runtime_connections_opened_total") == 2.0
    assert session.value("runtime_connections_closed_total") == 1.0
    assert session.value("runtime_connections_active") == 1.0
    assert session.value("runtime_session_state", state="streaming") == 1.0


def test_the_last_client_to_leave_orphans_the_session() -> None:
    session = _Session()
    session.open_session()

    session.send(SessionEvent.CONNECTION_OPENED, conn_id=1002)
    session.send(SessionEvent.CONNECTION_CLOSED, conn_id=1002)

    assert session.value("runtime_connections_active") == 0.0
    assert session.value("runtime_session_state", state="orphaned") == 1.0


def test_teardown_clears_the_clients_it_closed_wholesale() -> None:
    session = _Session()
    session.open_session()
    session.send(SessionEvent.CONNECTION_OPENED, conn_id=1002)
    session.send(SessionEvent.CONNECTION_OPENED, conn_id=1003)

    session.send(SessionEvent.STOP_SESSION, reason=EndReason.STOPPED)
    session.send(SessionEvent.CLEANUP_COMPLETE, reason=EndReason.STOPPED)

    # Teardown closes every wire at once and reports no per-connection loss, so
    # the gauge would hold two dead clients if it only followed the close facts.
    assert session.value("runtime_connections_active") == 0.0
    # Neither client closed itself, so the two counters differ by what teardown
    # reaped. A reader subtracts them to get that number.
    assert session.value("runtime_connections_opened_total") == 2.0
    assert session.value("runtime_connections_closed_total") == 0.0


def test_counts_a_session_a_crash_evicted() -> None:
    session = _Session()
    session.open_session()

    session.clock.tick(12.0)
    session.send(SessionEvent.EVICTION, reason=EndReason.ERROR, error="the model crashed")

    assert session.value("runtime_sessions_total", reason="error") == 1.0
    assert session.value("runtime_session_duration_seconds_sum", reason="error") == 12.0
    assert session.value("runtime_session_state", state="terminated") == 1.0
    assert session.value("runtime_connections_active") == 0.0


def test_an_eviction_with_no_session_counts_nothing() -> None:
    session = _Session()
    session.send(SessionEvent.INITIALIZATION_SUCCESS)

    session.send(SessionEvent.EVICTION, reason=EndReason.EVICTED)

    # The model sat idle, so no session ran and none ended.
    assert session.value("runtime_sessions_total", reason="evicted") == 0.0
    assert session.value("runtime_session_duration_seconds_count", reason="evicted") is None


def test_a_model_that_fails_to_load_counts_no_session() -> None:
    session = _Session()

    session.send(SessionEvent.INITIALIZATION_FAIL)

    assert session.value("runtime_session_state", state="terminated") == 1.0
    for reason in EndReason:
        assert session.value("runtime_sessions_total", reason=reason.value) == 0.0


def test_counts_errors_and_keeps_the_message_out_of_the_labels() -> None:
    session = _Session()
    session.open_session()

    session.send(SessionEvent.ERROR, message="command 'set_mode' references an unresolved upload")
    session.send(SessionEvent.ERROR, message="command 'set_seed' rejected (seed: out of range)")

    assert session.value("runtime_session_errors_total") == 2.0
    # A message is unbounded text. One in a label would cost a series for every
    # distinct error the runtime ever writes.
    assert "set_mode" not in session.rendered()


def test_a_second_session_measures_itself_from_its_own_start() -> None:
    session = _Session()
    session.open_session()
    session.clock.tick(30.0)
    session.send(SessionEvent.STOP_SESSION, reason=EndReason.STOPPED)
    session.send(SessionEvent.CLEANUP_COMPLETE, reason=EndReason.STOPPED)

    session.send(SessionEvent.START_SESSION)
    session.clock.tick(5.0)
    session.send(SessionEvent.CONNECTION_OPENED, conn_id=1002)
    session.send(SessionEvent.STOP_SESSION, reason=EndReason.STOPPED)
    session.send(SessionEvent.CLEANUP_COMPLETE, reason=EndReason.STOPPED)

    assert session.value("runtime_sessions_total", reason="stopped") == 2.0
    assert session.value("runtime_session_duration_seconds_sum", reason="stopped") == 35.0
    assert session.value("runtime_session_time_to_first_client_seconds_sum") == 5.0
    assert session.value("runtime_session_state", state="ready") == 1.0


def test_the_machine_state_and_the_gauge_agree_after_a_full_round() -> None:
    session = _Session()
    session.open_session()
    session.send(SessionEvent.CONNECTION_OPENED, conn_id=1002)
    session.send(SessionEvent.CONNECTION_CLOSED, conn_id=1002)
    session.send(SessionEvent.TIMEOUT, reason=EndReason.TIMED_OUT)
    session.send(SessionEvent.CLEANUP_COMPLETE, reason=EndReason.TIMED_OUT)

    assert session.machine.current_state is SessionState.READY
    for member in SessionState:
        expected = 1.0 if member is SessionState.READY else 0.0
        assert session.value("runtime_session_state", state=member.name.lower()) == expected
