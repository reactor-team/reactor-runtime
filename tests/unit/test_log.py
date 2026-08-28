import io
import json
import logging

import pytest

from reactor_runtime.log import (
    JsonFormatter,
    StructuredLogger,
    TextFormatter,
    clear_session_id,
    configure,
    get_logger,
    get_session_id,
    release_session_id,
    set_session_id,
)


def configured_logger(
    monkeypatch: pytest.MonkeyPatch, fmt: str
) -> tuple[StructuredLogger, io.StringIO]:
    monkeypatch.setenv("REACTOR_LOG_FORMAT", fmt)
    buffer = io.StringIO()
    configure(level=logging.DEBUG, stream=buffer)
    return get_logger("test.logger"), buffer


def test_get_logger_returns_structured_logger() -> None:
    assert isinstance(get_logger("anything"), StructuredLogger)


def test_text_mode_renders_fields_after_the_message(monkeypatch: pytest.MonkeyPatch) -> None:
    log, buffer = configured_logger(monkeypatch, "text")
    log.info("session started", session_id="s-1", count=3)
    line = buffer.getvalue()
    assert "session started" in line
    assert "session_id=s-1" in line
    assert "count=3" in line


def test_text_mode_quotes_values_that_need_it(monkeypatch: pytest.MonkeyPatch) -> None:
    log, buffer = configured_logger(monkeypatch, "text")
    log.info("evt", note="two words")
    assert 'note="two words"' in buffer.getvalue()


def test_default_mode_is_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REACTOR_LOG_FORMAT", raising=False)
    buffer = io.StringIO()
    configure(level=logging.DEBUG, stream=buffer)
    get_logger("test.logger").info("plain", k="v")
    line = buffer.getvalue()
    assert "plain" in line
    assert "k=v" in line


def test_json_mode_emits_one_object_with_envelope_and_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log, buffer = configured_logger(monkeypatch, "json")
    log.info("session started", session_id="s-1")
    payload = json.loads(buffer.getvalue().strip())
    assert payload["msg"] == "session started"
    assert payload["level"] == "info"
    assert payload["logger"] == "test.logger"
    assert payload["session_id"] == "s-1"


def test_json_envelope_keys_cannot_be_overwritten(monkeypatch: pytest.MonkeyPatch) -> None:
    log, buffer = configured_logger(monkeypatch, "json")
    log.info("real", level="HACK", logger="HACK", ts="HACK")
    payload = json.loads(buffer.getvalue().strip())
    assert payload["msg"] == "real"
    assert payload["level"] == "info"
    assert payload["logger"] == "test.logger"
    assert payload["ts"] != "HACK"


def test_none_fields_are_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    log, buffer = configured_logger(monkeypatch, "text")
    log.info("evt", present=1, absent=None)
    line = buffer.getvalue()
    assert "present=1" in line
    assert "absent=" not in line


def test_exception_attaches_a_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    log, buffer = configured_logger(monkeypatch, "json")
    try:
        raise ValueError("boom")
    except ValueError:
        log.exception("handling failed")
    payload = json.loads(buffer.getvalue().strip())
    assert payload["level"] == "error"
    assert "ValueError: boom" in payload["exc_info"]


def test_text_mode_exception_keeps_fields_on_the_message_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log, buffer = configured_logger(monkeypatch, "text")
    try:
        raise ValueError("boom")
    except ValueError:
        log.exception("handling failed", session_id="s-1")
    out = buffer.getvalue()
    first_line = out.splitlines()[0]
    assert "handling failed" in first_line
    assert "session_id=s-1" in first_line
    assert "Traceback (most recent call last)" in out
    assert "ValueError: boom" in out
    # The fields belong on the message line, never after the traceback.
    assert out.index("session_id=s-1") < out.index("Traceback")


def test_level_below_threshold_is_suppressed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REACTOR_LOG_FORMAT", "text")
    buffer = io.StringIO()
    configure(level=logging.WARNING, stream=buffer)
    get_logger("test.logger").info("should not appear")
    assert buffer.getvalue() == ""


def test_text_formatter_without_fields_is_just_the_message() -> None:
    formatter = TextFormatter("%(levelname)s %(message)s")
    record = logging.LogRecord("n", logging.INFO, __file__, 1, "hello", (), None)
    assert formatter.format(record) == "INFO hello"


def test_json_formatter_without_fields_has_only_the_envelope() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord("n", logging.INFO, __file__, 1, "hello", (), None)
    payload = json.loads(formatter.format(record))
    assert set(payload) == {"ts", "level", "logger", "msg"}


# --- the session id stamped on every record ------------------------------


def test_no_session_id_field_between_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    log, buffer = configured_logger(monkeypatch, "json")
    log.info("idle")
    payload = json.loads(buffer.getvalue().strip())
    assert "session_id" not in payload


def test_the_live_session_id_is_stamped_without_the_call_site_naming_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log, buffer = configured_logger(monkeypatch, "json")
    set_session_id("s-live")
    log.info("generating", chunk_idx=7)
    payload = json.loads(buffer.getvalue().strip())
    assert payload["session_id"] == "s-live"
    assert payload["chunk_idx"] == 7


def test_text_mode_renders_the_stamped_session_id(monkeypatch: pytest.MonkeyPatch) -> None:
    log, buffer = configured_logger(monkeypatch, "text")
    set_session_id("s-live")
    log.info("generating")
    assert "session_id=s-live" in buffer.getvalue()


def test_a_call_site_that_names_the_session_id_keeps_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log, buffer = configured_logger(monkeypatch, "json")
    set_session_id("s-live")
    log.info("recorder started", session_id="s-explicit")
    payload = json.loads(buffer.getvalue().strip())
    assert payload["session_id"] == "s-explicit"


def test_a_plain_stdlib_logger_is_stamped_too(monkeypatch: pytest.MonkeyPatch) -> None:
    # Model code that reaches for the standard library rather than get_logger
    # still lands on the configured handler by propagation, which is where the
    # stamp happens — so its records name the session without any model change.
    _, buffer = configured_logger(monkeypatch, "json")
    set_session_id("s-live")
    logging.getLogger("some.model.module").info("loaded weights")
    payload = json.loads(buffer.getvalue().strip())
    assert payload["session_id"] == "s-live"
    assert payload["logger"] == "some.model.module"


def test_clearing_stops_the_stamp(monkeypatch: pytest.MonkeyPatch) -> None:
    log, buffer = configured_logger(monkeypatch, "json")
    set_session_id("s-live")
    log.info("in session")
    clear_session_id()
    log.info("after session")
    first, second = (json.loads(line) for line in buffer.getvalue().splitlines() if line.strip())
    assert first["session_id"] == "s-live"
    assert "session_id" not in second


def test_get_session_id_reports_what_is_stamped() -> None:
    assert get_session_id() is None
    set_session_id("s-live")
    assert get_session_id() == "s-live"
    clear_session_id()
    assert get_session_id() is None


def test_releasing_the_live_session_unbinds_it() -> None:
    set_session_id("s-one")
    release_session_id("s-one")
    assert get_session_id() is None


def test_releasing_an_earlier_session_leaves_the_live_one_bound() -> None:
    # A release is deferred until the session's teardown finishes, so the next
    # session can already be bound by the time it runs.
    set_session_id("s-one")
    set_session_id("s-two")
    release_session_id("s-one")
    assert get_session_id() == "s-two"
