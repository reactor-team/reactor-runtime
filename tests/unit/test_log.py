import io
import json
import logging
from collections.abc import Iterator

import pytest

from reactor_runtime.log import (
    JsonFormatter,
    StructuredLogger,
    TextFormatter,
    configure,
    get_logger,
)


@pytest.fixture(autouse=True)
def _restore_root_logging() -> Iterator[None]:
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


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
