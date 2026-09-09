"""Wire contract: the runtime's JSON records satisfy the reactor CLI's log renderer.

The renderer pretty-renders a line when it decodes as one JSON object with
``CLI_REQUIRED_RECORD_FIELDS`` present as strings; anything else prints
verbatim. This pins the runtime's emitted envelope against that rule.
"""

from __future__ import annotations

import io
import json
import logging

import pytest

from reactor_runtime import log

# The fields the CLI's acceptance predicate requires, all strings.
CLI_REQUIRED_RECORD_FIELDS = ("ts", "level", "logger", "msg")


def _cli_accepts_as_log_record(line: str) -> bool:
    """Return whether the reactor CLI would pretty-render *line*."""
    try:
        record = json.loads(line)
    except ValueError:
        return False
    if not isinstance(record, dict):
        return False
    return all(isinstance(record.get(k), str) for k in CLI_REQUIRED_RECORD_FIELDS)


@pytest.fixture
def json_stream(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """Configure root logging in JSON mode into a captured stream."""
    monkeypatch.setenv("REACTOR_LOG_FORMAT", "json")
    buffer = io.StringIO()
    log.configure(level=logging.DEBUG, stream=buffer)
    return buffer


def test_record_satisfies_cli_acceptance(json_stream: io.StringIO) -> None:
    log.get_logger("test.contract").info("hello")
    lines = [ln for ln in json_stream.getvalue().splitlines() if ln.strip()]
    assert len(lines) == 1, f"expected one record, got {len(lines)}"
    assert _cli_accepts_as_log_record(lines[0]), (
        "the CLI would print this line verbatim (raw JSON in a TTY): " + lines[0]
    )


@pytest.mark.parametrize(
    ("line", "why"),
    [
        ("not json", "a non-JSON line has no envelope"),
        ('{"ts":"2026-01-01T00:00:00+0000","level":"info"}', "missing msg"),
        ('{"ts":1,"level":"info","msg":"m"}', "ts is not a string"),
        ('{"ts":"t","level":"info","msg":42}', "msg is not a string"),
    ],
    ids=["non-json", "missing-required-field", "non-string-ts", "non-string-msg"],
)
def test_predicate_rejects_non_records(line: str, why: str) -> None:
    """Negative control: the predicate fails what the CLI would echo verbatim."""
    assert not _cli_accepts_as_log_record(line), why
