"""Structured logging for the runtime.

A thin wrapper over the standard library that renders one record per line in one
of two shapes, chosen by the ``REACTOR_LOG_FORMAT`` environment variable:

- ``text`` (default): the message followed by ``key=value`` tokens, easy to read
  in a terminal and to grep.
- ``json``: one JSON object per line, ready for a log pipeline to parse.

Call sites pass structured context as keyword arguments —
``log.info("session started", session_id=sid)`` — and the active formatter
renders them; the wire shape is the formatter's concern, not the call site's.
``configure`` installs the chosen formatter on the root logger, and
``get_logger`` returns a logger to write through.
"""

from __future__ import annotations

import json
import logging
import os
from typing import IO, Any

LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S%z"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# Structured fields from ``log.info(..., k=v)`` are stashed on the record under a
# single attribute, so the individual keys never collide with the reserved names
# the standard library puts on a ``LogRecord``.
_REACTOR_FIELDS_ATTR = "reactor_fields"

# Envelope keys the JSON formatter owns; a field of the same name from a call
# site cannot overwrite them.
_JSON_RESERVED = frozenset({"ts", "level", "logger", "msg", "exc_info"})

_QUOTE_TRIGGERS = (" ", "=", '"', "\n", "\r", "\t")


def _logfmt_value(value: Any) -> str:
    """Render *value* as a logfmt-safe token.

    A value containing whitespace, ``=``, or a quote is wrapped in quotes, and
    control characters are escaped, so a multi-line value stays on one line.
    """
    text = "" if value is None else str(value)
    if text and not any(c in text for c in _QUOTE_TRIGGERS):
        return text
    escaped = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _record_fields(record: logging.LogRecord) -> dict[str, Any]:
    """Return the structured fields attached to *record*, dropping ``None`` values."""
    raw = getattr(record, _REACTOR_FIELDS_ATTR, None)
    if not isinstance(raw, dict):
        return {}
    return {key: value for key, value in raw.items() if value is not None}


class TextFormatter(logging.Formatter):
    """Render a record as its message followed by ``key=value`` field tokens."""

    def formatMessage(self, record: logging.LogRecord) -> str:  # noqa: N802 — overrides logging.Formatter
        """Append the record's structured fields to the formatted message line.

        Appending happens on the message line so that when the record carries an
        exception, the parent's traceback block (added after this returns) lands
        below the fields rather than the fields landing after the traceback.
        """
        base = super().formatMessage(record)
        fields = _record_fields(record)
        if not fields:
            return base
        rendered = " ".join(f"{key}={_logfmt_value(value)}" for key, value in fields.items())
        return f"{base} {rendered}"


class JsonFormatter(logging.Formatter):
    """Render a record as one JSON object per line.

    The envelope (``ts``, ``level``, ``logger``, ``msg``, and ``exc_info`` when an
    exception is attached) is owned by the formatter; structured fields are merged
    in alongside it but cannot overwrite an envelope key.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format *record* as a single JSON line."""
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, LOG_DATEFMT),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in _record_fields(record).items():
            if key not in _JSON_RESERVED:
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class StructuredLogger:
    """A logger whose level methods take structured fields as keyword arguments.

    ``log.info("message", key=value)`` attaches ``{"key": value}`` to the record
    for the active formatter to render. The object wraps a standard-library logger
    by name, so configuration (level, handlers) flows through the usual hierarchy.
    """

    __slots__ = ("_log",)

    def __init__(self, name: str) -> None:
        """Wrap the standard-library logger named *name*."""
        self._log = logging.getLogger(name)

    def _emit(
        self, level: int, msg: str, fields: dict[str, Any], *, exc_info: bool = False
    ) -> None:
        if not self._log.isEnabledFor(level):
            return
        extra = {_REACTOR_FIELDS_ATTR: fields} if fields else None
        self._log.log(level, msg, exc_info=exc_info, extra=extra)

    def debug(self, msg: str, **fields: Any) -> None:
        """Log *msg* at DEBUG with structured *fields*."""
        self._emit(logging.DEBUG, msg, fields)

    def info(self, msg: str, **fields: Any) -> None:
        """Log *msg* at INFO with structured *fields*."""
        self._emit(logging.INFO, msg, fields)

    def warning(self, msg: str, **fields: Any) -> None:
        """Log *msg* at WARNING with structured *fields*."""
        self._emit(logging.WARNING, msg, fields)

    def error(self, msg: str, **fields: Any) -> None:
        """Log *msg* at ERROR with structured *fields*."""
        self._emit(logging.ERROR, msg, fields)

    def exception(self, msg: str, **fields: Any) -> None:
        """Log *msg* at ERROR with the active exception's traceback attached."""
        self._emit(logging.ERROR, msg, fields, exc_info=True)


def get_logger(name: str) -> StructuredLogger:
    """Return a :class:`StructuredLogger` named *name*, typically ``__name__``."""
    return StructuredLogger(name)


def configure(*, level: int = logging.INFO, stream: IO[str] | None = None) -> None:
    """Install the structured formatter on the root logger.

    The shape is chosen by ``REACTOR_LOG_FORMAT``: ``json`` for one JSON object
    per line, anything else (the default) for human-readable ``key=value`` text.
    Replaces any handlers already on the root logger so output has a single,
    predictable shape.

    Args:
        level: The level the root logger is set to.
        stream: Where lines are written; defaults to standard error.
    """
    fmt = os.getenv("REACTOR_LOG_FORMAT", "text").strip().lower()
    formatter: logging.Formatter = (
        JsonFormatter() if fmt == "json" else TextFormatter(LOG_FORMAT, LOG_DATEFMT)
    )
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    for existing in root.handlers[:]:
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)


__all__ = [
    "JsonFormatter",
    "StructuredLogger",
    "TextFormatter",
    "configure",
    "get_logger",
]
