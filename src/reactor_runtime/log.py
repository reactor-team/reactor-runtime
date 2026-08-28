"""Structured logging for the runtime.

A thin wrapper over the standard library that renders one record per line in one
of two shapes, chosen by the ``REACTOR_LOG_FORMAT`` environment variable:

- ``text`` (default): the message followed by ``key=value`` tokens, easy to read
  in a terminal and to grep.
- ``json``: one JSON object per line, ready for a log pipeline to parse.

Call sites pass structured context as keyword arguments —
``log.info("chunk encoded", chunk_idx=idx)`` — and the active formatter renders
them; the wire shape is the formatter's concern, not the call site's.
``configure`` installs the chosen formatter on the root logger, and
``get_logger`` returns a logger to write through.

Two fields arrive without a call site naming them, stamped by
:class:`SessionContextFilter` on the handler ``configure`` installs. While a
session is live, every record carries its ``session_id``; from the moment the
runtime boots, every record carries ``state``, the lifecycle state the process
was in when the record was written — so the model-load window is retrievable as
``state="created"`` even though no session exists yet. The vocabulary is the
session state machine's, the same words the session descriptor's ``state``
field serves; the health route's coarse ``state`` (``loading`` / ``available``
/ ``serving``) is a projection of it for outside observers and deliberately
not what the records carry. Because the stamp happens where records are
written rather than where they are made, it reaches a model's own
``logging.getLogger(__name__)`` and any third-party library that propagates to
root, so a line can be traced to the session and phase that produced it without
model code threading either through its call sites.
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

# The live session's id, stamped on every record while it is set. A module global
# rather than a ContextVar because a session fans its work across plain worker
# threads, which do not inherit context; the runtime hosts one session at a time,
# so a single value is unambiguous.
_session_id: str | None = None

# Counts bindings, so a release can name the one it retires. Two sessions may
# carry the same id — nothing stops a caller reusing one — and a release that
# matched on the id alone would unbind the session that reused it.
_session_binding = 0


def set_session_id(session_id: str | None) -> int:
    """Stamp *session_id* on every record written from now on.

    Args:
        session_id: The live session's id, or ``None`` to stamp nothing.

    Returns:
        A token naming this binding, which :func:`release_session_id` takes to
        retire it.
    """
    global _session_id, _session_binding
    _session_id = session_id
    _session_binding += 1
    return _session_binding


def clear_session_id() -> None:
    """Stop stamping a session id, for the window between sessions."""
    set_session_id(None)


def release_session_id(binding: int) -> None:
    """Retire the binding *binding* names, leaving a later one in place.

    A session's teardown outlives the move that ends it, so the release that
    follows one is deferred until that work has finished. By then the next
    session may already have bound its own id — the same id, even, since nothing
    stops a caller reusing one — so a release names the binding it retires rather
    than the value that binding held.

    Args:
        binding: The token :func:`set_session_id` returned, ignored once a later
            binding has replaced the one it names.
    """
    global _session_id
    if _session_binding == binding:
        _session_id = None


def get_session_id() -> str | None:
    """Return the id currently being stamped, or ``None`` between sessions."""
    return _session_id


# The runtime's lifecycle state, stamped on every record while it is set. Unlike
# the session id it needs no binding token: there is always exactly one current
# state and the latest write is by definition the truth, so last-write-wins is
# the correct semantics rather than a race to guard against.
_state: str | None = None


def set_state(state: str | None) -> None:
    """Stamp *state* on every record written from now on.

    Args:
        state: The runtime's lifecycle state, or ``None`` to stamp nothing.
    """
    global _state
    _state = state


def get_state() -> str | None:
    """Return the state currently being stamped, or ``None`` before boot."""
    return _state


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


class SessionContextFilter(logging.Filter):
    """Stamp the live session's id and the runtime's state on every record.

    Sits on the handler rather than on one logger, so it sees every record a
    handler writes: the runtime's own, a model's ``logging.getLogger(__name__)``,
    and a third-party library's that propagates to root. A call site that names
    ``session_id`` or ``state`` itself keeps its own value — a model logging its
    own ``state`` claims that record's field, deliberately — and a field with
    nothing bound, the session id between sessions or the state before boot, is
    absent rather than empty.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Merge the ambient context into *record*'s structured fields."""
        stamped = {"session_id": _session_id, "state": _state}
        context = {key: value for key, value in stamped.items() if value is not None}
        if not context:
            return True
        fields = getattr(record, _REACTOR_FIELDS_ATTR, None)
        if not isinstance(fields, dict):
            setattr(record, _REACTOR_FIELDS_ATTR, context)
            return True
        for key, value in context.items():
            fields.setdefault(key, value)
        return True


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
    predictable shape. The handler carries a :class:`SessionContextFilter`, so
    every record written through it is stamped with the live session's id and
    the runtime's lifecycle state.

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
    handler.addFilter(SessionContextFilter())
    root = logging.getLogger()
    for existing in root.handlers[:]:
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)


__all__ = [
    "JsonFormatter",
    "SessionContextFilter",
    "StructuredLogger",
    "TextFormatter",
    "clear_session_id",
    "configure",
    "get_logger",
    "get_session_id",
    "get_state",
    "release_session_id",
    "set_session_id",
    "set_state",
]
