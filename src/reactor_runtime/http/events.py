"""Rendering the egress stream as Server-Sent Events.

The stream carries two envelope types:

- ``transition``: a journal fact, a
  :class:`~reactor_runtime.core.model.TransitionEvent`. Framed with its sequence
  number as the SSE ``id``, so a consumer can mirror the runtime's view and
  resume after an id it has seen.
- ``stats``: a live quality reading, a
  :class:`~reactor_runtime.core.model.StatsEvent`. Framed without an ``id``: it
  has no sequence number and is never replayed, so it moves no consumer's resume
  point.

This module is the one place that decides both envelopes' wire shape.
"""

from __future__ import annotations

import json
from typing import Any

from reactor_runtime.core import StatsEvent, TransitionEvent


def runner_event_to_dict(event: TransitionEvent) -> dict[str, Any]:
    """Render a transition event as a JSON-serialisable ``transition`` envelope.

    Args:
        event: The transition event to render.

    Returns:
        A plain dict whose ``type`` is ``"transition"``, whose ``event``/
        ``from``/``to`` name the move, whose ``ts`` is the move's Unix epoch
        millisecond timestamp, and whose ``detail`` carries the move's payload
        verbatim.
    """
    transition = event.transition
    return {
        "type": "transition",
        "event": transition.event.name.lower(),
        "from": transition.from_state.name.lower(),
        "to": transition.to_state.name.lower(),
        "ts": transition.ts_ms,
        "detail": dict(transition.detail),
    }


def format_sse(seq: int, event: TransitionEvent) -> str:
    """Frame a transition event as one SSE message carrying its sequence number.

    Args:
        seq: The event's sequence number, emitted as the SSE ``id`` so a
            consumer can resume after it.
        event: The transition event to frame.

    Returns:
        A complete SSE message, terminated by the blank line that ends an event.
    """
    payload = json.dumps(runner_event_to_dict(event))
    return f"id: {seq}\ndata: {payload}\n\n"


def stats_event_to_dict(event: StatsEvent) -> dict[str, Any]:
    """Render a live reading as a JSON-serialisable ``stats`` envelope.

    Args:
        event: The reading to render.

    Returns:
        A plain dict whose ``type`` is ``"stats"``, whose ``event`` names the
        reading (e.g. ``"client_stats"``), whose ``ts`` is when the runtime
        recorded it as Unix epoch milliseconds, and whose ``detail`` carries
        the reading verbatim.
    """
    return {
        "type": "stats",
        "event": event.name,
        "ts": event.ts_ms,
        "detail": dict(event.detail),
    }


def format_live_sse(event: StatsEvent) -> str:
    """Frame a live reading as one SSE message with no ``id``.

    Args:
        event: The reading to frame.

    Returns:
        A complete SSE message, terminated by the blank line that ends an event.
        With no ``id`` line, a client's ``Last-Event-ID`` keeps naming the last
        journal fact it received.
    """
    payload = json.dumps(stats_event_to_dict(event), allow_nan=False)
    return f"data: {payload}\n\n"
