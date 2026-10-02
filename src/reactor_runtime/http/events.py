"""Rendering transition events as Server-Sent Events.

The egress journal is transitions only: every fact it holds is a
:class:`~reactor_runtime.core.model.TransitionEvent`, so the stream carries
exactly one envelope type, ``transition``. This module is the one place that
decides that envelope's wire shape and frames it for the ``text/event-stream``
body, so a consumer can mirror the runtime's view and resume from a sequence
number it has seen.
"""

from __future__ import annotations

import json
from typing import Any

from reactor_runtime.core import TransitionEvent


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
