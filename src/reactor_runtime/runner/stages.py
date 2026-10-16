"""Hold the stages the runtime times itself, per connection, until a report takes them.

A transport times the stages it sees on the wire and reports them on its
reading. The runner and the model loop time the rest, for each command:
``dispatch``, from the command reaching the runtime until its handler starts,
and ``handler``, the handler running and its reply being written.
:class:`RunnerStages` keeps those per connection, and :meth:`RunnerStages.merge`
adds them to each connection's reading beside the transport's own.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from reactor_runtime.core import ConnId, MessageStage, TransportReading
from reactor_runtime.core.stages import StageTotals

MESSAGE_STAGES = ("dispatch", "handler")
"""The message stages the runtime times, in the order a command goes through them."""


class RunnerStages:
    """The stages the runtime timed for each connection, since the last take.

    The runner adds a stage when the model reports it, and takes them when it
    builds a report. :meth:`reset` starts a new session from nothing.
    """

    def __init__(self) -> None:
        self._messages = StageTotals(MESSAGE_STAGES)

    def add_message(self, conn_id: ConnId, stage: str, ms: float) -> None:
        """Add one message from *conn_id* that spent *ms* milliseconds in *stage*."""
        self._messages.add(conn_id, stage, ms)

    def reset(self) -> None:
        """Drop every stage, for a session that starts."""
        self._messages.clear()

    def merge(
        self, readings: Iterable[tuple[ConnId, TransportReading]]
    ) -> list[tuple[ConnId, TransportReading]]:
        """Add each connection's stages to its reading, after the transport's, and take them.

        A connection with no reading here keeps its stages until one has.
        """
        readings = list(readings)
        taken = self._messages.take([cid for cid, _ in readings])
        out: list[tuple[ConnId, TransportReading]] = []
        for cid, reading in readings:
            found = taken.get(cid)
            if found:
                reading = replace(
                    reading,
                    message_stages=reading.message_stages
                    + tuple(
                        MessageStage(name, total_ms=total, messages=count)
                        for name, total, count in found
                    ),
                )
            out.append((cid, reading))
        return out
