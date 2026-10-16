"""Hold the stages the runtime times itself, per connection, until a report takes them.

A transport times the stages it sees on the wire and reports them on its
reading. The runner and the model loop time the rest. For each command:
``dispatch``, from the command reaching the runtime until its handler starts,
and ``handler``, the handler running and its reply being written. For each
frame: ``input_wait``, the frame in the model's input buffer until a step
reads it, timed for the connection that sent it; and the step's
``process_input``, ``generate`` and ``process_output``, timed once per step
and counted on each video track the step emits, for every connection.
:class:`RunnerStages` keeps those per connection, and :meth:`RunnerStages.merge`
adds them to each connection's reading beside the transport's own.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace

from reactor_runtime.core import (
    ConnId,
    FrameStage,
    MessageStage,
    TransportReading,
)
from reactor_runtime.core.stages import StageTotals

MESSAGE_STAGES = ("dispatch", "handler")
"""The message stages the runtime times, in the order a command goes through them."""

FRAME_STAGES = ("input_wait", "process_input", "generate", "process_output")
"""The frame stages the runtime times, in the order a frame goes through them."""


class RunnerStages:
    """The stages the runtime timed for each connection, since the last take.

    The runner adds a stage when the model reports it, and takes them when it
    builds a report. :meth:`reset` starts a new session from nothing.
    """

    def __init__(self) -> None:
        self._messages = StageTotals(MESSAGE_STAGES)
        # Keyed by (connection, track).
        self._frames = StageTotals(FRAME_STAGES)

    def add_message(self, conn_id: ConnId, stage: str, ms: float) -> None:
        """Add one message from *conn_id* that spent *ms* milliseconds in *stage*."""
        self._messages.add(conn_id, stage, ms)

    def add_frame(self, conn_id: ConnId, track: str, stage: str, ms: float) -> None:
        """Add one frame on *conn_id*'s *track* that spent *ms* milliseconds in *stage*."""
        self._frames.add((conn_id, track), stage, ms)

    def add_step(
        self, conn_ids: Iterable[ConnId], tracks: Iterable[str], stages: Mapping[str, float]
    ) -> None:
        """Add a step's *stages* once on each of *tracks*, for each of *conn_ids*."""
        tracks = tuple(tracks)
        for conn_id in conn_ids:
            for track in tracks:
                for stage, ms in stages.items():
                    self._frames.add((conn_id, track), stage, ms)

    def reset(self) -> None:
        """Drop every stage, for a session that starts."""
        self._messages.clear()
        self._frames.clear()

    def merge(
        self, readings: Iterable[tuple[ConnId, TransportReading]]
    ) -> list[tuple[ConnId, TransportReading]]:
        """Add each connection's stages to its reading, after the transport's, and take them.

        A track's frame stages go to that track's reading. A connection with no
        reading here, or a track its reading does not carry yet, keeps its
        stages until one does.
        """
        readings = list(readings)
        messages = self._messages.take([cid for cid, _ in readings])
        frames = self._frames.take(
            [(cid, track.track_name) for cid, reading in readings for track in reading.tracks]
        )
        out: list[tuple[ConnId, TransportReading]] = []
        for cid, reading in readings:
            found = messages.get(cid)
            if found:
                reading = replace(
                    reading,
                    message_stages=reading.message_stages
                    + tuple(
                        MessageStage(name, total_ms=total, messages=count)
                        for name, total, count in found
                    ),
                )
            if any((cid, track.track_name) in frames for track in reading.tracks):
                reading = replace(
                    reading,
                    tracks=tuple(
                        replace(
                            track,
                            frame_stages=track.frame_stages
                            + tuple(
                                FrameStage(name, total_ms=total, frames=count)
                                for name, total, count in frames.get((cid, track.track_name), [])
                            ),
                        )
                        for track in reading.tracks
                    ),
                )
            out.append((cid, reading))
        return out
