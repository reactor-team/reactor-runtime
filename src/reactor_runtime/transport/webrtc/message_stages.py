"""Add up where a connection's data-channel messages spend their time, stage by stage.

The engine's chunked data channel keeps running totals of four stages, which a
stats sample turns into a window's worth: ``queue_wait``, ``send`` and
``stall`` for the messages this side sends, and ``reassembly`` for the
messages it receives.

Only the data channel that carries commands is timed. A channel without
chunking has no totals, so it reports no stages.

:class:`MessageStageWindow` holds all of them until a report takes them.
"""

from __future__ import annotations

from reactor_runtime.core import MessageStage
from reactor_runtime.core.stages import StageTotals
from reactor_runtime.transport.webrtc.stats import ChunkingSample

STAGES = (
    "queue_wait",
    "send",
    "stall",
    "reassembly",
)
"""Every stage, in the order a message goes through them."""


class MessageStageWindow:
    """The time a connection's messages spent in each stage, since the last take.

    The stats sampler folds in the channel's totals and a report takes them
    from the event loop, so every access holds a lock.
    """

    def __init__(self) -> None:
        self._totals = StageTotals(STAGES)

    def add(self, stage: str, ms: float, messages: int = 1) -> None:
        """Add *messages* messages that spent *ms* milliseconds in *stage*, together.

        A count below one or a time that is negative or not finite is no
        measurement, and is left out.
        """
        self._totals.add(None, stage, ms, messages)

    def fold(self, sample: ChunkingSample | None, previous: ChunkingSample | None) -> None:
        """Add the stages the chunked channel timed between *previous* and *sample*.

        The totals run for the channel's life, so a window's worth is the
        difference between two samples. No sample, no previous one, or a total
        that went backwards (a channel opened again) adds nothing.

        ``stall`` is the time the pump stopped, spread over the messages sent
        in the window, so its average is the stall time per message.
        """
        if sample is None or previous is None:
            return
        sent = sample.messages_sent - previous.messages_sent
        received = sample.messages_received - previous.messages_received
        if sent > 0:
            for stage, now, before in (
                ("queue_wait", sample.queue_wait_s, previous.queue_wait_s),
                ("send", sample.send_s, previous.send_s),
                ("stall", sample.stall_s, previous.stall_s),
            ):
                self.add(stage, (now - before) * 1000.0, sent)
        if received > 0:
            self.add("reassembly", (sample.reassembly_s - previous.reassembly_s) * 1000.0, received)

    def take(self) -> tuple[MessageStage, ...]:
        """Return the stages in trip order, and start them again."""
        return tuple(
            MessageStage(name, total_ms=total, messages=count)
            for name, total, count in self._totals.take().get(None, [])
        )
