"""Add up where a connection's data-channel messages spend their time, stage by stage.

The engine's chunked data channel keeps running totals, which a stats sample
turns into a window's worth. For the messages this side sends: ``enqueue``
(from ``send`` until the message is in the queue), ``queue_wait``, ``send``
and ``stall``. ``queue_wait`` is also timed in parts, the steps of handing a
message's first chunk to libwebrtc: ``first_check``, ``first_build`` and
``first_native``. ``rest_native`` is libwebrtc's send of the later chunks,
within ``send``. For the messages it receives: ``reassembly``. The engine also
times ``handoff`` on a received message: the wait for the GIL and the copy into
Python. The peer times one more itself, ``loop_wait``: a message from reaching
the peer's sink until the event loop takes it.

Only the data channel that carries commands is timed. A channel without
chunking has no totals, so it reports no stages.

:class:`MessageStageWindow` holds all of them until a report takes them.
"""

from __future__ import annotations

from reactor_runtime.core import MessageStage
from reactor_runtime.core.stages import StageTotals
from reactor_runtime.transport.webrtc.stats import ChunkingSample

STAGES = (
    "enqueue",
    "queue_wait",
    "first_check",
    "first_build",
    "first_native",
    "send",
    "rest_native",
    "stall",
    "reassembly",
    "handoff",
    "loop_wait",
)
"""Every stage, in the order a message goes through them."""

PARTS = {
    "first_check": "queue_wait",
    "first_build": "queue_wait",
    "first_native": "queue_wait",
    "rest_native": "send",
}
"""The stages timed in parts, by part: each part's time is within its stage's."""


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
        ``rest_native`` is libwebrtc's send of every chunk but each message's
        first, spread the same way.
        """
        if sample is None or previous is None:
            return
        sent = sample.messages_sent - previous.messages_sent
        received = sample.messages_received - previous.messages_received
        if sent > 0:
            native_s = sample.native_send_s - previous.native_send_s
            first_native_s = sample.first_native_s - previous.first_native_s
            for stage, seconds in (
                ("enqueue", sample.enqueue_s - previous.enqueue_s),
                ("queue_wait", sample.queue_wait_s - previous.queue_wait_s),
                ("first_check", sample.first_check_s - previous.first_check_s),
                ("first_build", sample.first_build_s - previous.first_build_s),
                ("first_native", first_native_s),
                ("send", sample.send_s - previous.send_s),
                ("rest_native", native_s - first_native_s),
                ("stall", sample.stall_s - previous.stall_s),
            ):
                self.add(stage, seconds * 1000.0, sent)
        if received > 0:
            self.add("reassembly", (sample.reassembly_s - previous.reassembly_s) * 1000.0, received)

    def take(self) -> tuple[MessageStage, ...]:
        """Return the stages in trip order, and start them again."""
        return tuple(
            MessageStage(name, total_ms=total, messages=count, part_of=PARTS.get(name, ""))
            for name, total, count in self._totals.take().get(None, [])
        )
