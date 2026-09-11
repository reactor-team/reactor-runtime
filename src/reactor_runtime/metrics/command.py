# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Record command outcomes and ingress times."""

from __future__ import annotations

import time
from collections.abc import Iterable

from prometheus_client import Counter, Histogram

from reactor_runtime.metrics.registry import RuntimeMetrics

# Ingress is the runtime's own work, and it reaches the model in under a
# millisecond while the loop is free. The upper buckets are a loop that is
# starved, which is the condition the measurement exists to expose.
_COMMAND_INGRESS_BUCKETS = (0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
UNKNOWN_COMMAND = "unknown"
"""The command label for a name the model does not declare.

A client sends whatever name it likes, so the name on the wire is not a bounded
value and cannot be a label. Only the commands in the model's schema are bounded.
Every other name shares this one series, which keeps a client that spells a
command wrong in a loop from minting a series for each attempt.
"""


class CommandMetrics:
    """Records what a client asked the model to do, and how long the ask took.

    Every command the runtime admits passes one choke point, which already
    branches on the three outcomes a command can have: the model accepted it, the
    contract rejected it, or it referenced an upload the store could not produce.
    This class names those three outcomes as three methods, so the choke point
    reads as the outcome it just decided and no label value appears at the call
    site.

    Ingress covers what the runtime does with a command before the model sees it:
    the wait for the event loop, the decode, the contract validation, and the
    enqueue. It stops when the command is enqueued, so it measures the runtime and
    not the model — a handler that runs for a minute does not appear here. It also
    excludes the wait for the bytes of an upload the command references, which is
    the client's own latency and would otherwise bury the runtime's.
    """

    def __init__(self, metrics: RuntimeMetrics) -> None:
        """Declare the command instruments on the registry of *metrics*.

        Args:
            metrics: The holder whose registry the instruments register against.
        """
        self._commands = Counter(
            "runtime_commands_total",
            "Commands a client sent, by command and by what the runtime did with it.",
            ["command", "outcome"],
            registry=metrics.registry,
        )
        self._ingress = Histogram(
            "runtime_command_ingress_seconds",
            "How long the runtime took to carry a command from the wire to the model.",
            ["command"],
            buckets=_COMMAND_INGRESS_BUCKETS,
            registry=metrics.registry,
        )

    def declare(self, commands: Iterable[str]) -> None:
        """Seed the series of every command the model declares.

        A command nobody has sent yet has no series at all, which reads the same
        as a command the model does not have. Seeding the declared names answers
        "which of my commands do clients use" off one scrape, with a zero for the
        ones nobody sends.

        Only the two outcomes an ordinary command has are seeded. An unresolved
        upload is a fault, and a row of zeroes for a fault that never happened
        says nothing a missing series does not.
        """
        for command in commands:
            for outcome in ("accepted", "rejected"):
                self._commands.labels(command=command, outcome=outcome)

    def accepted(self, command: str, *, since: float) -> None:
        """Count a command the model took, and measure how long it waited."""
        self._commands.labels(command=command, outcome="accepted").inc()
        self._ingress.labels(command=command).observe(time.monotonic() - since)

    def rejected(self, command: str, *, since: float) -> None:
        """Count a command the contract refused, and measure how long that took.

        A rejection reaches the same choke point as an acceptance and costs the
        same work, so it belongs in the ingress measurement.
        """
        self._commands.labels(command=command, outcome="rejected").inc()
        self._ingress.labels(command=command).observe(time.monotonic() - since)

    def unresolved_upload(self, command: str) -> None:
        """Count a command dropped because an upload it references never arrived.

        This one records no ingress. Ingress measures a command the runtime
        carried to the model, and this one never got there.
        """
        self._commands.labels(command=command, outcome="unresolved_upload").inc()
