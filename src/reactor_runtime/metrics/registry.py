# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Own the shared Prometheus registry and process identity."""

from prometheus_client import CollectorRegistry, Info, generate_latest


class RuntimeMetrics:
    """The registry one process observes on, and the identity it publishes.

    Every instrument in the process registers against :attr:`registry`, and
    :meth:`render` serves it. The process states its own version and model once,
    as a ``runtime_info`` series, so no other instrument needs to repeat
    that identity on each observation.
    """

    def __init__(self, *, version: str, model: str) -> None:
        """Create an empty registry and publish the identity of the process on it.

        Args:
            version: The version of the runtime that runs in this process.
            model: The reference of the model this process hosts.
        """
        self.registry = CollectorRegistry()
        Info(
            "runtime",
            "The version of the runtime and the model this process hosts.",
            registry=self.registry,
        ).info({"version": version, "model": model})

    def render(self) -> bytes:
        """Render the registry in the Prometheus text format.

        The registry exists before the model starts to load, so a scrape during a
        slow load answers with the identity of the process and every observation
        made so far.
        """
        return generate_latest(self.registry)
