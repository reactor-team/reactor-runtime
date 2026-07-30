"""The runtime's metrics registry.

Metrics leave this process only when somebody asks for them. The runtime holds
one Prometheus registry, renders it on ``GET /metrics``, and waits. It opens no
connection of its own to report telemetry, so a deployment scrapes the endpoint
on whatever schedule it likes, and a runtime nobody scrapes pays nothing beyond
the memory of its own counters.

The assembly creates the registry and injects it into the components that observe
on it. No module here holds a registry of its own, so one process renders exactly
the instruments its own components registered, and a test builds a holder, reads
it back, and never sees another test's numbers.

An observation carries only labels with a small, fixed set of values, such as the
name of a command in the model's schema or the name of a member of an enum. The
identity of the process is not one of them: it rides a single ``runtime_info``
series, and the scraper attaches whatever else it knows about where the process
runs.

Every instrument is named ``runtime_`` followed by what it measures. A metric name
is global to the store that ingests it, so the prefix is what keeps a series about
a session in this process distinct from a series about a session in whatever else
a deployment scrapes.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Info, generate_latest

CONTENT_TYPE = CONTENT_TYPE_LATEST
"""The media type of a rendered registry, which is the Prometheus text format."""


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
