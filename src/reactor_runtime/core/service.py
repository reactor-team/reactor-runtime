"""Service-lifecycle vocabulary.

The contract the service supervises and the config threaded through ``serve``. A
``ServiceComponent`` declares its startup dependencies and supports an ordered
drain (stop taking new work) distinct from stop (release resources), so the
service alone arbitrates start, drain, and stop ordering. ``RuntimeConfig`` is
the single object that configures one runtime process.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from reactor_runtime.core.values import Health


@runtime_checkable
class ServiceComponent(Protocol):
    """A piece the service supervises through the process lifecycle.

    A component names the components it must start after and exposes the four
    lifecycle verbs the service drives. It never manages its own place in startup
    or shutdown: the service is the sole arbiter of ordering, and the drain verb
    — stop accepting new work, let in-flight finish — is what graceful shutdown
    of active sessions needs.
    """

    name: str
    depends_on: tuple[str, ...]

    async def start(self) -> None:
        """Bring the component up. Called in dependency order."""

    async def drain(self) -> None:
        """Stop accepting new work and let in-flight work finish."""

    async def stop(self) -> None:
        """Release resources. Called after draining, in reverse order."""

    def health(self) -> Health:
        """Report current readiness, for aggregation into process health."""


@dataclass(frozen=True)
class RuntimeConfig:
    """The single configuration object threaded through ``serve``.

    Names where to find the model, how to reach it, and the lifecycle tunables
    the service and runner need. Concrete components read the fields they care
    about; later flows extend it as they grow needs.

    Attributes:
        model_ref: Import reference to the model class, ``"module:Class"``.
        model_config: Opaque settings handed to the model at load time.
        host: Address the HTTP ingress binds.
        port: Port the HTTP ingress binds.
        grace_period: Seconds a draining session is given to end before stop.
        orphan_timeout: Seconds a session may stay client-less before it closes.
        recording_dir: Directory recordings are written to, or ``None`` to
            disable recording.
    """

    model_ref: str
    model_config: Mapping[str, Any] = field(default_factory=dict)
    host: str = "0.0.0.0"
    port: int = 8080
    grace_period: float = 30.0
    orphan_timeout: float = 60.0
    recording_dir: str | None = None
