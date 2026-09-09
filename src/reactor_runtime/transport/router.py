"""The transport-router base and the session-control surface it binds to.

A :class:`TransportRouter` is a connection-type endpoint group: it mounts the
HTTP routes for one transport onto the shared application and owns that
transport's acceptor (or plays the acceptor role itself when there is no
handshake). Adding a transport is adding a router; nothing else in the runtime
changes.

Every router funnels into the same two runner entry points regardless of how it
negotiated — ``connection_opened`` to register a live connection and the
:class:`~reactor_runtime.core.transport.ConnectionSink` callbacks wired onto
each connection — so several transports can share one session. That neutral
session-facing surface is :class:`SessionControl`; the runner satisfies it by
shape, and signalling never crosses it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from fastapi import FastAPI
from pydantic import BaseModel

from reactor_runtime.core import ConnectionSink, ConnId, SessionState


class ErrorDetail(BaseModel):
    """The body every rejected request carries: a human-readable motive.

    The one error shape of the HTTP surface — the fixed route groups and the
    transport route groups alike declare their non-2xx responses with it, so a
    client generating types from the published contract sees the same
    ``{"detail": ...}`` body everywhere.
    """

    detail: str


class SessionNotRunningError(RuntimeError):
    """Raised by :meth:`SessionControl.require_session_running` when no session is live.

    A neutral signal a router translates into a client-facing rejection. It
    carries no transport detail so the session-control surface stays free of any
    one transport's wire vocabulary.
    """


class UnknownSessionError(RuntimeError):
    """Raised by :meth:`SessionControl.require_session_running` for a wrong session id.

    A request addressed a session id the runtime does not host. Distinct from
    :class:`SessionNotRunningError` — the session-control surface is live but the
    addressed id is not its own — so a router can map it to a not-found rather
    than a not-running rejection.
    """


class TooManyConnectionsError(RuntimeError):
    """Raised when a new connection would exceed the session's concurrent limit.

    The acceptor holds one media peer per live connection, so an unbounded
    number of offers is an unbounded number of native peers. A fresh connection
    offered past the configured ceiling is refused rather than negotiated; a
    re-offer on an already-known connection is a reconnect and is always
    admitted. A router maps this to a transient rejection so a client can retry
    once a slot frees.

    Attributes:
        limit: The concurrent-connection ceiling that was reached.
    """

    def __init__(self, limit: int) -> None:
        """Record the ceiling the offer was refused against."""
        self.limit = limit
        super().__init__(f"connection limit reached ({limit})")


class ConnectionsExhaustedError(RuntimeError):
    """Raised when the session's connection-id space is used up.

    Ids are minted within a bounded per-session range and never reused within
    the session, so a session that mints its whole allowance can mint no more.
    Surfacing this rather than looping keeps id allocation from spinning the
    event loop when the space is full.
    """


class SessionTransitionError(RuntimeError):
    """Raised when a session start or stop is rejected from the current state.

    The session is opened only from ``READY`` and stopped only from a running
    state; a request that does not fit is rejected without changing state. This
    carries the attempted *action* and the :class:`SessionState` the session is
    in, so a router can map the rejection to a precise client response — the
    motive is explicit rather than a silent no-op idempotent success.

    Attributes:
        action: The control verb that was rejected (``"start"`` or ``"stop"``).
        state: The state the session was in when the transition was rejected.
    """

    def __init__(self, action: str, state: SessionState) -> None:
        """Record the rejected action and the state it was rejected from."""
        self.action = action
        self.state = state
        super().__init__(f"cannot {action} session from state {state.name.lower()}")


@runtime_checkable
class SessionControl(ConnectionSink, Protocol):
    """The session-facing surface a router drives, beyond the upward sink.

    Composes the upward :class:`~reactor_runtime.core.transport.ConnectionSink`
    with the few session-control operations a router needs to admit a
    connection: guarding that a session is live, minting connection ids
    centrally so transports cannot collide, and reporting the model's declared
    track manifest for connection setup. The runner implements it; a router only
    ever holds it as this shape.
    """

    def require_session_running(self, sid: str) -> None:
        """Admit a request only against the live session.

        Raise :class:`SessionNotRunningError` unless a session is live, and
        :class:`UnknownSessionError` when *sid* is not the id the runtime hosts.

        Args:
            sid: The session id the request addressed.
        """

    def new_conn_id(self) -> ConnId:
        """Mint a fresh connection id, unique within the session."""

    def offer_admitted(self, conn_id: ConnId) -> None:
        """Stamp an admitted connection offer with the live session.

        Called as a transport accepts an offer, before negotiation begins. The
        runner uses the stamp to refuse the wire if it only reaches its
        connected state once a later session is running.
        """

    def track_map(self) -> Mapping[str, Any]:
        """Return the model's declared track manifest for connection setup."""


class TransportRouter(ABC):
    """Mount one transport's routes onto the shared application.

    Constructed ahead of serving and handed the application and the runner once.
    A concrete router adds its route group, owns its acceptor (or accepts
    connections inline), and references the runner only as the
    :class:`SessionControl` surface — never reaching into transport internals.
    """

    @abstractmethod
    def mount(self, app: FastAPI, runner: SessionControl) -> None:
        """Register this transport's routes against *app*, bound to *runner*."""
