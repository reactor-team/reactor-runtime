"""The single door to the model — :class:`ModelBridge`.

Everything that crosses between the session/transport world and the model's own
thread goes through here, and nowhere else. The bridge has three inbound faces
split by *authority* — a client-authored command is validated against the
contract before it is admitted, a reactor-authoritative event is trusted and
passed straight through, and media rides its own plane — plus one outbound face
bound once at start, and the start/stop of the model itself.

Each method takes only serialisable arguments. Today the call hops a thread onto
the model's queues; the same surface is what later becomes the process boundary,
so nothing here reaches into model internals beyond the handful of entrypoints
:class:`~reactor_runtime.interface.internal.reactor_core.ReactorCore` exposes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reactor_runtime.core.model import ReactorEvent
from reactor_runtime.core.values import ConnId, InputFrame
from reactor_runtime.interface.internal.reactor_core import (
    AddressedSink,
    BroadcastSink,
    FailureSink,
    MediaOps,
    MediaSink,
    ReactorCore,
    RequestId,
)
from reactor_runtime.interface.model.contract import ContractError, ModelContract


@dataclass(frozen=True)
class CommandOutcome:
    """The result of admitting a command at the bridge.

    Validation is the only thing that can reject a command synchronously; once
    admitted it is dispatched asynchronously on the model thread, so an accepted
    outcome means *valid and enqueued*, not *handled*.

    Attributes:
        accepted: Whether the command passed validation and was enqueued.
        field: The offending field on rejection, else ``None``.
        reason: Why it was rejected, else ``None``.
    """

    accepted: bool
    field: str | None = None
    reason: str | None = None

    @classmethod
    def accept(cls) -> CommandOutcome:
        """Return an accepted outcome."""
        return cls(accepted=True)

    @classmethod
    def reject(cls, field: str, reason: str) -> CommandOutcome:
        """Return a rejected outcome carrying the failing field and reason."""
        return cls(accepted=False, field=field, reason=reason)


class ModelBridge:
    """The one boundary between the runner and the model.

    Built with a model and the contract computed from its class — the bridge
    never builds the contract, it receives both fully formed. Outbound is bound
    once via :meth:`bind_outbound`, then :meth:`start` spawns the model.

    Args:
        model: The engine to drive.
        contract: The model's contract, used to validate inbound commands.
    """

    def __init__(self, model: ReactorCore, contract: ModelContract) -> None:
        self._model = model
        self._contract = contract
        self._outbound_bound = False

    @property
    def contract(self) -> ModelContract:
        """The model's contract."""
        return self._contract

    # -- inbound: user-authored, validated --------------------------------

    async def submit_command(
        self,
        name: str,
        raw_args: dict[str, Any],
        *,
        conn_id: ConnId | None,
        request_id: RequestId,
    ) -> CommandOutcome:
        """Validate a client command against the contract and admit it.

        The single place request-time validation runs. A payload that fails the
        contract is rejected here and never reaches the model; a valid one is
        turned into a typed command and enqueued with its addressing for a reply.

        Args:
            name: The command name from the wire.
            raw_args: The raw argument mapping from the wire.
            conn_id: The connection that sent it, when known.
            request_id: The request id to correlate a reply against.

        Returns:
            An accepted outcome when the command was valid and enqueued, else a
            rejected one carrying the failing field and reason.
        """
        try:
            command = self._contract.validate(name, raw_args)
        except ContractError as error:
            return CommandOutcome.reject(error.field, error.reason)
        self._model.submit_command(command, conn_id, request_id)
        return CommandOutcome.accept()

    # -- inbound: reactor-authoritative, trusted --------------------------

    def dispatch_reactor_event(self, event: ReactorEvent) -> None:
        """Hand the model an authoritative fact, unvalidated.

        Reactor events are authored by the runtime, never by a client, so they
        are trusted and pass straight to the model. Uploads belong here too — the
        runtime has already fetched and vouched for the bytes.
        """
        self._model.post_reactor_event(event)

    # -- inbound: media plane ---------------------------------------------

    def push_media(self, track: str, frame: InputFrame) -> None:
        """Route an inbound frame to its track's buffer."""
        self._model.push_media(track, frame)

    # -- outbound: bound once before start --------------------------------

    def bind_outbound(
        self,
        *,
        broadcast: BroadcastSink,
        addressed: AddressedSink,
        media: MediaSink,
        media_ops: MediaOps | None = None,
        failure: FailureSink | None = None,
    ) -> None:
        """Wire the model's outbound paths down into the runner. Call once.

        ``broadcast`` delivers a message to every client; ``addressed`` delivers
        one to a single connection, correlated to a request id when it is a reply
        (a ``None`` message is the bodyless acknowledgement of a command that
        returned nothing); ``media`` receives each emitted :class:`MediaChunk` —
        a chunk asking for backpressure (``chunk.wait``) may hold the caller
        until downstream has room, and the model side dispatches the call to a
        worker thread accordingly. ``media_ops`` carries the playout operations
        behind the model's ``output`` handle, each a one-way call with plain
        arguments. ``failure`` receives the exception that ends the model's run
        loop, at most once, on the model thread — it is how the owner learns
        the model died rather than idled.

        Args:
            broadcast: Sink for a message sent to all clients.
            addressed: Sink for a reply sent to one connection.
            media: Sink for each emitted media chunk.
            media_ops: Operations over the downstream media consumers, for
                the model's ``output`` handle.
            failure: Sink for an unrecoverable crash of the model's run loop.

        Raises:
            RuntimeError: If outbound has already been bound.
        """
        if self._outbound_bound:
            raise RuntimeError("bind_outbound must be called once")
        self._model.bind_output(
            broadcast=broadcast, addressed=addressed, media=media, media_ops=media_ops
        )
        if failure is not None:
            self._model.bind_failure(failure)
        self._outbound_bound = True

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Spawn the model thread.

        Raises:
            RuntimeError: If :meth:`bind_outbound` has not been called.
        """
        if not self._outbound_bound:
            raise RuntimeError("bind_outbound must be called before start")
        self._model.start_thread()

    async def stop(self) -> None:
        """Cancel the model loop."""
        self._model.stop()
