"""The stepped path of a runner: a client's step requests and the answer to each.

A stepped session runs one step per ``RequestStep`` a client sends on the data
channel. :class:`StepRequests` takes each request from the connection that sent
it, checks it whole against the model's contract and input tracks, and hands it
to the model as plain calls and its step, or refuses it with a ``StepResult``
that takes no number. When the model reports the step, it encodes the step's
media on the model's thread and sends the ``StepResult`` back on the data
channel.

A connection's step requests, and in a stepped session its commands, are
submitted one at a time in the order they arrived (:class:`StepOrder`), so one
that waits for an upload holds back the ones sent after it.

The runner owns the session; the stepped path reads it through a
:class:`StepHost`, so it can be driven, and tested, without one.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Mapping
from dataclasses import dataclass
from typing import Any

from reactor_runtime.core import (
    CompletedStep,
    ConnId,
    SessionEvent,
    StepRefusal,
    StepRefusalKind,
    TrackDirection,
    TrackKind,
)
from reactor_runtime.interface.internal.bridge import CommandOutcome, ModelBridge
from reactor_runtime.interface.internal.reactor_core import FrameSource
from reactor_runtime.log import get_logger
from reactor_runtime.message_gateway import InboundStep
from reactor_runtime.metrics import CommandMetrics
from reactor_runtime.protocol import Channel, Codec, ProtocolVersion, select
from reactor_runtime.protocol.frames import (
    FrameEncodingError,
    check_frame,
    decode_frame,
    media_size,
    track_frames,
)
from reactor_runtime.runner.connection_manager import ConnectionManager
from reactor_runtime.runner.session_start import Generation
from reactor_runtime.step_results import content_type_of
from reactor_runtime.upload_store import UnknownUploadError
from reactor_wire.v1 import platform_pb2, step_pb2

logger = get_logger(__name__)

# The wire code a step the model did not run is answered with, by why it did not.
REFUSAL_CODES: dict[StepRefusalKind, step_pb2.StepFailureCode] = {
    StepRefusalKind.REFUSED: step_pb2.STEP_FAILURE_CODE_REFUSED,
    StepRefusalKind.NOT_READY: step_pb2.STEP_FAILURE_CODE_NOT_READY,
    StepRefusalKind.COMMAND_FAILED: step_pb2.STEP_FAILURE_CODE_COMMAND_FAILED,
}

# The runtime's track kind for each kind a wire message names.
WIRE_TRACK_KINDS = {
    platform_pb2.TRACK_KIND_VIDEO: TrackKind.VIDEO,
    platform_pb2.TRACK_KIND_AUDIO: TrackKind.AUDIO,
}


@dataclass(frozen=True)
class EncodedResult:
    """A requested step's result as the model's thread encoded it, without its files.

    Exactly one is set: *frame*, the v1 frame ready to send; *oversized*, the
    size of media that does not fit the connection, never copied; or *error*,
    why the media cannot be carried as frames.
    """

    frame: bytes | None = None
    oversized: int = 0
    error: str | None = None


class StepInputError(ValueError):
    """A step request's input frames cannot be written to the model's tracks."""

    def __init__(self, entry: str, message: str) -> None:
        super().__init__(message)
        self.entry = entry


def refused(key: str, outcome: CommandOutcome) -> str:
    """Return why the command written under *key* failed the contract."""
    reason = str(outcome.reason)
    # A constraint's reason already opens with the field it names.
    if not reason.startswith(f"{outcome.field}:"):
        reason = f"{outcome.field}: {reason}"
    return f"{key} is refused: {reason}"


def step_file_refs(step: CompletedStep) -> list[step_pb2.StepFileRef]:
    """Reference each extra file a saved step keeps, as its step result is saved.

    A failed step keeps no files, so its result references none.
    """
    if step.error is not None:
        return []
    return [
        step_pb2.StepFileRef(name=name, content_type=content_type_of(name), size=len(data))
        for name, data in step.files.items()
    ]


@dataclass(frozen=True)
class StepHost:
    """What the stepped path reads from, and does through, the runner that owns it.

    The values that change over the runner's life (its model, the session, its
    mode, the session count, the starting input's progress) are read through a
    call each time, so the stepped path always sees the current ones.

    Attributes:
        bridge: The loaded model's bridge, or ``None`` before a model loads.
        session_running: Whether a session is in a state that runs steps.
        generation: The mode the current session was started in.
        sessions_posted: How many sessions the runner has posted to the model,
            which a refusal from an earlier session is told apart by.
        starting_input_done: The event set once the current session's starting
            input is submitted.
        connections: The runner's connections, which step replies go out on.
        journal: Record a session event on the runner's journal.
        resolve_uploads: Resolve a command's upload references to files,
            waiting up to the given seconds for each one's bytes.
        command_metrics: The runner's command instruments.
        command_label: A command name made safe to label a metric with.
        codec_for: The codec for a connection's protocol version.
        spawn: Run a coroutine as a task the runner tracks.
    """

    bridge: Callable[[], ModelBridge | None]
    session_running: Callable[[], bool]
    generation: Callable[[], Generation]
    sessions_posted: Callable[[], int]
    starting_input_done: Callable[[], asyncio.Event]
    connections: ConnectionManager
    journal: Callable[..., object]
    resolve_uploads: Callable[
        [str, Mapping[str, Any], Mapping[str, str], float], Awaitable[dict[str, Any]]
    ]
    command_metrics: CommandMetrics
    command_label: Callable[[str], str]
    codec_for: Callable[[ProtocolVersion], Codec]
    spawn: Callable[[Coroutine[Any, Any, None]], None]


class StepOrder:
    """Each connection's step requests, and stepped commands, one at a time in arrival order.

    Holds, per connection, its lock and how many messages hold or wait for it.
    """

    def __init__(self) -> None:
        self._order: dict[ConnId, tuple[asyncio.Lock, int]] = {}

    def __len__(self) -> int:
        """Return how many connections have a message holding or waiting for their order."""
        return len(self._order)

    @contextlib.asynccontextmanager
    async def hold(self, conn_id: ConnId) -> AsyncIterator[None]:
        """Hold *conn_id*'s step order until the step request or command in hand is submitted.

        Each message is handled on its own task, and the tasks start in the
        order the messages arrived, so taking the lock before the first await
        queues the messages in that order. The entry goes once no message of the
        connection holds or waits for it.
        """
        lock, users = self._order.get(conn_id, (asyncio.Lock(), 0))
        self._order[conn_id] = (lock, users + 1)
        try:
            async with lock:
                yield
        finally:
            _, users = self._order[conn_id]
            if users == 1:
                del self._order[conn_id]
            else:
                self._order[conn_id] = (lock, users - 1)


class StepRequests:
    """A runner's stepped path: submit each step request, and answer it.

    Args:
        host: What the stepped path reads from the runner that owns it.
        upload_wait: How long, in seconds, a request waits for each of its
            uploads' bytes by default.
    """

    def __init__(self, host: StepHost, *, upload_wait: float) -> None:
        self._host = host
        self._upload_wait = upload_wait
        self.order = StepOrder()
        # Step results are v1 frames: only the v1 codec carries step traffic.
        self._codec = select(ProtocolVersion.V1)

    async def submit_client_step(self, step: InboundStep) -> None:
        """Submit a client's step request once the session's starting input is submitted.

        A request waits behind the starting input as a client's command does,
        so the model sees the starting input, and its ``StartingInputApplied``,
        before the first requested step. A connection's requests are submitted
        one at a time, in the order they arrived, so one that waits for an
        upload holds back the requests sent after it.
        """
        async with self.order.hold(step.conn_id):
            await self._host.starting_input_done().wait()
            await self.submit(step)

    async def submit(self, step: InboundStep, *, upload_wait: float | None = None) -> None:
        """Check a step request whole, then hand it to the model as plain calls and its step.

        Each ``state`` key becomes the model's ``set_<key>`` command, in
        ascending key order, then each of the request's commands follows, in
        order. Every call has its uploads resolved, whether referenced in its
        ``uploads`` map or inline in an argument (a ``state`` value of an
        upload-typed field carries its file as an inline ``{"upload_id": ...}``
        reference), and is checked against the contract; every frame is checked
        against its input track, and each track's frames against its buffer.
        Only a request that passes every check reaches the model: its calls are
        posted and journalled, each carrying the step's request id and its
        entry, so a handler that fails answers the step with
        ``COMMAND_FAILED``: the calls after it are skipped and the step does
        not run.

        A request that does not run is answered with a ``StepResult`` that
        takes no number. It is ``NOT_READY`` when the session is not running.
        It is ``REFUSED`` when the session does not step or the connection did
        not negotiate data-channel chunking, or takes less than one plain
        message (256 KiB) a message, and, naming the entry, when an entry's
        upload does not arrive or it fails the contract, or when a media input
        entry names a track the model does not take, holds a frame that cannot
        be decoded, or holds more than the track's buffer does; such a request
        changes nothing. Neither is journalled or counted toward the session's
        steps. A request with no entries and no frames runs a step on the
        model's current state.
        """
        if upload_wait is None:
            upload_wait = self._upload_wait
        bridge = self._host.bridge()
        if bridge is None:
            return
        # The session's state first: the start it would read the mode from
        # outlives its session, so outside one it is the previous session's.
        if not self._host.session_running():
            self.refuse(
                step.conn_id,
                step.request_id,
                "the session is not running a step yet",
                failure_code=step_pb2.STEP_FAILURE_CODE_NOT_READY,
            )
            return
        if self._host.generation() is not Generation.STEPPED:
            self.refuse(
                step.conn_id,
                step.request_id,
                "this session is not stepped; start it with generation 'stepped' to request steps",
            )
            return
        if not self._host.connections.message_chunking(step.conn_id):
            self.refuse(
                step.conn_id,
                step.request_id,
                "this connection cannot carry a step's result; a stepped session needs a "
                "WebRTC connection that negotiated data-channel chunking",
            )
            return

        def refuse(entry: str, reason: str) -> None:
            self._host.journal(SessionEvent.ERROR, message=f"step request: {reason}")
            self.refuse(step.conn_id, step.request_id, reason, entry=entry)

        entries = [
            (f"state.{key}", f"set_{key}", {key: value}, {})
            for key, value in sorted(step.state.items())
        ]
        entries += [
            (f"commands[{index}]", command.name, dict(command.args), dict(command.uploads))
            for index, command in enumerate(step.commands)
        ]
        calls: list[tuple[str, str, dict[str, Any]]] = []
        for key, name, args, uploads in entries:
            try:
                resolved = await self._host.resolve_uploads(name, args, uploads, upload_wait)
            except UnknownUploadError:
                self._host.command_metrics.unresolved_upload(self._host.command_label(name))
                refuse(key, f"{key} references an unresolved upload")
                return
            calls.append((key, name, resolved))
        try:
            media_input = self.check_media_input(step.media_input)
        except StepInputError as error:
            refuse(error.entry, str(error))
            return
        # Nothing awaits from here to the step, so the whole request is posted
        # or none of it is.
        rejected = bridge.submit_step(
            calls,
            media_input,
            conn_id=step.conn_id,
            request_id=step.request_id,
            max_message_size=self._host.connections.max_message_size(step.conn_id),
        )
        if rejected is not None:
            key, outcome = rejected
            name = next(name for entry, name, _ in calls if entry == key)
            self._host.command_metrics.rejected(
                self._host.command_label(name), since=step.received_at
            )
            refuse(key, refused(key, outcome))
            return
        # Counted and journalled as a client's own command is, so a step's
        # calls show up in the same metrics and errors.
        for _, name, args, _ in entries:
            self._host.command_metrics.accepted(
                self._host.command_label(name), since=step.received_at
            )
            self._host.journal(SessionEvent.COMMAND, name=name, args=args, conn_id=step.conn_id)

    def check_media_input(
        self, media_input: tuple[step_pb2.TrackFrames, ...]
    ) -> dict[str, list[FrameSource]]:
        """Check a step request's media input, by the input track each entry belongs to.

        Every frame is checked without copying its samples; the copy happens
        when the model's thread calls the source this returns for it.

        Raises:
            StepInputError: An entry names a track the model does not take, a
                track twice, or a kind the track is not, holds a frame that
                cannot be decoded, or holds more than the track's buffer does.
        """
        bridge = self._host.bridge()
        assert bridge is not None
        tracks = {
            name: info
            for name, info in bridge.contract.tracks.items()
            if info.direction is TrackDirection.IN
        }
        checked: dict[str, list[FrameSource]] = {}
        for index, entry in enumerate(media_input):
            where = f"media_input[{index}]"
            info = tracks.get(entry.track)
            if info is None:
                raise StepInputError(
                    where,
                    f"{where} names {entry.track!r}, which is not an input track of the model",
                )
            if entry.track in checked:
                raise StepInputError(where, f"{where} names {entry.track!r} a second time")
            kind = WIRE_TRACK_KINDS.get(entry.kind)
            if entry.kind != platform_pb2.TRACK_KIND_UNSPECIFIED and kind is not info.kind:
                raise StepInputError(
                    where, f"{where} says {entry.track!r} is not a {info.kind} track"
                )
            frames: list[FrameSource] = []
            samples = 0
            for position, frame in enumerate(entry.frames):
                try:
                    shape = check_frame(frame, info.kind)
                except FrameEncodingError as error:
                    raise StepInputError(where, f"{where}.frames[{position}]: {error}") from None
                if info.kind is TrackKind.AUDIO:
                    samples += shape[-1]
                frames.append(functools.partial(decode_frame, frame, info.kind))
            bound = bridge.input_overflow(entry.track, len(frames), samples)
            if bound is not None:
                raise StepInputError(
                    where,
                    f"{where} holds {len(frames)} frames for {entry.track!r}, "
                    f"whose input buffer holds {bound}",
                )
            checked[entry.track] = frames
        return checked

    def refuse(
        self,
        conn_id: ConnId,
        request_id: str,
        reason: str,
        *,
        failure_code: step_pb2.StepFailureCode = step_pb2.STEP_FAILURE_CODE_REFUSED,
        code: str | None = None,
        entry: str | None = None,
    ) -> None:
        """Answer a step request without a result, on the data channel.

        The answer is a ``StepResult`` with no number, no elapsed time, and no
        media, carrying *failure_code* and *reason* as its error, the failing
        handler's *code*, and the request's *entry* at fault. Nothing is
        journalled or counted for it.
        """
        logger.debug(
            "step request did not run",
            conn_id=conn_id,
            reason=reason,
            failure_code=step_pb2.StepFailureCode.Name(failure_code),
        )
        result = step_pb2.StepResult(
            failure_code=failure_code, error=reason, code=code, entry=entry
        )
        self._host.connections.send_step_reply(
            conn_id,
            lambda version: self._host.codec_for(version).encode_step_result(request_id, result),
        )

    def on_refusal(self, refusal: StepRefusal) -> None:
        """Answer a step request the model did not run.

        Called on the model's thread and sent from it, so the answer goes out
        behind any reply to the commands the model handled before the step. A
        refusal from an earlier session is answered as a session that ended,
        since that session's state is gone.
        """
        if refusal.session != self._host.sessions_posted():
            self.end_request(refusal.conn_id, refusal.request_id)
            return
        self.refuse(
            refusal.conn_id,
            refusal.request_id,
            refusal.reason,
            failure_code=REFUSAL_CODES[refusal.kind],
            code=refusal.code,
            entry=refusal.entry,
        )

    def end_request(self, conn_id: ConnId, request_id: str) -> None:
        """Answer a step request whose session ended before its result could go out."""
        self.refuse(
            conn_id,
            request_id,
            "the session ended before the step's result was sent",
            failure_code=step_pb2.STEP_FAILURE_CODE_SESSION_ENDED,
        )

    def encode_result(self, step: CompletedStep) -> EncodedResult | None:
        """Encode a requested step's result, on the model's thread.

        Copying and serializing a step's media is a result's heaviest work, so
        all of it happens here, before the hop that keeps the step's replies in
        order, and not on the runtime loop every connection shares: the loop
        gets the finished frame. Media larger than the connection took when the
        request was admitted is not copied at all. The step's files are left
        out, since whether the step store takes the step is decided on the loop.

        Returns:
            The encoded result, or ``None`` for a step no client requested.
        """
        request = step.request
        if request is None:
            return None
        limit = request.max_message_size
        size = media_size(step.bundle)
        if limit is not None and size > limit:
            return EncodedResult(oversized=size)
        try:
            tracks = track_frames(step.bundle, video_fps=step.fps)
        except FrameEncodingError as error:
            return EncodedResult(error=str(error))
        result = step_pb2.StepResult(
            step=request.index,
            tracks=tracks,
            # The default loop, the only one a stepped session runs, always
            # measures the step, so a missing time does not happen in practice.
            elapsed_seconds=step.elapsed or 0.0,
            error=step.error,
            failure_code=(
                step_pb2.STEP_FAILURE_CODE_FAILED
                if step.error is not None
                else step_pb2.STEP_FAILURE_CODE_UNSPECIFIED
            ),
        )
        _, frame = self._codec.encode_step_result(request.request_id, result)
        assert isinstance(frame, bytes)
        return EncodedResult(frame=frame)

    def send_result(
        self,
        step: CompletedStep,
        index: int,
        *,
        saved: bool,
        encoded: EncodedResult | None = None,
    ) -> None:
        """Answer a requested step with its result, built from the step's report.

        The result is the report's media, one entry per output track, so what
        the client receives is what the step reported and what a saved step
        keeps, with the seconds its ``generate()`` took (or what the model
        reported), the same measurement a saved step records as
        ``generate_s``. When the step store took the step (*saved*), the
        result names each extra file the model kept with it, which the client
        reads from the session's step results; the bytes stay out of the
        message, so a file counts toward the message's size only by its name,
        media type, and size. It goes out as one message, before the store has
        written the step: the files are readable once the step's
        ``result.json`` is, and a save that fails keeps none, saying why in
        ``save_error``.

        A step that ran always answers with its number and elapsed time. One
        whose model reported a failure answers ``FAILED`` with the model's
        message, as does one whose media cannot be carried as frames. One whose
        result is larger than the connection takes, by its own limit or by the
        media engine's refusal, answers ``RESULT_TOO_LARGE`` with no media or
        files, and the session goes on.

        *encoded* is the result the model's thread encoded (see
        :meth:`encode_result`); without it, it is encoded here.
        """
        request = step.request
        assert request is not None
        if encoded is None:
            encoded = self.encode_result(step)
            assert encoded is not None
        elapsed = step.elapsed or 0.0
        request_id = request.request_id

        def too_large(size: int, *, at_least: bool = False) -> bytes:
            logger.warning("a step result does not fit one message", step=index, size=size)
            measured = f"at least {size}" if at_least else f"{size}"
            _, frame = self._codec.encode_step_result(
                request_id,
                step_pb2.StepResult(
                    step=index,
                    elapsed_seconds=elapsed,
                    failure_code=step_pb2.STEP_FAILURE_CODE_RESULT_TOO_LARGE,
                    error=(
                        f"step {index}'s result is {measured} bytes, "
                        "more than this connection takes in one message"
                    ),
                ),
            )
            assert isinstance(frame, bytes)
            return frame

        if encoded.error is not None:
            logger.error("a step's output could not be encoded", step=index, error=encoded.error)
            _, failed = self._codec.encode_step_result(
                request_id,
                step_pb2.StepResult(
                    step=index,
                    elapsed_seconds=elapsed,
                    failure_code=step_pb2.STEP_FAILURE_CODE_FAILED,
                    error=f"step {index}'s output could not be encoded: {encoded.error}",
                ),
            )
            self._host.connections.send_step_reply(
                request.conn_id, lambda _: (Channel.DATA, failed)
            )
            return
        if encoded.frame is None:
            # Frames alone that outgrow one message were never copied.
            answer = too_large(encoded.oversized, at_least=True)
            self._host.connections.send_step_reply(
                request.conn_id, lambda _: (Channel.DATA, answer)
            )
            return
        frame = encoded.frame
        refs = step_file_refs(step) if saved else ()
        if refs:
            # Serialized protobuf messages merge when concatenated: the files
            # join the result the model's thread encoded. Only a saved step
            # that kept files pays this copy on the loop.
            _, files = self._codec.encode_step_result(request_id, step_pb2.StepResult(files=refs))
            assert isinstance(files, bytes)
            frame += files
        self._host.spawn(self._host.connections.send_step_result(request.conn_id, frame, too_large))
