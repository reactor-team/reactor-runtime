"""The stepped path of a runner: a client's step requests and the answer to each.

A stepped session runs one step per ``RequestStep`` a client sends on the data
channel. :class:`StepRequests` takes each request from the connection that sent
it, checks it whole against the model's contract and input tracks, and hands it
to the model as plain calls and its step, or refuses it with a ``StepResult``
that takes no number. When the model reports the step, it encodes the step's
media on the model's thread and sends the ``StepResult`` back on the data
channel.

Each output video track a connection negotiated an encoding for is one encoded
stream to that connection for its life (:class:`OutputStreams`): its frames
reference the frames of the steps before, and start at a keyframe only where
the connection's stream starts over.

A client can also push input frames outside any step (``PushFrames``). They
are never refused: each frame its track can take is written to the track's
input buffer, and any other is dropped without a word.

A connection's step requests, its pushed frames, and in a stepped session its
commands, are submitted one at a time in the order they arrived
(:class:`StepOrder`), so one that waits for an upload holds back the ones sent
after it, and frames pushed before a step are written before it runs.

The runner owns the session; the stepped path reads it through a
:class:`StepHost`, so it can be driven, and tested, without one.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import threading
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Iterable,
    Mapping,
    Sequence,
)
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from reactor_runtime.core import (
    CompletedStep,
    ConnId,
    SessionEvent,
    StepRefusal,
    StepRefusalKind,
    TrackDirection,
    TrackInfo,
    TrackKind,
)
from reactor_runtime.interface.internal.bridge import CommandOutcome
from reactor_runtime.interface.internal.reactor_core import FrameSource, RequestId
from reactor_runtime.log import get_logger
from reactor_runtime.message_gateway import InboundPush, InboundStep
from reactor_runtime.protocol import Channel, Codec, ProtocolVersion, select
from reactor_runtime.protocol.frames import (
    FrameEncodingError,
    check_frame,
    decode_frame,
    decoded_frame,
    media_size,
    track_frames,
)
from reactor_runtime.runner.session_start import Generation
from reactor_runtime.step_results import content_type_of
from reactor_runtime.transport.step_frames import TrackEncoder
from reactor_runtime.upload_store import UnknownUploadError
from reactor_wire.v1 import platform_pb2, step_pb2

logger = get_logger(__name__)

# The wire code a step the model did not run is answered with, by why it did not.
_REFUSAL_CODES: dict[StepRefusalKind, step_pb2.StepFailureCode] = {
    StepRefusalKind.REFUSED: step_pb2.STEP_FAILURE_CODE_REFUSED,
    StepRefusalKind.NOT_READY: step_pb2.STEP_FAILURE_CODE_NOT_READY,
    StepRefusalKind.COMMAND_FAILED: step_pb2.STEP_FAILURE_CODE_COMMAND_FAILED,
}

# The runtime's track kind for each kind a wire message names.
_WIRE_TRACK_KINDS = {
    platform_pb2.TRACK_KIND_VIDEO: TrackKind.VIDEO,
    platform_pb2.TRACK_KIND_AUDIO: TrackKind.AUDIO,
}


@dataclass(frozen=True)
class EncodedResult:
    """A requested step's result as the model's thread encoded it, without its files.

    Exactly one of *frame*, the v1 frame ready to send, *oversized*, the size
    of a result that does not fit the connection, or *error*, why the media
    cannot be carried as frames, is set. *streamed* names the output tracks
    whose frames in *frame* continue an encoded stream, which starts over when
    the frame is not sent.
    """

    frame: bytes | None = None
    oversized: int = 0
    error: str | None = None
    streamed: tuple[str, ...] = ()


class StepInputError(ValueError):
    """A step request's input frames cannot be written to the model's tracks."""

    def __init__(self, entry: str, message: str) -> None:
        super().__init__(message)
        self.entry = entry


def _step_file_refs(step: CompletedStep) -> list[step_pb2.StepFileRef]:
    """Reference each extra file a saved step keeps, as its step result is saved.

    A failed step keeps no files, so its result references none.
    """
    if step.error is not None:
        return []
    return [
        step_pb2.StepFileRef(
            name=name,
            content_type=content_type_of(name),
            size=len(data) if isinstance(data, bytes) else data.size,
        )
        for name, data in step.files.items()
    ]


class Journal(Protocol):
    """Records a session event on the runner's journal, as its state machine's ``send``."""

    def __call__(self, event: SessionEvent, /, **detail: Any) -> object:
        """Record *event*, with its *detail*."""
        ...


class StepConnections(Protocol):
    """What the stepped path reads from, and sends on, the runner's connections."""

    def message_chunking(self, cid: ConnId) -> bool:
        """Whether a connection negotiated chunked messages."""
        ...

    def max_message_size(self, cid: ConnId) -> int | None:
        """Return the largest message a connection takes; ``None`` when unknown."""
        ...

    def step_frame_encodings(self, cid: ConnId) -> Mapping[str, int]:
        """Return the ``FrameEncoding`` each track's step frames take on a connection."""
        ...

    def send_step_reply(
        self, cid: ConnId, encode: Callable[[ProtocolVersion], tuple[Channel, bytes | str]]
    ) -> None:
        """Send a reply to a step request on the data channel."""
        ...

    async def send_step_result(
        self, cid: ConnId, frame: bytes, too_large: Callable[[int], bytes]
    ) -> None:
        """Send a step's encoded result, or the one *too_large* renders when it does not fit."""
        ...


class StepMetrics(Protocol):
    """The command instruments a step request's calls are counted on."""

    def accepted(self, command: str, *, since: float) -> None:
        """Count a command the model took."""
        ...

    def rejected(self, command: str, *, since: float) -> None:
        """Count a command the contract refused."""
        ...

    def unresolved_upload(self, command: str) -> None:
        """Count a command dropped because an upload it references never arrived."""
        ...


class StepContract(Protocol):
    """What the stepped path reads from the model's contract."""

    @property
    def tracks(self) -> Mapping[str, TrackInfo]:
        """The model's media tracks, by name."""
        ...


class StepBridge(Protocol):
    """What the stepped path reads from, and hands to, the loaded model's bridge."""

    @property
    def contract(self) -> StepContract:
        """The model's contract."""
        ...

    def input_overflow(self, track: str, frames: int, samples: int = 0) -> str | None:
        """Say how *track*'s input buffer is bounded, when *frames* do not fit it together."""
        ...

    def submit_step(
        self,
        calls: Sequence[tuple[str, str, dict[str, Any]]],
        media_input: Mapping[str, Sequence[FrameSource]],
        *,
        conn_id: ConnId,
        request_id: RequestId,
        max_message_size: int | None = None,
    ) -> tuple[str, CommandOutcome] | None:
        """Admit a step request's calls and its step, or return the first call that fails."""
        ...

    def push_frames(
        self, media_input: Mapping[str, Sequence[FrameSource]], *, conn_id: ConnId
    ) -> None:
        """Admit frames a client pushed outside any step, in their place among its requests."""
        ...


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

    bridge: Callable[[], StepBridge | None]
    session_running: Callable[[], bool]
    generation: Callable[[], Generation]
    sessions_posted: Callable[[], int]
    starting_input_done: Callable[[], asyncio.Event]
    connections: StepConnections
    journal: Journal
    resolve_uploads: Callable[
        [str, Mapping[str, Any], Mapping[str, str], float], Awaitable[dict[str, Any]]
    ]
    command_metrics: StepMetrics
    command_label: Callable[[str], str]
    codec_for: Callable[[ProtocolVersion], Codec]
    spawn: Callable[[Coroutine[Any, Any, None]], None]


class StepOrder:
    """Each connection's step requests, pushes, and stepped commands, one at a time in order.

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


_STREAMLESS = frozenset({step_pb2.FRAME_ENCODING_UNSPECIFIED, step_pb2.FRAME_ENCODING_RAW})
"""Encodings whose frames stand alone, so a track in one keeps no encoder."""

_SEND_WAIT = 5.0
"""Seconds a connection's next result waits for the previous one's send outcome.

A result whose outcome does not come in time is taken as lost: the next
frames start at a keyframe, which decodes whether it went out or not.
"""


@dataclass
class _Streams:
    """One connection's encoded output streams.

    Holds its negotiated encodings, the encoder of each encoded output track,
    and the results whose frames continue a stream and whose send has no
    outcome yet.
    """

    encodings: Mapping[str, int]
    encoders: dict[str, TrackEncoder]
    sending: list[EncodedResult] = field(default_factory=list)


class OutputStreams:
    """Each connection's encoded output streams, one :class:`TrackEncoder` per track.

    A connection's entry opens with its first step request, holding the
    encoding each track negotiated, and lasts until the connection closes or
    is replaced. Each output track whose encoding is not raw gets its encoder
    the first time a step encodes for the connection; the encoder's backend
    opens with the first frames it encodes. A track that is raw on the
    connection, and every track of a connection that negotiated nothing,
    keeps no encoder.

    A track's stream starts at a keyframe on a new connection, since its
    encoder is new, and starts over at one after :meth:`restart`: frames that
    were encoded and not sent, or a track the client resumed because its
    decoder lost the stream.

    A connection's result whose frames continue a stream is held
    (:meth:`hold`) until its send has an outcome (:meth:`settle`), and the
    connection's next result is not encoded before then: a send the
    connection refuses restarts the stream before the next frames are
    encoded, so none of them references frames the client never got.

    The loop opens, closes, and restarts entries while the model's thread
    encodes on them, so every call is thread-safe.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._settled = threading.Condition(self._lock)
        self._conns: dict[ConnId, _Streams] = {}

    def __len__(self) -> int:
        """Return how many connections hold an entry."""
        with self._lock:
            return len(self._conns)

    def open(self, conn_id: ConnId, encodings: Mapping[str, int]) -> None:
        """Hold *conn_id*'s negotiated *encodings*, by track, unless it already has an entry."""
        with self._lock:
            if conn_id not in self._conns:
                self._conns[conn_id] = _Streams(dict(encodings), {})

    def encoders(self, conn_id: ConnId) -> dict[str, TrackEncoder]:
        """Return the encoder of each encoded track of *conn_id*, by name; empty without an entry.

        Called on the model's thread, the only one that encodes, to encode a
        step's result. Waits until every result held for the
        connection is settled, up to :data:`_SEND_WAIT`; past it, the held
        results are dropped and every track starts at a keyframe.
        """
        with self._lock:
            streams = self._conns.get(conn_id)
            if streams is None:
                return {}
            keyframe = False
            if streams.sending and not self._settled.wait_for(
                lambda: self._conns.get(conn_id) is not streams or not streams.sending,
                timeout=_SEND_WAIT,
            ):
                streams.sending.clear()
                keyframe = True
            if self._conns.get(conn_id) is not streams:
                return {}
            for track, encoding in streams.encodings.items():
                if encoding not in _STREAMLESS and track not in streams.encoders:
                    streams.encoders[track] = TrackEncoder(encoding)
            if keyframe:
                for encoder in streams.encoders.values():
                    encoder.request_keyframe()
            return dict(streams.encoders)

    def restart(self, conn_id: ConnId, tracks: Iterable[str] | None) -> None:
        """Start the next frames of *tracks* (every track for ``None``) at a keyframe.

        A track whose encoder does not exist yet starts at one anyway.
        """
        with self._lock:
            streams = self._conns.get(conn_id)
            if streams is None:
                return
            names = streams.encoders if tracks is None else tracks
            for encoder in [streams.encoders[n] for n in names if n in streams.encoders]:
                encoder.request_keyframe()

    def hold(self, conn_id: ConnId, result: EncodedResult) -> None:
        """Hold *conn_id*'s next encoding until *result*, which continues a stream, is settled."""
        with self._lock:
            streams = self._conns.get(conn_id)
            if streams is not None:
                streams.sending.append(result)

    def settle(self, conn_id: ConnId, result: EncodedResult) -> None:
        """Release a held *result* once its send has an outcome, or it will not be sent."""
        with self._lock:
            streams = self._conns.get(conn_id)
            if streams is not None:
                streams.sending[:] = [held for held in streams.sending if held is not result]
            self._settled.notify_all()

    def close(self, conn_id: ConnId) -> None:
        """Drop *conn_id*'s entry, freeing its encoders off the event loop.

        Freeing an encoder can wait on its backend, a hardware session or
        threads, so it is left to a worker thread when a loop runs here.
        """
        with self._lock:
            streams = self._conns.pop(conn_id, None)
            self._settled.notify_all()
        self._release([] if streams is None else [streams])

    def close_all(self) -> None:
        """Drop every connection's entry, freeing their encoders as :meth:`close` does."""
        with self._lock:
            dropped = list(self._conns.values())
            self._conns.clear()
            self._settled.notify_all()
        self._release(dropped)

    @staticmethod
    def _release(dropped: list[_Streams]) -> None:
        """Free the encoders of *dropped* entries, off the event loop when one runs here."""
        released = [encoder for streams in dropped for encoder in streams.encoders.values()]
        for streams in dropped:
            streams.encoders.clear()
        if not released:
            return
        # With no running loop, or its executor shut down, they are freed here.
        with contextlib.suppress(RuntimeError):
            asyncio.get_running_loop().run_in_executor(None, released.clear)


class StepRequests:
    """A runner's stepped path: submit each step request, and answer it.

    Args:
        host: What the stepped path reads from the runner that owns it.
        upload_wait: How long, in seconds, a request waits for each of its
            uploads' bytes.
    """

    def __init__(self, host: StepHost, *, upload_wait: float) -> None:
        self._host = host
        self._upload_wait = upload_wait
        self.order = StepOrder()
        self.streams = OutputStreams()
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

    async def submit(self, step: InboundStep) -> None:
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
        self.streams.open(step.conn_id, self._host.connections.step_frame_encodings(step.conn_id))

        def refuse_entry(entry: str, reason: str) -> None:
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
                resolved = await self._host.resolve_uploads(name, args, uploads, self._upload_wait)
            except UnknownUploadError:
                self._host.command_metrics.unresolved_upload(self._host.command_label(name))
                refuse_entry(key, f"{key} references an unresolved upload")
                return
            calls.append((key, name, resolved))
        try:
            media_input = self.check_media_input(step.media_input, step.pictures)
        except StepInputError as error:
            refuse_entry(error.entry, str(error))
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
            refuse_entry(key, outcome.refusal(key))
            return
        # Counted and journalled as a client's own command is, so a step's
        # calls show up in the same metrics and errors.
        for _, name, args, _ in entries:
            self._host.command_metrics.accepted(
                self._host.command_label(name), since=step.received_at
            )
            self._host.journal(SessionEvent.COMMAND, name=name, args=args, conn_id=step.conn_id)

    async def submit_client_push(self, push: InboundPush) -> None:
        """Submit frames a client pushed once the session's starting input is submitted.

        A push takes its place in the connection's step order, as a step
        request does, so its frames are written after the steps requested
        before it and before the steps requested after it.
        """
        async with self.order.hold(push.conn_id):
            await self._host.starting_input_done().wait()
            self.push(push)

    def push(self, push: InboundPush) -> None:
        """Hand the frames a client pushed to the model, every one its tracks can take.

        A push is never refused and never answered. Outside a running stepped
        session its frames are dropped. In one, each frame is written to its
        input track's buffer, which keeps the newest frames when it is full; a
        frame for a track that is not one of the model's input tracks, of a
        kind the track is not, that cannot be decoded, or that the transport
        could not decode, is dropped, and the frames beside it go on.
        """
        bridge = self._host.bridge()
        if bridge is None:
            return
        if not self._host.session_running() or self._host.generation() is not Generation.STEPPED:
            logger.debug(
                "pushed frames dropped: no stepped session is running", conn_id=push.conn_id
            )
            return
        media_input = self._pushed_frames(bridge, push)
        if media_input:
            bridge.push_frames(media_input, conn_id=push.conn_id)

    def _pushed_frames(self, bridge: StepBridge, push: InboundPush) -> dict[str, list[FrameSource]]:
        """Each pushed frame its input track can take, by track, in the order they were sent."""
        tracks = {
            name: info
            for name, info in bridge.contract.tracks.items()
            if info.direction is TrackDirection.IN
        }
        taken: dict[str, list[FrameSource]] = {}
        dropped: list[str] = []
        for index, entry in enumerate(push.tracks):
            info = tracks.get(entry.track)
            kind = _WIRE_TRACK_KINDS.get(entry.kind)
            if info is None or (
                entry.kind != platform_pb2.TRACK_KIND_UNSPECIFIED and kind is not info.kind
            ):
                dropped.append(f"tracks[{index}]")
                continue
            frames = taken.setdefault(entry.track, [])
            for position, frame in enumerate(entry.frames):
                picture = push.pictures.get((index, position))
                if picture is not None:
                    if info.kind is TrackKind.VIDEO:
                        frames.append(functools.partial(decoded_frame, frame, picture))
                    else:
                        dropped.append(f"tracks[{index}].frames[{position}]")
                    continue
                try:
                    check_frame(frame, info.kind)
                except FrameEncodingError:
                    # An encoded frame the transport did not decode lands here
                    # too: its track's stream is broken, and the client was told.
                    dropped.append(f"tracks[{index}].frames[{position}]")
                    continue
                frames.append(functools.partial(decode_frame, frame, info.kind))
        if dropped:
            logger.debug("pushed frames dropped", conn_id=push.conn_id, entries=dropped)
        return {track: frames for track, frames in taken.items() if frames}

    def check_media_input(
        self,
        media_input: tuple[step_pb2.TrackFrames, ...],
        pictures: Mapping[tuple[int, int], npt.NDArray[np.uint8]],
    ) -> dict[str, list[FrameSource]]:
        """Check a step request's media input, by the input track each entry belongs to.

        Every frame is checked without copying its samples; the copy happens
        when the model's thread calls the source this returns for it. A frame
        the transport already decoded is read as its picture in *pictures*,
        keyed by ``(entry, position)``, which only a video track takes.

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
            kind = _WIRE_TRACK_KINDS.get(entry.kind)
            if entry.kind != platform_pb2.TRACK_KIND_UNSPECIFIED and kind is not info.kind:
                raise StepInputError(
                    where, f"{where} says {entry.track!r} is not a {info.kind} track"
                )
            frames: list[FrameSource] = []
            samples = 0
            for position, frame in enumerate(entry.frames):
                picture = pictures.get((index, position))
                if picture is not None:
                    if info.kind is not TrackKind.VIDEO:
                        raise StepInputError(
                            where,
                            f"{where}.frames[{position}]: an encoded video frame cannot go "
                            f"to {info.kind} track {entry.track!r}",
                        )
                    frames.append(functools.partial(decoded_frame, frame, picture))
                    continue
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
            failure_code=_REFUSAL_CODES[refusal.kind],
            code=refusal.code,
            entry=refusal.entry,
        )

    def track_resumed(self, conn_id: ConnId, name: str) -> None:
        """Start the next frames of output track *name* on *conn_id* at a keyframe.

        A client resumes a track whose decoder lost its stream, so its next
        frames are a stream that decodes on its own. The connection's other
        tracks continue their streams.
        """
        self.streams.restart(conn_id, [name])

    def connection_closed(self, conn_id: ConnId) -> None:
        """Forget a connection that closed or was replaced, with its output streams.

        A connection that comes back on the same id starts each stream over,
        at a keyframe.
        """
        self.streams.close(conn_id)

    def connections_closed(self) -> None:
        """Forget every connection's output streams, as the session's teardown closes them all."""
        self.streams.close_all()

    def end_request(
        self, conn_id: ConnId, request_id: str, encoded: EncodedResult | None = None
    ) -> None:
        """Answer a step request whose session ended before its result could go out.

        The encoded streams of *encoded*, the result that is not sent, start
        over at a keyframe.
        """
        if encoded is not None and encoded.streamed:
            self.streams.restart(conn_id, encoded.streamed)
            self.streams.settle(conn_id, encoded)
        self.refuse(
            conn_id,
            request_id,
            "the session ended before the step's result was sent",
            failure_code=step_pb2.STEP_FAILURE_CODE_SESSION_ENDED,
        )

    def encode_result(self, step: CompletedStep) -> EncodedResult | None:
        """Encode a requested step's result, on the model's thread.

        Copying, encoding, and serializing a step's media is a result's
        heaviest work, so all of it happens here, before the hop that keeps the
        step's replies in order, and not on the runtime loop every connection
        shares: the loop gets the finished frame. The step's files are left
        out, since whether the step store takes the step is decided on the
        loop. The data the model returned with the step counts toward the
        result's size beside its media.

        Each output video track the connection negotiated an encoding for is
        encoded on its connection's encoder (see :class:`OutputStreams`), and
        every other track is raw. The result is measured against the largest
        message the connection took when the request was admitted once it is
        encoded; raw media and data that alone outgrow it are not copied, and
        no encoder moves on for them. An encoded track whose frames do not go
        out in the result starts its next frames at a keyframe.

        The connection's previous result that continues a stream is waited on
        until its send has an outcome, so a refused send restarts the stream
        before these frames are encoded; a result that continues one is held
        the same way until :meth:`send_result` or :meth:`end_request` settles
        it.

        Returns:
            The encoded result, or ``None`` for a step no client requested.
        """
        request = step.request
        if request is None:
            return None
        limit = request.max_message_size
        encoders = self.streams.encoders(request.conn_id)
        # A failed step's result carries no data, so its data takes no room.
        data = step.data if step.error is None else None
        size = media_size(step.bundle, encoded=encoders.keys()) + len(data or b"")
        if limit is not None and size > limit:
            return EncodedResult(oversized=size)
        try:
            tracks = track_frames(step.bundle, video_fps=step.fps, encoders=encoders)
        except FrameEncodingError as error:
            # Tracks encoded before the failure moved their streams on.
            self.streams.restart(request.conn_id, encoders.keys())
            return EncodedResult(error=str(error))
        streamed = tuple(
            entry.track
            for entry in tracks
            if any(frame.encoding not in _STREAMLESS for frame in entry.frames)
        )
        result = step_pb2.StepResult(
            step=request.index,
            tracks=tracks,
            # The default loop, the only one a stepped session runs, always
            # measures the step, so a missing time does not happen in practice.
            elapsed_seconds=step.elapsed or 0.0,
            error=step.error,
            data=data,
            failure_code=(
                step_pb2.STEP_FAILURE_CODE_FAILED
                if step.error is not None
                else step_pb2.STEP_FAILURE_CODE_UNSPECIFIED
            ),
        )
        _, frame = self._codec.encode_step_result(request.request_id, result)
        assert isinstance(frame, bytes)
        if limit is not None and len(frame) > limit:
            self.streams.restart(request.conn_id, streamed)
            return EncodedResult(oversized=len(frame))
        encoded = EncodedResult(frame=frame, streamed=streamed)
        if streamed:
            self.streams.hold(request.conn_id, encoded)
        return encoded

    def send_result(
        self,
        step: CompletedStep,
        index: int,
        *,
        saved: bool,
        encoded: EncodedResult | None = None,
    ) -> None:
        """Answer a requested step with its result, built from the step's report.

        The result is the report's media, one entry per output track, and the
        data the model returned with it, so what the client receives is what
        the step reported and, for its media, what a saved step keeps, with
        the seconds its ``generate()`` took (or what the model reported), the
        same measurement a saved step records as ``generate_s``. When the step
        store took the step (*saved*), the result names each extra file the
        model kept with it, which the client reads from the session's step
        results; the bytes stay out of the
        message, so a file counts toward the message's size only by its name,
        media type, and size. It goes out as one message, before the store has
        written the step: the files are readable once the step's
        ``result.json`` is, and a save that fails keeps none, saying why in
        ``save_error``.

        A step that ran always answers with its number and elapsed time. One
        whose model reported a failure answers ``FAILED`` with the model's
        message, as does one whose media cannot be carried as frames. One whose
        result is larger than the connection takes, by its own limit or by the
        media engine's refusal, answers ``RESULT_TOO_LARGE`` with no media,
        data, or files, and the session goes on; its encoded tracks start their
        next frames at a keyframe. A failed step's result carries no data.

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
        streamed = encoded.streamed

        def too_large(size: int, *, at_least: bool = False) -> bytes:
            logger.warning("a step result does not fit one message", step=index, size=size)
            self.streams.restart(request.conn_id, streamed)
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
            # Media and data that alone outgrow one message were never copied.
            answer = too_large(encoded.oversized, at_least=True)
            self._host.connections.send_step_reply(
                request.conn_id, lambda _: (Channel.DATA, answer)
            )
            return
        frame = encoded.frame
        refs = _step_file_refs(step) if saved else ()
        if refs:
            # Serialized protobuf messages merge when concatenated: the files
            # join the result the model's thread encoded. Only a saved step
            # that kept files pays this copy on the loop.
            _, files = self._codec.encode_step_result(request_id, step_pb2.StepResult(files=refs))
            assert isinstance(files, bytes)
            frame += files
        self._host.spawn(self._send_result(request.conn_id, frame, too_large, encoded))

    async def _send_result(
        self, conn_id: ConnId, frame: bytes, too_large: Callable[[int], bytes], held: EncodedResult
    ) -> None:
        """Send a result, then settle it: a refusal has restarted its streams by then."""
        try:
            await self._host.connections.send_step_result(conn_id, frame, too_large)
        finally:
            self.streams.settle(conn_id, held)
