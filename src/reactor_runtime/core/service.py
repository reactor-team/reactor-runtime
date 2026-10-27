"""Service-lifecycle vocabulary.

The contract the service supervises and the config threaded through ``serve``. A
``ServiceComponent`` declares its startup dependencies and supports an ordered
drain (stop taking new work) distinct from stop (release resources), so the
service alone arbitrates start, drain, and stop ordering. ``RuntimeConfig`` is
the single object that configures one runtime process.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
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
class RecordingConfig:
    """The recorder's tunables, drawn from the manifest's ``recording:`` block.

    A model with no ``recording:`` block records nothing (``enabled`` is
    ``False``); flipping ``enabled`` on is enough for the common single-track
    case. ``video_track`` / ``audio_track`` only need naming when the model
    declares more than one track of that kind — otherwise the recorder picks the
    lone track of each kind automatically. The encoder fields surface straight
    into the ffmpeg invocation.

    Attributes:
        enabled: Whether the runtime records the session at all.
        chunk_seconds: HLS segment length and the forced-keyframe interval.
        clip_max_seconds: Upper bound a snap-clip request is capped to.
        skip_leading_black: Drop gap-fill frames before the first real frame and
            anchor the timeline there, so a clip's markers and bytes share an
            origin.
        video_track: The output track to record, or ``None`` to auto-pick the
            lone video track.
        audio_track: The output track to record, or ``None`` to auto-pick the
            lone audio track (recording stays video-only when there is none).
        video_codec: ``"h264"`` or ``"h265"``.
        video_preset: The libx264/libx265 preset.
        video_crf: The constant-rate-factor quality target.
        target_width: Fixed encode width, or ``None`` to follow the first frame.
        target_height: Fixed encode height, or ``None`` to follow the first frame.
        audio_codec: The audio codec, e.g. ``"aac"``.
        audio_bitrate_kbps: The audio bitrate in kilobits per second.
        recording_dir: Directory clips are written under, or ``None`` to use a
            fresh temporary directory.
    """

    enabled: bool = False
    chunk_seconds: int = 4
    clip_max_seconds: int = 300
    skip_leading_black: bool = True
    video_track: str | None = None
    audio_track: str | None = None
    video_codec: str = "h264"
    video_preset: str = "veryfast"
    video_crf: int = 23
    target_width: int | None = None
    target_height: int | None = None
    audio_codec: str = "aac"
    audio_bitrate_kbps: int = 128
    recording_dir: str | None = None


@dataclass(frozen=True)
class StepResultsConfig:
    """How each finished step is kept, drawn from the manifest's ``step_results:`` block.

    A model with no ``step_results:`` block keeps nothing (``enabled`` is
    ``False``). When enabled, every step the model reports is saved as a
    folder: its output in one ``output.mp4``, with each track of the step's
    ``Output`` as its own stream, beside any extra files and a ``result.json``.
    Turning it on is the model's choice, not a session option.

    Attributes:
        enabled: Whether the runtime saves each finished step.
        video_codec: ``"h264"`` or ``"h265"``.
        video_preset: The libx264/libx265 preset.
        video_crf: The constant-rate-factor quality target.
        audio_codec: The audio codec, e.g. ``"aac"``.
        audio_bitrate_kbps: The audio bitrate in kilobits per second.
        queue: How many steps may wait to be saved. A step that finds the
            queue full is not saved, so the model never waits on saving.
    """

    enabled: bool = False
    video_codec: str = "h264"
    video_preset: str = "veryfast"
    video_crf: int = 23
    audio_codec: str = "aac"
    audio_bitrate_kbps: int = 128
    queue: int = 8


@dataclass(frozen=True)
class LiveKitRpcMethod:
    """One LiveKit RPC method a room session connects to the model.

    Exactly one of ``command`` and ``message`` is set. A method with a
    ``command`` is one the session's participant registers: a call to it runs
    the command and answers with its result. A method with a ``message`` is one
    the participant calls on others whenever the model sends that message.

    Attributes:
        method: The RPC method name, matched exactly.
        command: The command a call to the method runs.
        message: The message whose sending calls the method.
    """

    method: str
    command: str | None = None
    message: str | None = None


class LiveKitBlockError(ValueError):
    """Raised for a manifest ``runtime.livekit`` block that is malformed or does not fit the model.

    A manifest mistake, told apart from a model that fails to load, so it can
    be reported as one.
    """


@dataclass(frozen=True)
class LiveKitConfig:
    """How a LiveKit room reaches the model, drawn from the manifest's ``livekit:`` block.

    A model with no ``livekit:`` block can still join a room: participants
    reach it through the Reactor protocol, and remote tracks feed its input
    tracks by name. The block adds the room's own signals on top of that.

    Attributes:
        rpc: RPC methods connected to commands or messages.
        topics: Data topic prefixes. A data packet on ``<prefix>.<command>``
            runs that command.
        media_inputs: Byte stream topics or remote track names, each mapped to
            the declared input track it feeds. Held read-only, and left out of
            the hash, so the configuration stays hashable.
    """

    rpc: tuple[LiveKitRpcMethod, ...] = ()
    topics: tuple[str, ...] = ()
    media_inputs: Mapping[str, str] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        """Hold ``media_inputs`` as a read-only copy of what was passed."""
        object.__setattr__(self, "media_inputs", MappingProxyType(dict(self.media_inputs)))

    def check(
        self,
        *,
        commands: Collection[str],
        messages: Collection[str],
        input_tracks: Collection[str],
    ) -> None:
        """Refuse a block that names a command, message, or input track the model lacks.

        Args:
            commands: The commands the model declares.
            messages: The messages the model declares.
            input_tracks: The input tracks the model declares.

        Raises:
            LiveKitBlockError: Naming the first entry that refers to something
                the model does not declare.
        """
        for entry in self.rpc:
            if entry.command is not None and entry.command not in commands:
                raise LiveKitBlockError(
                    f"runtime.livekit.rpc: method {entry.method!r} runs command "
                    f"{entry.command!r}, which the model does not declare"
                )
            if entry.message is not None and entry.message not in messages:
                raise LiveKitBlockError(
                    f"runtime.livekit.rpc: method {entry.method!r} is called for message "
                    f"{entry.message!r}, which the model does not declare"
                )
        for source, track in self.media_inputs.items():
            if track not in input_tracks:
                raise LiveKitBlockError(
                    f"runtime.livekit.media_inputs: {source!r} feeds input track {track!r}, "
                    "which the model does not declare"
                )


WORKER_SETTINGS = (
    "max_batch",
    "batch_window",
    "max_request_bytes",
    "shm_input_bytes",
    "shm_output_bytes",
)
"""The serving settings a ``runtime.worker`` block may set beside ``import``."""


@dataclass(frozen=True)
class WorkerConfig:
    """The worker half of a model split in two, drawn from the manifest's ``runtime.worker`` block.

    Read by the model worker's entry point, ``python -m
    reactor_runtime.model_server``; the session runtime reads nothing from it.

    Attributes:
        import_ref: The worker half, a ``ModelServer`` subclass, as ``"module:Class"``.
        settings: The serving settings the block sets, by name (a subset of
            :data:`WORKER_SETTINGS`). The worker checks them against its class
            when it loads: the class is the authority on what it can serve.
            Held read-only, and left out of the hash.
    """

    import_ref: str
    settings: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        """Hold ``settings`` as a read-only copy of what was passed."""
        object.__setattr__(self, "settings", MappingProxyType(dict(self.settings)))


@dataclass(frozen=True)
class RuntimeConfig:
    """The single configuration object threaded through ``serve``.

    Names where to find the model, how to reach it, and the lifecycle tunables
    the service and runner need. Concrete components read the fields they care
    about; later flows extend it as they grow needs.

    Attributes:
        model_ref: Import reference to the model class, ``"module:Class"``.
        model_name: The name the model is published under (from ``model.name``
            in ``reactor.yaml``), which titles its schema. ``None`` when the
            manifest states none, leaving the name to be derived from the class.
        config_path: Absolute path to the model's config file (from
            ``runtime.config`` in ``reactor.yaml``), or ``None`` when the
            manifest names none. Handed to the model's ``load`` to read however
            it wants; the runtime never parses it.
        host: Address the HTTP ingress binds.
        port: Port the HTTP ingress binds.
        grace_period: Seconds a draining session is given to end before stop.
        exit_timeout: Seconds the orderly shutdown may take, on top of any
            session drain it includes, before the process is forced to exit.
        orphan_timeout: Seconds a session may stay client-less before it closes.
        recording: The recorder's configuration; disabled by default.
        step_results: How each finished step is saved; disabled by default.
        livekit: How a LiveKit room reaches the model; empty by default.
        worker: The worker half of a model split in two, from
            ``runtime.worker``; ``None`` for a model that does not split.
    """

    model_ref: str
    model_name: str | None = None
    config_path: Path | None = None
    host: str = "0.0.0.0"
    port: int = 8080
    grace_period: float = 30.0
    exit_timeout: float = 10.0
    orphan_timeout: float = 60.0
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    step_results: StepResultsConfig = field(default_factory=StepResultsConfig)
    livekit: LiveKitConfig = field(default_factory=LiveKitConfig)
    worker: WorkerConfig | None = None
