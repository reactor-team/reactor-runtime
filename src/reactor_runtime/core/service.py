"""Service-lifecycle vocabulary.

The contract the service supervises and the config threaded through ``serve``. A
``ServiceComponent`` declares its startup dependencies and supports an ordered
drain (stop taking new work) distinct from stop (release resources), so the
service alone arbitrates start, drain, and stop ordering. ``RuntimeConfig`` is
the single object that configures one runtime process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

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
class RuntimeConfig:
    """The single configuration object threaded through ``serve``.

    Names where to find the model, how to reach it, and the lifecycle tunables
    the service and runner need. Concrete components read the fields they care
    about; later flows extend it as they grow needs.

    Attributes:
        model_ref: Import reference to the model class, ``"module:Class"``.
        config_path: Absolute path to the model's config file (from
            ``runtime.config`` in ``reactor.yaml``), or ``None`` when the
            manifest names none. Handed to the model's ``load`` to read however
            it wants; the runtime never parses it.
        stepping: How an engine-backed model advances — ``"automatic"`` to run
            its own loop, ``"triggered"`` to advance one step per ``step``
            command. ``None`` leaves the model's own declaration in place. The
            same application streams under one deployment and steps under
            another, so this belongs to the deployment rather than the class.
        host: Address the HTTP ingress binds.
        port: Port the HTTP ingress binds.
        grace_period: Seconds a draining session is given to end before stop.
        orphan_timeout: Seconds a session may stay client-less before it closes.
        recording: The recorder's configuration; disabled by default.
    """

    model_ref: str
    config_path: Path | None = None
    stepping: str | None = None
    host: str = "0.0.0.0"
    port: int = 8080
    grace_period: float = 30.0
    orphan_timeout: float = 60.0
    recording: RecordingConfig = field(default_factory=RecordingConfig)
