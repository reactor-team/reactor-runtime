"""Reading the manifest and resolving the model it names.

The ``reactor.yaml`` manifest is the runtime's one configuration file:
``runtime.import`` names the model as a ``"module:Class"`` reference,
``runtime.config`` points at the model's own config file, the
``runtime.recording`` block configures the recorder, the
``runtime.step_results`` block says how each finished step is saved, the
``runtime.livekit`` block says how a LiveKit room reaches the model, and the
``runtime.worker`` block names the model half of a model split in two. This
module turns that file into
a :class:`~reactor_runtime.core.RuntimeConfig` and the reference into the model
class — and it is the only code that does either, so every entry point resolves
the same model from the same directory.

The module stays deliberately light: it needs YAML and the core config types,
nothing from the server or the transport stack. Rendering a model's schema
imports it without dragging an ASGI server or a media engine along.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from reactor_runtime.core import (
    LiveKitBlockError,
    LiveKitConfig,
    LiveKitRpcMethod,
    RecordingConfig,
    RuntimeConfig,
    StepResultsConfig,
    TrackDirection,
    WorkerConfig,
)
from reactor_runtime.interface.internal.reactor_core import ReactorCore

if TYPE_CHECKING:
    from reactor_runtime.interface.model.contract import ModelContract

MANIFEST = "reactor.yaml"


def load_config(manifest: Path) -> RuntimeConfig:
    """Read a ``reactor.yaml`` manifest into a :class:`RuntimeConfig`.

    ``runtime.import`` — the ``"module:Class"`` model reference — and
    ``runtime.config`` — the path to the model's own config file — name the
    model, ``model.name`` is the name it is published under, the
    ``runtime.recording`` block configures the recorder, the
    ``runtime.step_results`` block configures step results, and the
    ``runtime.livekit`` block configures room sessions; the rest of the manifest
    describes the model to the platform and is not the runtime's concern. The
    config path is passed to the model verbatim (resolved to an absolute path);
    the runtime never parses its contents.

    Args:
        manifest: Path to the ``reactor.yaml`` file.

    Returns:
        A configuration naming the model the manifest points at, the name it
        publishes under, the path to its config file when present, the
        recorder's settings, the step-results settings, and the room settings.

    Raises:
        SystemExit: If the manifest is not valid YAML, is not a mapping, or
            carries no ``runtime.import``.
        ValueError: If a block the runtime refuses to guess about has the
            wrong shape: a ``step_results.queue`` below 1, a malformed
            ``livekit`` block, or a malformed ``worker`` block.
    """
    try:
        document = yaml.safe_load(manifest.read_text())
    except yaml.YAMLError as error:
        raise SystemExit(f"{manifest}: invalid YAML: {error}") from None
    if not isinstance(document, dict):
        raise SystemExit(f"{manifest}: not a valid {MANIFEST}")
    runtime = document.get("runtime")
    runtime = runtime if isinstance(runtime, dict) else {}
    model_ref = runtime.get("import")
    if not isinstance(model_ref, str) or not model_ref:
        raise SystemExit(f"{manifest}: missing runtime.import (the model reference)")
    return RuntimeConfig(
        model_ref=model_ref,
        model_name=_model_name(document.get("model")),
        config_path=_resolve_config_path(runtime, manifest),
        recording=_recording_from_manifest(runtime, document),
        step_results=_step_results_from_manifest(runtime),
        livekit=_livekit_from_manifest(runtime),
        worker=_worker_from_manifest(runtime),
    )


def _model_name(block: Any) -> str | None:
    """Read ``model.name``, the name the model is published under.

    The name is the model's identity on the platform, and it carries characters
    a Python class name cannot — ``mage-vl`` and ``morpheus-v4`` are hyphenated.
    A manifest that omits it leaves the name to be derived from the class.

    Args:
        block: The raw ``model:`` value from the manifest, if any.

    Returns:
        The published name, or ``None`` when the manifest states none.
    """
    if not isinstance(block, dict):
        return None
    name = block.get("name")
    return name if isinstance(name, str) and name else None


_WORKER_KEYS = {
    "import": "import_ref",
    "max_batch": "max_batch",
    "batch_window": "batch_window",
    "max_request_bytes": "max_request_bytes",
    "shm_input_bytes": "shm_input_bytes",
    "shm_output_bytes": "shm_output_bytes",
}


def _worker_from_manifest(runtime: dict[str, Any]) -> WorkerConfig | None:
    """Parse the manifest's ``runtime.worker`` block into a :class:`WorkerConfig`.

    The block names the model half and its serving settings::

        runtime:
          import: plasma:Plasma
          worker:
            import: plasma_worker:PlasmaWorker
            max_batch: 2
            shm_output_bytes: 33554432

    An unknown key is refused rather than ignored: a misspelt size would
    otherwise leave a session's memory at its default without a word.

    Args:
        runtime: The manifest's ``runtime`` section.

    Returns:
        The parsed block, or ``None`` when the manifest has none.

    Raises:
        ValueError: The block is not a mapping, names a key it does not take, or
            holds a value of the wrong type or range.
    """
    block = runtime.get("worker")
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ValueError(
            f"runtime.worker must be a block with `import: module:Class`, got {block!r}"
        )
    if not isinstance(block.get("import"), str):
        raise ValueError("runtime.worker.import must name the model half as 'module:Class'")
    unknown = sorted(set(block) - set(_WORKER_KEYS))
    if unknown:
        raise ValueError(
            f"runtime.worker has unknown keys {unknown}; it takes {', '.join(_WORKER_KEYS)}"
        )
    return WorkerConfig(**{_WORKER_KEYS[key]: value for key, value in block.items()})


def import_model_class(model_ref: str) -> type[ReactorCore]:
    """Resolve a ``"module:Class"`` reference into the model class it names.

    Args:
        model_ref: An import reference of the form ``"package.module:Class"``.

    Returns:
        The referenced model class.

    Raises:
        ValueError: If the reference is not of the form ``"module:Class"``.
        TypeError: If the reference does not name a :class:`ReactorCore` subclass.
    """
    module_name, separator, class_name = model_ref.partition(":")
    if not separator or not module_name or not class_name:
        raise ValueError(f"model_ref must be 'module:Class', got {model_ref!r}")
    model_cls = getattr(importlib.import_module(module_name), class_name)
    if not isinstance(model_cls, type) or not issubclass(model_cls, ReactorCore):
        raise TypeError(f"{model_ref} does not name a ReactorCore subclass")
    return model_cls


def _recording_from_manifest(runtime: dict[str, Any], document: dict[str, Any]) -> RecordingConfig:
    """Parse the manifest's ``recording:`` block into a :class:`RecordingConfig`.

    The block nests under ``runtime:``. A top-level ``recording:`` block is
    still honored for older manifests, and the nested block wins when both are
    present. A missing or non-mapping block leaves recording disabled at its
    defaults. Unknown keys are ignored so a manifest can carry forward-looking
    settings without breaking an older runtime.

    Args:
        runtime: The manifest's ``runtime`` section.
        document: The whole manifest document, for the legacy top-level block.

    Returns:
        The parsed recorder configuration.
    """
    block = runtime.get("recording")
    if not isinstance(block, dict):
        block = document.get("recording")
    if not isinstance(block, dict):
        return RecordingConfig()
    raw_video = block.get("video")
    video: dict[str, Any] = raw_video if isinstance(raw_video, dict) else {}
    raw_audio = block.get("audio")
    audio: dict[str, Any] = raw_audio if isinstance(raw_audio, dict) else {}
    defaults = RecordingConfig()
    return RecordingConfig(
        enabled=bool(block.get("enabled", defaults.enabled)),
        chunk_seconds=int(block.get("chunk_seconds", defaults.chunk_seconds)),
        clip_max_seconds=int(block.get("clip_max_seconds", defaults.clip_max_seconds)),
        skip_leading_black=bool(block.get("skip_leading_black", defaults.skip_leading_black)),
        video_track=block.get("video_track"),
        audio_track=block.get("audio_track"),
        video_codec=str(video.get("codec", defaults.video_codec)),
        video_preset=str(video.get("preset", defaults.video_preset)),
        video_crf=int(video.get("crf", defaults.video_crf)),
        target_width=_optional_int(video.get("target_width")),
        target_height=_optional_int(video.get("target_height")),
        audio_codec=str(audio.get("codec", defaults.audio_codec)),
        audio_bitrate_kbps=int(audio.get("bitrate_kbps", defaults.audio_bitrate_kbps)),
    )


def _step_results_from_manifest(runtime: dict[str, Any]) -> StepResultsConfig:
    """Parse the manifest's ``runtime.step_results`` block into a :class:`StepResultsConfig`.

    A missing or non-mapping block leaves step results disabled at their
    defaults, and so does a non-mapping ``video`` or ``audio`` sub-block for
    its own settings. Unknown keys are ignored so a manifest can carry
    forward-looking settings without breaking an older runtime. The block names
    no tracks: every track of a step's output is kept as its own stream.

    Args:
        runtime: The manifest's ``runtime`` section.

    Returns:
        The parsed step-results configuration.

    Raises:
        ValueError: ``queue`` is below 1.
    """
    block = runtime.get("step_results")
    if not isinstance(block, dict):
        return StepResultsConfig()
    raw_video = block.get("video")
    video: dict[str, Any] = raw_video if isinstance(raw_video, dict) else {}
    raw_audio = block.get("audio")
    audio: dict[str, Any] = raw_audio if isinstance(raw_audio, dict) else {}
    defaults = StepResultsConfig()
    queue = int(block.get("queue", defaults.queue))
    if queue < 1:
        raise ValueError(f"runtime.step_results.queue must be at least 1, got {queue}")
    return StepResultsConfig(
        enabled=bool(block.get("enabled", defaults.enabled)),
        video_codec=str(video.get("codec", defaults.video_codec)),
        video_preset=str(video.get("preset", defaults.video_preset)),
        video_crf=int(video.get("crf", defaults.video_crf)),
        audio_codec=str(audio.get("codec", defaults.audio_codec)),
        audio_bitrate_kbps=int(audio.get("bitrate_kbps", defaults.audio_bitrate_kbps)),
        queue=queue,
    )


def _livekit_from_manifest(runtime: dict[str, Any]) -> LiveKitConfig:
    """Parse the manifest's ``runtime.livekit`` block into a :class:`LiveKitConfig`.

    A missing block, or a missing key in it, leaves that part empty. Unlike
    the recording and step-results blocks, a block of the wrong shape is
    refused rather than tolerated: every entry connects a room signal to the
    model, and one that is silently dropped is a signal the model never
    receives. A key present with no value (``livekit:`` or ``rpc:`` left
    empty) is refused the same way. Unknown keys are ignored so a manifest
    can carry forward-looking settings.

    Args:
        runtime: The manifest's ``runtime`` section.

    Returns:
        The parsed LiveKit configuration.

    Raises:
        LiveKitBlockError: The block, or one of its entries, has the wrong shape, or
            an inbound method is listed twice.
    """
    if "livekit" not in runtime:
        return LiveKitConfig()
    block = runtime["livekit"]
    if not isinstance(block, dict):
        raise LiveKitBlockError("runtime.livekit must be a mapping")
    return LiveKitConfig(
        rpc=_livekit_rpc(block["rpc"]) if "rpc" in block else (),
        topics=_livekit_topics(block["topics"]) if "topics" in block else (),
        media_inputs=(
            _livekit_media_inputs(block["media_inputs"]) if "media_inputs" in block else {}
        ),
    )


def check_livekit_block(config: LiveKitConfig, contract: ModelContract) -> None:
    """Refuse a ``runtime.livekit`` block that names something *contract* does not declare.

    The runtime runs this when the model loads, so a wrong block stops the
    model from loading instead of failing in a session.

    Raises:
        LiveKitBlockError: Naming the first command, message, or input track
            the block refers to that the model lacks.
    """
    config.check(
        commands=contract.commands.keys(),
        messages=contract.messages.keys(),
        input_tracks=[
            name for name, info in contract.tracks.items() if info.direction is TrackDirection.IN
        ],
    )


def _livekit_rpc(raw: Any) -> tuple[LiveKitRpcMethod, ...]:
    if not isinstance(raw, list):
        raise LiveKitBlockError("runtime.livekit.rpc must be a list")
    entries: list[LiveKitRpcMethod] = []
    inbound: set[str] = set()
    for index, item in enumerate(raw):
        where = f"runtime.livekit.rpc[{index}]"
        if not isinstance(item, dict):
            raise LiveKitBlockError(f"{where} must be a mapping")
        method = item.get("method")
        if not isinstance(method, str) or not method:
            raise LiveKitBlockError(f"{where}.method must be a non-empty string")
        if ("command" in item) == ("message" in item):
            raise LiveKitBlockError(f"{where} needs exactly one of 'command' and 'message'")
        key = "command" if "command" in item else "message"
        target = item.get(key)
        if not isinstance(target, str) or not target:
            raise LiveKitBlockError(f"{where}.{key} must be a non-empty string")
        if key == "message":
            entries.append(LiveKitRpcMethod(method=method, message=target))
            continue
        # A participant keeps one handler per method name.
        if method in inbound:
            raise LiveKitBlockError(f"{where}: method {method!r} already runs a command")
        inbound.add(method)
        entries.append(LiveKitRpcMethod(method=method, command=target))
    return tuple(entries)


def _livekit_topics(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list) or not all(isinstance(t, str) and t for t in raw):
        raise LiveKitBlockError("runtime.livekit.topics must be a list of non-empty strings")
    return tuple(raw)


def _livekit_media_inputs(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict) or not all(
        isinstance(k, str) and k and isinstance(v, str) and v for k, v in raw.items()
    ):
        raise LiveKitBlockError(
            "runtime.livekit.media_inputs must map non-empty source names to input track names"
        )
    return dict(raw)


def _optional_int(value: Any) -> int | None:
    """Coerce an optional manifest value to ``int``, leaving ``None`` as is."""
    return None if value is None else int(value)


def _resolve_config_path(runtime: dict[str, Any], manifest: Path) -> Path | None:
    """Resolve ``runtime.config`` to an absolute path, relative to the manifest.

    A relative ``config`` is resolved against the manifest's directory so it
    works regardless of the process's working directory. Returns ``None`` when
    no config file is named.

    Args:
        runtime: The manifest's ``runtime`` section.
        manifest: Path to the ``reactor.yaml`` file, whose parent anchors a
            relative config path.

    Returns:
        The absolute config path, or ``None`` when none is configured.
    """
    config = runtime.get("config")
    if not isinstance(config, str) or not config:
        return None
    candidate = Path(config)
    return candidate if candidate.is_absolute() else manifest.parent / candidate
