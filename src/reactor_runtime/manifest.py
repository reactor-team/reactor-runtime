"""Reading the manifest and resolving the model it names.

The ``reactor.yaml`` manifest is the runtime's one configuration file:
``runtime.import`` names the model as a ``"module:Class"`` reference,
``runtime.config`` points at the model's own config file, and the top-level
``recording:`` block configures the recorder. This module turns that file into
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
from typing import Any

import yaml

from reactor_runtime.core import RecordingConfig, RuntimeConfig
from reactor_runtime.interface.internal.reactor_core import ReactorCore

MANIFEST = "reactor.yaml"


def load_config(manifest: Path) -> RuntimeConfig:
    """Read a ``reactor.yaml`` manifest into a :class:`RuntimeConfig`.

    ``runtime.import`` — the ``"module:Class"`` model reference — and
    ``runtime.config`` — the path to the model's own config file — name the
    model, ``model.name`` is the name it is published under, and the top-level
    ``recording:`` block configures the recorder; the rest of the manifest
    describes the model to the platform and is not the runtime's concern. The
    config path is passed to the model verbatim (resolved to an absolute path);
    the runtime never parses its contents.

    Args:
        manifest: Path to the ``reactor.yaml`` file.

    Returns:
        A configuration naming the model the manifest points at, the name it
        publishes under, the path to its config file when present, and the
        recorder's settings.

    Raises:
        SystemExit: If the manifest is not valid YAML, is not a mapping, or
            carries no ``runtime.import``.
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
        recording=_recording_from_manifest(document.get("recording")),
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


def _recording_from_manifest(block: Any) -> RecordingConfig:
    """Parse the manifest's ``recording:`` block into a :class:`RecordingConfig`.

    A missing or non-mapping block leaves recording disabled at its defaults.
    Unknown keys are ignored so a manifest can carry forward-looking settings
    without breaking an older runtime.

    Args:
        block: The raw ``recording:`` value from the manifest, if any.

    Returns:
        The parsed recorder configuration.
    """
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
