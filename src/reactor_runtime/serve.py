"""Serving the runtime — assembly, the run entry, and the command.

The one place the runtime's concrete shape is named: build the runner, hand it
to the HTTP server, supervise the two with a service. :func:`main` is the
command that boots a model from the ``reactor.yaml`` in the working directory —
it reads only the model reference and leaves the rest of the manifest to the
platform.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import sys
from pathlib import Path
from typing import Any

import yaml

from reactor_runtime import log
from reactor_runtime.core import RuntimeConfig
from reactor_runtime.http import HttpServer
from reactor_runtime.runner import Runner
from reactor_runtime.service import Service
from reactor_runtime.transport.webrtc.config import WebRtcConfig
from reactor_runtime.transport.webrtc.gstreamer.peer import gstreamer_peer_factory
from reactor_runtime.transport.webrtc.router import WebRtcRouter

_MANIFEST = "reactor.yaml"

logger = log.get_logger(__name__)


def _version() -> str:
    """Return the installed runtime version, or ``"unknown"`` when running from source."""
    try:
        return importlib.metadata.version("reactor-runtime")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _assemble(cfg: RuntimeConfig) -> Service:
    """Assemble the service from the runtime's components.

    The runner is built once and shared with the HTTP server, which the routes
    drive and the transport reports into. The WebRTC transport is mounted with
    the GStreamer media engine, so a client can negotiate a peer connection and
    stream to and from the model. The runner's shutdown hook is wired to the
    service so a failed model load brings the whole process down.

    Args:
        cfg: The configuration for this runtime process.

    Returns:
        A service with the runner and the HTTP server hooked on.
    """
    service = Service()
    runner = Runner(cfg)
    runner.request_shutdown = service.request_shutdown
    service.add(runner)
    webrtc = WebRtcRouter(WebRtcConfig(), gstreamer_peer_factory)
    service.add(HttpServer(cfg, runner, transports=[webrtc]))
    return service


async def serve(cfg: RuntimeConfig) -> None:
    """Run the runtime to completion: assemble the service and supervise it.

    Args:
        cfg: The configuration for this runtime process.
    """
    await _assemble(cfg).run()


def _load_config(manifest: Path) -> RuntimeConfig:
    """Read a ``reactor.yaml`` manifest into a :class:`RuntimeConfig`.

    Only ``runtime.import`` — the ``"module:Class"`` model reference — and
    ``runtime.config`` — the path to the model's own config file — are read; the
    rest of the manifest describes the model to the platform and is not the
    runtime's concern. The config path is passed to the model verbatim (resolved
    to an absolute path); the runtime never parses its contents.

    Args:
        manifest: Path to the ``reactor.yaml`` file.

    Returns:
        A configuration naming the model the manifest points at and, when
        present, the path to its config file.

    Raises:
        SystemExit: If the manifest is not a mapping or carries no
            ``runtime.import``.
    """
    document = yaml.safe_load(manifest.read_text())
    if not isinstance(document, dict):
        raise SystemExit(f"{manifest}: not a valid {_MANIFEST}")
    runtime = document.get("runtime")
    runtime = runtime if isinstance(runtime, dict) else {}
    model_ref = runtime.get("import")
    if not isinstance(model_ref, str) or not model_ref:
        raise SystemExit(f"{manifest}: missing runtime.import (the model reference)")
    return RuntimeConfig(model_ref=model_ref, config_path=_resolve_config_path(runtime, manifest))


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


def main() -> None:
    """Boot the runtime from the ``reactor.yaml`` in the working directory.

    Refuses to start when no manifest is present. The manifest's directory is
    put first on the import path so a model referenced as ``"pipeline:Model"``
    resolves to the code sitting beside it.

    Raises:
        SystemExit: If no ``reactor.yaml`` is found in the working directory.
    """
    log.configure()
    manifest = Path.cwd() / _MANIFEST
    if not manifest.is_file():
        raise SystemExit(f"no {_MANIFEST} found in {Path.cwd()}")
    sys.path.insert(0, str(manifest.parent))
    cfg = _load_config(manifest)
    logger.info(
        "starting reactor runtime",
        version=_version(),
        model=cfg.model_ref,
        host=cfg.host,
        port=cfg.port,
    )
    asyncio.run(serve(cfg))


if __name__ == "__main__":
    main()
