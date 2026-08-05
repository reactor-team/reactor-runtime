"""Serving the runtime — assembly, the run entry, and the command.

The one place the runtime's concrete shape is named: build the runner, hand it
to the HTTP server, supervise the two with a service. :func:`main` is the
command that boots a model from the ``reactor.yaml`` in the working directory —
it reads only the model reference and leaves the rest of the manifest to the
platform.

This is also the runtime's one configuration boundary: the manifest names the
model, and the surrounding deployment names everything else (bind address, the
ICE servers and port range, the congestion-control bitrate limits, the
lifecycle timeouts) through environment variables. The transport and lifecycle
config objects stay free of environment reads; the small adapter here is the
only place that translates the outside world into them.
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib.metadata
import logging
import os
import sys
from pathlib import Path

from reactor_runtime import log
from reactor_runtime.core import RuntimeConfig
from reactor_runtime.http import HttpServer
from reactor_runtime.manifest import MANIFEST, load_config
from reactor_runtime.metrics import RuntimeMetrics
from reactor_runtime.runner import Runner
from reactor_runtime.service import Service
from reactor_runtime.transport.webrtc.config import (
    IceServer,
    IceTransportPolicy,
    WebRtcConfig,
)
from reactor_runtime.transport.webrtc.peer import WebRtcPeerFactory
from reactor_runtime.transport.webrtc.router import WebRtcRouter

# Public STUN server used when no STUN/TURN is configured, so the SDP answer
# carries a server-reflexive candidate. A same-host client still connects on
# host candidates without it, but anything across a NAT needs one. An operator
# overrides it (or adds TURN) through STUN_SERVERS / TURN_SERVERS.
_DEFAULT_STUN_SERVER = "stun:stun.l.google.com:19302"

logger = log.get_logger(__name__)


def _version() -> str:
    """Return the installed runtime version, or ``"unknown"`` when running from source."""
    try:
        return importlib.metadata.version("reactor-runtime")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _csv(name: str) -> list[str]:
    """Return the comma-separated, whitespace-trimmed values of env var *name*."""
    return [item.strip() for item in os.getenv(name, "").split(",") if item.strip()]


def _ice_servers_from_env() -> tuple[IceServer, ...]:
    """Build the ICE servers offered for candidate gathering from the environment.

    Reads ``STUN_SERVERS`` (comma-separated ``stun:`` URLs) and ``TURN_SERVERS``
    (comma-separated ``username;credential;url`` entries). When neither is set,
    falls back to a single public STUN server so the runtime still gathers a
    server-reflexive candidate.

    Raises:
        SystemExit: If a ``TURN_SERVERS`` entry is not ``username;credential;url``.
    """
    servers: list[IceServer] = [IceServer(urls=(url,)) for url in _csv("STUN_SERVERS")]
    for entry in _csv("TURN_SERVERS"):
        parts = [part.strip() for part in entry.split(";", 2)]
        if len(parts) != 3 or not all(parts):
            raise SystemExit(f"TURN_SERVERS entry {entry!r} must be 'username;credential;url'")
        username, credential, url = parts
        servers.append(IceServer(urls=(url,), username=username, credential=credential))
    return tuple(servers) or (IceServer(urls=(_DEFAULT_STUN_SERVER,)),)


def _port_range_from_env() -> tuple[int, int] | None:
    """Parse ``WEBRTC_PORT_RANGE`` (``min:max`` / ``:max`` / ``min:``) for ICE.

    Returns ``None`` when unset, letting the stack pick ephemeral ports.

    Raises:
        SystemExit: If the value is malformed, out of ``[1024, 65535]``, or inverted.
    """
    raw = os.getenv("WEBRTC_PORT_RANGE", "").strip()
    if not raw:
        return None
    if raw.count(":") != 1:
        raise SystemExit(f"WEBRTC_PORT_RANGE {raw!r} must be 'min:max', ':max', or 'min:'")
    low_str, high_str = (part.strip() for part in raw.split(":"))
    try:
        low = int(low_str) if low_str else 1024
        high = int(high_str) if high_str else 65535
    except ValueError:
        raise SystemExit(f"WEBRTC_PORT_RANGE {raw!r} has a non-integer bound") from None
    if not 1024 <= low <= high <= 65535:
        raise SystemExit(f"WEBRTC_PORT_RANGE {raw!r} out of [1024, 65535] or inverted")
    return low, high


def _ice_policy_from_env() -> IceTransportPolicy:
    """Read ``ICE_TRANSPORT_POLICY`` (``all`` default, or ``relay``).

    Raises:
        SystemExit: If the value is neither ``all`` nor ``relay``.
    """
    raw = os.getenv("ICE_TRANSPORT_POLICY", "all").strip().lower()
    try:
        return IceTransportPolicy(raw)
    except ValueError:
        raise SystemExit(f"ICE_TRANSPORT_POLICY {raw!r} must be 'all' or 'relay'") from None


def _float_env(name: str, default: float) -> float:
    """Return env var *name* as a float, or *default* when unset/empty.

    Raises:
        SystemExit: If the value is set but not a number.
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise SystemExit(f"{name} {raw!r} must be a number") from None


def _int_env(name: str, default: int) -> int:
    """Return env var *name* as an int, or *default* when unset/empty.

    Raises:
        SystemExit: If the value is set but not an integer.
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"{name} {raw!r} must be an integer") from None


def _webrtc_config_from_env() -> WebRtcConfig:
    """Build the WebRTC transport config from the environment.

    The transport config object itself reads no environment; this adapter is the
    single place the outside world is translated into it.
    """
    return WebRtcConfig(
        ice_servers=_ice_servers_from_env(),
        port_range=_port_range_from_env(),
        transport_policy=_ice_policy_from_env(),
        ping_timeout=_float_env("WEBRTC_CLIENT_PING_TIMEOUT_SECONDS", 20.0),
        bwe_min_kbps=_int_env("WEBRTC_BWE_MIN_KBPS", WebRtcConfig.bwe_min_kbps),
        bwe_max_kbps=_int_env("WEBRTC_BWE_MAX_KBPS", WebRtcConfig.bwe_max_kbps),
        bwe_target_kbps=_int_env("WEBRTC_BWE_TARGET_KBPS", WebRtcConfig.bwe_target_kbps),
    )


def _apply_env(cfg: RuntimeConfig) -> RuntimeConfig:
    """Overlay the lifecycle/bind tunables from the environment onto *cfg*.

    The manifest names the model; ``HOST`` / ``PORT`` name where to bind, and
    ``ORPHAN_TIMEOUT_SECONDS`` / ``SIGTERM_GRACE_PERIOD`` tune the session and
    shutdown windows. Unset variables leave the dataclass defaults in place.

    Raises:
        SystemExit: If ``PORT`` or a timeout is set but not numeric.
    """
    port_raw = os.getenv("PORT", "").strip()
    try:
        port = int(port_raw) if port_raw else cfg.port
    except ValueError:
        raise SystemExit(f"PORT {port_raw!r} must be an integer") from None
    recording = cfg.recording
    recordings_dir = os.getenv("REACTOR_RECORDINGS_DIR", "").strip()
    if recordings_dir:
        recording = dataclasses.replace(recording, recording_dir=recordings_dir)
    return dataclasses.replace(
        cfg,
        host=os.getenv("HOST", cfg.host),
        port=port,
        orphan_timeout=_float_env("ORPHAN_TIMEOUT_SECONDS", cfg.orphan_timeout),
        grace_period=_float_env("SIGTERM_GRACE_PERIOD", cfg.grace_period),
        recording=recording,
    )


def _log_level_from_env() -> int:
    """Resolve ``REACTOR_LOG_LEVEL`` to a logging level, defaulting to ``INFO``."""
    name = os.getenv("REACTOR_LOG_LEVEL", "INFO").strip().upper()
    return logging.getLevelNamesMapping().get(name, logging.INFO)


def _assemble(
    cfg: RuntimeConfig,
    webrtc: WebRtcConfig | None = None,
    *,
    peer_factory: WebRtcPeerFactory | None = None,
) -> Service:
    """Assemble the service from the runtime's components.

    The runner is built once and shared with the HTTP server, which the routes
    drive and the transport reports into. The WebRTC transport is mounted with
    the selected media engine, so a client can negotiate a peer connection and
    stream to and from the model. The runner's shutdown hook is wired to the
    service so a failed model load brings the whole process down, and the
    service's aggregate health is wired into the HTTP server so ``/health``
    answers for every component of the process. The metrics registry is created
    here too, the one place that knows the identity of the process, and handed
    both to the runner that observes on it and to the HTTP server that renders
    it.

    Args:
        cfg: The configuration for this runtime process.
        webrtc: The WebRTC transport configuration; defaults to the plain
            ``WebRtcConfig`` when omitted (as in tests that don't exercise it).
        peer_factory: The media engine to mount; defaults to the libwebrtc
            engine when omitted.

    Returns:
        A service with the runner and the HTTP server hooked on.
    """
    if peer_factory is None:
        try:
            from reactor_runtime.transport.webrtc.peer import libwebrtc_peer_factory
        except Exception as exc:
            detail = str(exc) or type(exc).__name__
            raise SystemExit(f"the libwebrtc media engine is unavailable: {detail}") from exc
        peer_factory = libwebrtc_peer_factory
    service = Service()
    metrics = RuntimeMetrics(version=_version(), model=cfg.model_ref)
    runner = Runner(cfg, metrics)
    runner.request_shutdown = service.request_shutdown
    service.add(runner)
    transport = WebRtcRouter(webrtc or WebRtcConfig(), peer_factory, metrics)
    service.add(
        HttpServer(
            cfg,
            runner,
            transports=[transport],
            process_health=service.health,
            metrics=metrics,
        )
    )
    return service


async def serve(
    cfg: RuntimeConfig,
    webrtc: WebRtcConfig | None = None,
    *,
    peer_factory: WebRtcPeerFactory | None = None,
) -> None:
    """Run the runtime to completion: assemble the service and supervise it.

    Args:
        cfg: The configuration for this runtime process.
        webrtc: The WebRTC transport configuration; defaults to the plain
            ``WebRtcConfig`` when omitted.
        peer_factory: The media engine to mount; defaults to the libwebrtc
            engine when omitted.
    """
    await _assemble(cfg, webrtc, peer_factory=peer_factory).run()


def main() -> None:
    """Boot the runtime from the ``reactor.yaml`` in the working directory.

    Refuses to start when no manifest is present. The manifest's directory is
    put first on the import path so a model referenced as ``"pipeline:Model"``
    resolves to the code sitting beside it. The manifest names the model; the
    bind address, ICE/transport configuration, lifecycle timeouts, and log level
    are read from the environment around it.

    Raises:
        SystemExit: If no ``reactor.yaml`` is found or an environment variable
            is set to a malformed value.
    """
    from reactor_runtime.transport.webrtc.peer import libwebrtc_peer_factory

    log.configure(level=_log_level_from_env())
    manifest = Path.cwd() / MANIFEST
    if not manifest.is_file():
        raise SystemExit(f"no {MANIFEST} found in {Path.cwd()}")
    sys.path.insert(0, str(manifest.parent))
    cfg = _apply_env(load_config(manifest))
    webrtc = _webrtc_config_from_env()
    logger.info(
        "starting reactor runtime",
        version=_version(),
        model=cfg.model_ref,
        host=cfg.host,
        port=cfg.port,
        ice_servers=[server.urls[0] for server in webrtc.ice_servers],
        port_range=webrtc.port_range,
        ice_policy=str(webrtc.transport_policy),
    )
    asyncio.run(serve(cfg, webrtc, peer_factory=libwebrtc_peer_factory))


if __name__ == "__main__":
    main()
