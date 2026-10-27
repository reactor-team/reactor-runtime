"""Serve a model's worker half: ``python -m reactor_runtime.model_server``.

The one process of a deployment that owns the weights and the GPU. Run it in
the model's directory, beside the session runtimes' ``python -m
reactor_runtime.serve``; it reads ``reactor.yaml`` there::

    cd my-model/                          # the directory with reactor.yaml
    python -m reactor_runtime.model_server

Every option has an environment variable and a default:

* ``--worker`` / ``REACTOR_WORKER_IMPORT`` / the manifest's
  ``runtime.worker.import``: the :class:`~reactor_runtime.model_server.ModelServer`
  to serve, as ``module:Class``. The flag and the variable override the manifest.
* The manifest's ``runtime.worker`` block: the serving settings
  (``max_batch``, ``batch_window``, ``max_request_bytes``,
  ``shm_input_bytes``, ``shm_output_bytes``) on top of the class's. The class
  is the authority: the block may lower ``max_batch`` and raise the shm sizes,
  never the reverse, and a value outside that stops the start before
  ``load()`` runs. The effective values are logged and served on ``/health``.
* ``--config`` / the manifest's ``runtime.config``: handed to ``load()``.
* ``--socket`` / ``REACTOR_MODEL_WORKER_SOCKET`` / ``/run/reactor-worker/sock``:
  where session processes connect. Its directory is created when missing.
* ``REACTOR_MAX_SESSIONS``: when set, the most sessions the deployment runs; the
  worker serves only slots 0 to that number minus one.
  The start fails, before ``load()``, when ``/dev/shm`` has less free space
  than that many sessions' blocks.
* ``--health-port`` / ``REACTOR_MODEL_WORKER_HEALTH_PORT``: when set, ``GET
  /health`` on that port answers 503 ``loading`` while ``load()`` runs, then
  200 ``serving`` with the effective settings, the protocol version, and the
  schema fingerprint. It answers 503 ``unhealthy`` with a reason once the
  model called :meth:`~reactor_runtime.model_server.ModelServer.mark_unhealthy`
  or while its :meth:`~reactor_runtime.model_server.ModelServer.health`
  returns one (called on every probe, on the health server's thread), and 503
  ``stuck`` while one step has run longer than
  ``REACTOR_MODEL_WORKER_STUCK_SECONDS`` (120 by default).
* ``REACTOR_SHM_PREFIX`` (``rrmw`` by default): the shared blocks of this
  worker and its sessions are named under it and a token of each start. A
  worker that starts unlinks the blocks the worker before it on the same
  socket left, which it reads from ``shm-prefix`` next to the socket.

Every line it logs carries ``runtime_role="model_server"``, and a line about
one session carries that session's ``slot``, the number its session runtime
stamps on its own lines (``REACTOR_SESSION_SLOT``) and logs with the slot's
generation in its ``model server link`` line. Each such generation has a worker
line with the same slot and generation: ``session opened`` for a session's
first, ``session reset`` for each reset after it.

Once ready it also writes ``ready`` next to the socket, for an exec probe, and
removes it when it stops. SIGTERM and SIGINT stop it once the running step
finishes: it accepts no more sessions, answers every request still queued with
``worker_died: the model worker is stopping``, closes every session's
connection, unlinks their shared blocks, and exits ``0``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import secrets
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from reactor_runtime import log
from reactor_runtime.manifest import MANIFEST, import_ref, load_config
from reactor_runtime.model_server import wire
from reactor_runtime.model_server.client import DEFAULT_SOCKET, SOCKET_ENV
from reactor_runtime.model_server.contract import ModelServer, ServerSettings, check_declarations
from reactor_runtime.model_server.server import WorkerServer
from reactor_runtime.model_server.shm import unlink_prefix

logger = log.get_logger("reactor_runtime.model_server")

WORKER_ENV = "REACTOR_WORKER_IMPORT"
HEALTH_PORT_ENV = "REACTOR_MODEL_WORKER_HEALTH_PORT"
STUCK_ENV = "REACTOR_MODEL_WORKER_STUCK_SECONDS"
SHM_PREFIX_ENV = "REACTOR_SHM_PREFIX"
MAX_SESSIONS_ENV = "REACTOR_MAX_SESSIONS"
SHM_DIR = Path("/dev/shm")


class _Status:
    """What ``/health`` reports, shared between the main thread and the probe server."""

    def __init__(self, stuck_after: float) -> None:
        self.phase = "loading"
        self.server: WorkerServer | None = None
        self.worker: ModelServer | None = None
        self.stuck_after = stuck_after
        self.serving: dict[str, Any] = {}

    def report(self) -> tuple[int, dict[str, Any]]:
        server, worker = self.server, self.worker
        if self.phase != "serving" or server is None or worker is None:
            return 503, {"status": self.phase}
        latched = worker.unhealthy
        if latched is not None:
            return 503, {"status": "unhealthy", "reason": latched}
        running = server.step_running_for
        if running > self.stuck_after:
            return 503, {"status": "stuck", "step_seconds": round(running, 1)}
        try:
            reason = worker.health()
        except Exception as exc:
            reason = f"health() raised {type(exc).__name__}: {exc}"
        if reason is not None:
            return 503, {"status": "unhealthy", "reason": str(reason)}
        return 200, {"status": "serving", **self.serving}


def main(argv: list[str] | None = None) -> int:
    """Load the worker half and serve it until signalled; return the exit status."""
    parser = argparse.ArgumentParser(prog="python -m reactor_runtime.model_server")
    parser.add_argument("--worker", default=None, help="module:Class of the ModelServer")
    parser.add_argument("--socket", default=None, help="the Unix socket sessions connect to")
    parser.add_argument("--config", default=None, help="the model's config file")
    parser.add_argument("--health-port", type=int, default=None, help="serve GET /health here")
    args = parser.parse_args(argv)
    name = os.getenv("REACTOR_LOG_LEVEL", "INFO").strip().upper()
    log.configure(level=logging.getLevelNamesMapping().get(name, logging.INFO))
    log.set_process_context(runtime_role="model_server")
    sys.path.insert(0, str(Path.cwd()))

    manifest = Path.cwd() / MANIFEST
    worker_ref = args.worker or os.getenv(WORKER_ENV, "").strip() or None
    config_path = Path(args.config).resolve() if args.config else None
    overrides: dict[str, Any] = {}
    if manifest.is_file():
        try:
            cfg = load_config(manifest)
        except ValueError as exc:
            raise SystemExit(f"{manifest}: {exc}") from None
        if cfg.worker is not None:
            worker_ref = worker_ref or cfg.worker.import_ref
            overrides = dict(cfg.worker.settings)
        if config_path is None:
            config_path = cfg.config_path
    if not worker_ref:
        raise SystemExit(
            f"no worker half named: name runtime.worker.import in {MANIFEST}, pass --worker, "
            f"or set {WORKER_ENV}"
        )
    socket_path = Path(args.socket or os.getenv(SOCKET_ENV, "").strip() or DEFAULT_SOCKET)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    ready_file = socket_path.parent / "ready"
    with contextlib.suppress(FileNotFoundError):
        ready_file.unlink()
    base = os.getenv(SHM_PREFIX_ENV, "").strip() or "rrmw"
    # The prefix of the worker that served this socket before, if one did and
    # died without cleaning up: its blocks are reclaimed, nobody else's.
    record = socket_path.parent / "shm-prefix"
    with contextlib.suppress(OSError):
        previous = record.read_text().strip()
        if previous.startswith(base) and previous.endswith("_"):
            stale = unlink_prefix(previous)
            if stale:
                logger.info("unlinked blocks a previous worker left", count=stale, prefix=previous)
    prefix = f"{base}{secrets.token_hex(4)}_"
    record.write_text(prefix)

    status = _Status(_stuck_seconds())
    port = args.health_port if args.health_port is not None else _port_env()
    probe = _serve_health(port, status) if port is not None else None

    server: WorkerServer | None = None
    stop_requested = threading.Event()

    def on_signal(signum: int, _frame: object) -> None:
        stop_requested.set()
        if server is not None:
            server.stop()

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, on_signal)

    try:
        worker_cls = import_ref(worker_ref)
        if not (isinstance(worker_cls, type) and issubclass(worker_cls, ModelServer)):
            raise SystemExit(f"{worker_ref} does not name a ModelServer subclass")
        try:
            check_declarations(worker_cls)
            settings = ServerSettings.resolve(worker_cls, overrides)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"{worker_ref}: {exc}") from None
        logger.info(
            "model worker settings",
            worker=worker_ref,
            max_batch=settings.max_batch,
            batch_window=settings.batch_window,
            max_request_bytes=settings.max_request_bytes,
            shm_input_bytes=settings.shm_input_bytes,
            shm_output_bytes=settings.shm_output_bytes,
            from_manifest=sorted(overrides),
        )
        max_sessions = _check_shm_room(settings)
        worker = worker_cls()
        worker.device = _device()
        started = time.monotonic()
        worker.load(config_path)
        logger.info(
            "model worker loaded",
            worker=worker_ref,
            device=worker.device,
            seconds=round(time.monotonic() - started, 2),
            shm_slot_bytes=settings.shm_slot_bytes,
        )
        server = WorkerServer(
            worker, socket_path, shm_prefix=prefix, settings=settings, max_sessions=max_sessions
        )
        status.worker = worker
        status.serving = {
            "protocol": wire.PROTOCOL,
            "schema": server.schema,
            "shm_slot_bytes": settings.shm_slot_bytes,
            "settings": {
                "max_batch": settings.max_batch,
                "batch_window": settings.batch_window,
                "max_request_bytes": settings.max_request_bytes,
                "shm_input_bytes": settings.shm_input_bytes,
                "shm_output_bytes": settings.shm_output_bytes,
            },
        }
        server.bind()
        if stop_requested.is_set():
            server.close()
            return 0
        status.server = server
        status.phase = "serving"
        ready_file.write_text("ready\n")
        server.serve()
        logger.info("model worker stopped")
        return 0
    finally:
        status.phase = "stopped"
        with contextlib.suppress(FileNotFoundError):
            ready_file.unlink()
        if probe is not None:
            probe.shutdown()
        unlink_prefix(prefix)
        with contextlib.suppress(OSError):
            record.unlink()


def _serve_health(port: int, status: _Status) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.split("?")[0] != "/health":
                self.send_error(404)
                return
            code, body = status.report()
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: Any) -> None:
            """Write nothing per probe."""

    host = os.getenv("HOST", "0.0.0.0")
    probe = ThreadingHTTPServer((host, port), Handler)
    probe.daemon_threads = True
    threading.Thread(target=probe.serve_forever, name="worker-health", daemon=True).start()
    logger.info("model worker health served", port=port)
    return probe


def _port_env() -> int | None:
    raw = os.getenv(HEALTH_PORT_ENV, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"{HEALTH_PORT_ENV} {raw!r} must be an integer") from None


def _check_shm_room(settings: ServerSettings) -> int | None:
    """Fail the start when ``/dev/shm`` cannot hold ``REACTOR_MAX_SESSIONS`` sessions' blocks.

    Returns:
        ``REACTOR_MAX_SESSIONS``, or ``None`` when it is unset.

    Raises:
        SystemExit: ``REACTOR_MAX_SESSIONS`` is malformed, or the free space is short.
    """
    raw = os.getenv(MAX_SESSIONS_ENV, "").strip()
    if not raw:
        return None
    try:
        sessions = int(raw)
    except ValueError:
        raise SystemExit(f"{MAX_SESSIONS_ENV} {raw!r} must be an integer") from None
    if sessions < 1:
        raise SystemExit(f"{MAX_SESSIONS_ENV} must be at least 1")
    try:
        stats = os.statvfs(SHM_DIR)
    except OSError as exc:
        raise SystemExit(f"cannot read the free space of {SHM_DIR}: {exc}") from None
    free = stats.f_bavail * stats.f_frsize
    needed = sessions * settings.shm_slot_bytes
    if free < needed:
        raise SystemExit(
            f"{SHM_DIR} has {free} bytes free, and {sessions} sessions ({MAX_SESSIONS_ENV}) need "
            f"{needed}: {settings.shm_slot_bytes} each (shm_input_bytes + shm_output_bytes). "
            "Size /dev/shm (in a pod, the emptyDir's sizeLimit) to fit them"
        )
    logger.info("shared memory has room", free_bytes=free, needed_bytes=needed, sessions=sessions)
    return sessions


def _stuck_seconds() -> float:
    raw = os.getenv(STUCK_ENV, "").strip()
    try:
        value = float(raw) if raw else 120.0
    except ValueError:
        raise SystemExit(f"{STUCK_ENV} {raw!r} must be a number") from None
    if not value > 0:
        raise SystemExit(f"{STUCK_ENV} must be above 0")
    return value


def _device() -> str:
    """``"cuda"`` when the worker's module brought torch in and it sees a GPU, else ``"cpu"``.

    Read only when torch is already imported, so a worker without torch does not
    pay for importing it.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return "cpu"
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


if __name__ == "__main__":
    sys.exit(main())
