"""Serve a model's worker half: ``reactor-runtime serve-worker``.

The one process of a deployment that owns the weights and the GPU. Run it in
the model's directory; it reads ``reactor.yaml`` there when there is one::

    reactor-runtime serve-worker
    python -m reactor_runtime.model_worker --worker pkg.module:Worker --socket /tmp/w.sock

Every option has an environment variable and a default:

* ``--worker`` / ``REACTOR_WORKER_IMPORT`` / the manifest's ``runtime.worker``:
  the :class:`~reactor_runtime.model_worker.ModelWorker` to serve.
* ``--config`` / the manifest's ``runtime.config``: handed to ``load()``.
* ``--socket`` / ``REACTOR_MODEL_WORKER_SOCKET`` / ``/run/reactor-worker/sock``:
  where session processes connect. Its directory is created when missing.
* ``--health-port`` / ``REACTOR_MODEL_WORKER_HEALTH_PORT``: when set, ``GET
  /health`` on that port answers 503 ``loading`` while ``load()`` runs, 200
  ``serving`` once sessions can connect, with ``shm_slot_bytes``, the
  ``/dev/shm`` each session holds, and 503 ``stuck`` while one step has run
  longer than ``REACTOR_MODEL_WORKER_STUCK_SECONDS`` (120 by default), or
  ``failing`` once ``REACTOR_MODEL_WORKER_FAILING_STEPS`` steps in a row (5 by
  default) have raised for every session that stepped in the last minute, as
  they do when the GPU's state is broken.
* ``REACTOR_SHM_PREFIX`` (``rrmw`` by default): the shared blocks of this
  worker and its sessions are named under it and a token of each start. A
  worker that starts unlinks the blocks the worker before it on the same
  socket left, which it reads from ``shm-prefix`` next to the socket.

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
from reactor_runtime.manifest import MANIFEST, import_ref, load_config, load_worker_ref
from reactor_runtime.model_worker.client import DEFAULT_SOCKET, SOCKET_ENV
from reactor_runtime.model_worker.contract import ModelWorker
from reactor_runtime.model_worker.server import WorkerServer
from reactor_runtime.model_worker.shm import unlink_prefix

logger = log.get_logger("reactor_runtime.model_worker")

WORKER_ENV = "REACTOR_WORKER_IMPORT"
HEALTH_PORT_ENV = "REACTOR_MODEL_WORKER_HEALTH_PORT"
STUCK_ENV = "REACTOR_MODEL_WORKER_STUCK_SECONDS"
SHM_PREFIX_ENV = "REACTOR_SHM_PREFIX"
FAILING_ENV = "REACTOR_MODEL_WORKER_FAILING_STEPS"


class _Status:
    """What ``/health`` reports, shared between the main thread and the probe server."""

    def __init__(self, stuck_after: float) -> None:
        self.phase = "loading"
        self.server: WorkerServer | None = None
        self.stuck_after = stuck_after
        self.failing_after = _int_env(FAILING_ENV, 5)
        self.slot_bytes: int | None = None

    def report(self) -> tuple[int, dict[str, Any]]:
        server = self.server
        if self.phase == "serving" and server is not None:
            running = server.step_running_for
            if running > self.stuck_after:
                return 503, {"status": "stuck", "step_seconds": round(running, 1)}
            if server.failing(self.failing_after):
                return 503, {"status": "failing", "failed_steps": server.failed_steps}
            return 200, {"status": "serving", "shm_slot_bytes": self.slot_bytes}
        return 503, {"status": self.phase}


def main(argv: list[str] | None = None) -> int:
    """Load the worker half and serve it until signalled; return the exit status."""
    parser = argparse.ArgumentParser(prog="reactor-runtime serve-worker")
    parser.add_argument("--worker", default=None, help="module:Class of the ModelWorker")
    parser.add_argument("--socket", default=None, help="the Unix socket sessions connect to")
    parser.add_argument("--config", default=None, help="the model's config file")
    parser.add_argument("--health-port", type=int, default=None, help="serve GET /health here")
    args = parser.parse_args(argv)
    name = os.getenv("REACTOR_LOG_LEVEL", "INFO").strip().upper()
    log.configure(level=logging.getLevelNamesMapping().get(name, logging.INFO))
    sys.path.insert(0, str(Path.cwd()))

    manifest = Path.cwd() / MANIFEST
    worker_ref = args.worker or os.getenv(WORKER_ENV, "").strip() or None
    config_path = Path(args.config).resolve() if args.config else None
    if manifest.is_file():
        worker_ref = worker_ref or load_worker_ref(manifest)
        if config_path is None:
            config_path = load_config(manifest).config_path
    if not worker_ref:
        raise SystemExit(
            f"no worker half named: pass --worker, set {WORKER_ENV}, or name runtime.worker "
            f"in {MANIFEST}"
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
        if not (isinstance(worker_cls, type) and issubclass(worker_cls, ModelWorker)):
            raise SystemExit(f"{worker_ref} does not name a ModelWorker subclass")
        worker = worker_cls()
        worker.device = _device()
        started = time.monotonic()
        worker.load(config_path)
        logger.info(
            "model worker loaded",
            worker=worker_ref,
            device=worker.device,
            seconds=round(time.monotonic() - started, 2),
            shm_slot_bytes=worker_cls.shm_slot_size(),
        )
        status.slot_bytes = worker_cls.shm_slot_size()
        server = WorkerServer(worker, socket_path, shm_prefix=prefix)
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


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        raise SystemExit(f"{name} {raw!r} must be an integer") from None
    if value < 1:
        raise SystemExit(f"{name} must be at least 1")
    return value


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
