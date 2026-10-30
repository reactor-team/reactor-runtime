"""The model server's process as a service component: load the model, then serve it.

:class:`ModelServerProcess` is what ``python -m reactor_runtime.model_server``
runs, beside a :class:`~reactor_runtime.http.HealthServer`, under the runtime's
:class:`~reactor_runtime.service.Service`. Its ``start()`` checks the
deployment before anything heavy happens (the model half's class, its settings,
and each slot's shared-memory directory), reclaims the blocks a previous run
left, loads the weights, and then starts a
:class:`~reactor_runtime.model_server.WorkerServer`. Its lifecycle word is
``loading`` until the weights are in, ``available`` while it serves, and
``terminated`` once stopped; every record the process writes carries it as
``runtime_state``.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
import time
from pathlib import Path

from reactor_runtime.core import Health, HealthStatus, RuntimeState, WorkerConfig
from reactor_runtime.log import get_logger, set_state
from reactor_runtime.model_server.config import ServerProcessConfig
from reactor_runtime.model_server.errors import WorkerUnreachable
from reactor_runtime.model_server.link import LinkTransport
from reactor_runtime.model_server.server import ModelServer, ServerSettings
from reactor_runtime.model_server.shm_link import UnixShmTransport
from reactor_runtime.model_server.worker import WorkerServer
from reactor_runtime.service import ShutdownHook, no_shutdown

logger = get_logger(__name__)


class ModelServerProcess:
    """Load the model half named in the manifest, then serve it to the session runtimes.

    Args:
        worker: The manifest's ``runtime.worker`` block.
        config_path: The model's config file, handed to ``load()``.
        process: What the deployment set: the socket, the session count, the
            slot directories, the stuck limit.
        transport: How links are made; the shared-memory link over a Unix
            socket, on the configured slot directories, when omitted.
    """

    name = "model_server"
    depends_on: tuple[str, ...] = ()

    def __init__(
        self,
        worker: WorkerConfig,
        config_path: Path | None,
        process: ServerProcessConfig,
        *,
        transport: LinkTransport | None = None,
    ) -> None:
        self._worker = worker
        self._config_path = config_path
        self._process = process
        self._transport = transport
        self._state = RuntimeState.LOADING
        self._server: WorkerServer | None = None
        self.request_shutdown: ShutdownHook = no_shutdown

    def state(self) -> RuntimeState:
        """The process's lifecycle word: ``loading``, ``available``, or ``terminated``."""
        return self._state

    async def start(self) -> None:
        """Check the deployment, load the model, and start serving.

        A ``load()`` that raises is handled as the session runtime handles a
        model that fails to load: the failure is logged with its traceback, the
        lifecycle word moves to ``terminated``, and the process is asked to shut
        down cleanly through :attr:`request_shutdown`, so it exits with status
        0 and its deployment restarts it. ``start`` then returns.

        Raises:
            SystemExit: The manifest names no ``ModelServer`` subclass or a
                module that does not exist, a setting is outside what the class
                allows, a slot's shared-memory path is not a directory or its
                volume is too small, or another model server already listens on
                the socket.
        """
        cls = _import(self._worker.import_ref)
        try:
            settings = ServerSettings.resolve(cls, self._worker)
        except ValueError as exc:
            raise SystemExit(f"{self._worker.import_ref}: {exc}") from None
        transport = self._transport or UnixShmTransport(
            slot_directory=self._process.slot_directory,
            input_bytes=settings.shm_input_bytes,
            output_bytes=settings.shm_output_bytes,
        )
        if isinstance(transport, UnixShmTransport):
            _check_directories(transport, self._process.max_sessions)
            reclaimed = transport.reclaim(self._process.max_sessions)
            if reclaimed:
                logger.info("reclaimed blocks a previous model server left", count=reclaimed)
            _check_room(transport, self._process.max_sessions, settings)
        model = cls()
        model.device = _device()
        started = time.monotonic()
        logger.info("loading the model", model=self._worker.import_ref, device=model.device)
        try:
            await asyncio.to_thread(model.load, self._config_path)
        except Exception:
            logger.exception("the model failed to load", model=self._worker.import_ref)
            self._move(RuntimeState.TERMINATED)
            self.request_shutdown(failure=False)
            return
        logger.info(
            "model loaded",
            model=self._worker.import_ref,
            seconds=round(time.monotonic() - started, 2),
            max_batch=settings.max_batch,
            shm_slot_bytes=settings.shm_slot_bytes,
        )
        self._server = WorkerServer(
            model,
            transport,
            self._process.socket,
            settings=settings,
            max_sessions=self._process.max_sessions,
            stuck_after=self._process.stuck_after,
        )
        try:
            await self._server.start()
        except WorkerUnreachable as exc:
            raise SystemExit(str(exc)) from None
        self._move(RuntimeState.AVAILABLE)

    async def drain(self) -> None:
        """Stop accepting sessions and finish the running step."""
        if self._server is not None:
            await self._server.drain()

    async def stop(self) -> None:
        """Close every session and release their memory."""
        if self._server is not None:
            await self._server.stop()
        self._move(RuntimeState.TERMINATED)

    def _move(self, state: RuntimeState) -> None:
        self._state = state
        set_state(None, state.value)

    def health(self) -> Health:
        """Healthy while loading, the model server's own health once it serves.

        Unhealthy once terminated, whether it served or its model failed to
        load, as the session runtime reports a terminated session.
        """
        if self._state is RuntimeState.TERMINATED:
            return Health(HealthStatus.UNHEALTHY, "the model server terminated")
        if self._server is None:
            return Health.healthy()
        return self._server.health()


def _import(ref: str) -> type[ModelServer]:
    module_name, _, class_name = ref.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        # Only the module the manifest names, or a package on its path: one
        # the model's own code imports and lacks is the model's error.
        if exc.name is None or not (
            module_name == exc.name or module_name.startswith(f"{exc.name}.")
        ):
            raise
        raise SystemExit(
            f"runtime.worker.import {ref!r} names module {module_name!r}, which does not exist"
        ) from None
    cls = getattr(module, class_name, None)
    if not (isinstance(cls, type) and issubclass(cls, ModelServer)):
        raise SystemExit(f"runtime.worker.import {ref!r} does not name a ModelServer subclass")
    return cls


def _check_directories(transport: UnixShmTransport, slots: int) -> None:
    for slot in range(slots):
        directory = transport.slot_directory(slot)
        if not directory.exists():
            raise SystemExit(
                f"slot {slot}'s shared-memory directory {directory} does not exist: mount the "
                "session's memory-backed volume there"
            )
        if not directory.is_dir():
            raise SystemExit(
                f"slot {slot}'s shared-memory path {directory} is not a directory: mount the "
                "session's memory-backed volume there"
            )


def _check_room(transport: UnixShmTransport, slots: int, settings: ServerSettings) -> None:
    """Check each slot's volume is large enough for one session.

    The volume's size, not its free space: a previous run's blocks may still
    fill it. The model server allocates nothing in a slot until a session opens
    a link there, and by then each side has removed its own previous run's
    blocks; a link that still finds no room is refused, saying so.
    """
    for slot in range(slots):
        directory = transport.slot_directory(slot)
        try:
            stats = os.statvfs(directory)
        except OSError as exc:
            raise SystemExit(
                f"slot {slot}'s shared-memory directory {directory} cannot be read: {exc}. Mount "
                "the session's memory-backed volume there"
            ) from None
        total = stats.f_blocks * stats.f_frsize
        if total < settings.shm_slot_bytes:
            raise SystemExit(
                f"slot {slot}'s shared-memory volume {directory} holds {total} bytes and one "
                f"session needs up to {settings.shm_slot_bytes} (2 * runtime.worker."
                "shm_input_bytes + shm_output_bytes, in whole 64 KiB pages): size the "
                "session's memory-backed volume to fit them"
            )


def _device() -> str:
    torch = sys.modules.get("torch")
    if torch is None:
        return "cpu"
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
