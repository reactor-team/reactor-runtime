"""Saving each finished step as a folder of files, :class:`StepStore`.

A step result is everything one step produced, kept together under
``<root>/<session_id>/<step>/``: the step's output as one ``output.mp4``, the
extra files the model kept with it, and ``result.json``, which lists them. The
store writes ``result.json`` last, so a folder that has one is complete, and a
reader never sees a step that is half written.

Saving never makes the model wait. A step is offered to the store as it is
reported; one that finds the queue full is not saved. Each folder is deleted a
fixed time after its ``result.json`` is written, whether or not the session is
still running, so disk use follows the step rate rather than the session's
length.
"""

from __future__ import annotations

import contextlib
import json
import mimetypes
import os
import queue
import re
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reactor_runtime.core import CompletedStep, StepResultsConfig
from reactor_runtime.log import get_logger
from reactor_runtime.step_results.mp4 import write_mp4

logger = get_logger(__name__)

# How long a step folder is kept after its result.json is written.
_RETENTION_SECONDS = 300.0
# How often the reaper sweeps the root for aged-out step folders.
_REAP_INTERVAL_SECONDS = 30.0
# How long close() lets pending saves finish before it abandons the rest.
_DRAIN_SECONDS = 10.0
# How often a full queue is reported. A model that steps faster than its steps
# encode fills the queue on every step, so only the warning is rate-limited.
_DROP_LOG_INTERVAL_SECONDS = 5.0
# How often the worker looks up from an empty queue to check for close().
_IDLE_POLL_SECONDS = 0.1
# A session id is a lowercase UUID. Only such an id is used as a folder name,
# so a caller-chosen id can never point outside the root.
_SESSION_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

RESULT_FILE = "result.json"
OUTPUT_FILE = "output.mp4"


@dataclass(frozen=True)
class SavedStep:
    """A step whose folder is complete.

    Attributes:
        session_id: The session the step belongs to.
        step: The step's number within the session.
        files: The names of the files in the folder beside ``result.json``.
    """

    session_id: str
    step: int
    files: tuple[str, ...]


@dataclass(frozen=True)
class _Pending:
    session_id: str
    step: int
    completed: CompletedStep
    messages: tuple[Mapping[str, Any], ...]


class StepStore:
    """Save each finished step as a folder, and age the folders out.

    One worker thread saves steps in the order they were admitted; a second
    thread deletes folders whose retention has passed. Both start with the
    first admitted step, and the root is a fresh temporary directory created
    then, so a runtime that saves nothing creates nothing.
    """

    def __init__(
        self,
        config: StepResultsConfig,
        on_saved: Callable[[SavedStep], None],
        *,
        root: Path | None = None,
    ) -> None:
        """Bind the store to the encoding settings and the callback for finished folders.

        Args:
            config: The manifest's step-results settings: the codecs and the
                queue size.
            on_saved: Called from the worker thread once a step's folder is
                complete.
            root: Where to keep the folders. A fresh temporary directory when
                omitted.
        """
        self._config = config
        self._on_saved = on_saved
        self._root = root
        self._queue: queue.Queue[_Pending] = queue.Queue(maxsize=config.queue)
        # Orders the worker's creation of a step folder against the reaper's
        # removal of an empty session folder, so neither undoes the other.
        self._layout_lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._reaper: threading.Thread | None = None
        self._closing = threading.Event()
        self._abandon = threading.Event()
        self._reaper_stop = threading.Event()
        self._closed = False
        self._dropped = 0
        self._dropped_logged_at = 0.0

    @property
    def root(self) -> Path | None:
        """The directory the folders are kept in, or ``None`` before the first step."""
        return self._root

    def admit(
        self,
        session_id: str,
        step: int,
        completed: CompletedStep,
        messages: Sequence[Mapping[str, Any]] = (),
    ) -> bool:
        """Offer a step for saving, without waiting.

        Args:
            session_id: The session the step belongs to. Only a lowercase UUID
                is saved.
            step: The step's number within the session.
            completed: What the step produced.
            messages: The messages the model broadcast since the step before,
                each in its ``{"type", "data"}`` wire form.

        Returns:
            Whether the step will be saved. ``True`` means a folder with a
            ``result.json`` follows; ``False`` means the store is closed, the
            session id cannot name a folder, or the queue is full.
        """
        if self._closed:
            return False
        if not _SESSION_ID_RE.match(session_id):
            logger.warning(
                "not saving a step for a session id that is not a UUID",
                session_id=session_id,
                step=step,
            )
            return False
        self._ensure_started()
        try:
            self._queue.put_nowait(_Pending(session_id, step, completed, tuple(messages)))
        except queue.Full:
            self._note_dropped(session_id, step)
            return False
        return True

    def close(self, drain_seconds: float = _DRAIN_SECONDS) -> None:
        """Stop taking steps, let pending saves finish, and stop the reaper.

        Steps still waiting after *drain_seconds* are abandoned. Blocks while
        the worker drains, so the runner calls it off the event loop.
        Idempotent; safe when never started.

        Args:
            drain_seconds: How long pending saves may keep running.
        """
        self._closed = True
        self._closing.set()
        worker = self._worker
        if worker is not None:
            worker.join(timeout=drain_seconds)
            if worker.is_alive():
                self._abandon.set()
                logger.warning(
                    "abandoning step results still waiting to be saved",
                    pending=self._queue.qsize(),
                )
        self._reaper_stop.set()
        reaper = self._reaper
        if reaper is not None:
            reaper.join(timeout=2.0)

    # -- reading -----------------------------------------------------------------

    def ready_steps(self, session_id: str) -> list[int] | None:
        """Return the numbers of a session's complete steps, in order.

        Args:
            session_id: The session to list.

        Returns:
            The numbers of the steps whose ``result.json`` is written, or
            ``None`` when no folder is kept for the session: it saved no step,
            its folders have aged out, or the id is not a UUID.
        """
        session_dir = self._session_dir(session_id)
        if session_dir is None:
            return None
        try:
            children = list(session_dir.iterdir())
        except FileNotFoundError:
            return None
        return sorted(
            int(child.name)
            for child in children
            if child.name.isdigit() and (child / RESULT_FILE).is_file()
        )

    def result_path(self, session_id: str, step: int) -> Path | None:
        """Return the path of a step's ``result.json``, or ``None`` until it is written."""
        session_dir = self._session_dir(session_id)
        if session_dir is None or step < 1:
            return None
        path = session_dir / str(step) / RESULT_FILE
        return path if path.is_file() else None

    def file_path(self, session_id: str, step: int, name: str) -> tuple[Path, str] | None:
        """Return the path and content type of a file a complete step lists.

        Only a name the step's ``result.json`` lists is served, so a file
        still being written, or a name that leaves the folder, is never found.

        Args:
            session_id: The session the step belongs to.
            step: The step's number.
            name: The file's name, as ``result.json`` lists it.

        Returns:
            The file's path and content type, or ``None`` when the step is not
            complete or does not list *name*.
        """
        result = self.result_path(session_id, step)
        if result is None:
            return None
        try:
            entries = json.loads(result.read_text()).get("files", [])
        except (OSError, ValueError):
            return None
        for entry in entries:
            if entry.get("name") == name:
                path = result.parent / name
                return (path, entry["content_type"]) if path.is_file() else None
        return None

    def _session_dir(self, session_id: str) -> Path | None:
        if self._root is None or not _SESSION_ID_RE.match(session_id):
            return None
        session_dir = self._root / session_id
        return session_dir if session_dir.is_dir() else None

    # -- saving ------------------------------------------------------------------

    def _ensure_started(self) -> None:
        with self._start_lock:
            if self._worker is not None:
                return
            if self._root is None:
                self._root = Path(tempfile.mkdtemp(prefix="reactor-step-results-"))
            self._root.mkdir(parents=True, exist_ok=True)
            self._worker = threading.Thread(
                target=self._work, name="step-results-writer", daemon=True
            )
            self._reaper = threading.Thread(
                target=self._reap_loop, name="step-results-reaper", daemon=True
            )
            self._worker.start()
            self._reaper.start()

    def _note_dropped(self, session_id: str, step: int) -> None:
        self._dropped += 1
        now = time.monotonic()
        if now - self._dropped_logged_at >= _DROP_LOG_INTERVAL_SECONDS:
            self._dropped_logged_at = now
            logger.warning(
                "step results queue full; not saving the step",
                session_id=session_id,
                step=step,
                dropped=self._dropped,
            )

    def _work(self) -> None:
        while not self._abandon.is_set():
            try:
                pending = self._queue.get(timeout=_IDLE_POLL_SECONDS)
            except queue.Empty:
                if self._closing.is_set():
                    return
                continue
            if self._abandon.is_set():
                return
            try:
                saved = self._save(pending)
            except Exception:
                logger.exception(
                    "failed to write a step result",
                    session_id=pending.session_id,
                    step=pending.step,
                )
                continue
            try:
                self._on_saved(saved)
            except Exception:
                logger.exception("step result callback failed", step=pending.step)

    def _save(self, pending: _Pending) -> SavedStep:
        """Write one step's folder, ``result.json`` last.

        A step that cannot be encoded or written still gets its
        ``result.json``, holding the reason in ``save_error`` and no files, so
        an admitted step always ends in a complete folder.
        """
        assert self._root is not None
        step_dir = self._root / pending.session_id / str(pending.step)
        with self._layout_lock:
            step_dir.mkdir(parents=True, exist_ok=True)
        completed = pending.completed
        started = time.perf_counter()
        files: list[dict[str, Any]] = []
        save_error: str | None = None
        try:
            if completed.bundle is not None and completed.bundle.tracks:
                path = step_dir / OUTPUT_FILE
                write_mp4(path, completed.bundle, completed.fps, self._config)
                files.append(_file_entry(path))
            for name, data in completed.files.items():
                path = step_dir / name
                path.write_bytes(data)
                files.append(_file_entry(path))
        except Exception as error:
            logger.exception(
                "failed to save a step result",
                session_id=pending.session_id,
                step=pending.step,
            )
            save_error = f"{type(error).__name__}: {error}"
            for name in (OUTPUT_FILE, *completed.files):
                (step_dir / name).unlink(missing_ok=True)
            files = []
        result = {
            "step": pending.step,
            "session_id": pending.session_id,
            "files": files,
            "messages": list(pending.messages),
            "error": completed.error,
            "save_error": save_error,
            "timings": {
                "generate_s": completed.elapsed,
                "encode_s": time.perf_counter() - started,
            },
        }
        _write_atomically(step_dir / RESULT_FILE, json.dumps(result, default=str).encode())
        return SavedStep(
            session_id=pending.session_id,
            step=pending.step,
            files=tuple(entry["name"] for entry in files),
        )

    # -- retention ---------------------------------------------------------------

    def _reap_loop(self) -> None:
        while not self._reaper_stop.is_set():
            try:
                self._reap_expired(time.time())
            except Exception:
                logger.exception("step results reaper sweep failed")
            self._reaper_stop.wait(_REAP_INTERVAL_SECONDS)

    def _reap_expired(self, now: float) -> None:
        """Delete every step folder whose ``result.json`` is older than the retention.

        A folder still being written has no ``result.json`` and is never
        deleted. A session folder left empty is removed too.
        """
        root = self._root
        if root is None:
            return
        for session_dir in root.iterdir():
            if not session_dir.is_dir():
                continue
            for step_dir in session_dir.iterdir():
                result = step_dir / RESULT_FILE
                try:
                    written_at = result.stat().st_mtime
                except FileNotFoundError:
                    continue
                if now - written_at > _RETENTION_SECONDS:
                    shutil.rmtree(step_dir, ignore_errors=True)
                    logger.debug(
                        "reaped an aged-out step result",
                        session_id=session_dir.name,
                        step=step_dir.name,
                    )
            with self._layout_lock, contextlib.suppress(OSError):
                session_dir.rmdir()


def _file_entry(path: Path) -> dict[str, Any]:
    content_type, _ = mimetypes.guess_type(path.name)
    return {
        "name": path.name,
        "content_type": content_type or "application/octet-stream",
        "size": path.stat().st_size,
    }


def _write_atomically(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)
