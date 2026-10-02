"""The step-result store: one folder per saved step, served from local disk.

A step result is the folder of everything one step produced: the step's
``Output`` encoded as ``output.mp4``, any extra files the model adds, and a
``result.json`` that lists them. It is not a recording. There is no timeline
across steps, no seeking, and nothing is written while the session streams; a
folder appears whole, when a step is saved.

The store owns the disk layout under a root directory::

    <root>/<session_id>/steps/1/output.mp4
    <root>/<session_id>/steps/1/last_frame.png
    <root>/<session_id>/steps/1/result.json

A step is written into a hidden ``.<n>.partial`` sibling first, ``result.json``
last, and the folder is renamed into place only then. So a reader that lists
``steps/`` never sees a half-written step, and ``result.json`` is still the
last file written inside it: a consumer that mirrors the folder elsewhere can
copy it last and know the mirror is complete when it lands.
"""

from __future__ import annotations

import json
import mimetypes
import queue
import re
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from reactor_runtime.core.service import StepResultsConfig
from reactor_runtime.core.values import CompletedStep, MediaBundle
from reactor_runtime.log import get_logger
from reactor_runtime.step_results.mp4 import encode_mp4
from reactor_runtime.step_results.result import (
    MEDIA_FILENAME,
    RESULT_FILENAME,
    StepResult,
    StepResultCancelledError,
    StepResultsDisabledError,
)
from reactor_runtime.step_results.validate import check_filenames, prepare_tracks

logger = get_logger(__name__)

StepReadyCallback = Callable[[StepResult], None]
"""Called once a step folder is complete and renamed into place, on the saving thread."""

_STEPS_DIRNAME = "steps"
# Marks a session's folder as finished, for the retention reaper. A live
# session's folder carries no marker and is never removed.
_END_MARKER = ".complete"
# How long a finished session's steps stay on disk, so a consumer told of a
# step has time to fetch it after the session ends, and how often that is checked.
_RETENTION_SECONDS = 300.0
_REAP_INTERVAL_SECONDS = 30.0

# How many announced steps may wait for the worker before the announcing
# model is made to wait too. A step holds its frames in memory until it is
# encoded, so the bound is what keeps a slow disk from turning into unbounded
# RAM; with it, a model that outruns the encoder is paced to it.
_PENDING_STEPS = 4
# How long stop() waits for the worker to finish the steps still queued, so
# the last step of a session lands before the session is marked finished.
_DRAIN_SECONDS = 60.0

_SESSION_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_STEP_DIR_RE = re.compile(r"^[1-9][0-9]*$")
_MEDIA_TYPE = "video/mp4"
_RESULT_MEDIA_TYPE = "application/json"
_FALLBACK_MEDIA_TYPE = "application/octet-stream"


class StepResultStore:
    """Saves one session's step results and serves them back from local disk.

    Owned by the runner and built from the step-results config. The runner
    calls :meth:`start` on the session-start boundary and :meth:`stop` on
    close; in between, :meth:`enqueue` takes each step the model announces
    and a worker of the store's own saves it, :meth:`fail` writes the error
    folder that ends a session, and :meth:`list_steps`, :meth:`result_path`,
    and :meth:`file_path` answer the ``/steps`` routes. Steps are numbered
    from one per session, in the order they are announced, and a session
    keeps every step until it ends unless ``keep_last`` bounds the window.

    The model is never held for a save: an announcement is a queue put, and
    the encode runs on the worker. Saves are serialised on that worker, so the
    numbers match the order on disk and the encoder runs one file at a time.
    """

    def __init__(
        self, config: StepResultsConfig, *, on_ready: StepReadyCallback | None = None
    ) -> None:
        """Bind the store to its config and the readiness notification.

        Args:
            config: The store's tunables, including the directory steps are
                written under.
            on_ready: Called once a step folder is complete, on the thread that
                saved it, so the step can be announced.
        """
        self._config = config
        self._on_ready = on_ready
        # The root is materialised on the first start, so a disabled store
        # (the common case) never creates a directory.
        self._root: Path | None = None
        self._session_id: str | None = None
        self._session_dir: Path | None = None
        self._steps_dir: Path | None = None
        self._count = 0
        # The highest step the rolling window has already removed, so a trim
        # deletes only the folders between it and the new cutoff.
        self._trimmed_through = 0
        self._save_lock = threading.Lock()
        self._closed = threading.Event()
        # The announced steps waiting for the worker, and the worker itself.
        # A None on the queue is the stop sentinel: the worker saves everything
        # queued before it, then exits.
        self._pending: queue.Queue[CompletedStep | None] = queue.Queue(maxsize=_PENDING_STEPS)
        self._worker: threading.Thread | None = None
        # The process-lifetime reaper that ages finished sessions out of the
        # root. Started once the root exists and stopped by close().
        self._reaper_thread: threading.Thread | None = None
        self._reaper_stop = threading.Event()
        # Orders a start's claim of the session directory against the reaper's
        # decision to delete one.
        self._reaper_lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        """Whether step results are turned on by config."""
        return self._config.enabled

    @property
    def saved_count(self) -> int:
        """How many steps the live session has saved, errors included."""
        return self._count

    # -- lifecycle ------------------------------------------------------------

    def start(self, session_id: str) -> None:
        """Open the session's step directory under *session_id* and reset the count.

        A no-op when step results are disabled.

        Args:
            session_id: The id the session's steps are stored and served under.

        Raises:
            ValueError: If *session_id* is not a lowercase UUID, which is what
                keeps a directory name from being a path.
        """
        if not self._config.enabled:
            return
        if not _SESSION_ID_RE.fullmatch(session_id):
            raise ValueError("step result session_id must be a canonical lowercase UUID")
        if self._root is None:
            self._root = (
                Path(self._config.step_results_dir)
                if self._config.step_results_dir
                else Path(tempfile.mkdtemp(prefix="reactor-step-results-"))
            )
            self._root.mkdir(parents=True, exist_ok=True)
        self._ensure_reaper()
        with self._save_lock, self._reaper_lock:
            self._session_id = session_id
            self._session_dir = self._root / session_id
            self._steps_dir = self._session_dir / _STEPS_DIRNAME
            self._steps_dir.mkdir(parents=True, exist_ok=True)
            (self._session_dir / _END_MARKER).unlink(missing_ok=True)
            self._count = 0
            self._trimmed_through = 0
        if self._worker is None:
            self._worker = threading.Thread(
                target=self._work, name="step-results-worker", daemon=True
            )
            self._worker.start()
        logger.info("step results started", session_id=session_id, dir=str(self._steps_dir))

    def stop(self) -> None:
        """Finish the queued saves and close the session's step directory.

        The steps announced before the stop are saved first, bounded by
        :data:`_DRAIN_SECONDS`, so the last step of a session lands before the
        directory is marked finished; the reaper removes it once the retention
        window passes. Blocks while the worker drains, so the runner runs it
        off the event loop.
        """
        worker = self._worker
        if worker is not None:
            self._pending.put(None)
            worker.join(timeout=_DRAIN_SECONDS)
            if worker.is_alive():
                logger.warning(
                    "step results worker did not finish its queued saves in time",
                    session_id=self._session_id,
                )
            self._worker = None
        with self._save_lock:
            session_dir = self._session_dir
            if session_dir is None:
                return
            try:
                (session_dir / _END_MARKER).write_text("")
            except OSError:
                logger.exception("failed to mark the step results finished", dir=str(session_dir))
            self._session_id = None
            self._session_dir = None
            self._steps_dir = None

    def close(self) -> None:
        """Stop the worker and the retention reaper, and refuse further saves. Idempotent.

        A save in flight is cancelled; a stop before this is what lets it
        finish.
        """
        self._closed.set()
        worker = self._worker
        self._worker = None
        if worker is not None:
            self._pending.put(None)
            worker.join(timeout=2.0)
        self._reaper_stop.set()
        reaper = self._reaper_thread
        self._reaper_thread = None
        if reaper is not None:
            reaper.join(timeout=2.0)

    # -- saving ---------------------------------------------------------------

    def enqueue(self, step: CompletedStep) -> None:
        """Take a step the model announced; the worker saves it.

        Called off the model loop by the runner's step sink. Returns as soon
        as the step is queued. When :data:`_PENDING_STEPS` steps are already
        waiting, the call waits for room, which paces a model that outruns the
        encoder instead of letting its frames pile up in memory. A step
        announced with no live session, or with step results off, is dropped
        with a warning: there is nowhere to put it.

        Args:
            step: The completed step, as the model announced it.
        """
        if not self._config.enabled or self._steps_dir is None or self._closed.is_set():
            logger.warning("step result announced with no session to save it into; dropped")
            return
        if self._pending.full():
            logger.warning(
                "step results worker is behind; the model waits for it",
                pending=self._pending.qsize(),
            )
        self._pending.put(step)

    def _work(self) -> None:
        """Save each queued step in turn until the stop sentinel."""
        while True:
            step = self._pending.get()
            if step is None:
                return
            try:
                self.save(
                    step.bundle,
                    step.fps,
                    step.files,
                    messages=step.messages,
                    timings=step.timings,
                )
            except StepResultCancelledError:
                logger.info("step result save cancelled")
            except Exception:
                logger.exception("failed to save the step result")

    def save(
        self,
        bundle: MediaBundle | None,
        fps: float,
        files: Mapping[str, bytes | Path] | None = None,
        *,
        messages: list[dict[str, Any]] | None = None,
        timings: Mapping[str, float] | None = None,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> StepResult:
        """Write one step folder and announce it. Blocks while it encodes.

        The worker calls this for each queued step; calling it directly saves
        on the current thread, serialised against the worker.

        Args:
            bundle: The step's media, every track of the ``Output``, or ``None``
                for a step that produced files only.
            fps: The rate the video streams play at.
            files: Extra files to add to the folder, by name: the bytes to
                write, or a path to copy.
            messages: The messages sent during the step, in wire form, when the
                caller recorded them. Left out of ``result.json`` otherwise.
            timings: Timings the caller measured, in seconds. The encode time
                is added here.
            cancelled: Polled while encoding; a true reading abandons the save.

        Returns:
            The saved step.

        Raises:
            StepResultsDisabledError: If step results are off or no session is live.
            StepResultCancelledError: If the save was cancelled or the store closed.
            ValueError: If the media, the frame rate, or a file name is invalid.
            OSError: If the folder cannot be written.
        """
        files = dict(files or {})
        check_filenames(files)
        prepared = prepare_tracks(bundle, fps) if bundle is not None and bundle.tracks else []
        if not prepared and not files:
            raise ValueError("a step result needs media or at least one file")

        def is_cancelled() -> bool:
            return self._closed.is_set() or cancelled()

        def check_cancelled() -> None:
            if is_cancelled():
                raise StepResultCancelledError("step result save cancelled")

        with self._save_lock:
            steps_dir = self._require_live()
            check_cancelled()
            step = self._count + 1
            partial = steps_dir / f".{step}.partial"
            shutil.rmtree(partial, ignore_errors=True)
            partial.mkdir()
            try:
                document: dict[str, Any] = {"step": step, "files": [], "tracks": []}
                measured = dict(timings or {})
                if prepared:
                    started = time.perf_counter()
                    streams = encode_mp4(
                        partial / MEDIA_FILENAME, prepared, fps, self._config, is_cancelled
                    )
                    measured["encode_s"] = round(time.perf_counter() - started, 4)
                    document["files"].append({"name": MEDIA_FILENAME, "content_type": _MEDIA_TYPE})
                    document["tracks"] = [stream.to_dict() for stream in streams]
                for name, payload in files.items():
                    check_cancelled()
                    _write_file(partial / name, payload)
                    document["files"].append({"name": name, "content_type": _guess_type(name)})
                if messages is not None:
                    document["messages"] = list(messages)
                document["timings"] = measured
                check_cancelled()
                _write_result(partial, document)
                partial.rename(steps_dir / str(step))
            except BaseException:
                shutil.rmtree(partial, ignore_errors=True)
                raise
            self._count = step
            self._trim(steps_dir, step)
            result = self._result(step, document)
        self._announce(result)
        return result

    def fail(self, code: str, message: str) -> StepResult | None:
        """Write a step folder that carries an error instead of a result, and announce it.

        Used when a step cannot be saved because the model failed: the folder
        holds only a ``result.json`` with an ``error`` object, so a consumer
        waiting on the step gets the reason instead of silence. Best-effort: a
        store with no live session writes nothing and returns ``None``.

        Args:
            code: A short, stable token the consumer branches on.
            message: The readable reason.

        Returns:
            The saved step, or ``None`` when there was no session to save into.
        """
        with self._save_lock:
            if not self._config.enabled or self._steps_dir is None or self._closed.is_set():
                return None
            steps_dir = self._steps_dir
            step = self._count + 1
            partial = steps_dir / f".{step}.partial"
            shutil.rmtree(partial, ignore_errors=True)
            try:
                partial.mkdir()
                document: dict[str, Any] = {
                    "step": step,
                    "files": [],
                    "error": {"code": code, "message": message},
                }
                _write_result(partial, document)
                partial.rename(steps_dir / str(step))
            except OSError:
                shutil.rmtree(partial, ignore_errors=True)
                logger.exception("failed to write the error step result", step=step)
                return None
            self._count = step
            self._trim(steps_dir, step)
            result = self._result(step, document)
        self._announce(result)
        return result

    def _trim(self, steps_dir: Path, latest: int) -> None:
        """Drop the steps that fell out of the rolling window once *latest* landed.

        With ``keep_last`` set, every step numbered at or below
        ``latest - keep_last`` is removed. The caller holds the save lock, so
        the window moves with the count and a step is never removed while it
        is being written. Without ``keep_last`` nothing is removed.
        """
        keep = self._config.keep_last
        if keep is None or keep < 1:
            return
        cutoff = latest - keep
        for number in range(self._trimmed_through + 1, cutoff + 1):
            shutil.rmtree(steps_dir / str(number), ignore_errors=True)
        if cutoff > self._trimmed_through:
            logger.debug("trimmed step results", through=cutoff, keep_last=keep)
            self._trimmed_through = cutoff

    def _require_live(self) -> Path:
        """Return the live session's steps directory, or refuse the save."""
        if not self._config.enabled:
            raise StepResultsDisabledError("step results are not enabled for this model")
        if self._steps_dir is None:
            raise StepResultsDisabledError("no session is live to save a step result into")
        return self._steps_dir

    def _result(self, step: int, document: Mapping[str, Any]) -> StepResult:
        """Build the result record for a step just written."""
        assert self._session_id is not None
        names = [entry["name"] for entry in document["files"]]
        return StepResult(self._session_id, step, [*names, RESULT_FILENAME])

    def _announce(self, result: StepResult) -> None:
        """Tell the owner a step is complete; a failing callback never fails the save."""
        if self._on_ready is None:
            return
        try:
            self._on_ready(result)
        except Exception:
            logger.exception("step result notification failed", step=result.step)

    # -- serving (read by the HTTP routes) ------------------------------------

    def list_steps(self, session_id: str) -> list[dict[str, Any]] | None:
        """List the complete steps of a session, in order.

        Args:
            session_id: The id the session's steps are stored under.

        Returns:
            ``{"step", "files"}`` per complete step, or ``None`` when the
            session has no step directory, which includes a store that never
            started and an id that is not a UUID.
        """
        steps_dir = self._steps_dir_for(session_id)
        if steps_dir is None:
            return None
        listed: list[dict[str, Any]] = []
        for child in steps_dir.iterdir():
            if not child.is_dir() or not _STEP_DIR_RE.fullmatch(child.name):
                continue
            names = _file_names(child)
            if names is not None:
                listed.append({"step": int(child.name), "files": names})
        listed.sort(key=lambda entry: entry["step"])
        return listed

    def result_path(self, session_id: str, step: int) -> Path | None:
        """Return the path of a complete step's ``result.json``, or ``None``."""
        step_dir = self._step_dir_for(session_id, step)
        if step_dir is None:
            return None
        path = step_dir / RESULT_FILENAME
        return path if path.is_file() else None

    def file_path(self, session_id: str, step: int, name: str) -> tuple[Path, str] | None:
        """Return one file of a complete step and its content type, or ``None``.

        Only a file ``result.json`` lists is served, and ``result.json``
        itself, so a name the model never wrote is not a path into the folder.
        """
        step_dir = self._step_dir_for(session_id, step)
        if step_dir is None:
            return None
        if name == RESULT_FILENAME:
            path = step_dir / RESULT_FILENAME
            return (path, _RESULT_MEDIA_TYPE) if path.is_file() else None
        entries = _entries(step_dir)
        if entries is None:
            return None
        for entry in entries:
            if entry.get("name") == name:
                path = step_dir / name
                if not path.is_file():
                    return None
                return path, str(entry.get("content_type") or _FALLBACK_MEDIA_TYPE)
        return None

    def _steps_dir_for(self, session_id: str) -> Path | None:
        """Resolve a session's steps directory, or ``None`` when there is none."""
        if self._root is None or not _SESSION_ID_RE.fullmatch(session_id):
            return None
        steps_dir = self._root / session_id / _STEPS_DIRNAME
        return steps_dir if steps_dir.is_dir() else None

    def _step_dir_for(self, session_id: str, step: int) -> Path | None:
        """Resolve a complete step's directory, or ``None`` when it is not there yet."""
        steps_dir = self._steps_dir_for(session_id)
        if steps_dir is None or step < 1:
            return None
        step_dir = steps_dir / str(step)
        if not step_dir.is_dir() or not (step_dir / RESULT_FILENAME).is_file():
            return None
        return step_dir

    # -- retention ------------------------------------------------------------

    def _ensure_reaper(self) -> None:
        """Start the retention reaper once the root exists, at most once."""
        if self._reaper_thread is not None:
            return
        self._reaper_stop.clear()
        self._reaper_thread = threading.Thread(
            target=self._reap_loop, name="step-results-reaper", daemon=True
        )
        self._reaper_thread.start()

    def _reap_loop(self) -> None:
        """Sweep aged-out sessions from the root until close() stops the reaper."""
        while not self._reaper_stop.is_set():
            try:
                self._reap_expired(time.time())
            except Exception:
                logger.exception("step results reaper sweep failed")
            self._reaper_stop.wait(_REAP_INTERVAL_SECONDS)

    def _reap_expired(self, now: float) -> None:
        """Delete every finished session whose retention window has passed.

        The live session's directory is skipped, and a session still running
        carries no end marker, so an active session is never removed however
        long it runs. Each directory is judged and deleted under the same lock
        a start claims the live directory with.
        """
        root = self._root
        if root is None:
            return
        for session_dir in root.iterdir():
            if not session_dir.is_dir():
                continue
            with self._reaper_lock:
                if session_dir == self._session_dir:
                    continue
                marker = session_dir / _END_MARKER
                try:
                    finished_at = marker.stat().st_mtime
                except OSError:
                    continue
                if now - finished_at <= _RETENTION_SECONDS:
                    continue
                shutil.rmtree(session_dir, ignore_errors=True)
            logger.info("reaped aged-out step results", session_id=session_dir.name)


def _write_file(path: Path, payload: bytes | Path) -> None:
    """Write bytes, or copy a file the model already has on disk."""
    if isinstance(payload, Path):
        shutil.copyfile(payload, path)
    else:
        path.write_bytes(payload)


def _write_result(folder: Path, document: Mapping[str, Any]) -> None:
    """Write ``result.json`` into *folder*, the last file of a step."""
    (folder / RESULT_FILENAME).write_text(json.dumps(document, indent=2) + "\n")


def _guess_type(name: str) -> str:
    """The content type a file's name says it is, or the binary fallback."""
    guessed, _ = mimetypes.guess_type(name)
    return guessed or _FALLBACK_MEDIA_TYPE


def _entries(step_dir: Path) -> list[dict[str, Any]] | None:
    """Read the file list out of a step's ``result.json``, or ``None`` if unreadable."""
    try:
        document = json.loads((step_dir / RESULT_FILENAME).read_text())
    except (OSError, ValueError):
        return None
    entries = document.get("files") if isinstance(document, dict) else None
    if not isinstance(entries, list):
        return None
    return [entry for entry in entries if isinstance(entry, dict)]


def _file_names(step_dir: Path) -> list[str] | None:
    """The names of a complete step's files, ``result.json`` last."""
    entries = _entries(step_dir)
    if entries is None:
        return None
    return [str(entry["name"]) for entry in entries if "name" in entry] + [RESULT_FILENAME]
