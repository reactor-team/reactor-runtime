"""Stop this process when the pod it runs in is asked to restart as a whole.

A pod that runs a model server and its session runtimes as separate containers
can be restarted container by container without a call to the Kubernetes API:
whatever manages the pod writes a new token to ``restart`` in a small volume
every container mounts (``REACTOR_POD_CONTROL_DIR``), mounted read-only in the
runtime's containers, and each runtime process stops on its own, as on
SIGTERM. The kubelet then starts it again.

A process reads the token once when it starts and stops only when it changes,
so the token that restarted it does not stop it again. Without
``REACTOR_POD_CONTROL_DIR`` nothing is watched.
"""

from __future__ import annotations

import os
import signal
import threading
from collections.abc import Callable
from pathlib import Path

from reactor_runtime.log import get_logger

logger = get_logger(__name__)

POD_CONTROL_DIR_ENV = "REACTOR_POD_CONTROL_DIR"
TOKEN_FILE = "restart"
# How often the token is read.
POLL_SECONDS = 2.0
# The most bytes of the token file read.
_TOKEN_LIMIT = 256


def read_token(directory: Path) -> str:
    """The restart token in *directory*, or ``""`` when there is none or it cannot be read."""
    try:
        with (directory / TOKEN_FILE).open("rb") as f:
            return f.read(_TOKEN_LIMIT).decode("ascii", "replace").strip()
    except OSError:
        return ""


def watch_restart(
    directory: Path | None = None,
    *,
    poll: float = POLL_SECONDS,
    on_request: Callable[[], None] | None = None,
) -> threading.Thread | None:
    """Start a daemon thread that stops this process when the restart token changes.

    Args:
        directory: The pod-control directory; ``REACTOR_POD_CONTROL_DIR`` when
            omitted. ``None`` there too watches nothing.
        poll: Seconds between reads of the token.
        on_request: Called once the token changed; sends this process SIGTERM
            when omitted.

    Returns:
        The watching thread, or ``None`` when nothing is watched.
    """
    if directory is None:
        raw = os.getenv(POD_CONTROL_DIR_ENV, "").strip()
        if not raw:
            return None
        directory = Path(raw)
    stop = on_request or _terminate_self
    seen = read_token(directory)

    def run() -> None:
        stopped = threading.Event()
        while not stopped.wait(poll):
            token = read_token(directory)
            if token and token != seen:
                logger.warning(
                    "pod restart requested; stopping",
                    token=token,
                    directory=str(directory),
                )
                stop()
                return

    thread = threading.Thread(target=run, name="pod-control", daemon=True)
    thread.start()
    logger.info("watching for pod restarts", directory=str(directory), token=seen or None)
    return thread


def _terminate_self() -> None:
    os.kill(os.getpid(), signal.SIGTERM)
