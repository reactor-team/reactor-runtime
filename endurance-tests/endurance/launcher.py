"""Boot the runtime the way ``reactor-runtime`` does, plus a probe route.

Run as a script with the model directory as the working directory. It wraps
``build_app`` so the served app also mounts ``GET /_endurance/probe``, checks
every media pacer as it stops, then hands over to ``serve.main()``. The
published package is not changed, and importing this module changes nothing:
the route and the check exist only in a runtime started from here.
"""

from __future__ import annotations

import asyncio
import gc
import sys
import threading
import types
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI

import reactor_runtime.http.server as http_server
from reactor_runtime import serve
from reactor_runtime.core import Health
from reactor_runtime.metrics import RuntimeMetrics
from reactor_runtime.runner import Runner
from reactor_runtime.transport.livekit import livekit_available
from reactor_runtime.transport.router import TransportRouter

_build_app = http_server.build_app

# Pacers that have been stopped, weakly: a probe re-checks the ones still alive.
_stopped_pacers: weakref.WeakSet[Any] = weakref.WeakSet()
# Stopped pacers found still reaching model output, since start.
_holding_after_stop = 0
# Whether the path of a hit has been logged: the first one per run is enough
# to show what holds the output.
_logged_a_path = False
# What the reachability walk never follows: code, frames and modules lead back
# to the model loop, which holds the newest output legitimately.
_OPAQUE = (
    types.FunctionType,
    types.MethodType,
    types.BuiltinFunctionType,
    types.FrameType,
    types.CoroutineType,
    types.GeneratorType,
    types.ModuleType,
    type,
    threading.Thread,
)


def path_to_output(root: object, batches: list[np.ndarray]) -> list[str] | None:
    """Return how *root*'s own data reaches one of *batches*, or None if it does not.

    Walks what *root* refers to, following an array view to the array it views,
    and never through code, frames, modules or threads. The path names each
    object's type on the way, with the attribute or key that led on from it
    where one does. *batches* is held for the walk, so an id cannot be freed
    and reused by another object meanwhile.
    """
    batch_ids = {id(batch) for batch in batches}
    if not batch_ids:
        return None
    # Each object reached, by id: the object it was reached from, and how.
    parents: dict[int, tuple[object, str]] = {}
    seen: set[int] = set()
    stack = [root]
    while stack:
        obj = stack.pop()
        if id(obj) in seen or isinstance(obj, _OPAQUE):
            continue
        seen.add(id(obj))
        if id(obj) in batch_ids:
            return _describe_path(obj, parents)
        if isinstance(obj, np.ndarray):
            children = [] if obj.base is None else [obj.base]
        else:
            children = gc.get_referents(obj)
        for child in children:
            if id(child) not in seen and id(child) not in parents:
                parents[id(child)] = (obj, _edge_name(obj, child))
            stack.append(child)
    return None


def reaches_output(root: object, batches: list[np.ndarray]) -> bool:
    """Return whether *root*'s own data reaches one of *batches*."""
    return path_to_output(root, batches) is not None


def _edge_name(parent: object, child: object) -> str:
    """Name the attribute or key of *parent* that holds *child*, or ''."""
    items: Any = None
    if isinstance(parent, dict):
        items = parent.items()
    elif isinstance(getattr(parent, "__dict__", None), dict):
        items = vars(parent).items()
    for key, value in items or ():
        if value is child:
            return f".{key}" if not isinstance(parent, dict) else f"[{key!r}]"
    return ""


def _describe_path(target: object, parents: dict[int, tuple[object, str]]) -> list[str]:
    steps = [type(target).__name__]
    obj = target
    while id(obj) in parents:
        parent, edge = parents[id(obj)]
        steps.append(f"{type(parent).__name__}{edge}")
        obj = parent
    return steps[::-1]


def _install_pacer_check() -> None:
    """Check every pacer for held model output the moment it stops."""
    from reactor_runtime.transport.webrtc.pacer import MediaPacer

    stop = MediaPacer.stop

    def checked_stop(self: MediaPacer) -> None:
        global _holding_after_stop, _logged_a_path
        stop(self)
        import tracked_echo

        _stopped_pacers.add(self)
        path = path_to_output(self, tracked_echo.tracked_batches())
        if path is not None:
            _holding_after_stop += 1
            if not _logged_a_path:
                _logged_a_path = True
                # A real leak shows its holder; a false positive shows which
                # exclusion the walk is missing.
                print(
                    "endurance: a stopped pacer reaches model output via " + " -> ".join(path),
                    file=sys.stderr,
                    flush=True,
                )

    MediaPacer.stop = checked_stop


def _live_transport_objects() -> dict[str, int]:
    from reactor_runtime.transport.webrtc.connection import WebRTCConnection
    from reactor_runtime.transport.webrtc.pacer import MediaPacer
    from reactor_runtime.transport.webrtc.peer import WebRTCPeer

    participant: type | None = None
    if livekit_available():
        from reactor_runtime.transport.livekit.participant import ParticipantConnection

        participant = ParticipantConnection

    counts = {"connections": 0, "peers": 0, "pacers": 0, "participants": 0}
    for obj in gc.get_objects():
        if isinstance(obj, WebRTCConnection):
            counts["connections"] += 1
        elif isinstance(obj, WebRTCPeer):
            counts["peers"] += 1
        elif isinstance(obj, MediaPacer):
            counts["pacers"] += 1
        elif participant is not None and isinstance(obj, participant):
            counts["participants"] += 1
    return counts


def build_app(
    runner: Runner,
    transports: list[TransportRouter],
    process_health: Callable[[], Health],
    metrics: RuntimeMetrics,
) -> FastAPI:
    """Build the runtime's app, with the probe route added."""
    app = _build_app(runner, transports, process_health, metrics)

    @app.get("/_endurance/probe")
    async def probe() -> dict[str, Any]:
        import tracked_echo

        batches = tracked_echo.tracked_batches()
        stopped_holding = sum(
            1 for pacer in list(_stopped_pacers) if reaches_output(pacer, batches)
        )
        # Released before collecting: a batch this list held would be freed by
        # its refcount afterwards, not by the collection, and go uncounted.
        del batches
        # Collect, so garbage from this cycle is counted in this cycle's
        # reading rather than whenever the next collection runs.
        gc.collect()
        return {
            "registered_connections": runner._connections.count,
            "client_stats_connections": len(runner._client_stats._accepted_at),
            "batches_freed_by_gc": tracked_echo.freed_by_gc(),
            "pacers_holding_after_stop": _holding_after_stop,
            "stopped_pacers_holding_now": stopped_holding,
            "live_batches": tracked_echo.alive_batches(),
            "live": _live_transport_objects(),
            "tasks": len(asyncio.all_tasks()),
        }

    return app


def main() -> None:
    """Install the probe and the pacer check, then run the runtime."""
    # examples/echo, which the tracked model subclasses.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "echo"))
    # Swapping in the wrapper is the point; ty refuses any assignment to a module function.
    http_server.build_app = build_app  # ty: ignore[invalid-assignment]
    _install_pacer_check()
    serve.main()


if __name__ == "__main__":
    main()
