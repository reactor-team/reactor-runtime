"""Boot the runtime the way ``reactor-runtime`` does, plus a probe route.

Run with the model directory as the working directory. It wraps
``build_app`` so the served app also mounts ``GET /_endurance/probe``, then
hands over to ``serve.main()``. The published package is not changed: the
route exists only in a runtime started from here.
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
from reactor_runtime.transport.router import TransportRouter

# examples/echo, which the tracked model subclasses.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "echo"))

_build_app = http_server.build_app

# Pacers that have been stopped, weakly: a probe re-checks the ones still alive.
_stopped_pacers: weakref.WeakSet[Any] = weakref.WeakSet()
# Stopped pacers found still reaching model output, since start.
_holding_after_stop = 0
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


def _reaches_output(root: object, batch_ids: set[int]) -> bool:
    """Whether *root*'s own data reaches one of the model's output batches.

    Walks what *root* refers to, following an array view to the array it views,
    and never through code, frames, modules or threads.
    """
    if not batch_ids:
        return False
    seen: set[int] = set()
    stack = [root]
    while stack:
        obj = stack.pop()
        if id(obj) in seen or isinstance(obj, _OPAQUE):
            continue
        seen.add(id(obj))
        if id(obj) in batch_ids:
            return True
        if isinstance(obj, np.ndarray):
            if obj.base is not None:
                stack.append(obj.base)
            continue
        stack.extend(gc.get_referents(obj))
    return False


def _install_pacer_check() -> None:
    """Check every pacer for held model output the moment it stops."""
    from reactor_runtime.transport.webrtc.pacer import MediaPacer

    stop = MediaPacer.stop

    def checked_stop(self: MediaPacer) -> None:
        global _holding_after_stop
        stop(self)
        import tracked_echo

        _stopped_pacers.add(self)
        if _reaches_output(self, tracked_echo.tracked_ids()):
            _holding_after_stop += 1

    MediaPacer.stop = checked_stop


def _live_transport_objects() -> dict[str, int]:
    from reactor_runtime.transport.webrtc.connection import WebRTCConnection
    from reactor_runtime.transport.webrtc.pacer import MediaPacer
    from reactor_runtime.transport.webrtc.peer import WebRTCPeer

    counts = {"connections": 0, "peers": 0, "pacers": 0}
    for obj in gc.get_objects():
        if isinstance(obj, WebRTCConnection):
            counts["connections"] += 1
        elif isinstance(obj, WebRTCPeer):
            counts["peers"] += 1
        elif isinstance(obj, MediaPacer):
            counts["pacers"] += 1
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

        batch_ids = tracked_echo.tracked_ids()
        stopped_holding = sum(
            1 for pacer in list(_stopped_pacers) if _reaches_output(pacer, batch_ids)
        )
        # Collect, so garbage from this cycle is counted in this cycle's
        # reading rather than whenever the next collection runs.
        gc.collect()
        return {
            "registered_connections": runner._connections.count,
            "batches_freed_by_gc": tracked_echo.freed_by_gc(),
            "pacers_holding_after_stop": _holding_after_stop,
            "stopped_pacers_holding_now": stopped_holding,
            "live_batches": tracked_echo.alive_batches(),
            "live": _live_transport_objects(),
            "tasks": len(asyncio.all_tasks()),
        }

    return app


# Swapping in the wrapper is the point; ty refuses any assignment to a module function.
http_server.build_app = build_app  # ty: ignore[invalid-assignment]
_install_pacer_check()

if __name__ == "__main__":
    serve.main()
