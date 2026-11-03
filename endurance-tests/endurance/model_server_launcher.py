"""Run the model server with a probe for requests held by its workers."""

from __future__ import annotations

import gc
from typing import Any

from reactor_runtime.http import HealthServer
from reactor_runtime.model_server import ModelServerCore
from reactor_runtime.model_server import __main__ as entrypoint


def pending_requests() -> int:
    """Count requests still held by model-server workers after collection."""
    gc.collect()
    count = 0
    for obj in gc.get_objects():
        if isinstance(obj, ModelServerCore):
            with obj._cond:
                count += len(obj._pending)
    return count


class ProbedHealthServer(HealthServer):
    """Expose the worker's retained request count beside its health route."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        @self._app.get("/_endurance/model_server")
        async def probe() -> dict[str, int]:
            return {"pending_requests": pending_requests()}


def main() -> int:
    """Install the model-server probe and run its public entrypoint."""
    vars(entrypoint)["HealthServer"] = ProbedHealthServer
    return entrypoint.main()


if __name__ == "__main__":
    raise SystemExit(main())
