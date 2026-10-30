"""Run a split model's model half: ``python -m reactor_runtime.model_server``.

Run it in the model's workspace, beside the session runtimes' ``python -m
reactor_runtime.serve``. The manifest names the model half (``runtime.worker``)
and its serving settings; the deployment names the rest through the
environment (:func:`~reactor_runtime.model_server.config.server_process_config`).
The process is the runtime's service with two components, the model server and
its ``GET /health``, so it starts, drains, stops and reports health as a session
runtime does.
"""

from __future__ import annotations

import asyncio
import sys

from reactor_runtime import log
from reactor_runtime.core import RuntimeState
from reactor_runtime.environment import log_level_from_env
from reactor_runtime.http import HealthServer
from reactor_runtime.manifest import load_workspace
from reactor_runtime.model_server.config import server_process_config
from reactor_runtime.model_server.process import ModelServerProcess
from reactor_runtime.service import Service


def main() -> int:
    """Load the model half and serve it until the process is asked to stop; return the exit status.

    Raises:
        SystemExit: No ``reactor.yaml``, no ``runtime.worker`` block, a required
            variable unset, or a malformed value, naming it.
    """
    log.configure(level=log_level_from_env())
    log.set_process(role="model_server")
    log.set_state(None, RuntimeState.LOADING.value)
    cfg = load_workspace()
    if cfg.worker is None:
        raise SystemExit("reactor.yaml has no runtime.worker block naming the model half")
    process = server_process_config()
    model_server = ModelServerProcess(cfg.worker, cfg.config_path, process)
    service = Service(exit_timeout=process.exit_timeout, grace_period=process.grace_period)
    model_server.request_shutdown = service.request_shutdown
    service.add(model_server)
    service.add(
        HealthServer(
            host=process.host,
            port=process.health_port,
            process_health=service.health,
            state=model_server.state,
            depends_on=(model_server.name,),
        )
    )
    asyncio.run(service.run())
    service.bound_process_exit()
    return service.exit_code


if __name__ == "__main__":
    sys.exit(main())
