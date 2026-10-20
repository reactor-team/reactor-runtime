"""The tracked echo, its engine run in a worker process by a DistributedRunner.

``EchoModel`` already has a worker's shape (``load()``, ``generate()``,
``reset()``), and a ``DistributedRunner`` answers the same three calls, so it
takes the engine's place unchanged. Each step's frame then travels to the
worker and back through the package's shared-memory slots, and each session's
end resets the worker.
"""

from __future__ import annotations

from pathlib import Path

from echo_model import EchoModel
from tracked_echo import TrackedEcho

from reactor_runtime.distributed import DistributedRunner


class DistributedEcho(TrackedEcho):
    """The tracked echo with ``EchoModel`` in one worker process."""

    def load(self, config_path: Path | None) -> None:
        """Load as echo does, then start the engine in its worker process instead."""
        super().load(config_path)
        # One rank on the CPU: no process group to form.
        runner = DistributedRunner(EchoModel, init_process_group=False)
        runner.start()
        self.engine = runner
