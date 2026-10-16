"""A client that never closes: it runs in its own process and is killed mid-stream.

:class:`KilledClient` is the churn's side: it starts this module as a process,
waits for it to report that its frames came back, lets it stream on for a
moment, and kills it with ``SIGKILL``. The process gets no chance to close its
peer connection or its channels, so the runtime learns the client is gone only
from the wire going silent, the way it does when a client crashes or loses
its network.

Run as ``python -m endurance.abrupt <runtime url> <session id>``: it streams
as :class:`~endurance.harness.V1Client` does, prints one JSON line with the
frames received and the requests left unanswered, then streams until killed.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import IO

from endurance.harness import SUITE_DIR, RuntimeEndpoint, V1Client, gray_frame

# How long the process may take to start, connect and get its frames back
# before it is reported as never having streamed.
_READY_TIMEOUT_S = 90.0


class KilledClient:
    """A v1 client in its own process, killed with ``SIGKILL`` instead of closing.

    It has the shape of a :class:`~endurance.harness.BareClient` as the churn
    uses one: :meth:`run` returns the frames that came back, and
    :attr:`unanswered` lists the requests that were refused or never answered
    before the kill.
    """

    def __init__(
        self, runtime: RuntimeEndpoint, session_id: str, *, log: IO[str], linger_s: float
    ) -> None:
        """Create a client for *session_id* that streams *linger_s* past its burst.

        Args:
            runtime: The runtime to connect to.
            session_id: The session to join.
            log: Where the process's own output goes.
            linger_s: How long the process streams on after its frames came
                back, so the kill lands at different points of the stream.
        """
        self._runtime = runtime
        self._session_id = session_id
        self._log = log
        self._linger_s = linger_s
        self.unanswered: list[str] = []

    async def run(self, frame: tuple[bytes, int, int]) -> int:
        """Start the client process, kill it mid-stream, and return the frames it got back.

        The process streams its own frame of the same size: *frame* is taken
        for the churn's sake only.

        Raises:
            RuntimeError: The process exited before it reported its frames.
            TimeoutError: It did not report them within ``_READY_TIMEOUT_S``.
        """
        del frame
        self._log.flush()
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "endurance.abrupt",
            self._runtime.url,
            self._session_id,
            cwd=SUITE_DIR,
            stdout=asyncio.subprocess.PIPE,
            stderr=self._log,
        )
        try:
            assert proc.stdout is not None
            line = await asyncio.wait_for(proc.stdout.readline(), _READY_TIMEOUT_S)
            if not line:
                code = await proc.wait()
                raise RuntimeError(
                    f"the client process exited with {code} before its frames came back; "
                    f"see {Path(self._log.name).name}"
                )
            report = json.loads(line)
            await asyncio.sleep(self._linger_s)
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
        self.unanswered = report["unanswered"]
        return int(report["received"])


async def _main(url: str, session_id: str) -> None:
    client = V1Client(RuntimeEndpoint(url), session_id)

    def ready() -> None:
        report = {"received": client.received_video, "unanswered": client.unanswered}
        print(json.dumps(report), flush=True)

    await client.run_until_killed(gray_frame(), ready)


if __name__ == "__main__":
    asyncio.run(_main(sys.argv[1], sys.argv[2]))
