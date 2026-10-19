"""A client that never closes: it runs in its own process and is killed mid-stream.

:class:`KilledClient` is the churn's side: it starts this module as a process,
waits for it to report that its frames came back, lets it stream on for a
moment, and kills it with ``SIGKILL``. The process gets no chance to close its
peer connection or its channels, so the runtime learns the client is gone only
from the wire going silent, the way it does when a client crashes or loses
its network.

Run as ``python -m endurance.abrupt <runtime url> <session id>``: it streams
as :class:`~endurance.harness.V1Client` does, prints one ``REPORT`` line of
JSON with the frames received and the requests left unanswered, then streams
until killed, printing a ``FAILED`` line for each request it finds refused or
answered wrongly meanwhile.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import IO, Any

import psutil

from endurance.harness import SUITE_DIR, RuntimeEndpoint, V1Client, gray_frame

# How long the process may take to start, connect and get its frames back
# before it is reported as never having streamed.
_READY_TIMEOUT_S = 90.0
# How much of a failed process's output its error carries.
_OUTPUT_LINES = 60
# Mark the process's report line on its stdout, and each request it found
# refused or answered wrongly after the report, up to the kill.
_REPORT_PREFIX = "REPORT "
_FAILED_PREFIX = "FAILED "


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
        self._log_start = 0
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
        # Where this process's own output starts in the log the cycles share.
        self._log_start = self._log.tell()
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
            report = await asyncio.wait_for(self._read_report(proc), _READY_TIMEOUT_S)
            await asyncio.sleep(self._linger_s)
            # A client that died on its own was not killed: the cycle would not
            # have exercised the path it exists for.
            if _exited(proc):
                raise RuntimeError(
                    f"the client process exited on its own before it was killed"
                    f"{self._client_output()}"
                )
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
        # What the process found wrong while it streamed on, up to the kill,
        # is still in the pipe.
        rest = await proc.stdout.read()
        later = [
            json.loads(text[len(_FAILED_PREFIX) :])
            for line in rest.decode(errors="replace").splitlines()
            if (text := line.strip()).startswith(_FAILED_PREFIX)
        ]
        self.unanswered = [*report["unanswered"], *later]
        return int(report["received"])

    def _client_output(self) -> str:
        """Return the end of what this cycle's process wrote, its traceback included.

        A failing cycle carries it in its error, so the failure reads in full
        where the run ran, the PR smoke included, without the log file.
        """
        self._log.flush()
        try:
            with open(self._log.name, errors="replace") as log:
                log.seek(self._log_start)
                lines = log.read().splitlines()[-_OUTPUT_LINES:]
        except OSError:
            lines = []
        name = Path(self._log.name).name
        if not lines:
            return f"; it wrote nothing to {name}"
        return f"; the end of what it wrote to {name}:\n" + "\n".join(lines)

    async def _read_report(self, proc: asyncio.subprocess.Process) -> dict[str, Any]:
        """Return the report the process prints once its frames came back.

        Other lines on its stdout, from anything it imports, are skipped.
        """
        assert proc.stdout is not None
        while line := await proc.stdout.readline():
            text = line.decode(errors="replace").strip()
            if not text.startswith(_REPORT_PREFIX):
                continue
            try:
                return json.loads(text[len(_REPORT_PREFIX) :])
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"the client process printed an unreadable report{self._client_output()}"
                ) from error
        code = await proc.wait()
        raise RuntimeError(
            f"the client process exited with {code} before its frames came back"
            f"{self._client_output()}"
        )


def _exited(proc: asyncio.subprocess.Process) -> bool:
    """Whether *proc* has exited, reaped or not."""
    if proc.returncode is not None:
        return True
    try:
        return psutil.Process(proc.pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


async def _main(url: str, session_id: str) -> None:
    client = V1Client(RuntimeEndpoint(url), session_id)

    reported = 0

    def ready() -> None:
        nonlocal reported
        reported = len(client.failures())
        report = {"received": client.received_video, "unanswered": client.unanswered}
        print(_REPORT_PREFIX + json.dumps(report), flush=True)

    def tick() -> None:
        nonlocal reported
        failures = client.failures()
        for failure in failures[reported:]:
            print(_FAILED_PREFIX + json.dumps(failure), flush=True)
        reported = len(failures)

    await client.run_until_killed(gray_frame(), ready, tick)


if __name__ == "__main__":
    asyncio.run(_main(sys.argv[1], sys.argv[2]))
