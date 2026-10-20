"""session_churn: sessions started and stopped over and over, a client in each."""

from __future__ import annotations

import urllib.error
from collections.abc import Mapping

from endurance.churn import CycleOutcome, Released, run_cycles
from endurance.harness import CLOSE_TIMEOUT_S, RuntimeProcess, V1Client, gray_frame

DESCRIPTION = (
    "Each cycle starts a session under a fresh id, a v1 client joins and streams a "
    "webcam both ways until 60 frames come back, and the session is stopped. The stop "
    "rotates: a plain stop, the client leaving first and the orphaned session stopped "
    "after, a stop with a close reason, and a moderated stop, each of the last two "
    "owing the client its notice before the connection closes. The runtime is read "
    "once the session is ready again."
)

# How each cycle ends, in turn.
STOPS = ("stop", "leave_then_stop", "stop_with_reason", "moderated_stop")
# The notification the client is owed before its connection closes, per stop.
_NOTICES = {"stop_with_reason": "session_ended", "moderated_stop": "moderation"}
_BODIES = {
    "stop_with_reason": {"reason": "endurance stop"},
    "moderated_stop": {"moderate": True},
}
# How long an owed notice may trail the connection's close on this side.
_NOTICE_GRACE_S = 2.0


async def test_session_churn(runtime: RuntimeProcess) -> None:
    """Release everything a session and its client held once it has stopped, every time."""
    frame = gray_frame()

    async def cycle(n: int) -> CycleOutcome:
        stop = STOPS[n % len(STOPS)]
        session = runtime.request("POST", "/start_session", {"session_id": f"endurance-{n}"})[1]
        client = V1Client(runtime, session["session_id"])
        failures: list[str] = []
        released: Released | None = None
        try:
            received = await client.open(frame)
            if stop == "leave_then_stop":
                client.close()
                # Stopped once the runtime has dropped the client, so the
                # session is orphaned rather than streaming.
                _, released_after_s = runtime.settled_probe()
                released = Released(released_after_s)
                state = runtime.get("/session")["state"]
                if state != "orphaned":
                    failures.append(f"{stop}: the session was {state}, not orphaned, at the stop")
                _stop(stop, {}, failures)
            else:
                if notice := _NOTICES.get(stop):
                    client.expect_notice(notice)
                if _stop(stop, _BODIES.get(stop, {}), failures):
                    if not await client.stream_until_dropped(frame, CLOSE_TIMEOUT_S):
                        failures.append(
                            f"{stop}: the connection was still open {CLOSE_TIMEOUT_S:g}s after it"
                        )
                    await client.wait_for_notices(_NOTICE_GRACE_S)
                elif notice:
                    # A stop that did not happen owed no notice: the refusal
                    # is counted once, under failed_stops.
                    client.forget_notice(notice)
        finally:
            client.close()
        try:
            runtime.wait_ready(timeout_s=CLOSE_TIMEOUT_S)
        except TimeoutError:
            failures.append(f"{stop}: the session was not ready {CLOSE_TIMEOUT_S:g}s after it")
        return CycleOutcome(
            received=received,
            unanswered=client.collect_unanswered(),
            stop_failures=failures,
            released=released,
        )

    def _stop(stop: str, body: Mapping[str, object], failures: list[str]) -> bool:
        """Stop the session; record a refusal under *stop* and return whether it stopped."""
        try:
            runtime.request("POST", "/stop_session", body)
        except urllib.error.HTTPError as error:
            failures.append(f"{stop}: /stop_session answered {error.code}")
            return False
        # A stop that hangs past the request's timeout, or a connection the
        # runtime refuses: URLError and TimeoutError are both OSErrors.
        except OSError as error:
            failures.append(f"{stop}: /stop_session did not answer: {error}")
            return False
        return True

    await run_cycles(
        runtime,
        test_name="session-churn",
        description=DESCRIPTION,
        cycle=cycle,
        min_smoke_cycles=len(STOPS),
    )
