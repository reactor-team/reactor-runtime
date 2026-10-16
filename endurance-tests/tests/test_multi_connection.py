"""multi_connection: one session, a publisher and viewers joining and leaving together."""

from __future__ import annotations

import asyncio

from endurance.churn import CycleOutcome, run_cycles
from endurance.harness import RuntimeProcess, V1Client, Viewer, gray_frame

DESCRIPTION = (
    "One long-lived session. Each cycle a v1 client publishes a webcam until 60 "
    "frames come back, and while it streams on, three v1 viewers that publish nothing "
    "join and each wait for 60 frames of the same output. Then they leave in two "
    "steps, in turn the viewers first or the publisher first. When the publisher "
    "leaves first, a new one takes over the webcam while the viewers stay, and must "
    "be granted it and get its 60 frames back. Each step waits for the runtime to drop "
    "the clients that left; the runtime is read once every connection is gone and "
    "teardown has finished."
)

VIEWERS = 3


async def test_multi_connection(runtime: RuntimeProcess) -> None:
    """Release everything each connection of a shared session held, every time."""
    session = runtime.request("POST", "/start_session", {})[1]
    sid = session["session_id"]
    frame = gray_frame()

    async def cycle(n: int) -> CycleOutcome:
        publisher = V1Client(runtime, sid)
        viewers = [Viewer(runtime, sid) for _ in range(VIEWERS)]
        successor = V1Client(runtime, sid)
        clients = [publisher, *viewers]
        try:
            received = [await publisher.open(frame)]
            done = asyncio.Event()
            streaming = asyncio.create_task(publisher.stream_while(frame, done))
            try:
                received += await asyncio.gather(*(viewer.open(frame) for viewer in viewers))
            finally:
                done.set()
                await streaming
            # Streaming on sent requests after the settle in open().
            await publisher.wait_for_replies()
            if n % 2:
                publisher.close()
                runtime.wait_registered(VIEWERS)
                # The webcam passes to a new publisher while the viewers stay.
                clients.append(successor)
                received.append(await successor.open(frame))
                successor.close()
                runtime.wait_registered(VIEWERS)
                for viewer in viewers:
                    viewer.close()
            else:
                for viewer in viewers:
                    viewer.close()
                runtime.wait_registered(1)
        finally:
            for client in clients:
                client.close()
        unanswered = [f"publisher: {miss}" for miss in publisher.collect_unanswered()]
        for i, viewer in enumerate(viewers):
            unanswered += [f"viewer {i}: {miss}" for miss in viewer.collect_unanswered()]
        if successor in clients:
            unanswered += [f"successor: {miss}" for miss in successor.collect_unanswered()]
        return CycleOutcome(received=min(received), unanswered=unanswered)

    await run_cycles(runtime, test_name="multi-connection", description=DESCRIPTION, cycle=cycle)
