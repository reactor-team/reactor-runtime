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


class _Participant:
    """A client that streams on in the background, pinging, until it leaves.

    A connected client that goes quiet is dropped by the runtime's ping
    watchdog, so every participant keeps streaming while the others come and go.
    """

    def __init__(self, client: V1Client, frame: tuple[bytes, int, int]) -> None:
        self.client = client
        self._done = asyncio.Event()
        self._task = asyncio.create_task(client.stream_while(frame, self._done))

    async def leave(self) -> None:
        """Stop streaming, wait for the replies still owed, and close."""
        self._done.set()
        await self._task
        await self.client.wait_for_replies()
        self.client.close()

    async def abandon(self) -> None:
        """Stop streaming and close at once, for a cycle that failed."""
        self._done.set()
        await asyncio.gather(self._task, return_exceptions=True)
        self.client.close()


async def test_multi_connection(runtime: RuntimeProcess) -> None:
    """Release everything each connection of a shared session held, every time."""
    session = runtime.request("POST", "/start_session", {})[1]
    sid = session["session_id"]
    frame = gray_frame()

    async def cycle(n: int) -> CycleOutcome:
        publisher = V1Client(runtime, sid)
        viewers = [Viewer(runtime, sid) for _ in range(VIEWERS)]
        successor = V1Client(runtime, sid)
        joined: list[_Participant] = []
        missed: list[str] = []

        async def expect_registered(count: int, step: str) -> None:
            # Exactly: a participant the runtime dropped early is a failure too.
            # Waited in a thread, so the participants keep streaming meanwhile.
            reading = await asyncio.to_thread(runtime.wait_registered, count)
            if reading["registered_connections"] != count:
                missed.append(
                    f"{step}: {reading['registered_connections']} connections registered, "
                    f"{count} expected"
                )

        async def join(client: V1Client) -> int:
            received = await client.open(frame)
            joined.append(_Participant(client, frame))
            return received

        async def leave(client: V1Client) -> None:
            participant = next(p for p in joined if p.client is client)
            joined.remove(participant)
            await participant.leave()

        try:
            received = [await join(publisher)]
            # A task group: if one viewer fails to join, the others are
            # cancelled and awaited before the cleanup below runs, so none
            # can join after it and stream on into the next cycle.
            async with asyncio.TaskGroup() as group:
                joins = [group.create_task(join(viewer)) for viewer in viewers]
            received += [task.result() for task in joins]
            await expect_registered(1 + VIEWERS, "all joined")
            if n % 2:
                await leave(publisher)
                await expect_registered(VIEWERS, "publisher left")
                # The webcam passes to a new publisher while the viewers stay.
                received.append(await join(successor))
                await leave(successor)
                await expect_registered(VIEWERS, "successor left")
                for viewer in viewers:
                    await leave(viewer)
            else:
                for viewer in viewers:
                    await leave(viewer)
                await expect_registered(1, "viewers left")
                await leave(publisher)
        finally:
            for participant in joined:
                await participant.abandon()
            for client in (publisher, *viewers, successor):
                client.close()
        unanswered = [f"publisher: {miss}" for miss in publisher.collect_unanswered()]
        for i, viewer in enumerate(viewers):
            unanswered += [f"viewer {i}: {miss}" for miss in viewer.collect_unanswered()]
        if n % 2:
            unanswered += [f"successor: {miss}" for miss in successor.collect_unanswered()]
        return CycleOutcome(received=min(received), unanswered=[*unanswered, *missed])

    await run_cycles(runtime, test_name="multi-connection", description=DESCRIPTION, cycle=cycle)
