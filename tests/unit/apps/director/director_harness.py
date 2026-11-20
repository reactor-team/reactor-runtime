"""A director app driven on the current event loop, with every outbound sink recorded."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from reactor_runtime.apps.director import ReactorDirectorApp
from reactor_runtime.core.model import ClientConnected, SessionStarted, StartingInputApplied
from reactor_runtime.core.values import (
    CommandFailure,
    CompletedStep,
    ConnId,
    MediaChunk,
    StepRefusal,
)
from reactor_runtime.interface.events.messages import ModelMessage
from reactor_runtime.interface.internal.reactor_core import CommandEnvelope, MediaOps, StepEnvelope
from reactor_runtime.interface.model.contract import ModelContract

CLIENT = ConnId(1002)
SYSTEM = ConnId(0)


class Harness:
    """A director app with its sinks recorded, its inbound queue drained, and its loop running."""

    def __init__(self, app: ReactorDirectorApp) -> None:
        self.app = app
        self.broadcasts: list[ModelMessage] = []
        self.addressed: list[tuple[Any, ...]] = []
        self.chunks: list[MediaChunk] = []
        self.reports: list[CompletedStep] = []
        self.refusals: list[StepRefusal] = []
        self.flushes: list[bool] = []
        app._on_loop_ready()
        app.bind_output(
            broadcast=self.broadcasts.append,
            addressed=lambda *args: self.addressed.append(args),
            media=self.chunks.append,
            media_ops=MediaOps(
                flush=self.flushes.append, set_rate=lambda rate: None, set_depth=lambda depth: None
            ),
            step=self.reports.append,
        )
        app._bind_step_refusal(self.refusals.append)
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        app._inbound_q = self.queue
        self.tasks: list[asyncio.Task[None]] = []
        self._requests = 0

    async def start(self, *, stepped: bool = False, starting_input: bool = False) -> None:
        """Start a session; a starting input waits for :meth:`apply_starting_input`."""
        self.tasks = [
            asyncio.create_task(self.app._inbound_loop()),
            asyncio.create_task(self.app.run()),
        ]
        self.post(SessionStarted("s", stepped=stepped, starting_input=starting_input))
        if starting_input:
            self.post(ClientConnected(SYSTEM, 1, system=True))
        else:
            self.post(ClientConnected(CLIENT, 1))
        await self.settle()

    def apply_starting_input(self) -> None:
        self.post(StartingInputApplied())

    def post(self, item: Any) -> None:
        self.queue.put_nowait(item)

    def command(self, name: str, **args: Any) -> str:
        """Send a command as the client would; return the request id its reply carries."""
        self._requests += 1
        request_id = f"c{self._requests}"
        command = ModelContract.of(type(self.app)).validate(name, args)
        self.post(CommandEnvelope(command, CLIENT, request_id))
        return request_id

    def step(self, request_id: str, *commands: tuple[str, dict[str, Any]]) -> None:
        """Request one step; the runtime posts the request's commands ahead of it, tagged."""
        contract = ModelContract.of(type(self.app))
        for index, (name, args) in enumerate(commands):
            self.post(
                CommandEnvelope(
                    contract.validate(name, args),
                    CLIENT,
                    None,
                    step_request_id=request_id,
                    step_entry=f"commands[{index}]",
                )
            )
        self.post(StepEnvelope(media_input={}, conn_id=CLIENT, request_id=request_id))

    def reply(self, request_id: str) -> Any:
        """The reply to one command: a message, a failure, or ``None`` for an acknowledgement."""
        for _, message, reply_to in self.addressed:
            if reply_to == request_id:
                return message
        raise AssertionError(f"no reply to {request_id}")

    def failure(self, request_id: str) -> str:
        reply = self.reply(request_id)
        assert isinstance(reply, CommandFailure), reply
        return reply.code

    def sent(self, name: str) -> list[dict[str, Any]]:
        """The data of every broadcast message called *name*, in order."""
        return [m.to_wire_format()["data"] for m in self.broadcasts if m.name == name]

    def state(self) -> dict[str, Any]:
        return self.sent("director_state")[-1]

    def shot_id(self, request_id: str) -> str:
        reply = self.reply(request_id)
        return reply.to_wire_format()["data"]["shot"]["shot_id"]

    async def settle(self) -> None:
        for _ in range(40):
            await asyncio.sleep(0.005)
            if self.queue.empty() and self.app._pending_step is None:
                return

    async def until(self, predicate: Callable[[], bool], within: float = 5.0) -> None:
        deadline = asyncio.get_running_loop().time() + within
        while not predicate():
            if asyncio.get_running_loop().time() > deadline:
                raise AssertionError("timed out waiting for the condition")
            await asyncio.sleep(0.005)

    async def stop(self) -> None:
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)


def track(output: Any, name: str) -> Any:
    """One track's payload of an output, a step bundle, or a command's field."""
    return getattr(output, name)


def encode_mp4(
    path: Path,
    frames: npt.NDArray[np.uint8],
    fps: float,
    audio: npt.NDArray[np.int16] | None = None,
    rate: int = 48_000,
) -> Path:
    """Write *frames*, and optional mono ``int16`` *audio*, to an MP4 at *path*."""
    import av

    with av.open(str(path), "w", format="mp4") as container:
        video = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000))
        video.width = frames.shape[2]
        video.height = frames.shape[1]
        video.pix_fmt = "yuv420p"
        sound = container.add_stream("aac", rate=rate) if audio is not None else None
        for frame in frames:
            for packet in video.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in video.encode(None):
            container.mux(packet)
        if sound is not None and audio is not None:
            chunk = av.AudioFrame.from_ndarray(audio.reshape(1, -1), format="s16", layout="mono")
            chunk.sample_rate = rate
            for packet in sound.encode(chunk):
                container.mux(packet)
            for packet in sound.encode(None):
                container.mux(packet)
    return path
