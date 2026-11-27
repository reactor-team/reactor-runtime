"""robot_observe_churn: the v1 churn on a robot model, the client sending stamped state."""

from __future__ import annotations

import asyncio
import importlib.util
import time
from collections.abc import Iterator
from typing import Any

import pytest

# A runtime from before the robot app (runtime_ref) has nothing for this
# scenario to test. Only a missing package skips: one that fails to import is
# a broken runtime, and fails here.
if importlib.util.find_spec("reactor_runtime.apps.robot") is None:
    pytest.skip("this runtime has no robot app", allow_module_level=True)
from endurance.churn import run_churn
from endurance.harness import _FPS, SUITE_DIR, RuntimeEndpoint, RuntimeProcess, V1Client
from endurance.report import DEFAULT_OUTPUT_DIR
from google.protobuf import struct_pb2

from reactor_runtime.apps.robot.observe import INVALID_OBSERVE
from reactor_wire.v1 import common_pb2, data_pb2, model_pb2

DESCRIPTION = (
    "One long-lived session on the tracked echo built on ReactorRobotApp. Each cycle a "
    "v1 client joins and streams until 60 frames come back, as in connection_churn_v1, "
    "and with every frame it sends an observe message carrying the frame's capture stamp "
    "and a robot state, more than the model's observe buffer holds, so the buffer fills "
    "and evicts. The model echoes a frame only once its state has joined it by stamp, so "
    "every frame back went through the robot app's camera reader and synchronizer. Every "
    "tenth frame the client also sends a malformed observe, which must be refused with "
    "invalid_observe. Once the client has left, the model must hold no stamped state "
    "and no input waiting to be joined (live_observed_states)."
)

# One malformed observe per this many frames.
_REFUSAL_EVERY = 10
_REQUEST = common_pb2.MessageKind.MESSAGE_KIND_REQUEST


class ObserveClient(V1Client):
    """A :class:`V1Client` that also sends its robot state with every frame, both stamped alike.

    A malformed observe it sends must come back refused with
    ``invalid_observe``; any other answer is listed in :attr:`unanswered`.
    """

    def __init__(self, runtime: RuntimeEndpoint, session_id: str) -> None:
        """Create a client for *session_id*; nothing connects until :meth:`run`."""
        super().__init__(runtime, session_id)
        self._frames = 0
        # The capture stamp of the frame about to be pushed and of its state.
        self._stamp = 0
        # Request ids of the malformed observes, owed a refusal.
        self._refusals: set[str] = set()

    def _on_tick(self) -> None:
        super()._on_tick()
        n = self._frames
        self._frames += 1
        self._stamp = time.monotonic_ns() // 1000
        self._send_observe(
            f"observe-{n}",
            {"capture_time_us": self._stamp, "state": {"joints": [n % 7, 0.5]}},
        )
        if n % _REFUSAL_EVERY == 0:
            request_id = f"observe-refused-{n}"
            with self._lock:
                self._pending[request_id] = ("observe", frozenset())
                self._refusals.add(request_id)
            self._send_observe(request_id, {"capture_time_us": "late", "state": [n]})

    async def _push_frame(self, frame: tuple[bytes, int, int]) -> None:
        """Push a webcam frame stamped with the capture stamp its observe carries."""
        if not self._stamp:
            await super()._push_frame(frame)
            return
        bgra, width, height = frame
        # The stamp rides in the frame's trailer, which this binding appends
        # only to a frame that carries user data, so the frame carries some.
        self._camera.push_video_frame(
            bgra, width, height, user_data=b"{}", capture_time_us=self._stamp
        )
        await asyncio.sleep(1 / _FPS)

    def _send_observe(self, request_id: str, data: dict[str, Any]) -> None:
        args = struct_pb2.Struct()
        args.update(data)
        self._data.send(
            data_pb2.DataClientMessage(
                request_id=request_id,
                kind=_REQUEST,
                command=model_pb2.Command(type="observe", data=args),
            ).SerializeToString(),
            binary=True,
        )

    def _answered(self, request_id: str, payload: str | None, message: Any) -> None:
        with self._lock:
            owed_refusal = request_id in self._refusals
            self._refusals.discard(request_id)
            if owed_refusal and self._pending.pop(request_id, None) is not None:
                code = message.error.code if payload == "error" else None
                if code != INVALID_OBSERVE:
                    self._failed.append(
                        f"observe ({request_id}): answered with {code or payload or 'nothing'}, "
                        f"not {INVALID_OBSERVE}"
                    )
                return
        super()._answered(request_id, payload, message)


@pytest.fixture
def runtime(request: pytest.FixtureRequest) -> Iterator[RuntimeProcess]:
    """A runtime serving the robot model, its log kept beside the reports."""
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    proc = RuntimeProcess(
        DEFAULT_OUTPUT_DIR / f"{request.node.name}-runtime.log",
        model_dir=SUITE_DIR / "model-robot",
    )
    try:
        proc.wait_ready()
        yield proc
    finally:
        proc.stop()


async def test_robot_observe_churn(runtime: RuntimeProcess) -> None:
    """Drop every stamped state a robot client sent once it has left, every time."""
    await run_churn(
        runtime, test_name="robot-observe-churn", description=DESCRIPTION, client=ObserveClient
    )
