import asyncio
import time

import numpy as np
import pytest

from reactor_runtime import Input, ModelMessage, Output, Video
from reactor_runtime.core import Command, SessionStarted
from reactor_runtime.core.values import ConnId, InputFrame, TrackDirection
from reactor_runtime.interface.internal.reactor_core import ReactorCore


class Out(Output):
    main: Video


class In(Input):
    camera: Video


class IdleCore(ReactorCore):
    output: Out
    input: In

    async def run(self) -> None:
        await asyncio.sleep(60)


class OutputOnlyCore(ReactorCore):
    output: Out

    async def run(self) -> None:
        await asyncio.sleep(60)


class Go(Command):
    pass


class Ping(ModelMessage):
    note: str


def frame(value: int = 0) -> InputFrame:
    return InputFrame(data=np.full((2, 2, 3), value, dtype=np.uint8), pts=float(value))


def test_input_buffers_are_wired_from_annotations() -> None:
    core = IdleCore()
    assert set(core._input_buffers) == {"camera"}
    assert vars(core.input)["camera"] is core._input_buffers["camera"]


def test_output_topology_is_read_from_annotations() -> None:
    core = IdleCore()
    assert set(core.output_buffer._video_tracks) == {"main"}
    assert core.output_buffer.fps == 30


def test_a_model_without_input_has_no_buffers() -> None:
    core = OutputOnlyCore()
    assert core._input_buffers == {}


def test_to_bundle_converts_a_typed_output() -> None:
    core = OutputOnlyCore()
    data = np.zeros((4, 4, 3), dtype=np.uint8)
    bundle = core._to_bundle(Out(main=data))
    assert set(bundle.tracks) == {"main"}
    assert bundle.tracks["main"].data is data
    assert bundle.tracks["main"].info.direction is TrackDirection.OUT


def test_emit_adapts_fps_to_measured_throughput() -> None:
    core = OutputOnlyCore()
    data = np.zeros((4, 4, 3), dtype=np.uint8)
    # drop=True enqueues without the running emission thread; one frame in 0.1s
    # is an effective 10 fps.
    asyncio.run(core.emit(Out(main=data), compute_time=0.1, drop=True))
    assert core.output_buffer.fps == 10


def test_emit_without_compute_time_leaves_fps_unchanged() -> None:
    core = OutputOnlyCore()
    asyncio.run(core.emit(Out(main=np.zeros((4, 4, 3), dtype=np.uint8)), drop=True))
    assert core.output_buffer.fps == 30


def test_emit_leaves_fps_unchanged_when_nothing_is_enqueued() -> None:
    core = OutputOnlyCore()
    # drop=False with emission stopped enqueues nothing, so no rate to adapt to.
    asyncio.run(core.emit(Out(main=np.zeros((4, 4, 3), dtype=np.uint8)), compute_time=0.1))
    assert core.output_buffer.fps == 30


def test_push_media_routes_to_the_track_buffer() -> None:
    core = IdleCore()
    core.push_media("camera", frame(1))
    assert core._input_buffers["camera"].available == 1


def test_push_media_for_an_unknown_track_is_a_noop() -> None:
    core = IdleCore()
    core.push_media("nope", frame(1))
    assert core._input_buffers["camera"].available == 0


def test_run_is_not_implemented_by_the_core() -> None:
    class Bare(ReactorCore):
        pass

    with pytest.raises(NotImplementedError):
        asyncio.run(Bare().run())


def test_send_routes_to_the_bound_broadcast_sink() -> None:
    core = OutputOnlyCore()
    sent: list[ModelMessage] = []
    core.bind_output(broadcast=sent.append, addressed=lambda conn, msg, req: None)
    asyncio.run(core.send(Ping(note="hi")))
    assert sent == [Ping(note="hi")]


def test_ingress_lands_on_the_two_typed_queues() -> None:
    core = IdleCore()
    core.start_thread()
    try:
        time.sleep(0.1)  # let the loop bootstrap its queues
        core.submit_command(Go(), ConnId(1), "req-1")
        core.post_reactor_event(SessionStarted(session_id="s-1"))
        time.sleep(0.1)
        assert core._command_q is not None
        assert core._reactor_q is not None
        assert core._command_q.qsize() == 1
        assert core._reactor_q.qsize() == 1
    finally:
        core.stop()
        time.sleep(0.1)
