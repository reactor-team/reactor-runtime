import asyncio
import time
from collections.abc import Callable

import numpy as np
import pytest

from reactor_runtime import Input, ModelMessage, Output, Video
from reactor_runtime.core import Command, MediaChunk, SessionStarted
from reactor_runtime.core.values import ConnId, InputFrame, TrackDirection
from reactor_runtime.interface.internal.reactor_core import MediaOps, ReactorCore


class Out(Output):
    main: Video


class In(Input):
    camera: Video


class IdleCore(ReactorCore):
    input: In

    async def run(self) -> None:
        await asyncio.sleep(60)


class OutputOnlyCore(ReactorCore):
    async def run(self) -> None:
        await asyncio.sleep(60)


class Go(Command):
    pass


class Ping(ModelMessage):
    note: str


@pytest.fixture(autouse=True)
def _seed_registries(isolate_interface_registries: None, register: Callable[..., None]) -> None:
    register(Out, In)


def frame(value: int = 0) -> InputFrame:
    return InputFrame(data=np.full((2, 2, 3), value, dtype=np.uint8), pts=float(value))


def _capture_media(core: ReactorCore) -> list[MediaChunk]:
    """Bind a media sink that records every emitted chunk."""
    chunks: list[MediaChunk] = []
    core.bind_output(
        broadcast=lambda msg: None,
        addressed=lambda conn, msg, req: None,
        media=chunks.append,
    )
    return chunks


def test_input_buffers_are_wired_from_annotations() -> None:
    core = IdleCore()
    assert set(core._input_buffers) == {"camera"}
    assert vars(core.input)["camera"] is core._input_buffers["camera"]


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


def test_emit_tags_the_chunk_with_measured_throughput() -> None:
    core = OutputOnlyCore()
    chunks = _capture_media(core)
    data = np.zeros((4, 4, 3), dtype=np.uint8)
    # One frame in 0.1s is an effective 10 fps.
    asyncio.run(core.emit(Out(main=data), compute_time=0.1))
    assert len(chunks) == 1
    assert chunks[0].fps == 10
    assert chunks[0].n_frames == 1


def test_emit_without_compute_time_uses_the_declared_fps() -> None:
    core = OutputOnlyCore()
    chunks = _capture_media(core)
    asyncio.run(core.emit(Out(main=np.zeros((4, 4, 3), dtype=np.uint8))))
    assert chunks[0].fps == 30


def test_emit_tags_a_batch_with_its_frame_count() -> None:
    core = OutputOnlyCore()
    chunks = _capture_media(core)
    asyncio.run(core.emit(Out(main=np.zeros((4, 8, 8, 3), dtype=np.uint8)), compute_time=0.2))
    # Four frames in 0.2s is 20 fps.
    assert chunks[0].n_frames == 4
    assert chunks[0].fps == 20


def _capture_ops(core: ReactorCore) -> tuple[list[MediaChunk], list[tuple[str, float | int]]]:
    """Bind a media sink and ops that record every emitted chunk and call."""
    chunks: list[MediaChunk] = []
    calls: list[tuple[str, float | int]] = []
    core.bind_output(
        broadcast=lambda msg: None,
        addressed=lambda conn, msg, req: None,
        media=chunks.append,
        media_ops=MediaOps(
            flush=lambda: calls.append(("flush", 0)),
            set_rate=lambda fps: calls.append(("rate", fps)),
            set_depth=lambda depth: calls.append(("depth", depth)),
        ),
    )
    return chunks, calls


def test_emit_requests_backpressure_by_default() -> None:
    core = OutputOnlyCore()
    chunks = _capture_media(core)
    asyncio.run(core.emit(Out(main=np.zeros((4, 4, 3), dtype=np.uint8))))
    assert chunks[0].wait is True


def test_emit_with_drop_requests_dropping_downstream() -> None:
    core = OutputOnlyCore()
    chunks = _capture_media(core)
    asyncio.run(core.emit(Out(main=np.zeros((4, 4, 3), dtype=np.uint8)), drop=True))
    assert chunks[0].wait is False


def test_emit_is_an_alias_of_the_output_handle() -> None:
    core = OutputOnlyCore()
    chunks = _capture_media(core)
    data = np.zeros((4, 4, 3), dtype=np.uint8)
    asyncio.run(core.emit(Out(main=data), compute_time=0.1))
    asyncio.run(core.output.emit(Out(main=data), compute_time=0.1))
    assert len(chunks) == 2
    assert (chunks[0].fps, chunks[0].n_frames, chunks[0].wait) == (
        chunks[1].fps,
        chunks[1].n_frames,
        chunks[1].wait,
    )


def test_output_fps_reads_the_declared_rate() -> None:
    assert OutputOnlyCore().output.fps == 30.0


def test_assigning_output_fps_repaces_and_retags() -> None:
    core = OutputOnlyCore()
    chunks, calls = _capture_ops(core)
    core.output.fps = 24
    assert ("rate", 24.0) in calls
    asyncio.run(core.emit(Out(main=np.zeros((4, 4, 3), dtype=np.uint8))))
    assert chunks[0].fps == 24


def test_output_fps_rejects_a_non_positive_rate() -> None:
    core = OutputOnlyCore()
    with pytest.raises(ValueError, match="fps must be positive"):
        core.output.fps = 0


def test_fps_assigned_before_bind_is_pushed_at_bind() -> None:
    core = OutputOnlyCore()
    core.output.fps = 24  # as a model would in load(), before binding
    _, calls = _capture_ops(core)
    assert ("rate", 24.0) in calls


def test_buffer_size_is_pushed_at_bind() -> None:
    class Sized(OutputOnlyCore):
        buffer_size = 8

    _, calls = _capture_ops(Sized())
    assert ("depth", 8) in calls


def test_an_undeclared_buffer_size_pushes_no_depth() -> None:
    _, calls = _capture_ops(OutputOnlyCore())
    assert all(name != "depth" for name, _ in calls)


def test_flush_fans_out_to_the_bound_ops() -> None:
    core = OutputOnlyCore()
    _, calls = _capture_ops(core)
    core.output.flush()
    assert ("flush", 0) in calls


def test_flush_before_bind_is_a_noop() -> None:
    OutputOnlyCore().output.flush()  # must not raise


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


def test_run_crash_reaches_the_bound_failure_sink() -> None:
    class CrashCore(OutputOnlyCore):
        async def run(self) -> None:
            raise RuntimeError("boom")

    core = CrashCore()
    failures: list[BaseException] = []
    core.bind_failure(failures.append)
    core.start_thread()
    assert core._thread is not None
    core._thread.join(timeout=2)
    assert not core._thread.is_alive()
    assert [str(failure) for failure in failures] == ["boom"]


def test_cancellation_does_not_reach_the_failure_sink() -> None:
    core = OutputOnlyCore()
    failures: list[BaseException] = []
    core.bind_failure(failures.append)
    core.start_thread()
    time.sleep(0.1)  # let the loop bootstrap before cancelling it
    core.stop()
    assert core._thread is not None
    core._thread.join(timeout=2)
    assert failures == []


def test_run_crash_without_a_bound_sink_still_ends_the_thread() -> None:
    class CrashCore(OutputOnlyCore):
        async def run(self) -> None:
            raise RuntimeError("boom")

    core = CrashCore()
    core.start_thread()
    assert core._thread is not None
    core._thread.join(timeout=2)
    assert not core._thread.is_alive()


def test_send_routes_to_the_bound_broadcast_sink() -> None:
    core = OutputOnlyCore()
    sent: list[ModelMessage] = []
    core.bind_output(
        broadcast=sent.append, addressed=lambda conn, msg, req: None, media=lambda chunk: None
    )
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
