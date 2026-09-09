# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Exercise the public authoring path with local and two-process CPU workers."""

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from examples.multi_gpu.model import FRAME_SHAPE, MultiGpuVideo, StripeWorker, VideoOutput
from reactor_runtime.core import MediaChunk
from reactor_runtime.core.model import (
    ClientConnected,
    ClientDisconnected,
    EndReason,
    SessionEnded,
    SessionStarted,
)
from reactor_runtime.core.values import ConnId
from reactor_runtime.distributed import DistributedVideoModel, WorkerError, WorkerGroup
from reactor_runtime.distributed import model as video_adapter
from reactor_runtime.interface.model.contract import ModelContract
from reactor_runtime.manifest import import_model_class, load_config

EXAMPLE = Path(__file__).parents[3] / "examples" / "multi_gpu"
CLIENT = ConnId(1)


@pytest.fixture(params=[1, 2])
def model(request, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(MultiGpuVideo, "init_process_group", False)
    monkeypatch.setattr(MultiGpuVideo, "worker_setup", lambda self, path: {"require_cuda": False})
    instance = MultiGpuVideo()
    instance.world_size = request.param
    instance.load(None)
    instance._on_loop_ready()
    yield instance
    assert instance._workers is not None
    assert instance._executor is not None
    if not instance._workers._shutdown_done:
        instance._executor.submit(instance._workers.shutdown).result()
        instance._executor.shutdown(wait=True)
    instance._loop.close()


@pytest.fixture
def group(model: MultiGpuVideo) -> WorkerGroup:
    assert model._workers is not None
    return model._workers


async def _connect(model: MultiGpuVideo, session: str = "first") -> None:
    await model._dispatch_reactor_event(SessionStarted(session))
    await model._dispatch_reactor_event(ClientConnected(CLIENT, total=1))


async def _stop(task: asyncio.Task) -> None:
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_manifest_and_command_surface(monkeypatch: pytest.MonkeyPatch, register_model) -> None:
    config = load_config(EXAMPLE / "reactor.yaml")
    assert config.world_size == 2
    monkeypatch.syspath_prepend(str(EXAMPLE))
    assert import_model_class(config.model_ref).__qualname__ == "MultiGpuVideo"
    register_model(MultiGpuVideo)
    contract = ModelContract.of(MultiGpuVideo)
    assert set(contract.commands) == {"set_brightness", "set_paused", "reset"}
    field = contract.commands["set_brightness"].command.__command_fields__["brightness"].info
    assert field.ge == 0.0
    assert field.le == 1.0
    assert set(contract.tracks) == {"main_video"}
    assert MultiGpuVideo.load is DistributedVideoModel.load
    assert MultiGpuVideo.run is DistributedVideoModel.run


def test_readme_authoring_snippet_executes() -> None:
    source = (EXAMPLE / "README.md").read_text().split("```python\n", 1)[1].split("```", 1)[0]
    namespace: dict[str, Any] = {"MyWorker": StripeWorker}
    exec(compile(source, str(EXAMPLE / "README.md"), "exec"), namespace)
    model = namespace["MyVideo"]()
    try:
        assert model.controls() == {"brightness": 1.0}
        frames = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        assert model.to_output(frames).main_video is frames
        assert "set_brightness" in ModelContract.of(type(model)).commands
    finally:
        model._loop.close()


async def test_every_rank_writes_its_band_and_sessions_reset(
    model: MultiGpuVideo,
    group: WorkerGroup,
) -> None:
    for _ in range(2):
        await model._call(group.start_session, {}, seed=0)
        frames = await model._call(group.generate, 0, {"brightness": 1.0})
        assert frames.shape == FRAME_SHAPE
        for rank in range(model.world_size):
            lo = rank * FRAME_SHAPE[1] // model.world_size
            hi = (rank + 1) * FRAME_SHAPE[1] // model.world_size
            expected = (
                np.arange(FRAME_SHAPE[2])[None, :] + np.arange(4)[:, None] + rank * 40
            ) % 256
            assert (frames[:, lo:hi] == expected[:, None, :, None]).all()
        await model._call(group.end_session)


async def test_runtime_thread_emits_real_media_and_shuts_down(
    model: MultiGpuVideo,
    group: WorkerGroup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready = threading.Event()
    emitted = threading.Event()
    chunks: list[MediaChunk] = []
    failures: list[BaseException] = []
    on_loop_ready = model._on_loop_ready

    def loop_ready():
        on_loop_ready()
        ready.set()

    def media(chunk: MediaChunk):
        chunks.append(chunk)
        emitted.set()

    monkeypatch.setattr(model, "_on_loop_ready", loop_ready)
    model.bind_output(broadcast=lambda _: None, addressed=lambda *args: None, media=media)
    model.bind_failure(failures.append)
    model.start_thread()
    try:
        assert await asyncio.to_thread(ready.wait, 5)
        model.post_reactor_event(SessionStarted("first"))
        model.post_reactor_event(ClientConnected(CLIENT, total=1))
        assert await asyncio.to_thread(emitted.wait, 5)
        chunk = chunks[0]
        assert chunk.fps == 24
        assert chunk.n_frames == FRAME_SHAPE[0]
        frames = chunk.bundle.tracks["main_video"].data
        assert frames.shape == FRAME_SHAPE
        assert frames.dtype == np.uint8
    finally:
        model.stop()
        assert model._thread is not None
        await asyncio.to_thread(model._thread.join, 10)
        assert not model._thread.is_alive()
    assert not failures
    assert group._shutdown_done


@pytest.mark.parametrize("transition", ["new_session", "reconnect", "reset", "paused_reset"])
async def test_session_transition_discards_inflight_output(
    model: MultiGpuVideo,
    group: WorkerGroup,
    monkeypatch: pytest.MonkeyPatch,
    transition: str,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    emitted = asyncio.Event()
    calls = []
    outputs = []
    original = group.generate

    def generate(index, controls, **kwargs):
        calls.append((index, dict(controls)))
        if len(calls) == 1:
            entered.set()
            assert release.wait(5), "test must release the in-flight computation"
        result = original(index, controls, **kwargs)
        result[:] = len(calls)
        return result

    async def emit(output: VideoOutput, **kwargs):
        outputs.append(output.main_video)
        emitted.set()
        model.paused = True

    monkeypatch.setattr(group, "generate", generate)
    monkeypatch.setattr(model, "emit", emit)
    await _connect(model)
    await model.set_brightness(0.25)
    task = asyncio.create_task(model.run())
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        await asyncio.wait_for(model.set_brightness(0.5), 1)
        if transition == "new_session":
            await model._dispatch_reactor_event(SessionEnded("first", EndReason.STOPPED))
            await _connect(model, "second")
        elif transition == "reconnect":
            await model._dispatch_reactor_event(ClientDisconnected(CLIENT, total=0))
            await model._dispatch_reactor_event(ClientConnected(CLIENT, total=1))
        else:
            if transition == "paused_reset":
                await model.set_paused(True)
            await model.reset()
        release.set()
        if transition == "paused_reset":
            await asyncio.sleep(0.02)
            assert not outputs
            assert model.paused
            await model.set_paused(False)
        await asyncio.wait_for(emitted.wait(), 5)
        assert len(outputs) == 1
        assert (outputs[0] == 2).all()
        expected = 1.0 if transition == "new_session" else 0.5
        assert calls == [(0, {"brightness": 0.25}), (0, {"brightness": expected})]
    finally:
        release.set()
        await _stop(task)
    assert group._shutdown_done


async def test_pause_holds_one_chunk_without_reinitializing_workers(
    model: MultiGpuVideo,
    group: WorkerGroup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    emitted = asyncio.Event()
    outputs = []
    starts = []
    start_session = group.start_session
    generate_chunk = group.generate
    clock = iter([100.0, 100.25])
    monkeypatch.setattr(video_adapter, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    monkeypatch.setattr(model, "adaptive_fps", True)

    def start(params, **kwargs):
        starts.append(params)
        start_session(params, **kwargs)

    def generate(index, controls, **kwargs):
        entered.set()
        assert release.wait(5), "test must release the in-flight computation"
        result = generate_chunk(index, controls, **kwargs)
        finished.set()
        return result

    async def emit(output, **kwargs):
        assert kwargs["compute_time"] == 0.25
        outputs.append(output)
        model.paused = True
        emitted.set()

    monkeypatch.setattr(group, "start_session", start)
    monkeypatch.setattr(group, "generate", generate)
    monkeypatch.setattr(model, "emit", emit)
    await _connect(model)
    task = asyncio.create_task(model.run())
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        await model.set_paused(True)
        release.set()
        assert await asyncio.to_thread(finished.wait, 5)
        await asyncio.sleep(0.02)
        assert not outputs
        assert group._session_open
        await model.set_paused(False)
        await asyncio.wait_for(emitted.wait(), 5)
        assert len(outputs) == 1
        assert len(starts) == 1
    finally:
        release.set()
        await _stop(task)


async def test_additional_viewers_share_one_sequence_and_no_audience_idles(
    model: MultiGpuVideo,
    group: WorkerGroup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted = asyncio.Event()
    starts = []
    indices = []
    start_session = group.start_session
    generate_chunk = group.generate

    def start(params, **kwargs):
        starts.append(params)
        start_session(params, **kwargs)

    def generate(index, controls, **kwargs):
        indices.append(index)
        return generate_chunk(index, controls, **kwargs)

    async def emit(output, **kwargs):
        model.paused = True
        emitted.set()

    monkeypatch.setattr(group, "start_session", start)
    monkeypatch.setattr(group, "generate", generate)
    monkeypatch.setattr(model, "emit", emit)
    await model._dispatch_reactor_event(SessionStarted("first"))
    task = asyncio.create_task(model.run())
    try:
        await asyncio.sleep(0.02)
        assert not starts
        await model._dispatch_reactor_event(ClientConnected(CLIENT, total=1))
        await asyncio.wait_for(emitted.wait(), 5)
        emitted.clear()
        epoch = model._epoch
        other = ConnId(2)
        await model._dispatch_reactor_event(ClientConnected(other, total=2))
        await model._dispatch_reactor_event(ClientDisconnected(CLIENT, total=1))
        assert model._epoch == epoch
        model.paused = False
        await asyncio.wait_for(emitted.wait(), 5)
        emitted.clear()
        assert indices == [0, 1]
        assert len(starts) == 1
        await model._dispatch_reactor_event(ClientDisconnected(other, total=0))
        model.paused = False
        await asyncio.sleep(0.02)
        assert not emitted.is_set()
        await model._dispatch_reactor_event(ClientConnected(CLIENT, total=1))
        await asyncio.wait_for(emitted.wait(), 5)
        assert indices == [0, 1, 0]
        assert len(starts) == 2
    finally:
        await _stop(task)


async def test_cancellation_drains_compute_before_releasing_buffers(
    model: MultiGpuVideo,
    group: WorkerGroup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    original = group.generate

    def generate(index, controls, **kwargs):
        entered.set()
        assert release.wait(5), "test must release the in-flight computation"
        return original(index, controls, **kwargs)

    monkeypatch.setattr(group, "generate", generate)
    await _connect(model)
    task = asyncio.create_task(model.run())
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert not group._shutdown_done
    finally:
        release.set()
        await _stop(task)
    assert group._shutdown_done
    assert all(not proc.is_alive() for proc in group._procs)


async def test_worker_failure_escapes_run_and_stops_the_group(
    model: MultiGpuVideo,
    group: WorkerGroup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args, **kwargs):
        raise WorkerError("worker failed")

    monkeypatch.setattr(group, "generate", fail)
    await _connect(model)
    with pytest.raises(WorkerError, match="worker failed"):
        await model.run()
    assert group._shutdown_done
