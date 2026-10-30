"""The shared-worker example: each session's frames are its own, batched or alone.

The model half is served on threads of the test process and every session is
a real ``ModelClient``, so the frames cross the socket and shared memory as
they do between processes. The application half's three step hooks are
checked on their own, against a stand-in for the model server, and against a
real one that goes away.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from plasma import Plasma, PlasmaOutput
from plasma_model import PlasmaModel
from plasma_types import HEIGHT, WIDTH, Frame, Tick

from reactor_runtime import StepOutcome
from reactor_runtime.core.model import SessionStarted
from reactor_runtime.manifest import load_config
from reactor_runtime.model_server import (
    LinkConfig,
    ModelClient,
    ServerRequest,
    ServerSettings,
    StepTimedOut,
    UnixShmTransport,
    WorkerLost,
    WorkerServer,
)

_EXAMPLE_DIR = Path(__file__).parents[3] / "examples" / "shared_worker"
_SESSIONS = 4


class CountingPlasma(PlasmaModel):
    """The example's model half, recording the size of every step it runs."""

    batch_window = 0.05

    def load(self, config_path: Path | None) -> None:
        super().load(config_path)
        self.sizes: list[int] = []

    def step(self, requests: list[ServerRequest[Tick]]) -> list[Frame]:
        self.sizes.append(len(requests))
        return super().step(requests)


class Server:
    """The example's model half behind a real model server core, on this process's threads."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.address = str(root / "sock")
        for slot in range(_SESSIONS):
            (root / f"s{slot}").mkdir(exist_ok=True)
        self.model = CountingPlasma()
        self.model.device = "cpu"
        self.model.load(None)
        self.server: WorkerServer | None = None

    def start(self) -> None:
        settings = ServerSettings.resolve(
            CountingPlasma, load_config(_EXAMPLE_DIR / "reactor.yaml").worker
        )
        transport = UnixShmTransport(slot_directory=str(self.root / "s{slot}"))
        self.server = WorkerServer(
            self.model, transport, self.address, settings=settings, max_sessions=_SESSIONS
        )
        asyncio.run(self.server.start())

    def stop(self) -> None:
        if self.server is not None:
            asyncio.run(self.server.drain())
            asyncio.run(self.server.stop())
            self.server = None

    def link(self, slot: int) -> LinkConfig:
        return LinkConfig(
            socket=self.address, slot=slot, shm_dir=self.root / f"s{slot}", connect_timeout=5
        )


@pytest.fixture
def server(tmp_path: Path) -> Iterator[Server]:
    one = Server(tmp_path)
    one.start()
    yield one
    one.stop()


def _payloads(slot: int) -> list[Tick]:
    return [Tick(tick=tick * (slot + 1), hue=40 * slot) for tick in range(6)]


def test_the_manifest_names_both_halves_and_the_serving_settings() -> None:
    cfg = load_config(_EXAMPLE_DIR / "reactor.yaml")
    assert cfg.model_ref == "plasma:Plasma"
    assert cfg.worker is not None
    assert cfg.worker.import_ref == "plasma_model:PlasmaModel"
    assert cfg.worker.max_batch == 4


def test_each_session_gets_the_same_frames_batched_as_alone(server: Server) -> None:
    clients = [ModelClient.connect(server.link(slot)) for slot in range(_SESSIONS)]
    try:
        for client in clients:
            client.reset()
        alone = {
            slot: [clients[slot].step(p).pixels for p in _payloads(slot)]
            for slot in range(_SESSIONS)
        }
        assert set(server.model.sizes) == {1}
        server.model.sizes.clear()

        batched: dict[int, list[np.ndarray]] = {slot: [] for slot in range(_SESSIONS)}
        barrier = threading.Barrier(_SESSIONS)

        def run(slot: int) -> None:
            for payload in _payloads(slot):
                barrier.wait(5)
                batched[slot].append(clients[slot].step(payload).pixels)

        threads = [threading.Thread(target=run, args=(slot,)) for slot in range(_SESSIONS)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        assert max(server.model.sizes) > 1, "the sessions were never batched together"
        for slot in range(_SESSIONS):
            for got, want in zip(batched[slot], alone[slot], strict=True):
                assert got.shape == (HEIGHT, WIDTH, 3)
                assert np.array_equal(got, want)
        # Different sessions, different frames: nothing crossed between them.
        assert not np.array_equal(batched[0][3], batched[1][3])
    finally:
        for client in clients:
            client.close()


def test_the_application_half_goes_down_when_the_model_server_goes_away(
    server: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    link = server.link(0)
    monkeypatch.setenv("REACTOR_MODEL_WORKER_SOCKET", link.socket)
    monkeypatch.setenv("REACTOR_SESSION_SLOT", "0")
    monkeypatch.setenv("REACTOR_SHM_DIR", str(link.shm_dir))
    app = Plasma()
    crashed: list[BaseException] = []
    heard = threading.Event()

    def crash(error: BaseException) -> None:
        crashed.append(error)
        heard.set()

    monkeypatch.setattr(app, "crash", crash)
    app.load(None)
    try:
        asyncio.run(_start(app))
        output = asyncio.run(_step(app))
        assert np.asarray(output.main_video).shape == (HEIGHT, WIDTH, 3)

        # Idle between steps: only on_lost can tell the application.
        server.stop()
        assert heard.wait(5)
        assert len(crashed) == 1
        assert isinstance(crashed[0], WorkerLost)
        with pytest.raises(WorkerLost):
            asyncio.run(_step(app))
    finally:
        app.model.close()


async def _step(app: Plasma) -> PlasmaOutput:
    """One step through the three hooks, as the runtime's loop runs it."""
    tick = await app.process_input()
    try:
        outcome = StepOutcome(result=app.generate(tick))
    except Exception as exc:
        outcome = StepOutcome(error=exc)
    return await app.process_output(outcome)


async def _start(app: Plasma) -> None:
    """Start a session on *app*, as the runtime does before its first step."""
    app._on_loop_ready()
    app.bind_output(
        broadcast=lambda message: None, addressed=lambda *args: None, media=lambda chunk: None
    )
    await app._dispatch_reactor_event(SessionStarted("s"))


class RecordingClient:
    """A ``ModelClient`` by shape: records the resets and answers each step with a blank frame."""

    def __init__(self) -> None:
        self.resets = 0
        self.steps: list[Tick] = []

    def reset(self) -> None:
        self.resets += 1

    def step(self, payload: Tick) -> Frame:
        self.steps.append(payload)
        return Frame(pixels=np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))


async def _app() -> tuple[Plasma, RecordingClient]:
    """The application half on a started session, its model server a recording stand-in."""
    app = Plasma()
    client = RecordingClient()
    app.model = client  # type: ignore[ty:invalid-assignment]  # a ModelClient by shape
    app.tick = 0
    await _start(app)
    return app, client


async def test_a_session_starts_from_the_first_frame_on_a_reset_slot() -> None:
    app, client = await _app()
    app.state.speed = 5
    await app.process_input()
    await app.process_input()

    await app._dispatch_reactor_event(SessionStarted("t"))

    assert client.resets == 2
    assert (await app.process_input()).tick == 0


async def test_generate_sends_the_tick_to_the_model_server_and_nothing_else() -> None:
    app, client = await _app()
    tick = Tick(tick=7, hue=9)

    frame = app.generate(tick)

    assert client.steps == [tick]
    assert frame.pixels.shape == (HEIGHT, WIDTH, 3)


async def test_process_input_builds_the_sessions_tick_from_its_state() -> None:
    app, _ = await _app()
    app.state.hue = 40
    app.state.speed = 3

    first = await app.process_input()
    second = await app.process_input()

    assert (first, second) == (Tick(tick=0, hue=40), Tick(tick=3, hue=40))


async def test_the_tick_wraps_at_the_palette_and_renders_the_same_frames() -> None:
    app, _ = await _app()
    app.state.speed = 16
    ticks = [(await app.process_input()).tick for _ in range(40)]

    assert all(0 <= tick < 256 for tick in ticks)
    assert ticks[16] == 0  # 16 steps of 16 is one turn of the palette
    model = PlasmaModel()
    model.load(None)
    unwrapped = Tick(tick=16 * 20, hue=0)
    wrapped = Tick(tick=ticks[20], hue=0)
    [a, b] = model.step([ServerRequest(0, 0, unwrapped), ServerRequest(1, 0, wrapped)])
    assert np.array_equal(a.pixels, b.pixels)


async def test_process_output_puts_the_frame_on_the_video_track() -> None:
    pixels = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    app, _ = await _app()
    output = await app.process_output(StepOutcome(result=Frame(pixels=pixels)))

    assert isinstance(output, PlasmaOutput)
    assert output.main_video is pixels


@pytest.mark.parametrize(
    "error", [WorkerLost("the model server went away"), StepTimedOut("no answer in 5 s")]
)
async def test_process_output_re_raises_a_lost_or_stuck_model_server(error: Exception) -> None:
    app, _ = await _app()
    with pytest.raises(type(error)) as raised:
        await app.process_output(StepOutcome(error=error))
    assert raised.value is error
