import asyncio
import gc
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pytest
from endurance.launcher import (
    _executor_threads,
    _live_transport_objects,
    path_to_output,
    reaches_output,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def tracked_echo(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """The tracking model module, its counters reset and its GC hook installed."""
    monkeypatch.syspath_prepend(str(ROOT / "examples" / "echo"))
    monkeypatch.syspath_prepend(str(ROOT / "endurance-tests" / "model"))
    import tracked_echo as module

    monkeypatch.setattr(module, "_batches", {})
    monkeypatch.setattr(module, "_freed_by_gc", 0)
    if module._on_gc not in gc.callbacks:
        gc.callbacks.append(module._on_gc)
    yield module
    # The hook is the model's, installed when a runtime imports it; keep it out
    # of every other test in this process.
    gc.callbacks.remove(module._on_gc)
    sys.modules.pop("tracked_echo", None)


def _batch() -> np.ndarray:
    return np.zeros((4, 8, 8, 3), dtype=np.uint8)


def test_a_batch_its_last_reference_frees_is_not_counted(tracked_echo: ModuleType) -> None:
    batch = _batch()
    tracked_echo.track(batch)
    del batch
    gc.collect()
    assert tracked_echo.alive_batches() == 0
    assert tracked_echo.freed_by_gc() == 0


def test_a_batch_only_garbage_held_is_counted_when_collected(tracked_echo: ModuleType) -> None:
    # No automatic collection between dropping the cycle and checking it is held.
    gc.disable()
    try:
        batch = _batch()
        tracked_echo.track(batch)
        holder: dict[str, object] = {"frame": batch[0]}
        holder["self"] = holder
        del batch, holder
        assert tracked_echo.alive_batches() == 1
        gc.collect()
    finally:
        gc.enable()
    assert tracked_echo.alive_batches() == 0
    assert tracked_echo.freed_by_gc() == 1


def test_a_held_view_reaches_its_batch() -> None:
    batch = _batch()

    class Pacer:
        def __init__(self) -> None:
            self.last_frame = {"video": batch[1]}

    assert reaches_output(Pacer(), [batch])


def test_an_object_holding_no_output_does_not_reach_it() -> None:
    batch = _batch()

    class Pacer:
        def __init__(self) -> None:
            self.last_frame = None
            self.queue: list[np.ndarray] = [np.zeros(3)]

    assert not reaches_output(Pacer(), [batch])


def test_a_batch_reachable_only_through_code_is_not_held_data() -> None:
    batch = _batch()

    class Pacer:
        def __init__(self) -> None:
            self.on_frame = lambda: batch

    assert not reaches_output(Pacer(), [batch])


def test_nothing_reaches_an_empty_set_of_batches() -> None:
    assert not reaches_output(object(), [])


def test_the_path_to_a_held_batch_names_its_holders() -> None:
    batch = _batch()

    class Pacer:
        def __init__(self) -> None:
            self.last_frame = {"video": batch[1]}

    path = path_to_output(Pacer(), [batch])
    assert path is not None
    assert path[0].startswith("Pacer")
    assert path[-1] == "ndarray"
    joined = " -> ".join(path)
    assert "last_frame" in joined
    assert "'video'" in joined


async def test_the_executor_count_is_the_default_executor_workers() -> None:
    assert _executor_threads() == 0
    await asyncio.to_thread(lambda: None)
    assert _executor_threads() == 1


class _Room:
    """Just enough of ``rtc.Room`` for a room connection to be built around it."""

    name = "the-room"

    def on(self, event: str, callback: object) -> None:
        pass

    def register_byte_stream_handler(self, topic: str, handler: object) -> None:
        pass


def _room_session() -> Any:
    from reactor_runtime.core import ConnId
    from reactor_runtime.transport.livekit.room import RoomSession

    return RoomSession(
        ConnId(1), "wss://rooms.example", "a-token", cast(Any, object()), room=cast(Any, _Room())
    )


def test_an_open_room_connection_is_not_counted_as_leaving() -> None:
    """A session that keeps its room open throughout reads no room leaving."""
    from reactor_runtime.transport.livekit.room import RoomState

    room = _room_session()
    room.state = RoomState.OPEN

    assert _live_transport_objects()["rooms_leaving"] == 0


def test_a_closed_room_connection_still_leaving_is_counted() -> None:
    from reactor_runtime.transport.livekit.room import RoomState

    room = _room_session()
    room.state = RoomState.CLOSED
    room._leave_pending = True

    assert _live_transport_objects()["rooms_leaving"] == 1
