import numpy as np

from reactor_runtime.core.values import InputFrame
from reactor_runtime.engine_contract import InputField, UserInput, VideoInput
from reactor_runtime.interface.engine.store import InputStore, MediaSpec


class Move(UserInput):
    direction: str = InputField(default="left")


class Look(UserInput):
    dx: float = 0.0


class Camera(VideoInput):
    chunk_size = 3


def _frame(value: int) -> InputFrame:
    return InputFrame(data=np.full((2, 2, 3), value, dtype=np.uint8), pts=float(value))


def _store(chunk_size: int = 3) -> InputStore:
    return InputStore({"camera": MediaSpec("camera", Camera, chunk_size)})


def _directions(window: list[UserInput]) -> list[str]:
    return [item.direction for item in window if isinstance(item, Move)]


def _chunk(window: list[UserInput]) -> Camera:
    assert isinstance(window[0], Camera)
    return window[0]


# -- the window ----------------------------------------------------------------


def test_a_drain_returns_everything_pushed_since_the_previous_one() -> None:
    store = InputStore()
    store.push(Move(direction="left"))
    store.push(Look(dx=1.0))

    assert len(store.drain(0)) == 2
    assert store.drain(1) == []


def test_the_window_is_ordered_by_arrival() -> None:
    store = InputStore()
    first, second = Move(direction="left"), Move(direction="right")
    store.push(first)
    store.push(second)

    window = store.drain(0)
    assert _directions(window) == ["left", "right"]
    assert window[0].timestamp_ms <= window[1].timestamp_ms


def test_pushing_stamps_the_arrival_the_caller_never_sets() -> None:
    store = InputStore()
    move = Move(direction="left")
    assert move.timestamp_ms == 0

    store.push(move)
    assert move.timestamp_ms > 0


# -- media ---------------------------------------------------------------------


def test_a_full_chunk_materializes_into_one_instance() -> None:
    store = _store()
    for value in (1, 2, 3):
        store.push_frame("camera", _frame(value))

    window = store.drain(0)
    assert len(window) == 1
    assert isinstance(window[0], Camera)
    assert window[0].data.shape == (3, 2, 2, 3)


def test_an_incomplete_chunk_waits_across_windows() -> None:
    store = _store()
    store.push_frame("camera", _frame(1))
    store.push_frame("camera", _frame(2))
    assert store.drain(0) == []

    store.push_frame("camera", _frame(3))
    window = store.drain(1)
    assert len(window) == 1
    assert [frame[0][0][0] for frame in _chunk(window).data] == [1, 2, 3]


def test_a_single_frame_chunk_carries_the_frame_unbatched() -> None:
    store = InputStore({"camera": MediaSpec("camera", Camera, 1)})
    store.push_frame("camera", _frame(7))

    assert _chunk(store.drain(0)).data.shape == (2, 2, 3)


def test_a_chunk_is_stamped_at_its_last_frames_arrival() -> None:
    store = _store()
    store.push_frame("camera", _frame(1))
    store.push_frame("camera", _frame(2))
    store.push_frame("camera", _frame(3))
    move = Move(direction="left")
    store.push(move)

    window = store.drain(0)
    # The chunk completed before the event arrived, so it leads the window.
    assert isinstance(window[0], Camera)
    assert window[1] is move


def test_a_chunk_carries_the_capture_time_it_starts_at() -> None:
    store = _store()
    for value in (4, 5, 6):
        store.push_frame("camera", _frame(value))

    assert _chunk(store.drain(0)).pts == 4.0


def test_a_frame_for_an_undeclared_track_is_dropped() -> None:
    store = _store()
    store.push_frame("lidar", _frame(1))

    assert store.drain(0) == []


# -- deferred injection --------------------------------------------------------


def test_a_deferred_input_is_held_until_its_step() -> None:
    store = InputStore()
    store.push(Move(direction="up"), at_step=2)

    assert store.drain(0) == []
    assert store.drain(1) == []
    assert len(store.drain(2)) == 1


def test_a_deferred_input_sorts_ahead_of_what_arrives_live() -> None:
    store = InputStore()
    store.push(Move(direction="scheduled"), at_step=1)
    store.drain(0)
    store.push(Move(direction="live"))

    assert _directions(store.drain(1)) == ["scheduled", "live"]


def test_clearing_deferred_drops_what_was_scheduled() -> None:
    store = InputStore()
    store.push(Move(direction="up"), at_step=3)
    store.clear_deferred()

    assert store.drain(3) == []


def test_reset_drops_events_deferrals_and_partial_chunks() -> None:
    store = _store()
    store.push(Move(direction="left"))
    store.push(Move(direction="right"), at_step=1)
    store.push_frame("camera", _frame(1))

    store.reset()
    store.push_frame("camera", _frame(2))
    store.push_frame("camera", _frame(3))

    assert store.drain(0) == []
    assert store.drain(1) == []
