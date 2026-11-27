"""A ReactorRobotApp takes stamped robot state as the ``observe`` command.

The messages go through the contract as a client's would, then to the app's
command dispatch; the replies are what the client would read back.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from reactor_runtime import ApplicationError, ReactorApp
from reactor_runtime.apps.robot import ReactorRobotApp
from reactor_runtime.apps.robot.observe import (
    INVALID_OBSERVE,
    OBSERVE_BUFFER_SIZE,
    ObserveBuffer,
    ObserveError,
    ObserveStats,
    StampedState,
    parse_observe,
)
from reactor_runtime.core.model import (
    ClientConnected,
    ClientDisconnected,
    EndReason,
    SessionEnded,
    SessionStarted,
)
from reactor_runtime.core.values import CommandFailure, ConnId
from reactor_runtime.interface.internal.reactor_core import CommandEnvelope
from reactor_runtime.interface.model.contract import ModelContract

ROBOT = ConnId(1001)
VIEWER = ConnId(1002)


class Policy(ReactorRobotApp):
    """A policy that never steps: only its commands run here."""

    async def process_input(self) -> None:
        raise ApplicationError("no observation")


@pytest.fixture(autouse=True)
def _seed_registries(
    isolate_interface_registries: None, register_model: Callable[[type], None]
) -> None:
    register_model(Policy)


Reply = tuple[ConnId, Any, Any]


def _ready(app: ReactorApp) -> list[Reply]:
    replies: list[Reply] = []
    app._on_loop_ready()
    app.bind_output(
        broadcast=lambda message: None,
        addressed=lambda conn, message, request: replies.append((conn, message, request)),
        media=lambda chunk: None,
        step=lambda step: None,
    )
    return replies


async def _live(app: ReactorApp) -> None:
    await app._dispatch_reactor_event(SessionStarted("s"))
    await app._dispatch_reactor_event(ClientConnected(ROBOT, 1))


async def _observe(app: ReactorApp, request_id: str | None = None, **data: Any) -> None:
    command = ModelContract.of(type(app)).validate("observe", data)
    await app._dispatch_command(CommandEnvelope(command, ROBOT, request_id))


def _stamps(app: ReactorRobotApp) -> list[int]:
    return [entry.capture_time_us for entry in app._observed.entries()]


def test_every_robot_app_declares_observe() -> None:
    schema = ModelContract.of(Policy).render_schema(name="policy")

    assert "observe" in schema.commands
    assert "observe" in ModelContract.of(ReactorRobotApp).commands


def test_a_subclass_cannot_declare_its_own_observe() -> None:
    from reactor_runtime import event

    with pytest.raises(ValueError, match="duplicate command name 'observe'"):

        class Shadowing(ReactorRobotApp):
            @event(name="observe")
            def mine(self, capture_time_us: int) -> None:
                pass


async def test_valid_messages_land_in_the_buffer_in_stamp_order() -> None:
    app = Policy()
    replies = _ready(app)
    await _live(app)

    for stamp in (3_000, 1_000, 2_000):
        await _observe(app, capture_time_us=stamp, state={"joints": [stamp / 1000]})

    assert _stamps(app) == [1_000, 2_000, 3_000]
    assert [dict(e.state) for e in app._observed.entries()] == [
        {"joints": [1.0]},
        {"joints": [2.0]},
        {"joints": [3.0]},
    ]
    assert app._observed.stats() == ObserveStats(accepted=3, buffered=3)
    assert replies == []


async def test_a_correlated_observe_is_acknowledged() -> None:
    app = Policy()
    replies = _ready(app)
    await _live(app)

    await _observe(app, request_id="r1", capture_time_us=1, state={})

    assert replies == [(ROBOT, None, "r1")]


async def test_an_integral_float_stamp_is_taken_as_an_integer() -> None:
    app = Policy()
    _ready(app)
    await _live(app)

    await _observe(app, capture_time_us=1_712_345_678_901.0, state={"gripper": 0.5})

    [entry] = app._observed.entries()
    assert entry.capture_time_us == 1_712_345_678_901
    assert type(entry.capture_time_us) is int


async def test_the_buffer_keeps_the_newest_states() -> None:
    app = Policy()
    _ready(app)
    await _live(app)

    for stamp in range(OBSERVE_BUFFER_SIZE + 3):
        await _observe(app, capture_time_us=stamp, state={})

    assert _stamps(app) == list(range(3, OBSERVE_BUFFER_SIZE + 3))
    assert app._observed.stats().evicted == 3


@pytest.mark.parametrize(
    ("data", "reason"),
    [
        ({"state": {}}, "needs capture_time_us"),
        ({"capture_time_us": "1000", "state": {}}, "must be an integer"),
        ({"capture_time_us": 1000.5, "state": {}}, "must be an integer"),
        ({"capture_time_us": True, "state": {}}, "must be an integer"),
        ({"capture_time_us": float("nan"), "state": {}}, "must be an integer"),
        ({"capture_time_us": -1, "state": {}}, "zero or more"),
        ({"capture_time_us": 1000}, "needs state"),
        ({"capture_time_us": 1000, "state": [1, 2]}, "must be a JSON object, got an array"),
        ({"capture_time_us": 1000, "state": "ready"}, "must be a JSON object, got a string"),
        ({"capture_time_us": 1000, "state": 3}, "must be a JSON object, got a number"),
    ],
)
async def test_a_malformed_message_is_refused_and_counted(
    data: dict[str, Any], reason: str
) -> None:
    app = Policy()
    replies = _ready(app)
    await _live(app)

    await _observe(app, **data)

    [(conn, failure, request)] = replies
    assert conn == ROBOT
    assert request is None
    assert isinstance(failure, CommandFailure)
    assert failure.code == INVALID_OBSERVE
    assert reason in failure.message
    assert len(app._observed) == 0
    assert app._observed.stats() == ObserveStats(refused=1)


async def test_a_refusal_leaves_the_states_already_kept() -> None:
    app = Policy()
    replies = _ready(app)
    await _live(app)

    await _observe(app, capture_time_us=1_000, state={"a": 1})
    await _observe(app, request_id="bad", capture_time_us="soon", state={"a": 2})
    await _observe(app, capture_time_us=2_000, state={"a": 3})

    assert _stamps(app) == [1_000, 2_000]
    assert app._observed.stats() == ObserveStats(accepted=2, refused=1, buffered=2)
    [(_, failure, request)] = replies
    assert request == "bad"
    assert isinstance(failure, CommandFailure)


async def test_the_buffer_is_dropped_when_the_last_client_leaves() -> None:
    app = Policy()
    _ready(app)
    await _live(app)
    await app._dispatch_reactor_event(ClientConnected(VIEWER, 2))
    await _observe(app, capture_time_us=1, state={})
    await _observe(app, capture_time_us="x", state={})

    await app._dispatch_reactor_event(ClientDisconnected(VIEWER, 1))
    assert len(app._observed) == 1

    await app._dispatch_reactor_event(ClientDisconnected(ROBOT, 0))
    assert app._observed.stats() == ObserveStats(accepted=1, refused=1)


@pytest.mark.parametrize(
    "boundary",
    [SessionEnded("s", EndReason.STOPPED), SessionStarted("s2")],
)
async def test_a_session_boundary_drops_the_states_and_their_counts(boundary: Any) -> None:
    app = Policy()
    _ready(app)
    await _live(app)
    await _observe(app, capture_time_us=1, state={})
    await _observe(app, capture_time_us="x", state={})

    await app._dispatch_reactor_event(boundary)

    assert app._observed.stats() == ObserveStats()


# -- the buffer and the parser on their own -------------------------------------


def _entry(stamp: int, **state: Any) -> StampedState:
    return StampedState(stamp, state, 0.0)


def test_a_repeated_stamp_replaces_the_state_held() -> None:
    buffer = ObserveBuffer(size=3)
    buffer.push(_entry(1, v=1))
    buffer.push(_entry(1, v=2))

    [entry] = buffer.entries()
    assert dict(entry.state) == {"v": 2}
    assert buffer.stats() == ObserveStats(accepted=2, buffered=1)


def test_a_state_older_than_a_full_buffer_is_dropped_on_arrival() -> None:
    buffer = ObserveBuffer(size=2)
    for stamp in (10, 20, 5):
        buffer.push(_entry(stamp))

    assert [e.capture_time_us for e in buffer.entries()] == [10, 20]
    assert buffer.stats() == ObserveStats(accepted=3, evicted=1, buffered=2)


def test_a_late_state_inside_a_full_buffer_takes_its_place_in_order() -> None:
    buffer = ObserveBuffer(size=3)
    for stamp in (10, 30, 40, 20):
        buffer.push(_entry(stamp))

    assert [e.capture_time_us for e in buffer.entries()] == [20, 30, 40]


def test_clear_keeps_the_counts_and_reset_zeroes_them() -> None:
    buffer = ObserveBuffer(size=2)
    buffer.push(_entry(1))
    buffer.refuse()

    buffer.clear()
    assert buffer.stats() == ObserveStats(accepted=1, refused=1)

    buffer.reset()
    assert buffer.stats() == ObserveStats()


def test_a_buffer_holds_at_least_one_state() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        ObserveBuffer(size=0)


def test_a_stamped_state_keeps_a_read_only_copy() -> None:
    state = {"joints": [0.0]}
    entry = StampedState(1, state, 0.0)
    state["gripper"] = 1.0

    assert dict(entry.state) == {"joints": [0.0]}
    with pytest.raises(TypeError):
        entry.state["gripper"] = 1.0  # type: ignore[ty:invalid-assignment] — the copy is read-only


def test_parse_refuses_a_state_with_a_non_string_key() -> None:
    with pytest.raises(ObserveError, match="keys must be strings"):
        parse_observe(1, {1: "a"})
