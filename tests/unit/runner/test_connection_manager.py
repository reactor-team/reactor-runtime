import logging
import sys
import threading
from collections.abc import Callable

import pytest

from reactor_runtime.core import (
    Connection,
    ConnectionCapabilities,
    ConnId,
    MediaBundle,
    MediaChunk,
    SessionEvent,
    SessionState,
    Transition,
)
from reactor_runtime.protocol import Channel, ProtocolVersion
from reactor_runtime.runner import ConnectionManager, SessionStateMachine
from reactor_runtime.transport import ConnectionsExhaustedError


class FakeConnection:
    """A shape-conforming stand-in that records what the manager calls on it.

    *on_outbound* runs at the head of every outbound call, so a test can make a
    connection act on the manager — drop a peer, or raise — from inside a
    fan-out that is iterating the registry.
    """

    def __init__(
        self,
        cid: int,
        *,
        capabilities: ConnectionCapabilities | None = None,
        protocol_version: ProtocolVersion = ProtocolVersion.V0,
        on_outbound: Callable[[], None] | None = None,
    ) -> None:
        self.id = ConnId(cid)
        self.capabilities = capabilities or ConnectionCapabilities(
            carries_video=True, carries_audio=True
        )
        self.protocol_version = protocol_version
        self.messages: list[bytes | str] = []
        self.control: list[bytes | str] = []
        self.media: list[MediaChunk] = []
        self.resumed: list[str] = []
        self.paused: list[str] = []
        self.closed = False
        self.flushed = False
        self.media_rate: float | None = None
        self.media_depth: int | None = None
        self._on_outbound = on_outbound

    def _outbound(self) -> None:
        if self._on_outbound is not None:
            self._on_outbound()

    def send_message(self, payload: bytes | str) -> None:
        self._outbound()
        self.messages.append(payload)

    def send_control(self, payload: bytes | str) -> None:
        self._outbound()
        self.control.append(payload)

    def send_media(self, chunk: MediaChunk) -> None:
        self._outbound()
        self.media.append(chunk)

    def flush_media(self) -> None:
        self._outbound()
        self.flushed = True

    def set_media_rate(self, fps: float) -> None:
        self._outbound()
        self.media_rate = fps

    def set_media_depth(self, depth: int) -> None:
        self._outbound()
        self.media_depth = depth

    def resume_track(self, name: str) -> None:
        self.resumed.append(name)

    def pause_track(self, name: str) -> None:
        self.paused.append(name)

    async def close(self) -> None:
        self.closed = True


class SilentConnection(FakeConnection):
    """A connection that discards what it is sent, for long concurrent runs.

    The recording fake accumulates one entry per call, which a fan-out spinning
    for the length of a churn loop turns into millions of them.
    """

    def send_message(self, payload: bytes | str) -> None:
        pass

    def send_media(self, chunk: MediaChunk) -> None:
        pass


def expect_state(sm: SessionStateMachine, state: SessionState) -> None:
    # The call boundary keeps the type checker from narrowing the property to a
    # single literal across the consecutive asserts of a walk.
    assert sm.current_state is state


def waiting_manager() -> tuple[ConnectionManager, SessionStateMachine]:
    sm = SessionStateMachine()
    sm.send(SessionEvent.INITIALIZATION_SUCCESS)
    sm.send(SessionEvent.START_SESSION)
    return ConnectionManager(state_machine=sm), sm


def test_fake_connection_conforms_to_the_protocol() -> None:
    assert isinstance(FakeConnection(1), Connection)


def test_new_conn_id_is_random_in_range_and_unique() -> None:
    cm, _ = waiting_manager()
    ids = [cm.new_conn_id() for _ in range(50)]
    assert len(set(ids)) == 50
    assert all(1002 <= int(cid) <= 9999 for cid in ids)


def test_new_conn_id_is_not_reused_after_a_drop() -> None:
    cm, _ = waiting_manager()
    first = cm.new_conn_id()
    cm.register(FakeConnection(first))
    cm.drop(first)
    # The dropped id stays in the session's used pool, so no later mint returns it.
    later = {cm.new_conn_id() for _ in range(100)}
    assert first not in later


def test_new_conn_id_raises_when_the_draw_keeps_colliding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A one-id range is used up after a single mint, so the next mint's draws all
    # collide and it refuses within the bounded attempts rather than spinning.
    monkeypatch.setattr("reactor_runtime.runner.connection_manager._MIN_CONN_ID", 1002)
    monkeypatch.setattr("reactor_runtime.runner.connection_manager._MAX_CONN_ID", 1002)
    cm, _ = waiting_manager()
    assert int(cm.new_conn_id()) == 1002
    with pytest.raises(ConnectionsExhaustedError):
        cm.new_conn_id()


def test_new_conn_id_still_mints_when_some_ids_are_taken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A dense but not full range still yields the free id within the attempt
    # bound: the allowance is not a lifetime cap, only a spin guard.
    monkeypatch.setattr("reactor_runtime.runner.connection_manager._MIN_CONN_ID", 1002)
    monkeypatch.setattr("reactor_runtime.runner.connection_manager._MAX_CONN_ID", 1003)
    cm, _ = waiting_manager()
    assert {int(cm.new_conn_id()), int(cm.new_conn_id())} == {1002, 1003}


@pytest.mark.asyncio
async def test_close_all_frees_the_id_pool_for_the_next_session() -> None:
    cm, _ = waiting_manager()
    cm.register(FakeConnection(cm.new_conn_id()))
    await cm.close_all()
    # Session teardown clears the used-id pool so the next session starts fresh.
    assert cm._used_conn_ids == set()


def test_first_connection_moves_session_to_streaming() -> None:
    cm, sm = waiting_manager()
    cm.register(FakeConnection(1))
    assert cm.count == 1
    assert sm.current_state is SessionState.STREAMING


def test_additional_connection_does_not_re_enter_streaming() -> None:
    cm, sm = waiting_manager()
    seen: list[Transition] = []
    sm.on_transition(seen.append)
    cm.register(FakeConnection(1))
    cm.register(FakeConnection(2))
    assert cm.count == 2
    assert sm.current_state is SessionState.STREAMING
    events = [t.event for t in seen]
    assert events == [
        SessionEvent.CONNECTION_OPENED,
        SessionEvent.CONNECTION_OPENED,
    ]


def test_last_connection_out_moves_session_to_orphaned() -> None:
    cm, sm = waiting_manager()
    cm.register(FakeConnection(1))
    cm.register(FakeConnection(2))
    cm.drop(ConnId(2))
    assert cm.count == 1
    expect_state(sm, SessionState.STREAMING)
    cm.drop(ConnId(1))
    assert cm.count == 0
    expect_state(sm, SessionState.ORPHANED)


def test_last_drop_sends_one_connection_closed_and_orphans() -> None:
    cm, sm = waiting_manager()
    cm.register(FakeConnection(1))
    seen: list[SessionEvent] = []
    sm.on_transition(lambda t: seen.append(t.event))
    cm.drop(ConnId(1))
    assert seen == [SessionEvent.CONNECTION_CLOSED]
    assert sm.current_state is SessionState.ORPHANED


def test_dropping_unknown_connection_is_ignored() -> None:
    cm, sm = waiting_manager()
    cm.register(FakeConnection(1))
    seen: list[Transition] = []
    sm.on_transition(seen.append)
    cm.drop(ConnId(99))
    assert cm.count == 1
    assert seen == []


def test_re_registering_an_id_replaces_without_re_driving() -> None:
    cm, sm = waiting_manager()
    first = FakeConnection(1)
    cm.register(first)
    seen: list[Transition] = []
    sm.on_transition(seen.append)
    replacement = FakeConnection(1)
    cm.register(replacement)
    assert cm.count == 1
    assert seen == []
    cm.broadcast(lambda _version: b"hi")
    assert replacement.messages == [b"hi"]
    assert first.messages == []


def test_broadcast_reaches_every_connection() -> None:
    cm, _ = waiting_manager()
    a, b = FakeConnection(1), FakeConnection(2)
    cm.register(a)
    cm.register(b)
    cm.broadcast(lambda _version: b"payload")
    cm.broadcast(lambda _version: '{"type":"text-frame"}')
    assert a.messages == [b"payload", '{"type":"text-frame"}']
    assert b.messages == [b"payload", '{"type":"text-frame"}']


def test_broadcast_encodes_per_connection_version() -> None:
    cm, _ = waiting_manager()
    v0 = FakeConnection(1, protocol_version=ProtocolVersion.V0)
    v1 = FakeConnection(2, protocol_version=ProtocolVersion.V1)
    cm.register(v0)
    cm.register(v1)
    cm.broadcast(lambda version: version.value)
    assert v0.messages == [ProtocolVersion.V0.value]
    assert v1.messages == [ProtocolVersion.V1.value]


def test_addressed_send_reaches_only_the_target() -> None:
    cm, _ = waiting_manager()
    a, b = FakeConnection(1), FakeConnection(2)
    cm.register(a)
    cm.register(b)
    cm.send(ConnId(2), lambda _version: b"only-b")
    assert a.messages == []
    assert b.messages == [b"only-b"]


def test_addressed_send_to_unknown_connection_is_ignored() -> None:
    cm, _ = waiting_manager()
    cm.register(FakeConnection(1))
    cm.send(ConnId(42), lambda _version: b"nowhere")


def test_send_control_reaches_only_the_target() -> None:
    cm, _ = waiting_manager()
    a, b = FakeConnection(1), FakeConnection(2)
    cm.register(a)
    cm.register(b)
    cm.send_control(ConnId(2), lambda _version: '{"type":"response"}')
    assert a.control == []
    assert b.control == ['{"type":"response"}']


def test_send_control_to_unknown_connection_is_ignored() -> None:
    cm, _ = waiting_manager()
    cm.register(FakeConnection(1))
    cm.send_control(ConnId(42), lambda _version: '{"type":"response"}')


def test_send_response_routes_by_the_codec_channel() -> None:
    cm, _ = waiting_manager()
    conn = FakeConnection(1)
    cm.register(conn)
    cm.send_response(ConnId(1), lambda _v: (Channel.DATA, "on-data"))
    cm.send_response(ConnId(1), lambda _v: (Channel.CONTROL, "on-control"))
    assert conn.messages == ["on-data"]
    assert conn.control == ["on-control"]


def test_broadcast_response_routes_each_connection_by_its_codec_channel() -> None:
    cm, _ = waiting_manager()
    v0 = FakeConnection(1, protocol_version=ProtocolVersion.V0)
    v1 = FakeConnection(2, protocol_version=ProtocolVersion.V1)
    cm.register(v0)
    cm.register(v1)
    cm.broadcast_response(
        lambda version: (
            (Channel.DATA, "legacy")
            if version is ProtocolVersion.V0
            else (Channel.CONTROL, "binary")
        )
    )
    assert v0.messages == ["legacy"]
    assert v0.control == []
    assert v1.control == ["binary"]
    assert v1.messages == []


def test_resume_and_pause_forward_to_the_connection() -> None:
    cm, _ = waiting_manager()
    conn = FakeConnection(1)
    cm.register(conn)
    cm.resume_track(ConnId(1), "main_video")
    cm.pause_track(ConnId(1), "main_audio")
    assert conn.resumed == ["main_video"]
    assert conn.paused == ["main_audio"]


def test_media_skips_data_only_connections() -> None:
    cm, _ = waiting_manager()
    media = FakeConnection(1, capabilities=ConnectionCapabilities(carries_video=True))
    data_only = FakeConnection(
        2, capabilities=ConnectionCapabilities(carries_video=False, carries_audio=False)
    )
    cm.register(media)
    cm.register(data_only)
    chunk = MediaChunk(bundle=MediaBundle(), fps=30.0, n_frames=1)
    cm.broadcast_media(chunk)
    assert media.media == [chunk]
    assert data_only.media == []


def test_broadcast_media_aborts_between_connections() -> None:
    cm, _ = waiting_manager()
    a, b = FakeConnection(1), FakeConnection(2)
    cm.register(a)
    cm.register(b)
    chunk = MediaChunk(bundle=MediaBundle(), fps=30.0, n_frames=1)
    # The abort flips once the first connection has media, so the second
    # never receives the chunk — the flush-mid-fan-out contract.
    cm.broadcast_media(chunk, abort=lambda: len(a.media) > 0)
    assert a.media == [chunk]
    assert b.media == []


def test_flush_media_reaches_every_connection() -> None:
    cm, _ = waiting_manager()
    a, b = FakeConnection(1), FakeConnection(2)
    cm.register(a)
    cm.register(b)
    cm.flush_media()
    assert a.flushed
    assert b.flushed


def test_set_media_rate_reaches_every_connection() -> None:
    cm, _ = waiting_manager()
    a, b = FakeConnection(1), FakeConnection(2)
    cm.register(a)
    cm.register(b)
    cm.set_media_rate(24.0)
    assert (a.media_rate, b.media_rate) == (24.0, 24.0)


def test_set_media_depth_reaches_every_connection() -> None:
    cm, _ = waiting_manager()
    a, b = FakeConnection(1), FakeConnection(2)
    cm.register(a)
    cm.register(b)
    cm.set_media_depth(8)
    assert (a.media_depth, b.media_depth) == (8, 8)


FanOut = Callable[[ConnectionManager], None]


def _refuse_to_encode_frame(_version: ProtocolVersion) -> bytes | str:
    """Stand in for a codec handed a payload it cannot render."""
    raise TypeError("value is not serialisable")


def _refuse_to_encode_response(_version: ProtocolVersion) -> tuple[Channel, bytes | str]:
    """The channel-picking form of :func:`_refuse_to_encode_frame`."""
    raise TypeError("value is not serialisable")


# Every fan-out that walks the whole registry, each invoked the way its caller
# does. A client dropping mid-fan-out has to be survivable on all of them, not
# only on the media path where it was first seen.
_FAN_OUTS: list[tuple[str, FanOut]] = [
    ("broadcast", lambda cm: cm.broadcast(lambda _version: "frame")),
    (
        "broadcast_response",
        lambda cm: cm.broadcast_response(lambda _version: (Channel.DATA, "frame")),
    ),
    (
        "broadcast_media",
        lambda cm: cm.broadcast_media(MediaChunk(bundle=MediaBundle(), fps=30.0, n_frames=1)),
    ),
    ("flush_media", lambda cm: cm.flush_media()),
    ("set_media_rate", lambda cm: cm.set_media_rate(24.0)),
    ("set_media_depth", lambda cm: cm.set_media_depth(8)),
]


@pytest.mark.parametrize(("name", "fan_out"), _FAN_OUTS, ids=[name for name, _ in _FAN_OUTS])
def test_fan_out_survives_a_drop_landing_inside_it(name: str, fan_out: FanOut) -> None:
    """A client leaving mid-fan-out must not break the fan-out.

    Dropping the second connection from inside the first one's outbound call is
    the deterministic form of the real race, where the drop arrives on the event
    loop while the model's thread is part-way through the registry.
    """
    cm, _ = waiting_manager()
    leaving = FakeConnection(2)
    staying = FakeConnection(1, on_outbound=lambda: cm.drop(leaving.id))
    cm.register(staying)
    cm.register(leaving)

    fan_out(cm)

    assert cm.count == 1


def test_fan_out_reaches_the_rest_when_one_connection_raises() -> None:
    """One failing wire is the fan-out's to absorb, not the session's to die on.

    The fan-out runs on the model's own thread, so an exception let through here
    would surface as a crash of the model's run loop and end the session for
    every client on it.
    """
    cm, _ = waiting_manager()

    def fail() -> None:
        raise RuntimeError("wire is gone")

    broken = FakeConnection(1, on_outbound=fail)
    healthy = FakeConnection(2)
    cm.register(broken)
    cm.register(healthy)
    chunk = MediaChunk(bundle=MediaBundle(), fps=30.0, n_frames=1)

    cm.broadcast_media(chunk)

    assert healthy.media == [chunk]


def test_an_absorbed_failure_names_the_wire_it_came_from(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An absorbed failure has to identify its connection to be actionable.

    Every client in a session logs through the same fan-out, so the operation
    alone cannot tell an operator which wire is the one failing.
    """
    cm, _ = waiting_manager()

    def fail() -> None:
        raise RuntimeError("wire is gone")

    cm.register(FakeConnection(1, on_outbound=fail))
    chunk = MediaChunk(bundle=MediaBundle(), fps=30.0, n_frames=1)

    with caplog.at_level(logging.ERROR, logger="reactor_runtime.runner.connection_manager"):
        cm.broadcast_media(chunk)

    assert len(caplog.records) == 1
    fields = getattr(caplog.records[0], "reactor_fields", {})
    assert fields["conn_id"] == ConnId(1)
    assert fields["operation"] == "broadcast_media"


_ENCODING_FAN_OUTS: list[tuple[str, FanOut]] = [
    ("broadcast", lambda cm: cm.broadcast(_refuse_to_encode_frame)),
    ("broadcast_response", lambda cm: cm.broadcast_response(_refuse_to_encode_response)),
]


@pytest.mark.parametrize(
    ("name", "fan_out"), _ENCODING_FAN_OUTS, ids=[name for name, _ in _ENCODING_FAN_OUTS]
)
def test_an_encode_failure_propagates_out_of_the_fan_out(name: str, fan_out: FanOut) -> None:
    """Encoding is not one client's to fail, so its failure is not absorbed.

    A payload the codec cannot render fails identically for every connection, so
    absorbing it would report an authoring mistake as N wire failures and let the
    message reach nobody. It belongs to the caller that built the payload.
    """
    cm, _ = waiting_manager()
    cm.register(FakeConnection(1))
    cm.register(FakeConnection(2))

    with pytest.raises(TypeError, match="not serialisable"):
        fan_out(cm)


def test_media_fan_out_is_safe_against_concurrent_connection_churn() -> None:
    """The reported shape: one thread fanning out while another registers and drops.

    Reads come off the model's thread and its emit worker while writes come off
    the runtime's event loop, so the two genuinely run at once in production. The
    residents widen each pass over the registry and the short switch interval
    lands the writer inside one, which is what makes the race reproducible here
    rather than merely possible.
    """
    cm, _ = waiting_manager()
    for cid in range(1, 64):
        cm.register(SilentConnection(cid))
    chunk = MediaChunk(bundle=MediaBundle(), fps=30.0, n_frames=1)
    failures: list[Exception] = []
    stop = threading.Event()

    def fan_out_until_stopped() -> None:
        # Held rather than swallowed: the main thread re-raises it below, so a
        # failure keeps the traceback that a thread would otherwise only print.
        try:
            while not stop.is_set():
                cm.broadcast_media(chunk)
        except Exception as error:
            failures.append(error)

    thread = threading.Thread(target=fan_out_until_stopped, name="fan-out")
    previous_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    thread.start()
    try:
        for cid in range(1000, 3000):
            cm.register(SilentConnection(cid))
            cm.drop(ConnId(cid))
    finally:
        stop.set()
        thread.join(timeout=10.0)
        sys.setswitchinterval(previous_interval)

    assert not thread.is_alive()
    if failures:
        raise failures[0]


def test_first_publisher_wins_and_resumes_the_track() -> None:
    cm, _ = waiting_manager()
    a, b = FakeConnection(1), FakeConnection(2)
    cm.register(a)
    cm.register(b)
    assert cm.publish_track(ConnId(1), "camera") is True
    assert a.resumed == ["camera"]
    assert cm.publish_track(ConnId(2), "camera") is False
    assert b.resumed == []


def test_publish_from_unregistered_connection_is_refused() -> None:
    cm, _ = waiting_manager()
    assert cm.publish_track(ConnId(7), "camera") is False


def test_unpublish_releases_the_slot_and_pauses_the_holder() -> None:
    cm, _ = waiting_manager()
    a, b = FakeConnection(1), FakeConnection(2)
    cm.register(a)
    cm.register(b)
    cm.publish_track(ConnId(1), "camera")
    cm.unpublish_track(ConnId(1), "camera")
    assert a.paused == ["camera"]
    assert cm.publish_track(ConnId(2), "camera") is True
    assert b.resumed == ["camera"]


def test_non_owner_cannot_release_a_track() -> None:
    cm, _ = waiting_manager()
    a, b = FakeConnection(1), FakeConnection(2)
    cm.register(a)
    cm.register(b)
    cm.publish_track(ConnId(1), "camera")
    cm.unpublish_track(ConnId(2), "camera")
    assert a.paused == []
    assert cm.publish_track(ConnId(2), "camera") is False


def test_drop_releases_held_tracks() -> None:
    cm, _ = waiting_manager()
    a, b = FakeConnection(1), FakeConnection(2)
    cm.register(a)
    cm.register(b)
    cm.publish_track(ConnId(1), "camera")
    cm.drop(ConnId(1))
    assert cm.publish_track(ConnId(2), "camera") is True


def test_keepalive_does_not_advance_the_session() -> None:
    cm, sm = waiting_manager()
    cm.register(FakeConnection(1))
    seen: list[Transition] = []
    sm.on_transition(seen.append)
    cm.note_keepalive(ConnId(1))
    cm.note_keepalive(ConnId(404))
    assert seen == []
    assert sm.current_state is SessionState.STREAMING


@pytest.mark.asyncio
async def test_close_is_awaitable_on_the_fake() -> None:
    conn = FakeConnection(1)
    await conn.close()
    assert conn.closed is True


@pytest.mark.asyncio
async def test_close_all_closes_every_connection_and_empties_the_registry() -> None:
    cm, _ = waiting_manager()
    a, b = FakeConnection(1), FakeConnection(2)
    cm.register(a)
    cm.register(b)
    cm.publish_track(ConnId(1), "camera")
    await cm.close_all()
    assert a.closed is True
    assert b.closed is True
    assert cm.count == 0
    assert cm.publish_track(ConnId(1), "camera") is False


@pytest.mark.asyncio
async def test_close_all_does_not_drive_the_session_machine() -> None:
    cm, sm = waiting_manager()
    cm.register(FakeConnection(1))
    seen: list[Transition] = []
    sm.on_transition(seen.append)
    await cm.close_all()
    assert seen == []
    assert sm.current_state is SessionState.STREAMING
