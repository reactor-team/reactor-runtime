import pytest

from reactor_runtime.core import (
    Connection,
    ConnectionCapabilities,
    ConnId,
    MediaBundle,
    SessionEvent,
    SessionState,
    Transition,
)
from reactor_runtime.runner import ConnectionManager, SessionStateMachine


class FakeConnection:
    """A shape-conforming stand-in that records what the manager calls on it."""

    def __init__(self, cid: int, *, capabilities: ConnectionCapabilities | None = None) -> None:
        self.id = ConnId(cid)
        self.capabilities = capabilities or ConnectionCapabilities(
            carries_video=True, carries_audio=True
        )
        self.messages: list[bytes | str] = []
        self.media: list[MediaBundle] = []
        self.resumed: list[str] = []
        self.paused: list[str] = []
        self.closed = False

    def send_message(self, payload: bytes | str) -> None:
        self.messages.append(payload)

    def send_media(self, bundle: MediaBundle) -> None:
        self.media.append(bundle)

    def resume_track(self, name: str) -> None:
        self.resumed.append(name)

    def pause_track(self, name: str) -> None:
        self.paused.append(name)

    async def close(self) -> None:
        self.closed = True


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
        SessionEvent.CLIENT_CONNECTED,
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


def test_close_precedes_disconnect_on_the_last_drop() -> None:
    cm, sm = waiting_manager()
    cm.register(FakeConnection(1))
    seen: list[SessionEvent] = []
    sm.on_transition(lambda t: seen.append(t.event))
    cm.drop(ConnId(1))
    assert seen == [SessionEvent.CONNECTION_CLOSED, SessionEvent.CLIENT_DISCONNECTED]


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
    cm.broadcast(b"hi")
    assert replacement.messages == [b"hi"]
    assert first.messages == []


def test_broadcast_reaches_every_connection() -> None:
    cm, _ = waiting_manager()
    a, b = FakeConnection(1), FakeConnection(2)
    cm.register(a)
    cm.register(b)
    cm.broadcast(b"payload")
    cm.broadcast('{"type":"text-frame"}')
    assert a.messages == [b"payload", '{"type":"text-frame"}']
    assert b.messages == [b"payload", '{"type":"text-frame"}']


def test_addressed_send_reaches_only_the_target() -> None:
    cm, _ = waiting_manager()
    a, b = FakeConnection(1), FakeConnection(2)
    cm.register(a)
    cm.register(b)
    cm.send(ConnId(2), b"only-b")
    assert a.messages == []
    assert b.messages == [b"only-b"]


def test_addressed_send_to_unknown_connection_is_ignored() -> None:
    cm, _ = waiting_manager()
    cm.register(FakeConnection(1))
    cm.send(ConnId(42), b"nowhere")


def test_media_skips_data_only_connections() -> None:
    cm, _ = waiting_manager()
    media = FakeConnection(1, capabilities=ConnectionCapabilities(carries_video=True))
    data_only = FakeConnection(
        2, capabilities=ConnectionCapabilities(carries_video=False, carries_audio=False)
    )
    cm.register(media)
    cm.register(data_only)
    bundle = MediaBundle()
    cm.broadcast_media(bundle, duplicate=False)
    assert media.media == [bundle]
    assert data_only.media == []


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
