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
from reactor_runtime.protocol import Channel, ProtocolVersion
from reactor_runtime.runner import ConnectionManager, SessionStateMachine


class FakeConnection:
    """A shape-conforming stand-in that records what the manager calls on it."""

    def __init__(
        self,
        cid: int,
        *,
        capabilities: ConnectionCapabilities | None = None,
        protocol_version: ProtocolVersion = ProtocolVersion.V0,
    ) -> None:
        self.id = ConnId(cid)
        self.capabilities = capabilities or ConnectionCapabilities(
            carries_video=True, carries_audio=True
        )
        self.protocol_version = protocol_version
        self.messages: list[bytes | str] = []
        self.control: list[bytes | str] = []
        self.media: list[MediaBundle] = []
        self.resumed: list[str] = []
        self.paused: list[str] = []
        self.closed = False

    def send_message(self, payload: bytes | str) -> None:
        self.messages.append(payload)

    def send_control(self, payload: bytes | str) -> None:
        self.control.append(payload)

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
    bundle = MediaBundle()
    cm.broadcast_media(bundle, is_fresh_black=False)
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
