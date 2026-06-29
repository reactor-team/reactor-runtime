import time
from collections.abc import Callable, Mapping
from typing import Any

from conftest import FakePeer
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reactor_runtime.core import Connection, ConnId, InputFrame
from reactor_runtime.protocol import Channel, ProtocolVersion
from reactor_runtime.transport import SessionNotRunningError, UnknownSessionError
from reactor_runtime.transport.webrtc import WebRtcConfig, WebRtcPeerFactory, WebRtcRouter
from reactor_runtime.transport.webrtc.config import IceServer

_SID = "s1"
_PREFIX = f"/sessions/{_SID}/transport/webrtc"


class FakeRunner:
    """A SessionControl that mints ids and records the facts pushed up to it."""

    def __init__(self, *, running: bool = True, session_id: str = _SID) -> None:
        self._running = running
        self._session_id = session_id
        self._next = 5000
        self.opened: list[ConnId] = []
        self.closed: list[ConnId] = []
        self.answered: list[tuple[ConnId, dict[str, str]]] = []

    def require_session_running(self, sid: str) -> None:
        if not self._running:
            raise SessionNotRunningError
        if sid != self._session_id:
            raise UnknownSessionError

    def new_conn_id(self) -> ConnId:
        self._next += 1
        return ConnId(self._next)

    def track_map(self) -> dict[str, Any]:
        return {"tracks": [{"name": "main_video", "kind": "video"}]}

    def connection_opened(self, conn: Connection) -> None:
        self.opened.append(conn.id)

    def connection_closed(self, conn_id: ConnId) -> None:
        self.closed.append(conn_id)

    def message_received(
        self, conn_id: ConnId, payload: bytes | str, version: ProtocolVersion, channel: Channel
    ) -> None:
        pass

    def media_received(self, conn_id: ConnId, track: str, frame: InputFrame) -> None:
        pass

    def keepalive(self, conn_id: ConnId) -> None:
        pass

    def resume_track(self, conn_id: ConnId, name: str) -> None:
        pass

    def pause_track(self, conn_id: ConnId, name: str) -> None:
        pass

    def publish_requested(self, conn_id: ConnId, name: str, request_id: str) -> None:
        pass

    def unpublish_track(self, conn_id: ConnId, name: str) -> None:
        pass

    def file_uploaded(self, conn_id: ConnId, upload_id: str) -> None:
        pass

    def schema_requested(self, conn_id: ConnId, request_id: str) -> None:
        pass

    def clip_requested(self, conn_id: ConnId, duration_seconds: float, request_id: str) -> None:
        pass

    def recording_requested(self, conn_id: ConnId, request_id: str) -> None:
        pass

    def connection_answered(self, conn_id: ConnId, answer: Mapping[str, str]) -> None:
        self.answered.append((conn_id, dict(answer)))


def _client(
    runner: FakeRunner,
    peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    config: WebRtcConfig | None = None,
) -> TestClient:
    app = FastAPI()
    WebRtcRouter(config or WebRtcConfig(ping_timeout=0.0), factory_for(peer)).mount(app, runner)
    return TestClient(app)


def _poll_answer(client: TestClient, cid: int, attempts: int = 50) -> Any:
    """Poll the answer route the way the client does, until it stops pending."""
    response = client.get(f"{_PREFIX}/connections/{cid}/sdp_params")
    for _ in range(attempts):
        if response.status_code == 200:
            break
        time.sleep(0.01)
        response = client.get(f"{_PREFIX}/connections/{cid}/sdp_params")
    return response


def test_register_mints_id_and_returns_track_map(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
) -> None:
    client = _client(FakeRunner(), fake_peer, factory_for)
    response = client.post(f"{_PREFIX}/connections")
    assert response.status_code == 201
    body = response.json()
    assert body["connection_id"] == 5001
    assert body["track_map"] == {"tracks": [{"name": "main_video", "kind": "video"}]}


def test_ice_servers_render_configured_servers(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
) -> None:
    config = WebRtcConfig(
        ping_timeout=0.0,
        ice_servers=(
            IceServer(urls=("stun:stun.example:3478",)),
            IceServer(urls=("turn:turn.example:3478",), username="u", credential="p"),
        ),
    )
    client = _client(FakeRunner(), fake_peer, factory_for, config=config)
    response = client.get(f"{_PREFIX}/ice_servers")
    assert response.status_code == 200
    assert response.json() == {
        "ice_servers": [
            {"uris": ["stun:stun.example:3478"]},
            {
                "uris": ["turn:turn.example:3478"],
                "credentials": {"username": "u", "password": "p"},
            },
        ]
    }


def test_offer_ice_servers_reach_the_peer_config(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
) -> None:
    with _client(FakeRunner(), fake_peer, factory_for) as client:
        accepted = client.post(
            f"{_PREFIX}/connections/5001/sdp_params",
            json={
                "sdp_offer": "the-offer",
                "ice_servers": [
                    {
                        "uris": ["turn:turn.example:3478"],
                        "credentials": {"username": "u", "password": "p"},
                    },
                    {"uris": ["stun:stun.example:3478"]},
                ],
            },
        )
        assert accepted.status_code == 202
        _poll_answer(client, 5001)
    assert fake_peer.last_config is not None
    assert fake_peer.last_config.ice_servers == (
        IceServer(urls=("turn:turn.example:3478",), username="u", credential="p"),
        IceServer(urls=("stun:stun.example:3478",)),
    )


def test_offer_without_ice_servers_uses_configured_servers(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
) -> None:
    config = WebRtcConfig(ping_timeout=0.0, ice_servers=(IceServer(urls=("stun:base:3478",)),))
    with _client(FakeRunner(), fake_peer, factory_for, config=config) as client:
        accepted = client.post(
            f"{_PREFIX}/connections/5001/sdp_params",
            json={"sdp_offer": "the-offer"},
        )
        assert accepted.status_code == 202
        _poll_answer(client, 5001)
    assert fake_peer.last_config is not None
    assert fake_peer.last_config.ice_servers == (IceServer(urls=("stun:base:3478",)),)


def test_offer_is_accepted_then_answer_is_polled(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
) -> None:
    with _client(FakeRunner(), fake_peer, factory_for) as client:
        accepted = client.post(
            f"{_PREFIX}/connections/5001/sdp_params",
            json={
                "sdp_offer": "the-offer",
                "track_mapping": [
                    {"mid": "0", "name": "main_video", "kind": "video", "direction": "recvonly"}
                ],
            },
        )
        assert accepted.status_code == 202
        assert accepted.json() == {"connection_id": 5001}
        answer = _poll_answer(client, 5001)
    assert answer.status_code == 200
    assert answer.json() == {"sdp_answer": "answer-sdp", "connection_id": 5001}
    # The codec the router mapped from the (absent) WebRTC version reached the
    # peer it built for the connection.
    assert fake_peer.protocol_version is ProtocolVersion.V0


def test_reconnect_reoffers_on_the_same_connection_with_put(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
) -> None:
    # A dropped client reconnects by PUT-ing a fresh offer to the same id,
    # without re-registering; the acceptor renegotiates that id.
    with _client(FakeRunner(), fake_peer, factory_for) as client:
        reconnected = client.put(
            f"{_PREFIX}/connections/5001/sdp_params",
            json={"sdp_offer": "new-offer", "track_mapping": []},
        )
        assert reconnected.status_code == 202
        assert reconnected.json() == {"connection_id": 5001}
        answer = _poll_answer(client, 5001)
    assert answer.status_code == 200
    assert answer.json() == {"sdp_answer": "answer-sdp", "connection_id": 5001}


def test_ice_candidates_reach_the_connection(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
) -> None:
    with _client(FakeRunner(), fake_peer, factory_for) as client:
        client.post(
            f"{_PREFIX}/connections/5001/sdp_params",
            json={"sdp_offer": "the-offer", "track_mapping": []},
        )
        # Drain the answer so negotiation has completed and the connection is
        # registered, then the candidate is delivered live.
        _poll_answer(client, 5001)
        response = client.post(
            f"{_PREFIX}/connections/5001/ice_candidates",
            json={"candidates": [{"candidate": "cand", "sdp_mid": "0", "sdp_mline_index": 0}]},
        )
    assert response.status_code == 202
    assert [c.candidate for c in fake_peer.ice] == ["cand"]


def test_routes_reject_when_no_session_running(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
) -> None:
    client = _client(FakeRunner(running=False), fake_peer, factory_for)
    response = client.post(f"{_PREFIX}/connections")
    assert response.status_code == 400


def test_routes_reject_unknown_session_id(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
) -> None:
    client = _client(FakeRunner(session_id="other"), fake_peer, factory_for)
    response = client.post(f"{_PREFIX}/connections")
    assert response.status_code == 404
