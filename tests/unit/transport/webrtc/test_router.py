from collections.abc import Callable
from typing import Any

from conftest import FakePeer
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reactor_runtime.core import Connection, ConnId, InputFrame
from reactor_runtime.transport import SessionNotRunningError
from reactor_runtime.transport.webrtc import WebRtcConfig, WebRtcPeerFactory, WebRtcRouter

_PREFIX = "/sessions/s1/transport/webrtc"


class FakeRunner:
    """A SessionControl that mints ids and records the facts pushed up to it."""

    def __init__(self, *, running: bool = True) -> None:
        self._running = running
        self._next = 5000
        self.opened: list[ConnId] = []
        self.closed: list[ConnId] = []

    def require_session_running(self) -> None:
        if not self._running:
            raise SessionNotRunningError

    def new_conn_id(self) -> ConnId:
        self._next += 1
        return ConnId(self._next)

    def track_map(self) -> dict[str, Any]:
        return {"tracks": [{"name": "main_video", "kind": "video"}]}

    def connection_opened(self, conn: Connection) -> None:
        self.opened.append(conn.id)

    def connection_closed(self, conn_id: ConnId) -> None:
        self.closed.append(conn_id)

    def message_received(self, conn_id: ConnId, payload: bytes) -> None:
        pass

    def media_received(self, conn_id: ConnId, track: str, frame: InputFrame) -> None:
        pass

    def keepalive(self, conn_id: ConnId) -> None:
        pass


def _client(
    runner: FakeRunner,
    peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
) -> TestClient:
    app = FastAPI()
    WebRtcRouter(WebRtcConfig(ping_timeout=0.0), factory_for(peer)).mount(app, runner)
    return TestClient(app)


def test_register_mints_id_and_returns_track_map(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
) -> None:
    client = _client(FakeRunner(), fake_peer, factory_for)
    response = client.post(f"{_PREFIX}/connections")
    assert response.status_code == 200
    body = response.json()
    assert body["connection_id"] == 5001
    assert body["track_map"] == {"tracks": [{"name": "main_video", "kind": "video"}]}


def test_offer_returns_sdp_answer(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
) -> None:
    client = _client(FakeRunner(), fake_peer, factory_for)
    response = client.post(
        f"{_PREFIX}/connections/5001/sdp_params",
        json={
            "sdp_offer": "the-offer",
            "track_mapping": [
                {"mid": "0", "name": "main_video", "kind": "video", "direction": "recvonly"}
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["sdp_answer"] == "answer-sdp"
    assert body["connection_id"] == 5001


def test_ice_candidates_reach_the_connection(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
) -> None:
    client = _client(FakeRunner(), fake_peer, factory_for)
    client.post(
        f"{_PREFIX}/connections/5001/sdp_params",
        json={"sdp_offer": "the-offer", "track_mapping": []},
    )
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
