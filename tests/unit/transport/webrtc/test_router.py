import time
from collections.abc import Callable, Mapping
from typing import Any

import pytest
from conftest import FakePeer
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reactor_runtime.core import ClientStatsBatch, Connection, ConnId, InputFrame
from reactor_runtime.metrics import RuntimeMetrics
from reactor_runtime.protocol import Channel, ProtocolVersion
from reactor_runtime.transport import (
    ConnectionIdClaimedError,
    ConnectionsExhaustedError,
    SessionNotRunningError,
    UnknownSessionError,
)
from reactor_runtime.transport.webrtc import WebRtcConfig, WebRtcPeerFactory, WebRtcRouter
from reactor_runtime.transport.webrtc.config import IceCredentials, IceServer

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
        self.admitted: list[ConnId] = []
        self.claims: dict[ConnId, str] = {}

    def require_session_running(self, sid: str) -> None:
        if not self._running:
            raise SessionNotRunningError
        if sid != self._session_id:
            raise UnknownSessionError

    def new_conn_id(self, transport: str) -> ConnId:
        self._next += 1
        self.claims[ConnId(self._next)] = transport
        return ConnId(self._next)

    def claim_conn_id(self, conn_id: ConnId, transport: str) -> bool:
        holder = self.claims.get(conn_id)
        if holder is None:
            self.claims[conn_id] = transport
            return True
        if holder != transport:
            raise ConnectionIdClaimedError(conn_id)
        return False

    def release_conn_id(self, conn_id: ConnId, transport: str) -> None:
        if self.claims.get(conn_id) == transport:
            del self.claims[conn_id]

    def connection_failed(self, conn_id: ConnId, transport: str, reason: str) -> None:
        pass

    def offer_admitted(self, conn_id: ConnId) -> None:
        self.admitted.append(conn_id)

    def track_map(self) -> dict[str, Any]:
        return {
            "main_video": {"kind": "video", "direction": "out", "rate": 0.0},
            "agent_audio": {"kind": "audio", "direction": "in", "rate": 24_000.0},
        }

    def connection_opened(self, conn: Connection) -> None:
        self.opened.append(conn.id)

    def connection_closed(self, conn_id: ConnId) -> None:
        self.closed.append(conn_id)

    def message_received(
        self, conn_id: ConnId, payload: bytes | str, version: ProtocolVersion, channel: Channel
    ) -> None:
        pass

    def command_received(
        self, conn_id: ConnId, name: str, args: Mapping[str, Any], request_id: str
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

    def client_stats_received(self, conn_id: ConnId, batch: ClientStatsBatch) -> None:
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
    metrics = RuntimeMetrics(version="0.0.0", model="fake:Model")
    WebRtcRouter(config or WebRtcConfig(ping_timeout=0.0), factory_for(peer), metrics).mount(
        app, runner
    )
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
    assert body["track_map"] == FakeRunner().track_map()


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


def test_offer_ice_credentials_and_port_range_reach_the_peer_config(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
) -> None:
    with _client(FakeRunner(), fake_peer, factory_for) as client:
        accepted = client.post(
            f"{_PREFIX}/connections/5001/sdp_params",
            json={
                "sdp_offer": "the-offer",
                "ice_credentials": {
                    "ufrag": "suppliedUfrag01",
                    "pwd": "aSuppliedPasswordOf22Chars",
                },
                "port_range": [51820, 51820],
            },
        )
        assert accepted.status_code == 202
        _poll_answer(client, 5001)
    assert fake_peer.last_config is not None
    assert fake_peer.last_config.ice_credentials == IceCredentials(
        ufrag="suppliedUfrag01", pwd="aSuppliedPasswordOf22Chars"
    )
    assert fake_peer.last_config.port_range == (51820, 51820)


# A well-formed pair, so each case below varies exactly one thing.
_UFRAG = "suppliedUfrag01"
_PWD = "aSuppliedPasswordOf22Chars"


@pytest.mark.parametrize(
    ("field", "value", "why"),
    [
        ("ice_credentials", {"ufrag": "abc", "pwd": _PWD}, "ufrag under 4"),
        ("ice_credentials", {"ufrag": "a" * 257, "pwd": _PWD}, "ufrag over 256"),
        ("ice_credentials", {"ufrag": "supplied Ufrag01", "pwd": _PWD}, "space is not ice-char"),
        ("ice_credentials", {"ufrag": _UFRAG, "pwd": "tooShortPassword"}, "pwd under 22"),
        ("ice_credentials", {"ufrag": _UFRAG, "pwd": "a=SuppliedPasswordOf22Ch"}, "= not ice-char"),
        ("port_range", [50000, 40000], "inverted range"),
        ("port_range", [0, 51820], "port 0 asks for an ephemeral port"),
        ("port_range", [51820, 70000], "port above 65535"),
    ],
)
def test_malformed_overrides_are_refused_at_request_time(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    field: str,
    value: Any,
    why: str,
) -> None:
    """A bad value must be a 422 naming the field, not a 202 and a stalled poll.

    Registering an offer answers 202 and negotiates in the background, so a
    value rejected downstream reaches the caller only as its answer poll timing
    out, with the reason in the runtime's logs. Nothing about that says which
    field was wrong.
    """
    with _client(FakeRunner(), fake_peer, factory_for) as client:
        response = client.post(
            f"{_PREFIX}/connections/5001/sdp_params",
            json={"sdp_offer": "the-offer", field: value},
        )
    assert response.status_code == 422, f"{why} was accepted: {response.status_code}"
    assert field in str(response.json()), f"the 422 does not name {field}: {response.json()}"
    assert fake_peer.last_config is None, "a refused offer must not reach the peer"


def test_a_single_port_range_is_allowed(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
) -> None:
    """min == max pins the connection to one port, which is a documented use."""
    with _client(FakeRunner(), fake_peer, factory_for) as client:
        accepted = client.post(
            f"{_PREFIX}/connections/5001/sdp_params",
            json={"sdp_offer": "the-offer", "port_range": [51820, 51820]},
        )
        assert accepted.status_code == 202
        _poll_answer(client, 5001)
    assert fake_peer.last_config is not None
    assert fake_peer.last_config.port_range == (51820, 51820)


def test_a_pinned_port_another_connection_holds_is_refused(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
) -> None:
    """A collision the caller can act on: a named 409, not a stalled poll.

    A caller pinning a port has no second port to fall back on, so a second
    connection on the same one gathers no host candidate and its negotiation is
    dropped in the background. The refusal has to reach the caller, and with a
    code distinct from the transient 503s — retrying the same port changes
    nothing, pinning another does.
    """
    with _client(FakeRunner(), fake_peer, factory_for) as client:
        accepted = client.post(
            f"{_PREFIX}/connections/5001/sdp_params",
            json={"sdp_offer": "the-offer", "port_range": [51820, 51820]},
        )
        assert accepted.status_code == 202
        _poll_answer(client, 5001)

        refused = client.post(
            f"{_PREFIX}/connections/5002/sdp_params",
            json={"sdp_offer": "the-offer", "port_range": [51820, 51820]},
        )
    assert refused.status_code == 409, refused.json()
    assert "51820" in refused.json()["detail"], refused.json()


def test_register_claims_the_minted_id_for_webrtc(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
) -> None:
    runner = FakeRunner()
    with _client(runner, fake_peer, factory_for) as client:
        cid = client.post(f"{_PREFIX}/connections").json()["connection_id"]
        # Offering on the id the route minted is the connection's own offer.
        accepted = client.post(
            f"{_PREFIX}/connections/{cid}/sdp_params", json={"sdp_offer": "the-offer"}
        )
    assert accepted.status_code == 202
    assert runner.claims == {ConnId(cid): "webrtc"}


@pytest.mark.parametrize("method", ["post", "put"])
def test_an_offer_on_another_transports_id_is_refused(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    method: str,
) -> None:
    """An id another transport holds is a 409 that names it, and nothing is negotiated.

    Without the refusal the WebRTC connection would replace the other
    transport's connection on the same id once its wire came up.
    """
    runner = FakeRunner()
    runner.claims[ConnId(5120)] = "room"
    with _client(runner, fake_peer, factory_for) as client:
        refused = client.request(
            method.upper(),
            f"{_PREFIX}/connections/5120/sdp_params",
            json={"sdp_offer": "the-offer"},
        )
    assert refused.status_code == 409, refused.json()
    assert refused.json()["detail"] == "connection 5120 belongs to another transport"
    assert runner.admitted == []
    assert fake_peer.last_config is None
    assert runner.claims == {ConnId(5120): "room"}


def test_a_put_on_the_connections_own_id_is_a_reconnect(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
) -> None:
    runner = FakeRunner()
    with _client(runner, fake_peer, factory_for) as client:
        first = client.post(f"{_PREFIX}/connections/5001/sdp_params", json={"sdp_offer": "one"})
        again = client.put(f"{_PREFIX}/connections/5001/sdp_params", json={"sdp_offer": "two"})
    assert (first.status_code, again.status_code) == (202, 202)
    assert runner.admitted == [ConnId(5001), ConnId(5001)]


def test_offer_without_ice_credentials_leaves_the_engine_to_generate_them(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
) -> None:
    """The ordinary case: the field is absent and nothing is imposed."""
    config = WebRtcConfig(ping_timeout=0.0, port_range=(40000, 40100))
    with _client(FakeRunner(), fake_peer, factory_for, config=config) as client:
        accepted = client.post(
            f"{_PREFIX}/connections/5001/sdp_params",
            json={"sdp_offer": "the-offer"},
        )
        assert accepted.status_code == 202
        _poll_answer(client, 5001)
    assert fake_peer.last_config is not None
    assert fake_peer.last_config.ice_credentials is None
    # The configured range is untouched by an offer that says nothing about it.
    assert fake_peer.last_config.port_range == (40000, 40100)


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


def test_register_reports_503_when_the_id_pool_is_exhausted(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
) -> None:
    class ExhaustedRunner(FakeRunner):
        def new_conn_id(self, transport: str) -> ConnId:
            raise ConnectionsExhaustedError

    client = _client(ExhaustedRunner(), fake_peer, factory_for)
    response = client.post(f"{_PREFIX}/connections")
    assert response.status_code == 503


def test_offer_past_the_connection_ceiling_reports_503(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
) -> None:
    config = WebRtcConfig(ping_timeout=0.0, max_connections=1)
    with _client(FakeRunner(), fake_peer, factory_for, config=config) as client:
        first = client.post(f"{_PREFIX}/connections/5001/sdp_params", json={"sdp_offer": "one"})
        assert first.status_code == 202
        # A second, distinct connection is past the single-slot ceiling.
        second = client.post(f"{_PREFIX}/connections/6001/sdp_params", json={"sdp_offer": "two"})
    assert second.status_code == 503


def test_an_offers_tracks_carry_the_rates_the_model_declares(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
) -> None:
    mapping = [{"mid": "0", "name": "agent_audio", "kind": "audio", "direction": "sendonly"}]
    with _client(FakeRunner(), fake_peer, factory_for) as client:
        client.post(
            f"{_PREFIX}/connections/5001/sdp_params",
            json={"sdp_offer": "the-offer", "track_mapping": mapping},
        )
        _poll_answer(client, 5001)

    assert fake_peer.last_tracks is not None
    [track] = fake_peer.last_tracks.tracks
    assert (track.info.name, track.info.rate) == ("agent_audio", 24_000.0)


@pytest.mark.parametrize("cid", [0, 1000, 10_000, 2**40])
def test_an_offer_on_an_id_outside_the_addressable_range_is_refused(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    cid: int,
) -> None:
    runner = FakeRunner()
    with _client(runner, fake_peer, factory_for) as client:
        refused = client.post(
            f"{_PREFIX}/connections/{cid}/sdp_params", json={"sdp_offer": "the-offer"}
        )

    assert refused.status_code == 422
    assert runner.claims == {}


def test_an_offer_on_the_legacy_id_is_accepted(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
) -> None:
    with _client(FakeRunner(), fake_peer, factory_for) as client:
        accepted = client.post(
            f"{_PREFIX}/connections/1001/sdp_params", json={"sdp_offer": "the-offer"}
        )

    assert accepted.status_code == 202


def test_a_refused_offer_releases_the_id_it_claimed(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
) -> None:
    runner = FakeRunner()
    config = WebRtcConfig(ping_timeout=0.0, max_connections=1)
    with _client(runner, fake_peer, factory_for, config=config) as client:
        client.post(f"{_PREFIX}/connections/5001/sdp_params", json={"sdp_offer": "one"})
        refused = client.post(f"{_PREFIX}/connections/5002/sdp_params", json={"sdp_offer": "two"})

    assert refused.status_code == 503
    assert runner.claims == {ConnId(5001): "webrtc"}


def test_a_refused_reoffer_keeps_the_id_it_already_held(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
) -> None:
    runner = FakeRunner()
    # Claimed by an earlier offer on the same id.
    runner.claims[ConnId(5002)] = "webrtc"
    config = WebRtcConfig(ping_timeout=0.0, max_connections=1)
    with _client(runner, fake_peer, factory_for, config=config) as client:
        client.post(f"{_PREFIX}/connections/5001/sdp_params", json={"sdp_offer": "one"})
        refused = client.post(f"{_PREFIX}/connections/5002/sdp_params", json={"sdp_offer": "two"})

    assert refused.status_code == 503
    assert runner.claims[ConnId(5002)] == "webrtc"
