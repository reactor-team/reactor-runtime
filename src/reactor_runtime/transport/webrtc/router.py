"""The WebRTC transport router.

Mounts the ``/sessions/{sid}/transport/webrtc`` route group and owns the
:class:`~reactor_runtime.transport.webrtc.acceptor.WebRTCAcceptor` bound to the
runner. A client reads the ICE servers, registers a connection to mint its id
and learn the track map, posts its SDP offer and polls for the answer (since
producing it can wait on ICE gathering), and trickles ICE candidates over the
connection's life. Every client registers an explicit connection: there is no
implicit default. A client whose transport dropped reconnects by re-offering on
the same connection — a PUT to its ``sdp_params`` — which renegotiates a fresh
peer for that id rather than minting a new one.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import FastAPI, Header, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from reactor_runtime.core import ConnId
from reactor_runtime.metrics import RuntimeMetrics, WebRtcMetrics
from reactor_runtime.transport.router import (
    ConnectionsExhaustedError,
    ErrorDetail,
    SessionControl,
    SessionNotRunningError,
    TooManyConnectionsError,
    TransportRouter,
    UnknownSessionError,
)
from reactor_runtime.transport.webrtc.acceptor import WebRTCAcceptor
from reactor_runtime.transport.webrtc.config import IceCredentials, IceServer, WebRtcConfig
from reactor_runtime.transport.webrtc.peer import WebRtcPeerFactory
from reactor_runtime.transport.webrtc.signaling import IceCandidate, SdpOffer, TrackMap
from reactor_runtime.transport.webrtc.version import protocol_for_transport

_PREFIX = "/sessions/{sid}/transport/webrtc"

# Every route in the group is guarded by require_session_running, whose
# rejections the app-level exception handlers render. Declared bare — statuses
# and the shared error shape — so the published contract carries the codes a
# client branches on.
_GUARD_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorDetail},
    404: {"model": ErrorDetail},
}

# Routes that create a connection add a 503: minting an id when the session's id
# space is used up, or offering past the concurrent-connection ceiling, is a
# transient refusal a client retries once a slot frees.
_CONNECT_RESPONSES: dict[int | str, dict[str, Any]] = {
    **_GUARD_RESPONSES,
    503: {"model": ErrorDetail},
}


class RegisterConnectionResponse(BaseModel):
    """A minted connection id and the model's track manifest for transceiver setup."""

    connection_id: int
    track_map: dict[str, Any]


class OfferAccepted(BaseModel):
    """The offer is accepted for asynchronous negotiation; poll for the answer."""

    connection_id: int


class SdpAnswerResponse(BaseModel):
    """The negotiated SDP answer for a connection."""

    sdp_answer: str
    connection_id: int


class TrackMappingEntry(BaseModel):
    """One track a client declares with its offer."""

    mid: str
    name: str
    kind: str
    direction: str


class TurnCredentials(BaseModel):
    """TURN authentication for an ICE server; absent on a plain STUN server."""

    username: str
    password: str


class IceServerEntry(BaseModel):
    """One STUN/TURN server offered for this connection's candidate gathering.

    Mirrors the wire shape the platform already uses for ICE servers:
    ``{uris, credentials: {username, password}}``, with ``credentials`` absent
    for a STUN server.
    """

    uris: list[str]
    credentials: TurnCredentials | None = None


# RFC 8445 \u00a715.4: ice-char = ALPHA / DIGIT / "+" / "/". Anchored, so a value
# is rejected for containing anything else rather than for merely starting with
# something valid.
_ICE_CHAR = r"^[A-Za-z0-9+/]+$"


class IceCredentialsEntry(BaseModel):
    """The ICE credentials a connection should answer with.

    The constraints are RFC 8445 \u00a715.4: both values are ``ice-char``
    (ALPHA / DIGIT / "+" / "/"), a ufrag is 4..256 of them and a password
    22..256.

    They are enforced here rather than left to the media engine because of where
    each failure surfaces. Registering an offer answers 202 and the negotiation
    runs in the background, so a malformed value rejected downstream reaches the
    caller only as its answer poll timing out, with the reason in the runtime's
    logs. Checked here it is a 422 naming the field.
    """

    ufrag: str = Field(min_length=4, max_length=256, pattern=_ICE_CHAR)
    pwd: str = Field(min_length=22, max_length=256, pattern=_ICE_CHAR)


# A UDP port. Zero is excluded: it asks the kernel for an ephemeral port, which
# is the one thing a caller pinning a range cannot mean.
_Port = Annotated[int, Field(ge=1, le=65535)]


class SdpParamsRequest(BaseModel):
    """A client's SDP offer, the tracks it declares, and optional overrides.

    ``ice_servers`` lets the caller supply the STUN/TURN servers this connection
    gathers against. Absent, the runtime uses its own configured servers; present
    (even empty), it is authoritative for the connection — so a reconnect can
    carry fresh credentials.

    ``ice_credentials`` and ``port_range`` follow the same rule and are likewise
    optional: absent — the usual case — the media engine generates its own
    credentials and the configured port range applies. They exist for a
    deployment that fronts the runtime with a relaying layer, which must know a
    connection's ICE credentials and media address before the connection exists.
    ``port_range`` is an inclusive ``[min, max]``; a single-port range pins the
    connection to one port.
    """

    sdp_offer: str
    track_mapping: list[TrackMappingEntry] = Field(default_factory=list)
    ice_servers: list[IceServerEntry] | None = None
    ice_credentials: IceCredentialsEntry | None = None
    port_range: tuple[_Port, _Port] | None = None

    @model_validator(mode="after")
    def _port_range_is_ordered(self) -> SdpParamsRequest:
        """Reject an inverted range here rather than at gathering.

        ``(50000, 40000)`` is accepted by the type and then fails when the
        engine gathers, which reaches the caller as an answer poll that times
        out. This makes it a 422 that names the field.
        """
        if self.port_range is not None:
            low, high = self.port_range
            if low > high:
                msg = f"port_range min {low} is above max {high}"
                raise ValueError(msg)
        return self


class IceCandidateEntry(BaseModel):
    """One trickle-ICE candidate from a client."""

    candidate: str
    sdp_mid: str | None = None
    sdp_mline_index: int | None = None


class IceCandidatesRequest(BaseModel):
    """A batch of trickle-ICE candidates from a client."""

    candidates: list[IceCandidateEntry] = Field(default_factory=list)
    is_final: bool = False


def _ice_credentials_from_request(
    entry: IceCredentialsEntry | None,
) -> IceCredentials | None:
    """Convert a connect request's ICE credentials to the transport's form.

    ``None`` (the field absent) means the media engine generates its own, which
    is the ordinary case.
    """
    if entry is None:
        return None
    return IceCredentials(ufrag=entry.ufrag, pwd=entry.pwd)


def _ice_servers_from_request(
    entries: list[IceServerEntry] | None,
) -> tuple[IceServer, ...] | None:
    """Convert a connect request's ICE servers to the transport's form.

    ``None`` (the field absent) means "use the runtime's configured servers"; a
    present list — even empty — is authoritative for the connection. Each entry
    mirrors the wire shape ``{uris, credentials: {username, password}}``.
    """
    if entries is None:
        return None
    return tuple(
        IceServer(
            urls=tuple(entry.uris),
            username=entry.credentials.username if entry.credentials else None,
            credential=entry.credentials.password if entry.credentials else None,
        )
        for entry in entries
    )


def _ice_servers_payload(config: WebRtcConfig) -> dict[str, Any]:
    """Render the configured ICE servers as the client's expected JSON shape.

    Each server is ``{"uris": [...]}``, with a ``credentials`` object carrying
    ``username``/``password`` only when the server is an authenticated TURN
    server. An empty list is valid — a local connection needs no STUN/TURN.
    """
    servers: list[dict[str, Any]] = []
    for server in config.ice_servers:
        entry: dict[str, Any] = {"uris": list(server.urls)}
        if server.username is not None and server.credential is not None:
            entry["credentials"] = {"username": server.username, "password": server.credential}
        servers.append(entry)
    return {"ice_servers": servers}


class WebRtcRouter(TransportRouter):
    """Mount the WebRTC routes and drive them through a WebRTC acceptor.

    Constructed with the WebRTC configuration and the peer factory the acceptor
    builds connections with; bound to the runner when mounted.
    """

    def __init__(
        self,
        config: WebRtcConfig,
        peer_factory: WebRtcPeerFactory,
        metrics: RuntimeMetrics,
    ) -> None:
        """Hold the configuration, peer factory, and instruments for the acceptor.

        The handshake instruments are declared here rather than at each mount, so
        the router owns one set of them for the life of the process.
        """
        self._config = config
        self._peer_factory = peer_factory
        self._metrics = WebRtcMetrics(metrics)

    def mount(self, app: FastAPI, runner: SessionControl) -> None:
        """Register the WebRTC route group against *app*, bound to *runner*."""
        acceptor = WebRTCAcceptor(
            sink=runner,
            config=self._config,
            peer_factory=self._peer_factory,
            metrics=self._metrics,
        )

        async def _session_not_running(request: Request, exc: Exception) -> Response:
            return JSONResponse(status_code=400, content={"detail": "No session running"})

        async def _unknown_session(request: Request, exc: Exception) -> Response:
            return JSONResponse(status_code=404, content={"detail": "Unknown session"})

        async def _connections_exhausted(request: Request, exc: Exception) -> Response:
            return JSONResponse(status_code=503, content={"detail": "No connection ids left"})

        async def _too_many_connections(request: Request, exc: Exception) -> Response:
            return JSONResponse(status_code=503, content={"detail": "Connection limit reached"})

        app.add_exception_handler(SessionNotRunningError, _session_not_running)
        app.add_exception_handler(UnknownSessionError, _unknown_session)
        app.add_exception_handler(ConnectionsExhaustedError, _connections_exhausted)
        app.add_exception_handler(TooManyConnectionsError, _too_many_connections)

        @app.get(f"{_PREFIX}/ice_servers", responses=_GUARD_RESPONSES)
        async def ice_servers(sid: str) -> dict[str, Any]:
            runner.require_session_running(sid)
            return _ice_servers_payload(self._config)

        @app.post(f"{_PREFIX}/connections", status_code=201, responses=_CONNECT_RESPONSES)
        async def register(sid: str) -> RegisterConnectionResponse:
            runner.require_session_running(sid)
            return RegisterConnectionResponse(
                connection_id=runner.new_conn_id(), track_map=dict(runner.track_map())
            )

        # One handler, registered once per method: a POST opens a connection's
        # first negotiation and a PUT re-offers on the same id. Separate
        # registrations give each verb its own stable operation id in the
        # OpenAPI document (a multi-method api_route derives one id from an
        # unordered method set — unstable and duplicated).
        async def offer(
            sid: str,
            cid: int,
            req: SdpParamsRequest,
            webrtc_version: Annotated[str | None, Header(alias="reactor-webrtc-version")] = None,
        ) -> OfferAccepted:
            runner.require_session_running(sid)
            tracks = TrackMap.from_client(entry.model_dump() for entry in req.track_mapping)
            conn_id = ConnId(cid)
            runner.offer_admitted(conn_id)
            acceptor.start_offer(
                conn_id,
                SdpOffer(req.sdp_offer),
                tracks,
                protocol_for_transport(webrtc_version),
                ice_servers=_ice_servers_from_request(req.ice_servers),
                ice_credentials=_ice_credentials_from_request(req.ice_credentials),
                port_range=req.port_range,
            )
            return OfferAccepted(connection_id=cid)

        offer_path = f"{_PREFIX}/connections/{{cid}}/sdp_params"
        app.post(offer_path, status_code=202, responses=_CONNECT_RESPONSES)(offer)
        app.put(offer_path, status_code=202, responses=_CONNECT_RESPONSES)(offer)

        @app.get(
            f"{_PREFIX}/connections/{{cid}}/sdp_params",
            responses={
                200: {"model": SdpAnswerResponse},
                202: {"description": "Negotiation still in flight; poll again."},
                **_GUARD_RESPONSES,
            },
        )
        async def sdp_answer(sid: str, cid: int) -> Response:
            runner.require_session_running(sid)
            answer = acceptor.take_answer(ConnId(cid))
            if answer is None:
                return Response(status_code=202)
            payload = SdpAnswerResponse(sdp_answer=answer.sdp, connection_id=cid)
            return JSONResponse(status_code=200, content=payload.model_dump())

        @app.post(
            f"{_PREFIX}/connections/{{cid}}/ice_candidates",
            status_code=202,
            responses=_GUARD_RESPONSES,
        )
        async def ice(sid: str, cid: int, req: IceCandidatesRequest) -> Response:
            runner.require_session_running(sid)
            for entry in req.candidates:
                await acceptor.add_ice(
                    ConnId(cid),
                    IceCandidate(entry.candidate, entry.sdp_mid, entry.sdp_mline_index),
                )
            return Response(status_code=202)
