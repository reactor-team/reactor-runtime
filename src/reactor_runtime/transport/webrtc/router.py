"""The WebRTC transport router.

Mounts the ``/sessions/{sid}/transport/webrtc`` route group and owns the
:class:`~reactor_runtime.transport.webrtc.acceptor.WebRTCAcceptor` bound to the
runner. A client registers a connection to mint its id and learn the track map,
posts its SDP offer to negotiate, and trickles ICE candidates — three routes over
one acceptor. Every client registers an explicit connection: there is no implicit
default.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from reactor_runtime.core import ConnId
from reactor_runtime.transport.router import (
    SessionControl,
    SessionNotRunningError,
    TransportRouter,
)
from reactor_runtime.transport.webrtc.acceptor import WebRTCAcceptor
from reactor_runtime.transport.webrtc.config import WebRtcConfig
from reactor_runtime.transport.webrtc.peer import WebRtcPeerFactory
from reactor_runtime.transport.webrtc.signaling import IceCandidate, SdpOffer, TrackMap

_PREFIX = "/sessions/{sid}/transport/webrtc"


class TrackMappingEntry(BaseModel):
    """One track a client declares with its offer."""

    mid: str
    name: str
    kind: str
    direction: str


class SdpParamsRequest(BaseModel):
    """A client's SDP offer plus the tracks it declares."""

    sdp_offer: str
    track_mapping: list[TrackMappingEntry] = Field(default_factory=list)


class IceCandidateEntry(BaseModel):
    """One trickle-ICE candidate from a client."""

    candidate: str
    sdp_mid: str | None = None
    sdp_mline_index: int | None = None


class IceCandidatesRequest(BaseModel):
    """A batch of trickle-ICE candidates from a client."""

    candidates: list[IceCandidateEntry] = Field(default_factory=list)
    is_final: bool = False


class WebRtcRouter(TransportRouter):
    """Mount the WebRTC routes and drive them through a WebRTC acceptor.

    Constructed with the WebRTC configuration and the peer factory the acceptor
    builds connections with; bound to the runner when mounted.
    """

    def __init__(self, config: WebRtcConfig, peer_factory: WebRtcPeerFactory) -> None:
        """Hold the configuration and peer factory for the acceptor."""
        self._config = config
        self._peer_factory = peer_factory

    def mount(self, app: FastAPI, runner: SessionControl) -> None:
        """Register the WebRTC route group against *app*, bound to *runner*."""
        acceptor = WebRTCAcceptor(sink=runner, config=self._config, peer_factory=self._peer_factory)

        async def _session_not_running(request: Request, exc: Exception) -> Response:
            return JSONResponse(status_code=400, content={"detail": "No session running"})

        app.add_exception_handler(SessionNotRunningError, _session_not_running)

        @app.post(f"{_PREFIX}/connections")
        async def register(sid: str) -> dict[str, Any]:
            runner.require_session_running()
            return {"connection_id": runner.new_conn_id(), "track_map": runner.track_map()}

        @app.post(f"{_PREFIX}/connections/{{cid}}/sdp_params")
        async def offer(sid: str, cid: int, req: SdpParamsRequest) -> dict[str, Any]:
            runner.require_session_running()
            tracks = TrackMap.from_client(entry.model_dump() for entry in req.track_mapping)
            answer = await acceptor.offer(ConnId(cid), SdpOffer(req.sdp_offer), tracks)
            return {"sdp_answer": answer.sdp, "connection_id": cid}

        @app.post(f"{_PREFIX}/connections/{{cid}}/ice_candidates")
        async def ice(sid: str, cid: int, req: IceCandidatesRequest) -> Response:
            runner.require_session_running()
            for entry in req.candidates:
                await acceptor.add_ice(
                    ConnId(cid),
                    IceCandidate(entry.candidate, entry.sdp_mid, entry.sdp_mline_index),
                )
            return Response(status_code=202)
