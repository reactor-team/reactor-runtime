"""HTTP signalling shared by the WebRTC integration tests.

A test client drives the runtime's WebRTC routes the way a real client does:
it offers, polls for the answer, applies the candidates embedded in it, and
trickles its own. These helpers hold that exchange, and serve the full runtime
over a real runner, so each test module only brings its model and its peer.

Import this after ``pytest.importorskip("reactor_webrtc")``: it needs the
native wheel.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

rw = pytest.importorskip("reactor_webrtc")

from reactor_runtime import ReactorApp  # noqa: E402
from reactor_runtime.core import RuntimeConfig  # noqa: E402
from reactor_runtime.http.server import build_app  # noqa: E402
from reactor_runtime.metrics import RuntimeMetrics  # noqa: E402
from reactor_runtime.runner.runner import Runner  # noqa: E402
from reactor_runtime.transport.webrtc import WebRtcConfig, WebRtcRouter  # noqa: E402
from reactor_runtime.transport.webrtc.peer import libwebrtc_peer_factory  # noqa: E402

PREFIX = "/sessions/00000000-0000-0000-0000-000000000000/transport/webrtc"
TIMEOUT_S = 25.0


@asynccontextmanager
async def serve(model: type[ReactorApp], model_ref: str) -> AsyncIterator[httpx.AsyncClient]:
    """Serve the full HTTP surface over a real runner of *model* and a real libwebrtc peer."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("reactor_runtime.runner.runner.import_model_class", lambda ref: model)
        metrics = RuntimeMetrics(version="0.0.0", model=model_ref)
        runner = Runner(RuntimeConfig(model_ref=model_ref), metrics)
        await runner.start()
        router = WebRtcRouter(
            WebRtcConfig(ping_timeout=0.0, ice_gathering_timeout_ms=4000),
            libwebrtc_peer_factory,
            metrics,
        )
        app = build_app(runner, [router], runner.health, metrics)
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://runtime") as client:
                response = await client.post("/start_session", json={})
                assert response.status_code == 200, response.text
                try:
                    yield client
                finally:
                    # Ending the session closes its connections and their media threads.
                    await client.post("/stop_session", json={})
        finally:
            await runner.stop()


async def new_connection(http: httpx.AsyncClient) -> int:
    """Register a connection and return its id."""
    response = await http.post(f"{PREFIX}/connections")
    assert response.status_code == 201, response.text
    return response.json()["connection_id"]


async def negotiate(
    http: httpx.AsyncClient,
    pc: Any,
    cid: int,
    track_mapping: list[dict[str, str]],
    *,
    method: str = "POST",
) -> None:
    """Offer *pc* on *cid*, apply the answer, and add the candidates it carries.

    ``PUT`` re-offers on a connection that already negotiated, as a reconnect does.
    """
    offer = await pc.create_offer()
    await pc.set_local_description(offer)
    response = await http.request(
        method,
        f"{PREFIX}/connections/{cid}/sdp_params",
        json={"sdp_offer": offer.sdp, "track_mapping": track_mapping},
    )
    assert response.status_code == 202, response.text

    response = await _poll_answer(http, cid)
    assert response.status_code == 200, response.text
    answer = response.json()["sdp_answer"]
    await pc.set_remote_description(rw.SessionDescription("answer", answer))
    for candidate in _ice_from_answer(answer):
        await pc.add_ice_candidate(candidate)


async def trickle(
    http: httpx.AsyncClient, cid: int, candidates: list[Any], stop: asyncio.Event
) -> None:
    """Send the candidates a peer gathers into *candidates* until *stop* is set."""
    while not stop.is_set():
        pending, candidates[:] = candidates[:], []
        if pending:
            await http.post(
                f"{PREFIX}/connections/{cid}/ice_candidates",
                json={
                    "candidates": [
                        {
                            "candidate": c.candidate,
                            "sdp_mid": c.sdp_mid,
                            "sdp_mline_index": c.sdp_mline_index,
                        }
                        for c in pending
                    ]
                },
            )
        await asyncio.sleep(0.05)


async def _poll_answer(http: httpx.AsyncClient, cid: int) -> httpx.Response:
    """Poll the answer route until negotiation finishes, as a client does over HTTP."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + TIMEOUT_S
    response = await http.get(f"{PREFIX}/connections/{cid}/sdp_params")
    while response.status_code == 202 and loop.time() < deadline:
        await asyncio.sleep(0.05)
        response = await http.get(f"{PREFIX}/connections/{cid}/sdp_params")
    return response


def _ice_from_answer(sdp: str) -> list[Any]:
    """Pull the candidates the runtime embedded into its non-trickle answer."""
    candidates: list[Any] = []
    mline_index = -1
    mid: str | None = None
    for line in sdp.replace("\r\n", "\n").split("\n"):
        if line.startswith("m="):
            mline_index += 1
            mid = None
        elif line.startswith("a=mid:"):
            mid = line[len("a=mid:") :].strip()
        elif line.startswith("a=candidate:"):
            candidates.append(
                rw.IceCandidate(
                    candidate=line[len("a=") :].strip(),
                    sdp_mid=mid,
                    sdp_mline_index=max(mline_index, 0),
                )
            )
    return candidates
