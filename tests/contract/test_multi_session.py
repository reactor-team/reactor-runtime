"""The HTTP contract of a runtime hosting several sessions in one process.

A batched model declaring ``max_sessions`` is served by a session host: each
session is admitted into a free slot under its own id, every route keyed by a
session id resolves to that session, and one session's end, timeout, or
failure leaves the others running.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, cast

import httpx
import numpy as np
import pytest
from contract_helpers import FakePeer, read_sse
from fastapi import FastAPI

from reactor_runtime import BatchedReactorApp, InputField, InputState, Output, Video
from reactor_runtime.core import RecordingConfig, RuntimeConfig
from reactor_runtime.http.server import build_app
from reactor_runtime.metrics import RuntimeMetrics
from reactor_runtime.protocol import ProtocolVersion
from reactor_runtime.runner.session_host import SessionHost
from reactor_runtime.step_results import SavedStep
from reactor_runtime.transport.webrtc import WebRtcConfig, WebRtcRouter
from reactor_runtime.transport.webrtc.config import WebRtcConfig as _Config
from reactor_runtime.transport.webrtc.peer import WebRtcPeerFactory
from reactor_runtime.transport.webrtc.signaling import SdpAnswer, SdpOffer, TrackMap

pytestmark = pytest.mark.asyncio

_SLOTS = 3

# Session ids are lowercase UUIDs; these name the sessions each test starts.
ALPHA = "00000001-0000-4000-8000-000000000000"
BETA = "00000002-0000-4000-8000-000000000000"
GAMMA = "00000003-0000-4000-8000-000000000000"
GOOD = "00000004-0000-4000-8000-000000000000"
BAD = "00000005-0000-4000-8000-000000000000"
WATCHED = "00000006-0000-4000-8000-000000000000"
ABANDONED = "00000007-0000-4000-8000-000000000000"
NOPE = "00000008-0000-4000-8000-000000000000"


class Picture(Output):
    main_video: Video


class Dial(InputState):
    value: int = InputField(default=0, ge=0, le=255)


class Painter(BatchedReactorApp):
    """Paints each session's dial value; a value of 13 fails that session's step."""

    state: Dial
    max_sessions = _SLOTS

    def generate_batch(self, inputs: dict[str, Dial]) -> Mapping[str, Any]:
        results: dict[str, Any] = {}
        for sid, dial in inputs.items():
            if dial.value == 13:
                results[sid] = ValueError("unlucky dial")
            else:
                results[sid] = Picture(main_video=np.full((2, 2, 3), dial.value, dtype=np.uint8))
        return results


@dataclass
class Host:
    client: httpx.AsyncClient
    app: FastAPI
    host: SessionHost
    peers: dict[str, FakePeer]


def _peers(peers: dict[str, FakePeer]) -> WebRtcPeerFactory:
    """Negotiate each offer instantly, keeping its peer under the offer's SDP."""

    async def factory(
        conn_id: int,
        offer: SdpOffer,
        tracks: TrackMap,
        config: _Config,
        version: ProtocolVersion,
    ) -> tuple[FakePeer, SdpAnswer]:
        peer = FakePeer()
        peer.protocol_version = version
        peers[offer.sdp] = peer
        return peer, SdpAnswer(f"answer-to-{offer.sdp}")

    return cast(WebRtcPeerFactory, factory)


@asynccontextmanager
async def hosting(cfg: RuntimeConfig | None = None) -> AsyncIterator[Host]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("reactor_runtime.runner.session_host.import_model_class", lambda ref: Painter)
        metrics = RuntimeMetrics(version="0.0.0", model="multi:Painter")
        host = SessionHost(cfg or RuntimeConfig(model_ref="multi:Painter"), metrics, _SLOTS)
        await host.start()
        peers: dict[str, FakePeer] = {}
        router = WebRtcRouter(WebRtcConfig(ping_timeout=0.0), _peers(peers), metrics)
        app = build_app(host, [router], host.health, metrics)
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://runtime"
            ) as client:
                yield Host(client=client, app=app, host=host, peers=peers)
        finally:
            await host.drain()
            await host.stop()


async def _start(client: httpx.AsyncClient, **body: Any) -> httpx.Response:
    return await client.post("/start_session", json=body)


async def _until(predicate: Callable[[], Any]) -> None:
    async with asyncio.timeout(5.0):
        while True:
            if predicate():
                return
            await asyncio.sleep(0.01)


def _states(host: SessionHost) -> dict[str, str]:
    return {descriptor["session_id"]: descriptor["state"] for descriptor in host.sessions()}


async def test_health_reports_capacity_and_turns_serving_when_full() -> None:
    async with hosting() as rt:
        body = (await rt.client.get("/health")).json()
        assert body["state"] == "available"
        assert body["capacity"] == {"slots": _SLOTS, "free": _SLOTS}

        for _ in range(_SLOTS - 1):
            assert (await _start(rt.client)).status_code == 200
        body = (await rt.client.get("/health")).json()
        assert body["state"] == "available"
        assert body["capacity"] == {"slots": _SLOTS, "free": 1}

        assert (await _start(rt.client)).status_code == 200
        body = (await rt.client.get("/health")).json()
        assert body["state"] == "serving"
        assert body["capacity"] == {"slots": _SLOTS, "free": 0}


async def test_a_start_with_every_slot_serving_is_the_busy_conflict() -> None:
    async with hosting() as rt:
        for _ in range(_SLOTS):
            await _start(rt.client)

        response = await _start(rt.client)

        assert response.status_code == 409
        assert (
            response.json()["detail"]
            == f"cannot start session: all {_SLOTS} session slots are serving"
        )


async def test_a_draining_host_refuses_starts_and_reports_no_free_slot() -> None:
    async with hosting() as rt:
        await _start(rt.client, session_id=ALPHA)

        await rt.host.drain()

        assert _states(rt.host) == {}
        response = await _start(rt.client)
        assert response.status_code == 409
        assert response.json()["detail"] == "cannot start session while closing"
        body = (await rt.client.get("/health")).json()
        assert body["state"] == "serving"
        assert body["capacity"] == {"slots": _SLOTS, "free": 0}


async def test_a_start_takes_the_callers_id_or_mints_one() -> None:
    async with hosting() as rt:
        chosen = (await _start(rt.client, session_id=ALPHA)).json()
        minted = (await _start(rt.client)).json()

        assert chosen["session_id"] == ALPHA
        assert minted["session_id"] not in (ALPHA, "00000000-0000-0000-0000-000000000000")
        assert set(_states(rt.host)) == {ALPHA, minted["session_id"]}


@pytest.mark.parametrize(
    "session_id",
    [
        "alpha",
        "../etc",
        "a/b",
        "x" * 5000,
        5,
        ["list"],
        "0000000A-0000-4000-8000-00000000000B",
        "{" + ALPHA + "}",
    ],
)
async def test_a_start_refuses_an_id_that_is_not_a_lowercase_uuid(session_id: Any) -> None:
    async with hosting() as rt:
        response = await _start(rt.client, session_id=session_id)

        assert response.status_code == 400
        assert "session_id" in response.json()["detail"]
        assert rt.host.capacity() == (_SLOTS, _SLOTS)


async def test_starting_a_running_id_again_conflicts() -> None:
    async with hosting() as rt:
        await _start(rt.client, session_id=ALPHA)

        response = await _start(rt.client, session_id=ALPHA)

        assert response.status_code == 409
        assert rt.host.capacity() == (_SLOTS, _SLOTS - 1)


async def test_session_reads_and_stops_are_addressed_by_id() -> None:
    async with hosting() as rt:
        await _start(rt.client, session_id=ALPHA)
        await _start(rt.client, session_id=BETA)

        assert (await rt.client.get("/session")).status_code == 400
        assert (await rt.client.get("/session", params={"session_id": NOPE})).status_code == 404
        beta = (await rt.client.get("/session", params={"session_id": BETA})).json()
        assert beta["session_id"] == BETA
        listed = (await rt.client.get("/sessions")).json()
        assert listed["capacity"] == {"slots": _SLOTS, "free": _SLOTS - 2}
        assert {entry["session_id"] for entry in listed["sessions"]} == {ALPHA, BETA}

        assert (await rt.client.post("/stop_session", json={})).status_code == 400
        assert (await rt.client.post("/stop_session", json={"session_id": NOPE})).status_code == 404
        assert (
            await rt.client.post("/stop_session", json={"session_id": ALPHA})
        ).status_code == 200
        await _until(lambda: set(_states(rt.host)) == {BETA})
        await _until(lambda: rt.host.capacity() == (_SLOTS, _SLOTS - 1))


async def test_the_shared_journal_keeps_a_runtimes_worth_of_history_per_slot() -> None:
    async with hosting() as rt:
        assert rt.host.events._history.maxlen == 4096 * _SLOTS
        assert rt.host.events._subscriber_limit == 4096 * _SLOTS


async def test_a_closing_session_is_listed_as_it_is_described() -> None:
    async with hosting() as rt:
        await _start(rt.client, session_id=ALPHA)
        await _start(rt.client, session_id=BETA)

        rt.host.stop_session(session_id=ALPHA)

        assert _states(rt.host) == {ALPHA: "closing", BETA: "waiting"}
        described = rt.host.descriptor(ALPHA)
        assert described["state"] == "closing"


async def test_transport_routes_resolve_the_addressed_session() -> None:
    async with hosting() as rt:
        await _start(rt.client, session_id=ALPHA)
        await _start(rt.client, session_id=BETA)
        prefix = "/sessions/{}/transport/webrtc/connections"

        assert (await rt.client.post(prefix.format(NOPE))).status_code == 404
        alpha = (await rt.client.post(prefix.format(ALPHA))).json()["connection_id"]
        beta = (await rt.client.post(prefix.format(BETA))).json()["connection_id"]
        assert alpha != beta
        for sid, cid in ((ALPHA, alpha), (BETA, beta)):
            offer = {"sdp_offer": f"offer-{sid}", "track_mapping": []}
            response = await rt.client.post(f"{prefix.format(sid)}/{cid}/sdp_params", json=offer)
            assert response.status_code == 202
        await _until(lambda: f"offer-{ALPHA}" in rt.peers)
        rt.peers[f"offer-{ALPHA}"].fire_connected()

        await _until(lambda: _states(rt.host).get(ALPHA) == "streaming")
        assert _states(rt.host)[BETA] == "waiting"


async def test_each_session_gets_its_own_steps_and_one_stopping_leaves_the_others() -> None:
    async with hosting() as rt:
        for sid, value in ((ALPHA, 1), (BETA, 2), (GAMMA, 3)):
            response = await _start(
                rt.client,
                session_id=sid,
                steps=10_000,
                starting_input={"state": {"value": value}},
            )
            assert response.status_code == 200
        counts: dict[str, int] = {}

        async def follow() -> None:
            async for _, event in rt.host.events.subscribe(0):
                if event.transition.event.name == "STEP_COMPLETED":
                    assert event.session_id is not None
                    counts[event.session_id] = counts.get(event.session_id, 0) + 1

        reader = asyncio.create_task(follow())
        try:
            await _until(lambda: all(counts.get(sid, 0) >= 3 for sid in (ALPHA, BETA, GAMMA)))
            await rt.client.post("/stop_session", json={"session_id": BETA})
            await _until(lambda: BETA not in _states(rt.host))
            before = dict(counts)
            await _until(
                lambda: counts[ALPHA] > before[ALPHA] + 3 and counts[GAMMA] > before[GAMMA] + 3
            )
            assert _states(rt.host) == {ALPHA: "streaming", GAMMA: "streaming"}
        finally:
            reader.cancel()


async def test_a_session_whose_step_fails_ends_alone_with_an_error() -> None:
    async with hosting() as rt:
        await _start(rt.client, session_id=GOOD, steps=10_000)
        await _start(
            rt.client,
            session_id=BAD,
            steps=10_000,
            starting_input={"state": {"value": 13}},
        )

        await _until(lambda: BAD not in _states(rt.host))
        assert _states(rt.host) == {GOOD: "streaming"}
        assert rt.host.health().status.value == "healthy"
        ended = [
            event
            for _, event in list(rt.host.events._history)
            if event.session_id == BAD and event.transition.event.name == "STOP_SESSION"
        ]
        assert ended
        assert ended[0].transition.detail["close_reason"] == "unlucky dial"


async def test_an_orphaned_session_times_out_alone() -> None:
    async with hosting(RuntimeConfig(model_ref="multi:Painter", orphan_timeout=0.2)) as rt:
        await _start(rt.client, session_id=WATCHED, steps=10_000)
        await _start(rt.client, session_id=ABANDONED)

        await _until(lambda: ABANDONED not in _states(rt.host))
        assert _states(rt.host) == {WATCHED: "streaming"}


async def test_every_event_names_its_session_and_a_session_stream_filters_to_it() -> None:
    async with hosting() as rt:
        await _start(rt.client, session_id=ALPHA)
        await _start(rt.client, session_id=BETA)
        await rt.client.post("/stop_session", json={"session_id": BETA})
        await _until(lambda: BETA not in _states(rt.host))

        everything = await read_sse(rt.app, "/events?since=0", count=8)
        beta_only = await read_sse(rt.app, f"/sessions/{BETA}/events?since=0", count=3)

        assert all("session_id" in frame.payload for frame in everything)
        named = {frame.payload["session_id"] for frame in everything}
        assert {ALPHA, BETA} <= named
        assert [frame.payload["session_id"] for frame in beta_only] == [BETA] * 3
        assert [frame.payload["event"] for frame in beta_only] == [
            "start_session",
            "stop_session",
            "cleanup_complete",
        ]


async def test_uploads_land_in_the_addressed_session() -> None:
    async with hosting() as rt:
        await _start(rt.client, session_id=ALPHA)
        await _start(rt.client, session_id=BETA)

        reserved = await rt.client.post(
            f"/sessions/{BETA}/uploads",
            json={"name": "a.bin", "size": 3, "mime_type": "application/octet-stream"},
        )
        upload_id = reserved.json()["presigned_id"]
        written = await rt.client.put(f"/uploads/{upload_id}", content=b"abc")

        assert written.status_code == 200
        assert rt.host.uploads_for(BETA).expected_size is not None
        assert (await rt.host.uploads_for(BETA).fetch(upload_id, wait_seconds=0)).data == b"abc"
        assert (
            await rt.client.post(
                f"/sessions/{NOPE}/uploads",
                json={"name": "a.bin", "size": 3, "mime_type": "application/octet-stream"},
            )
        ).status_code == 404


async def test_sessions_on_two_slots_connect_on_the_same_unregistered_connection_id() -> None:
    """The relaying caller's sequence: no registration, the same id in two sessions.

    A caller that mints connection ids itself offers straight to
    ``/connections/{cid}/sdp_params`` and trickles ICE there, and its ids are
    only unique within a session. Each session must get its own answer, its
    own open wire, and its own close, all journaled under the id it used.
    """
    first, second = str(uuid.uuid4()), str(uuid.uuid4())
    async with hosting() as rt:
        for sid in (first, second):
            assert (await _start(rt.client, session_id=sid)).status_code == 200
        for sid in (first, second):
            path = f"/sessions/{sid}/transport/webrtc/connections/1001"
            offer = {"sdp_offer": f"offer-{sid}", "track_mapping": []}
            assert (await rt.client.post(f"{path}/sdp_params", json=offer)).status_code == 202
            ice = {"candidates": [{"candidate": f"candidate-{sid}"}], "is_final": True}
            assert (await rt.client.post(f"{path}/ice_candidates", json=ice)).status_code == 202

        await _until(lambda: {f"offer-{first}", f"offer-{second}"} <= set(rt.peers))
        for sid in (first, second):
            peer = rt.peers[f"offer-{sid}"]
            await _until(lambda peer=peer: len(peer.ice) == 1)
            assert peer.ice[0].candidate == f"candidate-{sid}"
            peer.fire_connected()
        await _until(lambda: _states(rt.host) == {first: "streaming", second: "streaming"})
        rt.peers[f"offer-{first}"].fire_disconnect()
        await _until(lambda: _states(rt.host) == {first: "orphaned", second: "streaming"})

        for sid, count in ((first, 4), (second, 3)):
            frames = await read_sse(rt.app, f"/sessions/{sid}/events?since=0", count=count)
            moves = [(frame.payload["event"], frame.payload["detail"]) for frame in frames]
            answer = {"type": "answer", "sdp": f"answer-to-offer-{sid}"}
            assert (
                moves[1:]
                == [
                    ("connection_answered", {"conn_id": 1001, "answer": answer}),
                    ("connection_opened", {"conn_id": 1001}),
                    ("connection_closed", {"conn_id": 1001}),
                ][: count - 1]
            )


async def test_a_callers_upload_id_is_reserved_in_one_session_only() -> None:
    async with hosting() as rt:
        await _start(rt.client, session_id=ALPHA)
        await _start(rt.client, session_id=BETA)
        slot = {"name": "a.bin", "size": 3, "mime_type": "application/octet-stream"}

        first = await rt.client.post(f"/sessions/{ALPHA}/uploads", json={**slot, "upload_id": "u1"})
        second = await rt.client.post(f"/sessions/{BETA}/uploads", json={**slot, "upload_id": "u1"})
        written = await rt.client.put("/uploads/u1", content=b"abc")

        assert first.status_code == 201
        assert second.status_code == 409
        assert written.status_code == 200
        assert (await rt.host.uploads_for(ALPHA).fetch("u1", wait_seconds=0)).data == b"abc"


async def test_an_ended_sessions_recording_stays_reachable_after_its_slot_moves_on() -> None:
    cfg = RuntimeConfig(model_ref="multi:Painter", recording=RecordingConfig(enabled=True))
    async with hosting(cfg) as rt:
        await _start(rt.client, session_id=WATCHED)
        await _start(rt.client, session_id=ALPHA)
        await rt.client.post("/stop_session", json={"session_id": ALPHA})
        await _until(lambda: ALPHA not in _states(rt.host))
        await _start(rt.client, session_id=BETA)

        for sid in (WATCHED, BETA, ALPHA):
            response = await rt.client.get(f"/clips/chunks/{sid}/chunk_99999.m4s")
            assert response.status_code == 404, sid


async def test_a_step_saved_after_its_slot_moves_on_is_journaled_under_its_own_session() -> None:
    async with hosting() as rt:
        await _start(rt.client, session_id=WATCHED)
        await _start(rt.client, session_id=ALPHA)
        await rt.client.post("/stop_session", json={"session_id": ALPHA})
        await _until(lambda: ALPHA not in _states(rt.host))
        await _start(rt.client, session_id=BETA)

        rt.host._on_step_saved(SavedStep(session_id=ALPHA, step=7, files=("result.json",)))

        await _until(
            lambda: any(
                event.transition.event.name == "STEP_RESULT_READY"
                for _, event in rt.host.events._history
            )
        )
        saved = [
            event
            for _, event in rt.host.events._history
            if event.transition.event.name == "STEP_RESULT_READY"
        ]
        assert [(event.session_id, event.transition.detail["step"]) for event in saved] == [
            (ALPHA, 7)
        ]
        assert _states(rt.host) == {WATCHED: "waiting", BETA: "waiting"}
