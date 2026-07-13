"""Lock the WebRTC signalling relay surface under the fixed session id.

A consumer relaying signalling for a client posts SDP offers and trickle-ICE
batches to ``/sessions/{sid}/transport/webrtc/...`` with the fixed all-zero
session id, expects exactly ``202`` for both, and reads the SDP answer off the
``/events`` journal (the ``connection_answered`` self-loop) rather than
polling. Connection ids are minted by the runtime in ``[1002, 9999]``; id
``1001`` is reserved for legacy single-connection callers and must stay
addressable.
"""

from __future__ import annotations

import asyncio

import httpx
from contract_helpers import FIXED_SESSION_ID, Harness, JournalReader

_PREFIX = f"/sessions/{FIXED_SESSION_ID}/transport/webrtc"

_OFFER_BODY = {
    "sdp_offer": "v=0\r\no=- 0 0 IN IP4 0.0.0.0",
    "track_mapping": [
        {"mid": "0", "name": "main", "kind": "video", "direction": "recvonly"},
        {"mid": "1", "name": "webcam", "kind": "video", "direction": "sendonly"},
    ],
    "ice_servers": [
        {"uris": ["stun:stun.example:3478"]},
        {
            "uris": ["turn:turn.example:3478"],
            "credentials": {"username": "user", "password": "secret"},
        },
    ],
}


async def _register(client: httpx.AsyncClient) -> int:
    response = await client.post(f"{_PREFIX}/connections")
    assert response.status_code == 201
    connection_id = response.json()["connection_id"]
    assert isinstance(connection_id, int)
    return connection_id


async def test_routes_require_a_running_session(harness: Harness) -> None:
    response = await harness.client.post(f"{_PREFIX}/connections")

    assert response.status_code == 400


async def test_routes_reject_any_other_session_id(harness: Harness) -> None:
    await harness.client.post("/start_session", json={})

    response = await harness.client.post("/sessions/not-the-fixed-id/transport/webrtc/connections")

    assert response.status_code == 404


async def test_register_mints_an_id_in_the_locked_range(harness: Harness) -> None:
    await harness.client.post("/start_session", json={})

    response = await harness.client.post(f"{_PREFIX}/connections")

    assert response.status_code == 201
    body = response.json()
    assert 1002 <= body["connection_id"] <= 9999
    assert "track_map" in body


async def test_an_offer_is_accepted_with_202(harness: Harness) -> None:
    await harness.client.post("/start_session", json={})
    cid = await _register(harness.client)

    response = await harness.client.post(
        f"{_PREFIX}/connections/{cid}/sdp_params", json=_OFFER_BODY
    )

    assert response.status_code == 202
    assert response.json() == {"connection_id": cid}


async def test_a_reoffer_on_the_same_connection_is_accepted_with_put(harness: Harness) -> None:
    await harness.client.post("/start_session", json={})
    cid = await _register(harness.client)

    response = await harness.client.put(f"{_PREFIX}/connections/{cid}/sdp_params", json=_OFFER_BODY)

    assert response.status_code == 202


async def test_the_reserved_legacy_connection_id_is_addressable(harness: Harness) -> None:
    # A caller relaying for a legacy single-connection client addresses
    # connection 1001 without registering it first.
    await harness.client.post("/start_session", json={})

    response = await harness.client.post(f"{_PREFIX}/connections/1001/sdp_params", json=_OFFER_BODY)

    assert response.status_code == 202


async def test_ice_candidates_are_accepted_with_202(harness: Harness) -> None:
    await harness.client.post("/start_session", json={})
    cid = await _register(harness.client)
    await harness.client.post(f"{_PREFIX}/connections/{cid}/sdp_params", json=_OFFER_BODY)
    await _drain_answer(harness, cid)

    response = await harness.client.post(
        f"{_PREFIX}/connections/{cid}/ice_candidates",
        json={
            "candidates": [
                {"candidate": "candidate:0 1 UDP", "sdp_mid": "0", "sdp_mline_index": 0}
            ],
            "is_final": True,
        },
    )

    assert response.status_code == 202


async def test_the_answer_rides_the_journal_as_a_connection_answered_self_loop(
    harness: Harness,
) -> None:
    journal = JournalReader(harness.runner)
    try:
        await harness.client.post("/start_session", json={})
        cid = await _register(harness.client)
        await harness.client.post(f"{_PREFIX}/connections/{cid}/sdp_params", json=_OFFER_BODY)

        answered = await journal.expect("connection_answered")
        assert (answered["from"], answered["to"]) == ("waiting", "waiting")
        assert answered["detail"] == {
            "conn_id": cid,
            "answer": {"type": "answer", "sdp": "answer-sdp"},
        }
    finally:
        await journal.aclose()


async def test_the_wire_connecting_and_dropping_drives_the_occupancy_edges(
    harness: Harness,
) -> None:
    journal = JournalReader(harness.runner)
    try:
        await harness.client.post("/start_session", json={})
        cid = await _register(harness.client)
        await harness.client.post(f"{_PREFIX}/connections/{cid}/sdp_params", json=_OFFER_BODY)
        await journal.expect("connection_answered")

        harness.peer.fire_connected()
        opened = await journal.expect("connection_opened")
        assert (opened["from"], opened["to"]) == ("waiting", "streaming")
        assert opened["detail"] == {"conn_id": cid}

        harness.peer.fire_disconnect()
        closed = await journal.expect("connection_closed")
        assert (closed["from"], closed["to"]) == ("streaming", "orphaned")
        assert closed["detail"] == {"conn_id": cid}
    finally:
        await journal.aclose()


async def _drain_answer(harness: Harness, cid: int) -> None:
    """Poll the answer route until negotiation completes, so ICE lands live."""
    async with asyncio.timeout(2.0):
        while True:
            response = await harness.client.get(f"{_PREFIX}/connections/{cid}/sdp_params")
            if response.status_code == 200:
                assert set(response.json()) == {"sdp_answer", "connection_id"}
                return
            assert response.status_code == 202
            await asyncio.sleep(0.01)
