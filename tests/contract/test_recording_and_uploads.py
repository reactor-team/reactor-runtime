"""Lock the recording mirror and upload seeding surfaces.

A consumer mirroring the recording fetches segments by the names the
``chunk_ready`` journal facts imply — ``init.mp4`` for index ``-1`` and
``chunk_NNNNN.m4s`` (five digits) for media segments — under the recording id
the caller supplied at ``start_session``. Seeding an upload reserves a slot
under a caller-chosen ``upload_id`` and writes the bytes with a ``PUT``; a
``409`` on either step means the slot is already seeded and is not an error.
"""

from __future__ import annotations

import uuid
from io import BytesIO
from pathlib import Path

import av
import httpx
import numpy as np
from contract_helpers import FIXED_SESSION_ID, ContractOutput, Harness, JournalReader

from reactor_runtime.runner.runner import Runner

_RECORDING_ID = "7d9f5c1e-1111-2222-3333-444444444444"


def _seed_recording(runner: Runner, root: Path, *segments: str) -> None:
    """Materialise a recording on disk for the serving routes to read.

    Points the recorder's serving root at *root* — the seam a real session fills
    through the encoder — and writes the named segment files under the
    recording id, so the ``/clips`` surface serves them exactly as it would a
    live recording.
    """
    runner.recorder._root = root
    directory = root / _RECORDING_ID
    directory.mkdir(parents=True, exist_ok=True)
    for name in segments:
        (directory / name).write_bytes(b"segment-bytes")


# -- clips ----------------------------------------------------------------------


async def test_completed_batch_downloads_before_playback_or_session_end(
    harness: Harness, tmp_path: Path
) -> None:
    harness.runner.recorder._root = tmp_path
    response = await harness.client.post("/start_session", json={})
    assert response.status_code == 200
    assert response.json()["recording"]["enabled"] is False
    bridge = harness.runner._bridge
    assert bridge is not None

    clip = await bridge._model.output.save_clip(
        ContractOutput(main=np.zeros((101, 16, 16, 3), dtype=np.uint8)), fps=24
    )

    response = await harness.client.get(clip.playlist_url)
    assert response.status_code == 200
    playlist = response.text
    assert "#EXTINF:0.208333," in playlist
    assert playlist.endswith("#EXT-X-ENDLIST\n")
    assert (await harness.client.get("/session")).json()["state"] == "waiting"
    assert harness.runner.recorder._markers is None
    paths = [
        line.split('"')[1] if line.startswith("#EXT-X-MAP:") else line
        for line in playlist.splitlines()
        if line.startswith("#EXT-X-MAP:") or line.startswith("/clips/chunks/")
    ]
    media = bytearray()
    for path in paths:
        response = await harness.client.get(path)
        assert response.status_code == 200
        media.extend(response.content)
    with av.open(BytesIO(media), mode="r") as container:
        assert sum(1 for _ in container.decode(video=0)) == 101

    assert (await harness.client.post("/stop_session")).status_code == 200
    assert (await harness.client.get(clip.playlist_url)).status_code == 200


async def test_saved_clip_ids_keep_the_owner_and_journal_a_complete_artifact(
    harness: Harness, tmp_path: Path
) -> None:
    harness.runner.recorder._root = tmp_path
    harness.runner.start_session({"session_id": _RECORDING_ID})
    journal = JournalReader(harness.runner)
    bridge = harness.runner._bridge
    assert bridge is not None
    ids = [str(uuid.uuid4()), str(uuid.uuid4())]
    urls = []
    recordings = []
    try:
        for clip_id, frame_count in zip(ids, (4, 9), strict=True):
            url = f"/clips?session_id={_RECORDING_ID}&clip_id={clip_id}"
            assert (await harness.client.get(url)).status_code == 202
            clip = await bridge._model.output.save_clip(
                ContractOutput(main=np.zeros((frame_count, 16, 16, 3), dtype=np.uint8)),
                fps=24,
                clip_id=clip_id,
            )
            assert clip.session_id == _RECORDING_ID
            assert clip.clip_id == clip_id
            assert clip.playlist_url == url
            detail = (await journal.expect("saved_clip_ready"))["detail"]
            assert detail == {
                "session_id": _RECORDING_ID,
                "clip_id": clip_id,
                "recording_id": detail["recording_id"],
                "playlist_url": url,
            }
            recordings.append(detail["recording_id"])
            response = await harness.client.get(url)
            assert response.status_code == 200
            assert response.text.endswith("#EXT-X-ENDLIST\n")
            assert f"/clips/chunks/{detail['recording_id']}/init.mp4" in response.text
            urls.append(url)
        assert len(set(recordings)) == 2
        assert (await harness.client.get(urls[0] + "&start=0")).status_code == 400
        assert (await harness.client.get(urls[0] + "&end=1")).status_code == 400
        assert (
            await harness.client.get(f"/clips?session_id={_RECORDING_ID}&clip_id=bad")
        ).status_code == 400
        harness.runner.stop_session()
        for url in urls:
            assert (await harness.client.get(url)).status_code == 200
    finally:
        await journal.aclose()


async def test_the_init_segment_is_served_at_init_mp4(harness: Harness, tmp_path: Path) -> None:
    _seed_recording(harness.runner, tmp_path, "init.mp4")

    response = await harness.client.get(f"/clips/chunks/{_RECORDING_ID}/init.mp4")

    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"


async def test_media_segments_are_served_at_five_digit_chunk_names(
    harness: Harness, tmp_path: Path
) -> None:
    _seed_recording(harness.runner, tmp_path, "init.mp4", "chunk_00000.m4s")

    response = await harness.client.get(f"/clips/chunks/{_RECORDING_ID}/chunk_00000.m4s")

    assert response.status_code == 200
    assert response.headers["content-type"] == "video/iso.segment"


async def test_non_artifact_names_are_not_served(harness: Harness, tmp_path: Path) -> None:
    _seed_recording(harness.runner, tmp_path, "init.mp4")

    response = await harness.client.get(f"/clips/chunks/{_RECORDING_ID}/secrets.txt")

    assert response.status_code == 404


async def test_a_missing_segment_is_404_and_an_unknown_recording_410(
    harness: Harness, tmp_path: Path
) -> None:
    _seed_recording(harness.runner, tmp_path, "init.mp4")

    missing = await harness.client.get(f"/clips/chunks/{_RECORDING_ID}/chunk_00007.m4s")
    unknown = await harness.client.get("/clips/chunks/no-such-recording/init.mp4")

    assert missing.status_code == 404
    assert unknown.status_code == 410


async def test_a_ready_manifest_references_the_locked_chunk_paths(
    harness: Harness, tmp_path: Path
) -> None:
    _seed_recording(harness.runner, tmp_path, "init.mp4", "chunk_00000.m4s", "chunk_00001.m4s")

    response = await harness.client.get(f"/clips?session_id={_RECORDING_ID}&start=0&end=4")

    assert response.status_code == 200
    assert "#EXT-X-MAP" in response.text
    assert "init.mp4" in response.text
    assert f"/clips/chunks/{_RECORDING_ID}/chunk_00000.m4s" in response.text


async def test_a_pending_clip_is_202_with_retry_after(harness: Harness, tmp_path: Path) -> None:
    _seed_recording(harness.runner, tmp_path, "init.mp4", "chunk_00000.m4s")

    response = await harness.client.get(f"/clips?session_id={_RECORDING_ID}&start=0&end=4")

    assert response.status_code == 202
    assert "retry-after" in response.headers


async def test_an_unknown_recording_manifest_is_410(harness: Harness, tmp_path: Path) -> None:
    harness.runner.recorder._root = tmp_path

    response = await harness.client.get(f"/clips?session_id={_RECORDING_ID}&start=0&end=4")

    assert response.status_code == 410


async def test_a_bad_clip_range_is_400(harness: Harness, tmp_path: Path) -> None:
    _seed_recording(harness.runner, tmp_path, "init.mp4", "chunk_00000.m4s", "chunk_00001.m4s")

    response = await harness.client.get(f"/clips?session_id={_RECORDING_ID}&start=5&end=4")

    assert response.status_code == 400


# -- uploads --------------------------------------------------------------------


async def _create_slot(client: httpx.AsyncClient, upload_id: str, size: int = 4) -> httpx.Response:
    return await client.post(
        f"/sessions/{FIXED_SESSION_ID}/uploads",
        json={"name": "cat.png", "size": size, "mime_type": "image/png", "upload_id": upload_id},
    )


async def test_a_slot_is_reserved_under_the_supplied_upload_id(harness: Harness) -> None:
    await harness.client.post("/start_session", json={})

    response = await _create_slot(harness.client, "platform-upload-1")

    assert response.status_code == 201
    body = response.json()
    assert body["presigned_id"] == "platform-upload-1"
    assert body["presigned_url"].endswith("/uploads/platform-upload-1")


async def test_reserving_a_taken_upload_id_conflicts(harness: Harness) -> None:
    await harness.client.post("/start_session", json={})
    await _create_slot(harness.client, "platform-upload-1")

    response = await _create_slot(harness.client, "platform-upload-1")

    # 409 is the idempotency signal: the slot is already seeded, not an error.
    assert response.status_code == 409


async def test_the_bytes_land_with_a_put_to_the_slot(harness: Harness) -> None:
    await harness.client.post("/start_session", json={})
    await _create_slot(harness.client, "platform-upload-1")

    response = await harness.client.put("/uploads/platform-upload-1", content=b"\x89PNG")

    assert response.status_code == 200


async def test_putting_twice_conflicts(harness: Harness) -> None:
    await harness.client.post("/start_session", json={})
    await _create_slot(harness.client, "platform-upload-1")
    await harness.client.put("/uploads/platform-upload-1", content=b"\x89PNG")

    response = await harness.client.put("/uploads/platform-upload-1", content=b"\x89PNG")

    assert response.status_code == 409


async def test_putting_to_an_unknown_slot_is_404(harness: Harness) -> None:
    await harness.client.post("/start_session", json={})

    response = await harness.client.put("/uploads/never-reserved", content=b"\x89PNG")

    assert response.status_code == 404


async def test_upload_slots_require_the_fixed_session_id(harness: Harness) -> None:
    await harness.client.post("/start_session", json={})

    response = await harness.client.post(
        "/sessions/some-other-id/uploads",
        json={"name": "cat.png", "size": 4, "mime_type": "image/png"},
    )

    assert response.status_code == 404


async def test_upload_slots_require_a_running_session(harness: Harness) -> None:
    response = await _create_slot(harness.client, "platform-upload-1")

    assert response.status_code == 400
