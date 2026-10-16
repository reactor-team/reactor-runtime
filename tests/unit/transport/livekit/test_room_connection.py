import asyncio
from typing import Any, cast

import numpy as np
from livekit import rtc

from reactor_runtime.core import (
    Connection,
    ConnectionOrigin,
    ConnectionOriginSource,
    ConnId,
    MediaBundle,
    MediaChunk,
    TrackData,
    TrackDirection,
    TrackInfo,
    TrackKind,
)
from reactor_runtime.transport.livekit.connection import RoomConnection

VIDEO = TrackInfo(name="main_video", kind=TrackKind.VIDEO, direction=TrackDirection.OUT)
AUDIO = TrackInfo(
    name="main_audio", kind=TrackKind.AUDIO, rate=24_000, direction=TrackDirection.OUT
)


class FakeVideoSource:
    def __init__(self) -> None:
        self.frames: list[rtc.VideoFrame] = []

    def capture_frame(self, frame: rtc.VideoFrame) -> None:
        self.frames.append(frame)


class FakeAudioSource:
    def __init__(self) -> None:
        self.frames: list[rtc.AudioFrame] = []
        self.cleared = 0

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        self.frames.append(frame)

    def clear_queue(self) -> None:
        self.cleared += 1


class Leaves:
    """Records each time the connection leaves its room."""

    def __init__(self) -> None:
        self.count = 0

    async def __call__(self) -> None:
        self.count += 1


def make_connection(
    leaves: Leaves | None = None,
) -> tuple[RoomConnection, FakeVideoSource, FakeAudioSource]:
    video, audio = FakeVideoSource(), FakeAudioSource()
    conn = RoomConnection(
        ConnId(4711),
        asyncio.get_running_loop(),
        {"main_video": cast(Any, video)},
        {"main_audio": cast(Any, audio)},
        {"main_video": VIDEO, "main_audio": AUDIO},
        on_close=leaves or Leaves(),
    )
    return conn, video, audio


def bundle(value: int) -> MediaBundle:
    return MediaBundle(
        tracks={
            "main_video": TrackData(info=VIDEO, data=np.full((2, 3, 3), value, np.uint8)),
            "main_audio": TrackData(
                info=AUDIO, data=np.arange(960, dtype=np.int16).reshape(1, -1) + value
            ),
        }
    )


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


async def test_the_room_connection_is_a_system_connection_that_carries_media() -> None:
    conn, _, _ = make_connection()

    assert isinstance(conn, Connection)
    assert conn.id == 4711
    assert isinstance(conn, ConnectionOriginSource)
    assert conn.origin == ConnectionOrigin(transport="livekit", system=True)
    assert conn.capabilities.carries_video
    assert conn.capabilities.carries_audio


async def test_closing_the_room_connection_leaves_the_room_once() -> None:
    leaves = Leaves()
    conn, _, _ = make_connection(leaves)
    conn.start()

    await conn.close()
    await conn.close()

    assert leaves.count == 1


async def test_a_paced_frame_reaches_its_video_and_audio_sources_unchanged() -> None:
    conn, video, audio = make_connection()
    conn.start()

    conn._send_frame(bundle(5))
    await _settle()

    (frame,) = video.frames
    assert (frame.width, frame.height, frame.type) == (3, 2, rtc.VideoBufferType.RGB24)
    assert bytes(frame.data) == np.full((2, 3, 3), 5, np.uint8).tobytes()
    (sound,) = audio.frames
    assert (sound.sample_rate, sound.num_channels, sound.samples_per_channel) == (24_000, 1, 960)
    assert bytes(sound.data) == (np.arange(960, dtype=np.int16) + 5).tobytes()
    await conn.close()


async def test_audio_reaches_the_source_in_the_order_it_was_paced() -> None:
    conn, _, audio = make_connection()
    conn.start()

    for value in range(10):
        conn._send_frame(bundle(value))
    await _settle()

    firsts = [np.frombuffer(bytes(f.data), np.int16)[0] for f in audio.frames]
    assert firsts == list(range(10))
    await conn.close()


async def test_a_flush_clears_the_audio_waiting_for_the_room() -> None:
    conn, _, audio = make_connection()
    # No pump running: the frames wait, as they would for a room that is behind.
    for value in range(3):
        conn._send_frame(bundle(value))
    await _settle()

    conn.flush_media(hold=True)
    await _settle()

    assert conn._audio["main_audio"].pending.empty()
    assert audio.cleared == 1
    await conn.close()


async def test_messages_to_the_system_client_are_dropped() -> None:
    conn, video, audio = make_connection()

    conn.send_message(b"frame")
    conn.send_control(b"frame")

    assert video.frames == []
    assert audio.frames == []


async def test_media_sent_through_the_connection_is_paced_into_the_room() -> None:
    conn, video, _ = make_connection()
    conn.start()
    data = np.stack([np.full((2, 3, 3), v, np.uint8) for v in (1, 2)])
    chunk = MediaChunk(
        bundle=MediaBundle(tracks={"main_video": TrackData(info=VIDEO, data=data)}),
        fps=100.0,
        n_frames=2,
    )

    conn.send_media(chunk)
    await asyncio.sleep(0.1)

    assert [bytes(f.data)[0] for f in video.frames] == [1, 2]
    await conn.close()
