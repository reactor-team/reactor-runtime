"""In-process loopback for the libwebrtc WebRTC peer.

Drives :func:`libwebrtc_peer_factory` against a second libwebrtc peer standing in
for a browser client: the stand-in offers one track the model sends on and one it
receives on, plus a data channel, and the two negotiate a real connection over
loopback ICE. The test then asserts the seam actually carries traffic both ways —
outbound video reaches the client carrying the metadata the model attached to it,
inbound video surfaces through ``on_media`` carrying what the client attached, and
a data-channel frame surfaces through ``on_message``.

Requires the native ``reactor_webrtc`` wheel; the module skips cleanly when it is
absent.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import numpy as np
import pytest

rw = pytest.importorskip("reactor_webrtc")

from reactor_runtime.core import ConnId, MediaBundle  # noqa: E402
from reactor_runtime.core.values import (  # noqa: E402
    InputFrame,
    TrackData,
    TrackDirection,
    TrackInfo,
    TrackKind,
)
from reactor_runtime.protocol import Channel, ProtocolVersion  # noqa: E402
from reactor_runtime.transport.webrtc.config import WebRtcConfig  # noqa: E402
from reactor_runtime.transport.webrtc.frames import rgb_to_bgra  # noqa: E402
from reactor_runtime.transport.webrtc.peer import (  # noqa: E402
    WebRTCPeer,
    _get_factory,
    libwebrtc_peer_factory,
)
from reactor_runtime.transport.webrtc.signaling import (  # noqa: E402
    IceCandidate,
    MappedTrack,
    SdpOffer,
    TrackMap,
)

pytestmark = pytest.mark.asyncio

_WIDTH, _HEIGHT = 320, 240
_TIMEOUT_S = 25.0


async def _reached(event: asyncio.Event) -> bool:
    """Await an event up to the shared timeout, reporting whether it fired."""
    try:
        await asyncio.wait_for(event.wait(), _TIMEOUT_S)
    except TimeoutError:
        return False
    return True


def _solid_frame(value: int) -> np.ndarray:
    """A solid ``(H, W, 3)`` RGB frame; distinct values keep the encoder busy."""
    return np.full((_HEIGHT, _WIDTH, 3), value % 256, dtype=np.uint8)


def _ice_from_answer(sdp: str) -> list[rw.IceCandidate]:
    """Pull the candidates the peer embedded into its answer.

    The peer answers non-trickle — its gathered candidates ride inside the SDP,
    which a browser parses on ``setRemoteDescription``. The libwebrtc stand-in
    does not, so the test lifts them back out and feeds them in explicitly, one
    per m-section in offer order.
    """
    candidates: list[rw.IceCandidate] = []
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


class _Client:
    """A libwebrtc peer standing in for the browser, driven synchronously."""

    def __init__(self, factory: rw.PeerConnectionFactory) -> None:
        self.ice: list[rw.IceCandidate] = []
        self.received_video = 0
        self.received_audio = 0
        self.received_metadata: list[bytes] = []
        self.video_track_seen = threading.Event()
        self._recv_tracks: list[rw.Track] = []

        observer = rw.PeerConnectionObserver()
        ice = self.ice
        observer.on_ice_candidate = ice.append
        observer.on_track = self._on_track
        self.pc = factory.create_peer_connection(rw.RtcConfiguration(), observer)

        # Two transceivers the model sends on (client receives) — video and audio —
        # one the client sends on (model receives), plus the data channel.
        self.recv = self.pc.add_transceiver(rw.MediaKind.Video, rw.TransceiverDirection.RecvOnly)
        self.recv_audio = self.pc.add_transceiver(
            rw.MediaKind.Audio, rw.TransceiverDirection.RecvOnly
        )
        self.send_track = factory.create_video_track("client-cam")
        self.send = self.pc.add_transceiver(rw.MediaKind.Video, rw.TransceiverDirection.SendOnly)
        self.send.set_track(self.send_track)
        # Nothing to attach for metadata in either direction: reactor-webrtc
        # advertises the capability in the offer, the runtime's answer mirrors it,
        # and both peers install their own embed/strip steps on negotiation.
        self.data = self.pc.create_data_channel("data")

    def _on_track(self, kind: rw.MediaKind, track: rw.Track) -> None:
        self._recv_tracks.append(track)
        if kind == rw.MediaKind.Video:

            def _count_video(_bgra: bytes, _w: int, _h: int, meta: Any = None) -> None:
                self.received_video += 1
                if meta is not None:
                    self.received_metadata.append(bytes(meta.user_data))

            track.on_video_frame(_count_video)
            self.video_track_seen.set()
        elif kind == rw.MediaKind.Audio:

            def _count_audio(_pcm: bytes, _sr: int, _ch: int, _frames: int) -> None:
                self.received_audio += 1

            track.on_audio_frame(_count_audio)

    async def create_offer(self) -> str:
        offer = await self.pc.create_offer()
        await self.pc.set_local_description(offer)
        return offer.sdp

    def track_map(self) -> TrackMap:
        return TrackMap(
            tracks=(
                MappedTrack(
                    mid=self.recv.mid() or "0",
                    info=TrackInfo(
                        name="out_video", kind=TrackKind.VIDEO, direction=TrackDirection.OUT
                    ),
                ),
                MappedTrack(
                    mid=self.recv_audio.mid() or "1",
                    info=TrackInfo(
                        name="out_audio", kind=TrackKind.AUDIO, direction=TrackDirection.OUT
                    ),
                ),
                MappedTrack(
                    mid=self.send.mid() or "2",
                    info=TrackInfo(
                        name="in_video", kind=TrackKind.VIDEO, direction=TrackDirection.IN
                    ),
                ),
            )
        )

    async def accept_answer(self, sdp: str) -> None:
        await self.pc.set_remote_description(rw.SessionDescription("answer", sdp))
        for candidate in _ice_from_answer(sdp):
            await self.pc.add_ice_candidate(candidate)


async def _trickle_until(client: _Client, peer: WebRTCPeer, stop: asyncio.Event) -> None:
    """Forward the client's ICE candidates to the peer as they are gathered.

    Candidates arrive on a libwebrtc thread after the offer is created, so a
    single drain races the gathering; this keeps forwarding until the wire
    connects. The peer's own candidates travel the other way embedded in the
    answer, so only this direction trickles.
    """
    while not stop.is_set():
        pending, client.ice[:] = client.ice[:], []
        for candidate in pending:
            await peer.add_ice(
                IceCandidate(
                    candidate=candidate.candidate,
                    sdp_mid=candidate.sdp_mid,
                    sdp_mline_index=candidate.sdp_mline_index,
                )
            )
        await asyncio.sleep(0.05)


async def test_loopback_carries_media_and_messages() -> None:
    factory = _get_factory()
    client = _Client(factory)
    offer_sdp = await client.create_offer()
    tracks = client.track_map()

    messages: list[tuple[bytes | str, ProtocolVersion, Channel]] = []
    inbound_media: dict[str, int] = {}
    inbound_metadata: list[bytes] = []
    connected = asyncio.Event()
    loop = asyncio.get_running_loop()

    peer, answer = await libwebrtc_peer_factory(
        ConnId(1),
        SdpOffer(sdp=offer_sdp),
        tracks,
        WebRtcConfig(ice_gathering_timeout_ms=4000),
        ProtocolVersion.V0,
    )

    message_arrived = asyncio.Event()

    def _record_message(payload: bytes | str, version: ProtocolVersion, channel: Channel) -> None:
        messages.append((payload, version, channel))
        message_arrived.set()

    def _record_media(name: str, frame: InputFrame) -> None:
        inbound_media[name] = inbound_media.get(name, 0) + 1
        if frame.metadata is not None:
            inbound_metadata.append(frame.metadata)

    peer.on_message(_record_message)
    peer.on_media(_record_media)
    peer.on_ping(lambda: None)
    peer.on_connected(connected.set)
    peer.on_disconnect(lambda: None)

    stop_trickle = asyncio.Event()
    trickle_task = asyncio.create_task(_trickle_until(client, peer, stop_trickle))
    try:
        await client.accept_answer(answer.sdp)

        assert await _reached(connected), "peer connection never reached connected"

        # Wait for the model's outbound track to reach the client before pumping:
        # the frame sink is attached when it arrives.
        assert await asyncio.to_thread(client.video_track_seen.wait, _TIMEOUT_S), (
            "the model's outbound video track never reached the client"
        )

        # Pump media until every leg has produced output: outbound model video
        # and audio must both reach the client, the metadata attached to a frame
        # must arrive with it, and the client's inbound video must surface
        # through on_media. Video and audio ride the same bundle, so the seam
        # pushes them together each tick. A single frame is not enough for the
        # codecs to emit, so pump on a frame-rate cadence.
        video_info = tracks.tracks[0].info
        audio_info = tracks.tracks[1].info
        samples_per_tick = 48_000 // 30  # one 30 fps frame's worth of 48 kHz audio
        deadline = loop.time() + _TIMEOUT_S
        value = 0
        while loop.time() < deadline:
            if (
                client.received_video
                and client.received_audio
                and client.received_metadata
                and inbound_media.get("in_video")
                and inbound_metadata
            ):
                break
            value += 1
            audio = np.full((1, samples_per_tick), (value % 100) - 50, dtype=np.int16)
            peer.send_media(
                MediaBundle(
                    tracks={
                        "out_video": TrackData(
                            info=video_info,
                            data=_solid_frame(value),
                            metadata=f'{{"frame":{value}}}'.encode(),
                        ),
                        "out_audio": TrackData(info=audio_info, data=audio),
                    }
                )
            )
            bgra, width, height = rgb_to_bgra(_solid_frame(value + 128))
            client.send_track.push_video_frame(
                bgra, width, height, user_data=f'{{"client":{value}}}'.encode()
            )
            await asyncio.sleep(0.033)

        assert client.received_video, "outbound model video never reached the client"
        assert client.received_audio, "outbound model audio never reached the client"
        assert inbound_media.get("in_video"), "inbound client video never surfaced via on_media"
        assert client.received_metadata, "frame metadata never reached the client"
        assert client.received_metadata[0].startswith(b'{"frame":'), (
            f"unexpected metadata on the wire: {client.received_metadata[0]!r}"
        )
        assert inbound_metadata, "the client's frame metadata never surfaced via on_media"
        assert inbound_metadata[0].startswith(b'{"client":'), (
            f"unexpected inbound metadata: {inbound_metadata[0]!r}"
        )

        # A client data-channel frame must surface through on_message, tagged with
        # the sniffed codec version and the channel it arrived on.
        client.data.send(b'{"type": "hello"}', binary=False)
        assert await _reached(message_arrived), "inbound data-channel frame not surfaced"
        payload, version, channel = messages[0]
        assert channel is Channel.DATA
        assert version is ProtocolVersion.V0
        assert isinstance(payload, str)
        assert "hello" in payload
    finally:
        stop_trickle.set()
        trickle_task.cancel()
        await peer.close()
        client.pc = None  # type: ignore[assignment]
