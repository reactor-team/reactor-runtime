from collections.abc import Callable

import pytest

from reactor_runtime.core import InputFrame, MediaBundle
from reactor_runtime.protocol import ProtocolVersion
from reactor_runtime.transport.webrtc import (
    PeerStats,
    SdpAnswer,
    SdpOffer,
    TrackMap,
    WebRtcConfig,
    WebRtcPeerFactory,
)
from reactor_runtime.transport.webrtc.signaling import IceCandidate


class FakePeer:
    """A WebRtcPeer that records commands and lets tests fire inbound events."""

    def __init__(self, stats: PeerStats | None = None) -> None:
        self.ice: list[IceCandidate] = []
        self.messages: list[bytes | str] = []
        self.sent_media: list[MediaBundle] = []
        self.resumed: list[str] = []
        self.paused: list[str] = []
        self.closed = False
        self.protocol_version = ProtocolVersion.V0
        self.stats_fail_times = 0
        self._stats = stats if stats is not None else PeerStats(rtt_seconds=0.1)
        self._on_message: Callable[[bytes | str, ProtocolVersion], None] | None = None
        self._on_media: Callable[[str, InputFrame], None] | None = None
        self._on_ping: Callable[[], None] | None = None
        self._on_connected: Callable[[], None] | None = None
        self._on_disconnect: Callable[[], None] | None = None

    async def add_ice(self, candidate: IceCandidate) -> None:
        self.ice.append(candidate)

    def send_message(self, payload: bytes | str) -> None:
        self.messages.append(payload)

    def send_media(self, bundle: MediaBundle) -> None:
        self.sent_media.append(bundle)

    def resume_track(self, name: str) -> None:
        self.resumed.append(name)

    def pause_track(self, name: str) -> None:
        self.paused.append(name)

    async def stats(self) -> PeerStats:
        if self.stats_fail_times > 0:
            self.stats_fail_times -= 1
            raise RuntimeError("transient stats failure")
        return self._stats

    async def close(self) -> None:
        self.closed = True

    def on_message(self, callback: Callable[[bytes | str, ProtocolVersion], None]) -> None:
        self._on_message = callback

    def on_media(self, callback: Callable[[str, InputFrame], None]) -> None:
        self._on_media = callback

    def on_ping(self, callback: Callable[[], None]) -> None:
        self._on_ping = callback

    def on_connected(self, callback: Callable[[], None]) -> None:
        self._on_connected = callback

    def on_disconnect(self, callback: Callable[[], None]) -> None:
        self._on_disconnect = callback

    # --- test drivers: fire the inbound events a real peer would ---

    def fire_connected(self) -> None:
        assert self._on_connected is not None
        self._on_connected()

    def fire_disconnect(self) -> None:
        assert self._on_disconnect is not None
        self._on_disconnect()

    def fire_ping(self) -> None:
        assert self._on_ping is not None
        self._on_ping()

    def fire_message(self, payload: bytes | str) -> None:
        assert self._on_message is not None
        self._on_message(payload, self.protocol_version)

    def fire_media(self, track: str, frame: InputFrame) -> None:
        assert self._on_media is not None
        self._on_media(track, frame)


@pytest.fixture
def fake_peer() -> FakePeer:
    return FakePeer()


@pytest.fixture
def factory_for() -> Callable[..., WebRtcPeerFactory]:
    def make(peer: FakePeer, answer: str = "answer-sdp") -> WebRtcPeerFactory:
        async def factory(
            conn_id: int,
            offer: SdpOffer,
            tracks: TrackMap,
            config: WebRtcConfig,
            version: ProtocolVersion,
        ) -> tuple[FakePeer, SdpAnswer]:
            peer.protocol_version = version
            return peer, SdpAnswer(answer)

        return factory

    return make


@pytest.fixture
def out_av_tracks() -> TrackMap:
    """Two outbound tracks (video + audio) and one inbound track."""
    return TrackMap.from_client(
        [
            {"mid": "0", "name": "main_video", "kind": "video", "direction": "recvonly"},
            {"mid": "1", "name": "main_audio", "kind": "audio", "direction": "recvonly"},
            {"mid": "2", "name": "webcam", "kind": "video", "direction": "sendonly"},
        ]
    )
