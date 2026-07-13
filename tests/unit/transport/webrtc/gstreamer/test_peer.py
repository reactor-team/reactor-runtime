"""Unit tests for the GStreamer peer's seam adaptation.

These exercise the code that adapts the ported GStreamer transport onto the
WebRtcPeer seam — the inbound callbacks, the binary send path, the ICE
threading, and the stats mapping — without standing up a full negotiation,
which belongs to the standalone end-to-end proof. The GStreamer media body
itself is covered by the media-library tests.
"""

import asyncio
import inspect

import pytest

from reactor_runtime.core import TrackDirection
from reactor_runtime.protocol import Channel, ProtocolVersion
from reactor_runtime.transport.webrtc.gstreamer.peer import (
    GStreamerPeer,
    gstreamer_peer_factory,
)
from reactor_runtime.transport.webrtc.peer import PeerStats, TrackStat, WebRtcPeer
from reactor_runtime.transport.webrtc.signaling import IceCandidate


class FakeChannel:
    """Records the GObject signals a data channel would emit."""

    def __init__(self) -> None:
        self.emitted: list[tuple[str, object]] = []

    def emit(self, signal: str, arg: object) -> None:
        self.emitted.append((signal, arg))


class _FakeBytes:
    """Stands in for a ``GLib.Bytes`` carrying a binary data-channel frame."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def get_data(self) -> bytes:
        return self._data


def _peer() -> GStreamerPeer:
    peer = GStreamerPeer(ping_timeout_seconds=5.0)
    peer._stopping = False
    return peer


def test_conforms_to_webrtc_peer_protocol() -> None:
    assert isinstance(_peer(), WebRtcPeer)


def test_factory_has_the_peer_factory_shape() -> None:
    assert inspect.iscoroutinefunction(gstreamer_peer_factory)
    params = list(inspect.signature(gstreamer_peer_factory).parameters)
    assert params == ["conn_id", "offer", "tracks", "config", "version"]


def test_send_text_uses_send_string() -> None:
    peer = _peer()
    channel = FakeChannel()
    peer._data_channel = channel

    peer._gst_send_datachannel_msg('{"type":"current_mode"}')

    assert channel.emitted == [("send-string", '{"type":"current_mode"}')]


def test_send_binary_uses_send_data_with_bytes() -> None:
    peer = _peer()
    channel = FakeChannel()
    peer._data_channel = channel

    peer._gst_send_datachannel_msg(b"\x08\x96\x01")

    assert len(channel.emitted) == 1
    signal, payload = channel.emitted[0]
    assert signal == "send-data"
    assert payload.get_data() == b"\x08\x96\x01"


def test_send_control_uses_the_control_channel() -> None:
    peer = _peer()
    channel = FakeChannel()
    peer._control_channel = channel

    peer._gst_send_control_msg('{"type":"response"}')

    assert channel.emitted == [("send-string", '{"type":"response"}')]


async def test_inbound_data_frame_fires_message_and_ping() -> None:
    peer = _peer()
    peer._loop = asyncio.get_running_loop()
    messages: list[tuple[bytes | str, ProtocolVersion, Channel]] = []
    pings: list[int] = []
    peer.on_message(lambda payload, version, channel: messages.append((payload, version, channel)))
    peer.on_ping(lambda: pings.append(1))

    peer._gst_on_data_channel_message(None, "hello")
    await asyncio.sleep(0)

    assert messages == [("hello", ProtocolVersion.V0, Channel.DATA)]
    assert pings == [1]


async def test_inbound_binary_frame_carries_the_negotiated_version() -> None:
    peer = _peer()
    peer._loop = asyncio.get_running_loop()
    peer.protocol_version = ProtocolVersion.V1
    messages: list[tuple[bytes | str, ProtocolVersion, Channel]] = []
    peer.on_message(lambda payload, version, channel: messages.append((payload, version, channel)))

    peer._gst_on_data_channel_data(None, _FakeBytes(b"\x08\x01"))
    await asyncio.sleep(0)

    assert messages == [(b"\x08\x01", ProtocolVersion.V1, Channel.DATA)]


async def test_inbound_control_frame_surfaces_up_and_pings() -> None:
    peer = _peer()
    peer._loop = asyncio.get_running_loop()
    messages: list[tuple[bytes | str, ProtocolVersion, Channel]] = []
    pings: list[int] = []
    peer.on_message(lambda payload, version, channel: messages.append((payload, version, channel)))
    peer.on_ping(lambda: pings.append(1))

    peer._gst_on_control_channel_message(None, '{"type":"notification"}')
    await asyncio.sleep(0)

    assert messages == [('{"type":"notification"}', ProtocolVersion.V0, Channel.CONTROL)]
    assert pings == [1]


async def test_binary_first_frame_sniffs_and_latches_v1() -> None:
    peer = _peer()  # defaults to v0 until a frame reveals the codec
    peer._loop = asyncio.get_running_loop()
    messages: list[tuple[bytes | str, ProtocolVersion, Channel]] = []
    peer.on_message(lambda payload, version, channel: messages.append((payload, version, channel)))

    peer._gst_on_data_channel_data(None, _FakeBytes(b"\x08\x01"))
    await asyncio.sleep(0)

    assert peer.protocol_version is ProtocolVersion.V1
    assert messages == [(b"\x08\x01", ProtocolVersion.V1, Channel.DATA)]


async def test_json_text_first_frame_sniffs_v0() -> None:
    peer = _peer()
    peer._loop = asyncio.get_running_loop()
    peer.on_message(lambda payload, version, channel: None)

    peer._gst_on_control_channel_message(None, '{"type":"notification"}')
    await asyncio.sleep(0)

    assert peer.protocol_version is ProtocolVersion.V0


async def test_first_frame_latches_the_codec_for_the_connections_life() -> None:
    peer = _peer()
    peer._loop = asyncio.get_running_loop()
    versions: list[ProtocolVersion] = []
    peer.on_message(lambda payload, version, channel: versions.append(version))

    # A binary frame latches v1; a later text frame does not flip the codec.
    peer._gst_on_control_channel_data(None, _FakeBytes(b"\x08\x01"))
    peer._gst_on_control_channel_message(None, '{"type":"notification"}')
    await asyncio.sleep(0)

    assert peer.protocol_version is ProtocolVersion.V1
    assert versions == [ProtocolVersion.V1, ProtocolVersion.V1]


async def test_fire_is_a_noop_without_a_registered_callback() -> None:
    peer = _peer()
    peer._loop = asyncio.get_running_loop()

    # No callbacks registered: must not raise.
    peer._gst_on_data_channel_message(None, "hello")
    await asyncio.sleep(0)


async def test_stats_maps_rtc_stats_to_peerstats() -> None:
    peer = _peer()
    sample = (
        0.042,
        [TrackStat(name="VideoOut", direction=TrackDirection.OUT, fps=30.0, bitrate_bps=4000)],
    )

    async def fake_get() -> tuple[float | None, list[TrackStat]]:
        return sample

    peer._get_rtc_stats = fake_get  # type: ignore[method-assign]

    stats = await peer.stats()

    assert isinstance(stats, PeerStats)
    assert stats.rtt_seconds == 0.042
    assert stats.tracks == (sample[1][0],)


async def test_add_ice_without_mline_index_is_dropped() -> None:
    peer = _peer()
    scheduled: list[object] = []
    peer._run_on_gst_thread = lambda fn, *a, **k: scheduled.append(fn)  # type: ignore[method-assign]

    await peer.add_ice(IceCandidate(candidate="candidate:1 1 udp ...", sdp_mline_index=None))

    assert scheduled == []


async def test_add_ice_with_mline_index_is_scheduled() -> None:
    peer = _peer()
    scheduled: list[object] = []
    peer._run_on_gst_thread = lambda fn, *a, **k: scheduled.append(fn)  # type: ignore[method-assign]

    await peer.add_ice(IceCandidate(candidate="candidate:1 1 udp ...", sdp_mline_index=0))

    assert len(scheduled) == 1
