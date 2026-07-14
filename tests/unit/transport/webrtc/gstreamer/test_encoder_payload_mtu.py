
"""Tests that every encoder bin caps its RTP payloader ``mtu``.

GStreamer payloaders default to ``mtu=1400``. With SRTP/UDP/IP overhead that
exceeds the effective MTU of tunneled network paths (WireGuard/Tailscale links
are 1280 bytes), where oversized packets are silently dropped: ICE checks and
data-channel traffic fit and succeed while every full-size media packet
vanishes. The bins pin the payloader to :data:`RTP_PAYLOAD_MTU` (1200, the
same ceiling libwebrtc uses) so media survives those paths.
"""

import pytest

from reactor_runtime.transport.webrtc.gstreamer.gst import Gst
from reactor_runtime.transport.webrtc.gstreamer.settings import RTP_PAYLOAD_MTU
from reactor_runtime.transport.webrtc.gstreamer.encoders.av1 import AV1EncoderBin
from reactor_runtime.transport.webrtc.gstreamer.encoders.h264 import H264EncoderBin
from reactor_runtime.transport.webrtc.gstreamer.encoders.h265 import H265EncoderBin
from reactor_runtime.transport.webrtc.gstreamer.encoders.opus import OpusEncoderBin
from reactor_runtime.transport.webrtc.gstreamer.encoders.vp8 import VP8EncoderBin
from reactor_runtime.transport.webrtc.gstreamer.encoders.vp9 import VP9EncoderBin

_CASES = [
    ("vp8enc", "rtpvp8pay", lambda: VP8EncoderBin(pt=96)),
    ("vp9enc", "rtpvp9pay", lambda: VP9EncoderBin(pt=98)),
    ("x264enc", "rtph264pay", lambda: H264EncoderBin(pt=109)),
    ("x265enc", "rtph265pay", lambda: H265EncoderBin(pt=113)),
    ("av1enc", "rtpav1pay", lambda: AV1EncoderBin(pt=45)),
    ("opusenc", "rtpopuspay", lambda: OpusEncoderBin(pt=111)),
]


@pytest.mark.parametrize(
    "enc_factory,pay_name,make_bin",
    _CASES,
    ids=[c[1] for c in _CASES],
)
def test_payloader_mtu_is_capped(enc_factory, pay_name, make_bin) -> None:
    # The bin needs both the encoder and its payloader; some GStreamer builds
    # ship one without the other (e.g. av1enc without rtpav1pay), so skip unless
    # both factories are present.
    if (
        Gst.ElementFactory.find(enc_factory) is None
        or Gst.ElementFactory.find(pay_name) is None
    ):
        pytest.skip(f"{enc_factory}/{pay_name} not available in this GStreamer build")

    enc_bin = make_bin()
    payloader = enc_bin.get_by_name(pay_name)
    assert payloader is not None
    # Track the configured value, not the literal default: a WebRtcConfig with
    # a different rtp_payload_mtu changes RTP_PAYLOAD_MTU, and the payloader
    # must follow it. The default itself (1200) is covered in test_config.py.
    assert payloader.get_property("mtu") == RTP_PAYLOAD_MTU
