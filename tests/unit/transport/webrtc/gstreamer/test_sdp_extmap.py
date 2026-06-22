
"""Tests for SDP extmap injection."""

import pytest

from reactor_runtime.transport.webrtc.gstreamer.settings import (
    RTP_HEADER_EXTENSION_URI_TRANSPORT_WIDE_CC,
)
from reactor_runtime.transport.webrtc.gstreamer.sdp.extmap import (
    SdpExtmap,
    add_extmaps_per_mid_to_sdp,
    add_extmaps_per_mid_to_sdp_text,
    add_extmaps_to_sdp_media,
    add_extmaps_to_webrtc_session_description,
    extmap_id_by_mid_for_uri,
    negotiated_sdp_extmaps_by_mid,
)


def _norm_lines(s: str) -> str:
    return s.replace("\r\n", "\n").strip()


def test_adds_extmap_after_mid_for_video_and_audio():
    sdp = """v=0
o=- 1 2 IN IP4 0.0.0.0
s=-
t=0 0
m=video 9 UDP/TLS/RTP/SAVPF 96
a=mid:v0
a=sendonly
m=audio 9 UDP/TLS/RTP/SAVPF 111
a=mid:a0
a=sendonly
"""
    out = add_extmaps_to_sdp_media(
        sdp,
        [(4, RTP_HEADER_EXTENSION_URI_TRANSPORT_WIDE_CC)],
    )
    lines = _norm_lines(out).split("\n")
    v_mid = lines.index("a=mid:v0")
    assert lines[v_mid + 1] == (
        f"a=extmap:4 {RTP_HEADER_EXTENSION_URI_TRANSPORT_WIDE_CC}"
    )
    a_mid = lines.index("a=mid:a0")
    assert lines[a_mid + 1] == (
        f"a=extmap:4 {RTP_HEADER_EXTENSION_URI_TRANSPORT_WIDE_CC}"
    )


def test_skips_duplicate_id():
    sdp = """v=0
o=- 1 2 IN IP4 0.0.0.0
s=-
t=0 0
m=video 9 UDP/TLS/RTP/SAVPF 96
a=mid:v0
a=extmap:4 urn:existing:uri
a=sendonly
"""
    out = add_extmaps_to_sdp_media(sdp, [(4, RTP_HEADER_EXTENSION_URI_TRANSPORT_WIDE_CC)])
    text = _norm_lines(out)
    assert text.count("a=extmap:4") == 1
    assert "urn:existing:uri" in text


def test_uses_sdp_extmap_dataclass():
    sdp = """v=0
o=- 1 2 IN IP4 0.0.0.0
s=-
t=0 0
m=video 9 UDP/TLS/RTP/SAVPF 96
a=mid:v0
"""
    out = add_extmaps_to_sdp_media(sdp, [SdpExtmap(id=3, uri="urn:test")])
    assert "a=extmap:3 urn:test" in _norm_lines(out)


def test_media_types_filters_sections():
    sdp = """v=0
o=- 1 2 IN IP4 0.0.0.0
s=-
t=0 0
m=video 9 UDP/TLS/RTP/SAVPF 96
a=mid:v0
m=audio 9 UDP/TLS/RTP/SAVPF 111
a=mid:a0
"""
    out = add_extmaps_to_sdp_media(
        sdp,
        [(1, RTP_HEADER_EXTENSION_URI_TRANSPORT_WIDE_CC)],
        media_types=("video",),
    )
    text = _norm_lines(out)
    assert f"a=extmap:1 {RTP_HEADER_EXTENSION_URI_TRANSPORT_WIDE_CC}" in text
    assert text.count("a=extmap:") == 1


def test_extmap_id_by_mid_for_uri_differs_per_section():
    u = RTP_HEADER_EXTENSION_URI_TRANSPORT_WIDE_CC
    sdp = f"""v=0
o=- 1 2 IN IP4 0.0.0.0
s=-
t=0 0
m=video 9 UDP/TLS/RTP/SAVPF 96
a=mid:v0
a=extmap:2 {u}
m=audio 9 UDP/TLS/RTP/SAVPF 111
a=mid:a0
a=extmap:5 {u}
"""
    m = extmap_id_by_mid_for_uri(sdp, u)
    assert m["v0"] == 2
    assert m["a0"] == 5


def test_extmap_id_by_mid_for_uri_none_when_uri_missing():
    sdp = """v=0
o=- 1 2 IN IP4 0.0.0.0
s=-
t=0 0
m=video 9 UDP/TLS/RTP/SAVPF 96
a=mid:v0
"""
    m = extmap_id_by_mid_for_uri(sdp, RTP_HEADER_EXTENSION_URI_TRANSPORT_WIDE_CC)
    assert m["v0"] is None


def test_extmap_id_parses_sendrecv_direction():
    u = RTP_HEADER_EXTENSION_URI_TRANSPORT_WIDE_CC
    sdp = f"""v=0
o=- 1 2 IN IP4 0.0.0.0
s=-
t=0 0
m=video 9 UDP/TLS/RTP/SAVPF 96
a=mid:v0
a=extmap:7/sendrecv {u}
"""
    m = extmap_id_by_mid_for_uri(sdp, u)
    assert m["v0"] == 7


def test_negotiated_sdp_extmaps_by_mid_order_and_dedup():
    other = "urn:x-other:ext"
    u = RTP_HEADER_EXTENSION_URI_TRANSPORT_WIDE_CC
    sdp = f"""v=0
o=- 1 2 IN IP4 0.0.0.0
s=-
t=0 0
m=video 9 UDP/TLS/RTP/SAVPF 96
a=mid:v0
a=extmap:3 {u}
a=extmap:9 {other}
"""
    by_mid = negotiated_sdp_extmaps_by_mid(sdp, [other, u])
    assert by_mid["v0"] == [
        SdpExtmap(id=9, uri=other),
        SdpExtmap(id=3, uri=u),
    ]
    deduped = negotiated_sdp_extmaps_by_mid(sdp, [u, u])
    assert deduped["v0"] == [SdpExtmap(id=3, uri=u)]


def test_add_extmaps_per_mid_only_targets_listed_mids():
    u = RTP_HEADER_EXTENSION_URI_TRANSPORT_WIDE_CC
    sdp = """v=0
o=- 1 2 IN IP4 0.0.0.0
s=-
t=0 0
m=video 9 UDP/TLS/RTP/SAVPF 96
a=mid:v0
m=audio 9 UDP/TLS/RTP/SAVPF 111
a=mid:a0
"""
    out = add_extmaps_per_mid_to_sdp_text(
        sdp,
        {"v0": [SdpExtmap(id=3, uri=u)]},
    )
    text = _norm_lines(out)
    lines = text.split("\n")
    assert f"a=extmap:3 {u}" in text
    extmap_idxs = [i for i, ln in enumerate(lines) if ln.startswith("a=extmap:")]
    assert len(extmap_idxs) == 1
    v0_i = lines.index("a=mid:v0")
    a0_i = lines.index("a=mid:a0")
    assert v0_i < extmap_idxs[0] < a0_i


def test_add_extmaps_to_webrtc_session_description_preserves_type():
    pytest.importorskip("gi.repository.GstWebRTC")
    from reactor_runtime.transport.webrtc.gstreamer.gst import Gst, GstSdp, GstWebRTC

    Gst.init(None)

    u = RTP_HEADER_EXTENSION_URI_TRANSPORT_WIDE_CC
    sdp = """v=0
o=- 1 2 IN IP4 0.0.0.0
s=-
t=0 0
m=video 9 UDP/TLS/RTP/SAVPF 96
a=mid:v0
a=sendonly
"""
    res, sdpmsg = GstSdp.SDPMessage.new()
    GstSdp.sdp_message_parse_buffer(bytes(sdp, "utf-8"), sdpmsg)
    desc = GstWebRTC.WebRTCSessionDescription.new(
        GstWebRTC.WebRTCSDPType.ANSWER,
        sdpmsg,
    )
    out = add_extmaps_to_webrtc_session_description(
        desc,
        [SdpExtmap(id=4, uri=u)],
    )
    assert out.type == GstWebRTC.WebRTCSDPType.ANSWER
    assert f"a=extmap:4 {u}" in out.sdp.as_text().replace("\r\n", "\n")


def test_add_extmaps_per_mid_to_sdp_webrtc_session_description_preserves_type():
    """``add_extmaps_per_mid_to_sdp`` updates a :class:`GstWebRTC.WebRTCSessionDescription`."""
    pytest.importorskip("gi.repository.GstWebRTC")
    from reactor_runtime.transport.webrtc.gstreamer.gst import Gst, GstSdp, GstWebRTC

    Gst.init(None)

    u = RTP_HEADER_EXTENSION_URI_TRANSPORT_WIDE_CC
    sdp = """v=0
o=- 1 2 IN IP4 0.0.0.0
s=-
t=0 0
m=video 9 UDP/TLS/RTP/SAVPF 96
a=mid:v0
a=sendonly
"""
    res, sdpmsg = GstSdp.SDPMessage.new()
    GstSdp.sdp_message_parse_buffer(bytes(sdp, "utf-8"), sdpmsg)
    desc = GstWebRTC.WebRTCSessionDescription.new(
        GstWebRTC.WebRTCSDPType.ANSWER,
        sdpmsg,
    )
    out = add_extmaps_per_mid_to_sdp(
        desc,
        {"v0": [SdpExtmap(id=4, uri=u)]},
    )
    assert isinstance(out, GstWebRTC.WebRTCSessionDescription)
    assert out.type == GstWebRTC.WebRTCSDPType.ANSWER
    assert f"a=extmap:4 {u}" in out.sdp.as_text().replace("\r\n", "\n")
