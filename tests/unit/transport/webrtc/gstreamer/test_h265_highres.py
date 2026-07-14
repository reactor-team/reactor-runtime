
"""Regression: the H.265 encoder must produce RTP above the offered level.

Repro of the Safari zero-RTP-at-2K bug: Safari offers H265 with
``level-id=93`` (level 3.1, ~720p budget), and the encoder bin used to force
that into x265 as ``level-idc=3.1``. At 2560x1472 x265 then refuses to
initialize ("Can not initialize x265 encoder") and the send branch stalls
with ``not-negotiated`` — zero RTP ever leaves. The fix stops mapping the
offered ``level-id`` into the option-string; x265 derives a resolution-valid
level. This test drives the real ``H265EncoderBin`` with Safari's fmtp at 2K
and asserts encoded RTP flows.
"""

from reactor_runtime.transport.webrtc.gstreamer.gst import GLib, Gst
from reactor_runtime.transport.webrtc.gstreamer.encoders.h265 import H265EncoderBin

# 2K — above HEVC level 3.1's ~720p ceiling, so a pinned level-idc=3.1 stalls.
_WIDTH, _HEIGHT = 2560, 1472

# Safari's H265 offer fmtp (level-id=93 -> level 3.1).
_SAFARI_FMTP = {
    "level-id": "93",
    "profile-id": "1",
    "tier-flag": "0",
    "tx-mode": "SRST",
}


def _count_encoded_buffers_at_2k(num_buffers: int = 15) -> int:
    """Run videotestsrc(2K) -> H265EncoderBin -> appsink; return RTP buffers seen."""
    pipeline = Gst.Pipeline.new("h265_2k_regression")

    src = Gst.ElementFactory.make("videotestsrc")
    src.set_property("num-buffers", num_buffers)
    raw_caps = Gst.ElementFactory.make("capsfilter")
    raw_caps.set_property(
        "caps",
        Gst.Caps.from_string(
            f"video/x-raw,format=I420,width={_WIDTH},height={_HEIGHT},framerate=24/1"
        ),
    )
    convert = Gst.ElementFactory.make("videoconvert")
    encoder = H265EncoderBin(pt=96, format=_SAFARI_FMTP)
    sink = Gst.ElementFactory.make("appsink")
    sink.set_property("emit-signals", True)
    sink.set_property("sync", False)

    for element in (src, raw_caps, convert, encoder, sink):
        pipeline.add(element)
    assert src.link(raw_caps) and raw_caps.link(convert)
    assert (
        convert.get_static_pad("src").link(encoder.pad_sink()) == Gst.PadLinkReturn.OK
    )
    assert encoder.pad_src().link(sink.get_static_pad("sink")) == Gst.PadLinkReturn.OK

    seen = {"n": 0}

    def _on_sample(appsink):
        appsink.emit("pull-sample")
        seen["n"] += 1
        return Gst.FlowReturn.OK

    sink.connect("new-sample", _on_sample)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def _on_message(_bus, message):
        # EOS = clean finish; ERROR (e.g. x265 init failure from a pinned
        # level) ends the run too so a stalled pipeline can't hang the test.
        if message.type in (Gst.MessageType.EOS, Gst.MessageType.ERROR):
            loop.quit()

    bus.connect("message", _on_message)
    GLib.timeout_add_seconds(20, lambda: (loop.quit(), False)[1])

    pipeline.set_state(Gst.State.PLAYING)
    loop.run()
    pipeline.set_state(Gst.State.NULL)
    return seen["n"]


def test_h265_encoder_produces_rtp_at_2k_with_safari_fmtp():
    # level-id=93 -> level 3.1 (max ~720p). Forcing it into x265 as
    # level-idc=3.1 stalls 2K at encoder init (zero buffers); the unpinned
    # option-string must let RTP flow.
    buffers = _count_encoded_buffers_at_2k()
    assert buffers > 0, (
        "H.265 encoder produced zero RTP at 2K with Safari's level-id=93 — "
        "level-idc pin regression"
    )
