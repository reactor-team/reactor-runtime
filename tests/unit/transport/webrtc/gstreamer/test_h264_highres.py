
"""Regression: the H.264 encoder must produce RTP at >720p (2K).

Repro of the zero-RTP-at-2K bug: the encoder capsfilter used to pin the H.264
*level* derived from the SDP profile-level-id (``42e01f`` -> level 3.1). At
2560x1472 the encoded stream is ~level 5.x, so a level-3.1 capsfilter fails to
negotiate and the whole send branch stalls with ``not-negotiated`` — zero RTP
ever leaves (VP8, which carries no level, works over the same path). The fix
pins the profile only. This test drives the real ``H264EncoderBin`` at 2K and
asserts encoded RTP flows.
"""

from reactor_runtime.transport.webrtc.gstreamer.gst import GLib, Gst
from reactor_runtime.transport.webrtc.gstreamer.encoders.h264 import H264EncoderBin

# 2K — above H.264 level 3.1's ~720p ceiling, so a level-pinned capsfilter stalls.
_WIDTH, _HEIGHT = 2560, 1472


def _count_encoded_buffers_at_2k(profile_level_id: str, num_buffers: int = 15) -> int:
    """Run videotestsrc(2K) -> H264EncoderBin -> appsink; return RTP buffers seen."""
    pipeline = Gst.Pipeline.new("h264_2k_regression")

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
    encoder = H264EncoderBin(pt=109, profile_level_id=profile_level_id)
    sink = Gst.ElementFactory.make("appsink")
    sink.set_property("emit-signals", True)
    sink.set_property("sync", False)

    for element in (src, raw_caps, convert, encoder, sink):
        pipeline.add(element)
    assert src.link(raw_caps) and raw_caps.link(convert)
    assert convert.get_static_pad("src").link(encoder.pad_sink()) == Gst.PadLinkReturn.OK
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
        # EOS = clean finish; ERROR (e.g. not-negotiated from a bad capsfilter)
        # ends the run too so a stalled pipeline can't hang the test.
        if message.type in (Gst.MessageType.EOS, Gst.MessageType.ERROR):
            loop.quit()

    bus.connect("message", _on_message)
    GLib.timeout_add_seconds(20, lambda: (loop.quit(), False)[1])

    pipeline.set_state(Gst.State.PLAYING)
    loop.run()
    pipeline.set_state(Gst.State.NULL)
    return seen["n"]


def test_h264_encoder_produces_rtp_at_2k():
    # 42e01f -> level 3.1 (max ~720p). With a level-pinned capsfilter 2K
    # stalls with not-negotiated (zero buffers); profile-only caps must flow.
    buffers = _count_encoded_buffers_at_2k("42e01f")
    assert buffers > 0, "H.264 encoder produced zero RTP at 2K — level-pin regression"
