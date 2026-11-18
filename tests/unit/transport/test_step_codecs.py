"""The step frame codec table that negotiation and the frame codecs share."""

from __future__ import annotations

import reactor_webrtc as rw

from reactor_runtime.transport.step_codecs import BY_ENCODING, BY_SDP_NAME, STEP_FRAME_CODECS
from reactor_wire.v1 import step_pb2


def test_each_codec_is_named_once_in_every_vocabulary() -> None:
    codecs = [c.codec for c in STEP_FRAME_CODECS]

    assert len(BY_SDP_NAME) == len(BY_ENCODING) == len(STEP_FRAME_CODECS)
    assert all(codecs.count(codec) == 1 for codec in codecs)


def test_every_encoded_wire_encoding_has_a_codec() -> None:
    encoded = set(step_pb2.FrameEncoding.values()) - {
        step_pb2.FRAME_ENCODING_UNSPECIFIED,
        step_pb2.FRAME_ENCODING_RAW,
    }

    assert set(BY_ENCODING) == encoded


def test_each_sdp_name_is_the_bindings_codec_name() -> None:
    # The SDP name is the binding's variant name upper-cased, the same mapping
    # the peer uses for a configured codec preference.
    variants = {
        name.upper(): value
        for name in dir(rw.VideoCodec)
        if isinstance(value := getattr(rw.VideoCodec, name), rw.VideoCodec)
    }

    for codec in STEP_FRAME_CODECS:
        assert variants[codec.sdp_name] == codec.codec
