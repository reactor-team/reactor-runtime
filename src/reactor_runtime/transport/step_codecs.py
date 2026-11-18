"""The codecs an encoded step frame can be carried in, the one table every user reads.

A step frame's codec is named three ways: by the encoding its frames carry on
the wire (``FrameEncoding``), by the name an SDP ``rtpmap`` gives it (``H264``,
``VP8``...), and by the ``reactor_webrtc`` codec that encodes or decodes it.
:data:`STEP_FRAME_CODECS` holds the three together, so the negotiation that
picks a track's codec and the encoders and decoders that carry its frames
cannot disagree on which codecs exist, and :func:`video_codecs` is the one
set of codec backends they all open them with.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import reactor_webrtc as rw

from reactor_wire.v1 import step_pb2


@dataclass(frozen=True)
class StepFrameCodec:
    """One codec a step frame can be carried in.

    Attributes:
        sdp_name: The codec's name in an SDP ``rtpmap``, upper case.
        encoding: The wire encoding a frame in this codec is marked with.
        codec: The ``reactor_webrtc`` codec that encodes and decodes it.
    """

    sdp_name: str
    encoding: step_pb2.FrameEncoding.ValueType
    codec: rw.VideoCodec


STEP_FRAME_CODECS: tuple[StepFrameCodec, ...] = (
    StepFrameCodec("H264", step_pb2.FRAME_ENCODING_H264, rw.VideoCodec.H264),
    StepFrameCodec("VP8", step_pb2.FRAME_ENCODING_VP8, rw.VideoCodec.Vp8),
    StepFrameCodec("VP9", step_pb2.FRAME_ENCODING_VP9, rw.VideoCodec.Vp9),
    StepFrameCodec("AV1", step_pb2.FRAME_ENCODING_AV1, rw.VideoCodec.Av1),
)
"""Every codec a step frame can be carried in, in no particular order."""

BY_SDP_NAME: Mapping[str, StepFrameCodec] = MappingProxyType(
    {codec.sdp_name: codec for codec in STEP_FRAME_CODECS}
)
""":data:`STEP_FRAME_CODECS` by SDP name."""

BY_ENCODING: Mapping[step_pb2.FrameEncoding.ValueType, StepFrameCodec] = MappingProxyType(
    {codec.encoding: codec for codec in STEP_FRAME_CODECS}
)
""":data:`STEP_FRAME_CODECS` by wire encoding."""

_backends_lock = threading.Lock()
_backends: list[rw.VideoCodecs] = []


def video_codecs() -> rw.VideoCodecs:
    """Return the process's video codec backends, collected on first use.

    Safe to call from any thread: the step frame decoders and encoders open
    their codecs on worker threads.
    """
    with _backends_lock:
        if not _backends:
            _backends.append(rw.VideoCodecs())
        return _backends[0]
