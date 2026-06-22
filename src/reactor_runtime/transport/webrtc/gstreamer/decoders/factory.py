from typing import Any

from .vp8 import VP8DecoderBin
from .vp9 import VP9DecoderBin
from .av1 import AV1DecoderBin
from .h264 import H264DecoderBin
from .h265 import H265DecoderBin
from .opus import OpusDecoderBin


class DecoderFactory:
    @classmethod
    def create(cls, encoding_name: str, **kwargs: Any):
        """
        Creates an RTP depay + decoder bin based on RTP 'encoding-name'.

        Note: Decoders do NOT need pt (payload type). PT is used by payloader/depayload
        caps matching, but not by the decoder bin constructor itself.

        Args:
            encoding_name: One of VP8, VP9, AV1, H264, H265, OPUS (case-insensitive).
            **kwargs: Forwarded to the specific decoder bin constructor.

        Returns:
            A Gst.Bin instance (e.g. VP9DecoderBin, H264DecoderBin, ...)

        Raises:
            ValueError if encoding_name is unknown.
        """

        if encoding_name == "VP8":
            return VP8DecoderBin(**kwargs)

        if encoding_name == "VP9":
            return VP9DecoderBin(**kwargs)

        if encoding_name == "AV1":
            return AV1DecoderBin(**kwargs)

        if encoding_name == "H264":
            return H264DecoderBin(**kwargs)

        if encoding_name == "H265":
            return H265DecoderBin(**kwargs)

        if encoding_name == "OPUS":
            return OpusDecoderBin(**kwargs)

        raise ValueError("Unsupported encoding-name for decoder: %r" % encoding_name)
