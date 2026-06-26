from typing import Dict, Optional

from .vp8 import VP8EncoderBin
from .vp9 import VP9EncoderBin
from .av1 import AV1EncoderBin
from .h264 import H264EncoderBin
from .h265 import H265EncoderBin
from .opus import OpusEncoderBin


class EncoderFactory:
    """
    Factory responsible for instantiating encoder+payloader bins
    based on the negotiated RTP encoding-name from SDP.

    This acts as the bridge between:
        - SDP negotiation layer (codec selection + fmtp parsing)
        - GStreamer pipeline construction

    It ensures:
        - Correct encoder implementation is chosen
        - RTP payload type (PT) matches negotiated value
        - Relevant fmtp parameters (e.g., profile-id) are applied
    """

    @classmethod
    def create(
        cls,
        encoding_name: str,
        pt: int,
        format: Dict[str, Optional[str]],
        ssrc: Optional[int] = None,
    ):
        """
        Create an encoder bin matching the negotiated codec.

        Args:
            encoding_name:
                RTP encoding-name from SDP (e.g., "VP8", "H264", "AV1").
                Expected to be already normalized (uppercase).

            pt:
                RTP payload type negotiated for this codec.
                Must match the PT used in SDP and in rtp payloader.

            format:
                Parsed fmtp parameters from SDP (key → value).
                Used to propagate codec-specific constraints such as:
                    - H264 profile-level-id
                    - VP9 profile-id
                    - future AV1 parameters

            ssrc:
                Optional RTP SSRC to set on the payloader so it matches
                the SDP answer (required for WebRTC remote track creation).

        Returns:
            A Gst.Bin implementing:
                sink -> encoder -> payloader -> src

        Raises:
            ValueError if encoding_name is unsupported.

        Notes:
            This factory intentionally keeps logic minimal.
            Codec-specific behavior should live inside each encoder bin.
        """

        if encoding_name == "VP8":
            return VP8EncoderBin(pt=pt, ssrc=ssrc)

        if encoding_name == "VP9":
            return VP9EncoderBin(
                pt=pt,
                profile_id=format.get("profile-id") or "0",
                ssrc=ssrc,
            )

        if encoding_name == "AV1":
            return AV1EncoderBin(pt=pt, ssrc=ssrc)

        if encoding_name == "H264":
            return H264EncoderBin(
                pt=pt,
                profile_level_id=format.get("profile-level-id") or "42e01f",
                ssrc=ssrc,
            )

        if encoding_name == "H265":
            return H265EncoderBin(pt=pt, ssrc=ssrc, format=format)

        if encoding_name == "OPUS":
            return OpusEncoderBin(pt=pt, ssrc=ssrc, format=format)

        # -----------------------------------------------------------------
        # Unsupported codec
        # -----------------------------------------------------------------
        raise ValueError("Unsupported encoding-name for encoder: %r" % encoding_name)
