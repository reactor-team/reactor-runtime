"""Default settings the GStreamer media engine reads.

These mirror the defaults on
:class:`~reactor_runtime.transport.webrtc.config.WebRtcConfig`, which is the
single source of truth for the transport's configuration. A connection's own
config overrides them where the peer threads it into the pipeline; the values
here are the fallbacks the media library uses when nothing overrides them.
"""

from __future__ import annotations

from reactor_runtime.transport.webrtc.config import CodecEntry, WebRtcConfig

_DEFAULTS = WebRtcConfig()

# When False, video encoder/decoder bins use software elements only.
HW_CODECS_ENABLED: bool = _DEFAULTS.hw_codecs_enabled

# Video BWE bounds (kbps) for rtpgccbwe / VideoSender.
VIDEO_BWE_MIN_BITRATE_KBPS = _DEFAULTS.bwe_min_kbps
VIDEO_BWE_MAX_BITRATE_KBPS = _DEFAULTS.bwe_max_kbps
VIDEO_BWE_TARGET_BITRATE_KBPS = _DEFAULTS.bwe_target_kbps

# Relative gap below which a new GCC estimate is not re-applied to the encoders.
VIDEO_BWE_TARGET_UPDATE_RELATIVE_THRESHOLD = _DEFAULTS.bwe_target_update_threshold

# rtprtxsend retransmission history. A time of 0 means no time limit.
VIDEO_RTX_MAX_SIZE_PACKETS = _DEFAULTS.rtx_max_size_packets
VIDEO_RTX_MAX_SIZE_TIME_MS = _DEFAULTS.rtx_max_size_time_ms

# The mtu (bytes) every encoder bin pins on its RTP payloader so full-size
# media packets survive low-MTU network paths (see WebRtcConfig).
RTP_PAYLOAD_MTU: int = _DEFAULTS.rtp_payload_mtu

# Supported codecs in preference order; the first that appears in the offer wins.
SUPPORTED_VIDEO_CODECS: list[CodecEntry] = list(_DEFAULTS.supported_video_codecs)
SUPPORTED_AUDIO_CODECS: list[CodecEntry] = list(_DEFAULTS.supported_audio_codecs)

# RTP header extension URIs mirrored in answers when the offer includes them.
SUPPORTED_RTP_HEADER_EXTENSION_URIS: tuple[str, ...] = _DEFAULTS.rtp_header_extensions
RTP_HEADER_EXTENSION_URI_TRANSPORT_WIDE_CC = _DEFAULTS.rtp_header_extensions[0]
