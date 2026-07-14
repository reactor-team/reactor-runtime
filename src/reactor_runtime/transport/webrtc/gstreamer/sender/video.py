
"""
Video sender bin: raw frames in, RTP out; RTX and BWE live on the transport aux sender.
"""

from typing import List, NamedTuple, Optional

import numpy as np

from reactor_runtime.transport.webrtc.gstreamer.encoders import EncoderFactory
from reactor_runtime.transport.webrtc.gstreamer.quality import video_qos_score
from reactor_runtime.transport.webrtc.gstreamer.settings import VIDEO_BWE_TARGET_BITRATE_KBPS
from reactor_runtime.transport.webrtc.gstreamer.sdp.codec import CodecEntry
from reactor_runtime.transport.webrtc.gstreamer.sdp.extmap import SdpExtmap
from reactor_runtime.transport.webrtc.gstreamer.gst import Gst
from reactor_runtime.transport.webrtc.gstreamer.gst_helpers import (
    add_many,
    link_many,
    link_pads,
    make_element,
)
from reactor_runtime.transport.webrtc.gstreamer.probes.fps_probe import FpsProbe
from .base import _SenderStreamBase

# WebRTC stream/track identification (matches SDP a=msid and stream-id)
_DEFAULT_STREAM_ID = "default"


def _rgb_bytes_padded_to_stride(frame: np.ndarray) -> bytes:
    """Serialize an RGB frame with each row padded to a 4-byte stride.

    Downstream reads raw RGB rows padded up to a 4-byte boundary, so a width
    whose byte length (``width * 3``) is not a multiple of 4 needs the padding
    added here; a tightly packed buffer is undersized for that stride and the
    frame is rejected.

    Args:
        frame: An ``(H, W, 3)`` uint8 RGB array.

    Returns:
        The frame's bytes with each row padded out to its 4-byte stride.
    """
    height, width = frame.shape[:2]
    row_bytes = width * 3
    stride = (row_bytes + 3) & ~3
    if stride == row_bytes:
        return frame.tobytes()
    padded = np.zeros((height, stride), dtype=np.uint8)
    padded[:, :row_bytes] = np.ascontiguousarray(frame).reshape(height, row_bytes)
    return padded.tobytes()


class RtpRtxSenderIds(NamedTuple):
    """Primary and RTX RTP identifiers for ``rtprtxsend`` map structures."""

    pt: int
    ssrc: Optional[int]
    rtx_pt: Optional[int]
    rtx_ssrc: Optional[int]


def _gst_caps_for_rtp_header_extensions(extmaps: List[SdpExtmap]) -> Gst.Caps:
    """
    Build ``application/x-rtp`` caps with ``extmap-<id>=(string)"<uri>"`` fields
    so downstream elements match negotiated SDP ``a=extmap`` lines.
    """
    if not extmaps:
        raise ValueError("extmaps must be non-empty")
    seen_ids: set[int] = set()
    fields = ["application/x-rtp"]
    for em in extmaps:
        if em.id in seen_ids:
            raise ValueError(f"duplicate RTP header extension id: {em.id}")
        seen_ids.add(em.id)
        if em.id <= 0:
            raise ValueError(f"extmap id must be > 0, got {em.id}")
        uri = em.uri.strip()
        if not uri:
            raise ValueError("extmap uri must be non-empty")
        # Gst caps string: escape backslashes and double-quotes in the URI token.
        safe_uri = uri.replace("\\", "\\\\").replace('"', '\\"')
        fields.append(f'extmap-{em.id}=(string)"{safe_uri}"')
    return Gst.Caps.from_string(", ".join(fields))


class VideoSender(_SenderStreamBase):
    """
    ``Gst.Bin`` that turns raw frames into RTP for ``webrtcbin``:

        appsrc ! videoconvert ! queue ! encoder ! capsfilter (RTP header extensions)

    The ghost ``src`` pad exposes the encoded RTP stream for ``webrtcbin``.
    Retransmission (``rtprtxsend``) and congestion control (``rtpgccbwe``) are
    handled by the shared aux sender pipeline in the transport
    (``GStreamerTransport._gst_on_request_aux_sender``), not here.

    ``rtp_header_extensions`` lists negotiated ``SdpExtmap`` entries for this mid;
    pass ``[]`` or omit when none apply.
    """

    def __init__(
        self,
        codec_entry: CodecEntry,
        name: str = "video_sender",
        ssrc: Optional[int] = None,
        rtx_ssrc: Optional[int] = None,
        rtx_payload_type: Optional[int] = None,
        rtp_header_extensions: Optional[List[SdpExtmap]] = None,
    ):
        super().__init__(name=name)

        encoding_name = codec_entry.get("codec")
        pt = codec_entry.get("payload_type")
        if not encoding_name or pt is None:
            raise ValueError("codec_entry must contain 'codec' and 'payload_type'")
        format_params = codec_entry.get("parameters") or {}

        self._pt = int(pt)
        self._codec_name: str = encoding_name
        self._ssrc = ssrc
        self._rtx_pt = int(rtx_payload_type) if rtx_payload_type is not None else None
        self._rtx_ssrc = int(rtx_ssrc) if rtx_ssrc is not None else None

        self._appsrc = make_element("appsrc", "appsrc")
        self._appsrc.set_property("format", Gst.Format.TIME)
        self._appsrc.set_property("is-live", True)
        self._appsrc.set_property("do-timestamp", True)
        self._appsrc.set_property("block", True)

        videoconvert = make_element("videoconvert", "videoconvert")
        queue_el = make_element("queue", "queue")
        queue_el.set_property("max-size-buffers", 1)
        queue_el.set_property("leaky", "downstream")

        self._encoder = EncoderFactory.create(
            encoding_name, pt, format_params, ssrc=ssrc
        )
        self._encoder.set_target_bitrate_kbps(VIDEO_BWE_TARGET_BITRATE_KBPS)

        # Negotiated RTP header extensions (e.g. TWCC); ids must match SDP per mid.
        exts = list(rtp_header_extensions) if rtp_header_extensions is not None else []
        self._rtp_hdr_ext_capsfilter = make_element(
            "capsfilter", "rtp_hdr_ext_capsfilter"
        )
        if exts:
            self._rtp_hdr_ext_capsfilter.set_property(
                "caps", _gst_caps_for_rtp_header_extensions(exts)
            )

        add_many(
            self,
            self._appsrc,
            videoconvert,
            queue_el,
            self._encoder,
            self._rtp_hdr_ext_capsfilter,
            sync_with_parent=True,
        )
        link_many(self._appsrc, videoconvert, queue_el)
        link_pads(queue_el.get_static_pad("src"), self._encoder.pad_sink())
        link_pads(
            self._encoder.pad_src(), self._rtp_hdr_ext_capsfilter.get_static_pad("sink")
        )

        pad_src = self._rtp_hdr_ext_capsfilter.get_static_pad("src")
        if not pad_src:
            raise RuntimeError("capsfilter has no src pad")
        ghost_src = Gst.GhostPad.new("src", pad_src)
        if not ghost_src or not self.add_pad(ghost_src):
            raise RuntimeError("Failed to add ghost src pad to VideoSender")
        self._ghost_src: Gst.GhostPad = ghost_src

        # Per-sender caps and PTS (must be used only on GLib thread)
        self._outgoing_width = 0
        self._outgoing_height = 0

        self._fps_probe = FpsProbe(f"{name}_encoded")
        self._fps_probe.attach(queue_el.get_static_pad("src"))

    @property
    def pt(self) -> int:
        """Negotiated RTP payload type for the primary (non-RTX) video codec."""
        return self._pt

    @property
    def ssrc(self) -> Optional[int]:
        """RTP SSRC for outgoing media (passed to the encoder / payloader)."""
        return self._ssrc

    @property
    def rtx_pt(self) -> Optional[int]:
        """Negotiated RTX payload type (``apt=`` primary PT), if retransmission is used."""
        return self._rtx_pt

    @property
    def rtx_ssrc(self) -> Optional[int]:
        """RTX SSRC paired with :attr:`ssrc` for FID / ``rtprtxsend`` ``ssrc-map``."""
        return self._rtx_ssrc

    def get_rtprtx_sender_ids(self) -> RtpRtxSenderIds:
        """Return stored RTP/RTX identifiers for building ``rtprtxsend`` maps."""
        return RtpRtxSenderIds(
            pt=self._pt,
            ssrc=self._ssrc,
            rtx_pt=self._rtx_pt,
            rtx_ssrc=self._rtx_ssrc,
        )

    @property
    def codec_name(self) -> str:
        return self._codec_name

    @property
    def width(self) -> Optional[int]:
        return self._outgoing_width or None

    @property
    def height(self) -> Optional[int]:
        return self._outgoing_height or None

    @property
    def fps(self) -> Optional[float]:
        """Frames per second pushed into the encoder from the last completed window."""
        return self._fps_probe.last_fps

    @property
    def current_bitrate_kbps(self) -> int:
        """Configured encoder bitrate in kbps (delegates to the encoder bin)."""
        return self._encoder.get_bitrate_kbps()

    def set_target_bitrate_kbps(self, new_kbps: int) -> None:
        """Forward target bitrate to the encoder bin."""
        self._encoder.set_target_bitrate_kbps(int(new_kbps))

    def qos(self) -> Optional[float]:
        """Return the 0–10 quality score for this sender, or None if not ready."""
        w, h, fps = self.width, self.height, self.fps
        if not w or not h or not fps:
            return None
        return video_qos_score(
            self.current_bitrate_kbps * 1000, w, h, fps, self._codec_name
        )

    def push_buffer(self, frame: np.ndarray) -> bool:
        """
        Extract dimensions and bytes from the frame (e.g. numpy array with .shape and .tobytes()),
        set appsrc caps if needed, create a Gst.Buffer, set PTS/duration, and push to the appsrc.
        Must be called from the thread that runs the GStreamer main loop.

        Returns:
            True if the buffer was pushed successfully (Gst.FlowReturn.OK), False otherwise
            or if frame has no .shape/.tobytes().
        """
        try:
            height, width = frame.shape[:2]
            data = _rgb_bytes_padded_to_stride(frame)
        except Exception:
            return False

        if width != self._outgoing_width or height != self._outgoing_height:
            caps = Gst.Caps.from_string(
                f"video/x-raw,format=RGB,width={width},height={height},framerate=30/1"
            )
            self._appsrc.set_property("caps", caps)
            self._outgoing_width = width
            self._outgoing_height = height

        buf = Gst.Buffer.new_wrapped(data)
        ret = self._appsrc.emit("push-buffer", buf)
        return ret == Gst.FlowReturn.OK
