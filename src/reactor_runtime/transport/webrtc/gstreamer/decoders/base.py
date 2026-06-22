from typing import Protocol, Optional
from reactor_runtime.transport.webrtc.gstreamer.gst import Gst


class RTPDecoder(Protocol):
    """
    Structural typing contract (PEP 544) for RTP decoder bins shaped as:

        sink (RTP) -> rtpdepay -> decoder -> src (raw video/audio)

    This allows higher-level components (e.g., dynamic pipeline builders,
    HotSwitch, WebRTC receive chains) to operate generically on any
    codec-specific RTP decoder without depending on concrete classes.

    Any implementation must expose:

        - pad_sink(): RTP input pad
        - pad_src():  decoded/raw output pad

    This mirrors the EncoderPayloader protocol used on the send side.
    """

    def pad_src(self) -> Gst.Pad: ...
    def pad_sink(self) -> Gst.Pad: ...


class BaseRTPDecoderBin(Gst.Bin):
    """
    Base class for RTP decoder bins.

    Responsibilities:

        - Create and manage ghost sink/src pads
        - Provide safe property setting helpers
        - Provide simple linking utility with error handling

    Intended topology:

        webrtcbin (RTP output)
            ↓
        BaseRTPDecoderBin
            ↓
        downstream raw video/audio processing

    This base class keeps codec-specific decoder bins small and focused.
    """

    def __init__(self, name: str):
        super().__init__(name=name)

        # Ghost pads allow this bin to behave like a simple element:
        #   RTP in → raw media out
        self._ghost_sink: Optional[Gst.GhostPad] = None
        self._ghost_src: Optional[Gst.GhostPad] = None

    # ---------------------------------------------------------------------
    # Ghost pad creation
    # ---------------------------------------------------------------------

    def _create_ghost_pads(self, sink_pad: Gst.Pad, src_pad: Gst.Pad) -> None:
        """
        Create ghost pads exposing internal pads externally.

        Typical internal chain:
            rtpdepay -> decoder

        Ghost pads allow linking this bin as if it were a single element.
        """
        ghost_sink = Gst.GhostPad.new("sink", sink_pad)
        ghost_src = Gst.GhostPad.new("src", src_pad)

        if not ghost_sink or not ghost_src:
            raise RuntimeError("Failed to create ghost pads")

        if not self.add_pad(ghost_sink):
            raise RuntimeError("Failed to add ghost sink pad to bin")

        if not self.add_pad(ghost_src):
            raise RuntimeError("Failed to add ghost src pad to bin")

        self._ghost_sink = ghost_sink
        self._ghost_src = ghost_src

    def pad_src(self) -> Gst.Pad:
        """
        Return decoded/raw output pad.

        Used when linking to:
            - videoconvert
            - videoscale
            - appsink
            - compositor
        """
        if self._ghost_src is None:
            raise RuntimeError("Ghost src pad has not been created yet")
        return self._ghost_src

    def pad_sink(self) -> Gst.Pad:
        """
        Return RTP input pad.

        Used when linking from:
            - webrtcbin src pads
            - rtpbin
            - custom RTP demuxers
        """
        if self._ghost_sink is None:
            raise RuntimeError("Ghost sink pad has not been created yet")
        return self._ghost_sink

    # ---------------------------------------------------------------------
    # Safe linking helper
    # ---------------------------------------------------------------------

    def _link_or_raise(self, a: Gst.Element, b: Gst.Element, what: str) -> None:
        """
        Link two GStreamer elements or raise a descriptive error.

        Args:
            a:
                Upstream element

            b:
                Downstream element

            what:
                Human-readable description of the link
                (used in error messages)

        This centralizes link failure handling and improves debugging,
        especially in dynamic pipeline construction.
        """
        if not a.link(b):
            raise RuntimeError("Failed to link: %s" % what)
