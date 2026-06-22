from typing import Protocol, Optional
from reactor_runtime.transport.webrtc.gstreamer.gst import Gst


class EncoderPayloader(Protocol):
    """
    Structural typing contract (PEP 544) for encoder bins that expose:

        sink -> encoder -> payloader -> src

    This allows higher-level components (e.g., HotSwitch, WebRTC pipeline
    builders) to operate generically on different codec implementations
    (VP8, VP9, H264, AV1, H265, etc.) without depending on concrete classes.

    Any class implementing this protocol must provide:

        - pad_src(): RTP output pad
        - pad_sink(): raw input pad
        - set_target_bitrate_kbps() / get_bitrate_kbps(): configured bitrate in kbps

    This avoids tight coupling to specific encoder implementations.
    """

    def pad_src(self) -> Gst.Pad: ...
    def pad_sink(self) -> Gst.Pad: ...
    def set_target_bitrate_kbps(self, bitrate_kbps: int) -> None: ...
    def get_bitrate_kbps(self) -> int: ...


class BaseEncoderBin(Gst.Bin):
    """
    Base class for encoder + payloader bins.

    Responsibilities:

        - Create and manage ghost sink/src pads
        - Provide safe property setting helpers

    Codec-specific bitrate handling lives on each encoder bin
    (:meth:`set_target_bitrate_kbps` / :meth:`get_bitrate_kbps`).
    """

    def __init__(self, name: str):
        super().__init__(name=name)

        # Ghost pads expose internal pads as external pads.
        # They allow this Bin to behave like a regular element:
        #     raw video in → RTP out
        self._ghost_sink: Optional[Gst.GhostPad] = None
        self._ghost_src: Optional[Gst.GhostPad] = None

    # ---------------------------------------------------------------------
    # Ghost pad creation
    # ---------------------------------------------------------------------

    def _create_ghost_pads(self, sink_pad: Gst.Pad, src_pad: Gst.Pad) -> None:
        """
        Create and attach ghost pads to the bin.

        Ghost pads are required so external pipeline elements
        (e.g., tee, webrtcbin) can link to this bin as if it were
        a simple encoder element.

        Typical topology:
            appsrc ! BaseEncoderBin ! webrtcbin
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
        Return RTP output pad (ghost pad).

        Used when linking encoder output to:
            - webrtcbin
            - rtpbin
            - custom RTP routing elements
        """
        if self._ghost_src is None:
            raise RuntimeError("Ghost src pad has not been created yet")
        return self._ghost_src

    def pad_sink(self) -> Gst.Pad:
        """
        Return raw video input pad (ghost pad).

        Used when linking from:
            - appsrc
            - videoconvert
            - videoscale
            - tee branches
        """
        if self._ghost_sink is None:
            raise RuntimeError("Ghost sink pad has not been created yet")
        return self._ghost_sink
