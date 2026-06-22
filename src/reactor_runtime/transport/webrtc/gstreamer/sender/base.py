
"""
Base class for sender bins (audio and video).
"""

import numpy as np

from reactor_runtime.transport.webrtc.gstreamer.gst import Gst
from reactor_runtime.transport.webrtc.gstreamer.gst_helpers import link_pads


class _SenderStreamBase(Gst.Bin):
    """
    Base for sender bins. Subclasses build their pipeline and set
    self._appsrc and self._ghost_src, and override push_buffer.
    """

    def __init__(self, name: str):
        super().__init__(name=name)

    def get_appsrc(self) -> Gst.Element:
        """Return the internal appsrc element (e.g. to set caps)."""
        return self._appsrc

    def push_buffer(self, data: np.ndarray) -> bool:
        """
        Push a buffer to the appsrc. Subclasses override and define the format of *data*
        (e.g. a video frame or audio samples) and how it is converted and pushed.

        Must be called from the thread that runs the GStreamer main loop.

        Returns:
            True if the buffer was pushed successfully (Gst.FlowReturn.OK),
            False otherwise.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.push_buffer must be overridden"
        )

    def get_src_pad(self) -> Gst.Pad:
        """Return the bin's ghost src pad for linking downstream."""
        return self._ghost_src

    def link_src_to(self, sink_pad: Gst.Pad) -> None:
        """
        Link this bin's src pad to the given sink pad.

        Equivalent to: link_pads(self.get_src_pad(), sink_pad)
        """
        link_pads(self.get_src_pad(), sink_pad)
