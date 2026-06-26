
"""
Base class for receiver bins (audio and video).
"""

from typing import Callable, Optional

from reactor_runtime.transport.webrtc.gstreamer.gst import Gst
from reactor_runtime.transport.webrtc.gstreamer.gst_helpers import link_pads


class _ReceiverStreamBase(Gst.Bin):
    """
    Base for receiver bins. Subclasses build their pipeline and set
    self._appsink and self._ghost_sink.
    """

    def __init__(self, name: str):
        super().__init__(name=name)
        self._on_new_sample_callback: Optional[
            Callable[[str, Gst.Element], Gst.FlowReturn]
        ] = None

    def get_appsink(self) -> Gst.Element:
        """Return the internal appsink element (e.g. to get caps)."""
        return self._appsink

    def get_sink_pad(self) -> Gst.Pad:
        """Return the bin's ghost sink pad for linking from a source pad."""
        return self._ghost_sink

    def link_sink_from(self, src_pad: Gst.Pad) -> None:
        """
        Link the given source pad to this bin's sink pad.

        Equivalent to: link_pads(src_pad, self.get_sink_pad())
        """
        link_pads(src_pad, self.get_sink_pad())

    def set_on_new_sample(
        self,
        callback: Optional[Callable[[str, Gst.Element], Gst.FlowReturn]],
    ) -> None:
        """
        Set the callback invoked when a new frame/sample is available from the appsink.

        The callback receives the appsink element; it should call
        sink.emit("pull-sample") to get the Gst.Sample, then process it and return
        Gst.FlowReturn.OK or Gst.FlowReturn.EOS. Call before the pipeline is set to PLAYING.

        Args:
            callback: Called with (bin name, appsink element) when a new sample is available.
                Pass None to clear.
        """
        self._on_new_sample_callback = callback

    def _on_new_sample(self, sink: Gst.Element) -> Gst.FlowReturn:
        """Internal handler: invoke user callback with the appsink element."""
        if self._on_new_sample_callback is None:
            return Gst.FlowReturn.OK
        return self._on_new_sample_callback(self.name, sink)
