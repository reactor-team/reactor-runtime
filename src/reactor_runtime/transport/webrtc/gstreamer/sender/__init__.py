
"""
Sender stream abstractions: Gst.Bin subclasses for audio and video.

  - VideoSender: appsrc ! videoconvert ! queue ! encoder ! capsfilter(RTP hdr ext) ! bwe
  - AudioSender: appsrc ! audioconvert ! audiorate ! encoder
"""

from reactor_runtime.transport.webrtc.gstreamer.sender.audio import AudioSender
from reactor_runtime.transport.webrtc.gstreamer.sender.video import VideoSender

__all__ = ["AudioSender", "VideoSender"]
