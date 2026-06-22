"""Errors raised while negotiating or running a GStreamer WebRTC connection."""

from __future__ import annotations


class WebRTCSupersededError(Exception):
    """A connection was stopped or superseded before it finished setting up."""


class WebRTCNoVideoError(Exception):
    """An SDP offer carried no video media section."""


class WebRTCNoMediaError(Exception):
    """An SDP offer carried no audio or video media section."""
