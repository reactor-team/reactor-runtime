
"""Unit tests for the video sender's egress helper.

Covers :func:`_rgb_bytes_padded_to_stride`, which pads each outgoing RGB row
out to its 4-byte stride so a width whose byte length isn't 4-aligned produces
a buffer downstream accepts, and round-trips against the ingress reader.
"""

from __future__ import annotations

import numpy as np

from reactor_runtime.transport.webrtc.gstreamer.peer import _rgb_frame_from_mapped_buffer
from reactor_runtime.transport.webrtc.gstreamer.sender.video import (
    _rgb_bytes_padded_to_stride,
)


def test_aligned_width_is_tightly_packed() -> None:
    """A 4-aligned width needs no padding: length is exactly width*3 per row."""
    img = (np.arange(2 * 4 * 3) % 256).astype(np.uint8).reshape(2, 4, 3)

    data = _rgb_bytes_padded_to_stride(img)

    assert len(data) == 2 * 4 * 3
    assert data == img.tobytes()


def test_unaligned_width_padded_to_stride() -> None:
    """A width whose byte length isn't 4-aligned is padded up to its stride."""
    width, height = 6, 4  # width*3 == 18, padded up to a 20-byte stride
    stride = 20
    img = (np.arange(height * width * 3) % 256).astype(np.uint8).reshape(
        height, width, 3
    )

    data = _rgb_bytes_padded_to_stride(img)

    assert len(data) == height * stride
    laid_out = np.frombuffer(data, dtype=np.uint8).reshape(height, stride)
    assert np.array_equal(laid_out[:, : width * 3], img.reshape(height, width * 3))


def test_egress_ingress_roundtrip() -> None:
    """Padding on the way out and stripping it on the way in restores the frame."""
    width, height = 6, 4
    stride = 20
    img = (np.arange(height * width * 3) % 256).astype(np.uint8).reshape(
        height, width, 3
    )

    data = _rgb_bytes_padded_to_stride(img)
    out = _rgb_frame_from_mapped_buffer(data, width, height, stride)

    assert np.array_equal(out, img)
