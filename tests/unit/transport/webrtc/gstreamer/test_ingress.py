
"""Unit tests for the peer's video ingress helper.

Covers :func:`_rgb_frame_from_mapped_buffer`, which drops GStreamer's
row-stride padding when copying a decoded RGB buffer into a numpy frame.
GStreamer pads each raw RGB row up to a 4-byte boundary, so widths whose
byte length isn't a multiple of 4 arrive with a stride larger than
``width * 3``; reading them tightly packed shears the image and rotates
the channels into a grayscale smear.
"""

from __future__ import annotations

import numpy as np

from reactor_runtime.transport.webrtc.gstreamer.peer import _rgb_frame_from_mapped_buffer


def _pack_with_stride(img: np.ndarray, stride: int) -> bytes:
    """Lay out ``img`` with each row padded out to ``stride`` bytes."""
    height, width, _ = img.shape
    buf = np.zeros((height, stride), dtype=np.uint8)
    buf[:, : width * 3] = img.reshape(height, width * 3)
    return buf.tobytes()


def test_rgb_frame_tightly_packed_roundtrips() -> None:
    """A 4-aligned width (stride == width*3) copies through unchanged."""
    img = (np.arange(4 * 4 * 3) % 256).astype(np.uint8).reshape(4, 4, 3)

    out = _rgb_frame_from_mapped_buffer(img.tobytes(), 4, 4, 4 * 3)

    assert out.shape == (4, 4, 3)
    assert np.array_equal(out, img)


def test_rgb_frame_padded_stride_drops_padding() -> None:
    """A width whose byte length isn't 4-aligned round-trips at the true stride."""
    width, height = 6, 4  # width*3 == 18, padded up to a 20-byte stride
    stride = 20
    img = (np.arange(height * width * 3) % 256).astype(np.uint8).reshape(
        height, width, 3
    )

    out = _rgb_frame_from_mapped_buffer(_pack_with_stride(img, stride), width, height, stride)

    assert out.shape == (height, width, 3)
    assert np.array_equal(out, img)


def test_rgb_frame_naive_read_would_corrupt() -> None:
    """Guards the regression: reading padded rows at width*3 shears the image."""
    width, height = 6, 4
    stride = 20
    img = (np.arange(height * width * 3) % 256).astype(np.uint8).reshape(
        height, width, 3
    )
    data = _pack_with_stride(img, stride)

    naive = np.ndarray((height, width, 3), dtype=np.uint8, buffer=data).copy()

    assert not np.array_equal(naive, img)
