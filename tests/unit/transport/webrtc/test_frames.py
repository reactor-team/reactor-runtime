"""Unit tests for the libwebrtc frame conversions.

Pure NumPy conversions with no native dependency: they exercise the module
directly, without the ``reactor_webrtc`` wheel.
"""

import numpy as np
import pytest

from reactor_runtime.transport.webrtc.frames import (
    bgra_to_rgb,
    inbound_audio_to_mono,
    rgb_to_bgra,
    to_int16_mono,
)


def test_rgb_to_bgra_reorders_channels_and_adds_opaque_alpha() -> None:
    rgb = np.array([[[10, 20, 30]]], dtype=np.uint8)  # one pixel, R=10 G=20 B=30
    data, width, height = rgb_to_bgra(rgb)
    assert (width, height) == (1, 1)
    assert list(data) == [30, 20, 10, 255]  # B, G, R, A


def test_rgb_to_bgra_takes_last_frame_of_a_batch() -> None:
    batch = np.stack([np.full((2, 2, 3), 1, np.uint8), np.full((2, 2, 3), 7, np.uint8)])
    data, width, height = rgb_to_bgra(batch)
    assert (width, height) == (2, 2)
    # Every pixel comes from the last frame (value 7), not the first.
    assert set(data) == {7, 255}


def test_rgb_to_bgra_rejects_non_rgb() -> None:
    with pytest.raises(ValueError, match="RGB frame"):
        rgb_to_bgra(np.zeros((4, 4), dtype=np.uint8))


def test_bgra_to_rgb_round_trips() -> None:
    rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    data, width, height = rgb_to_bgra(rgb)
    restored = bgra_to_rgb(data, width, height)
    assert restored.shape == (2, 3, 3)
    assert np.array_equal(restored, rgb)


def test_bgra_to_rgb_returns_a_contiguous_copy_not_aliasing_the_source() -> None:
    data, width, height = rgb_to_bgra(np.zeros((1, 1, 3), dtype=np.uint8))
    restored = bgra_to_rgb(data, width, height)
    assert restored.flags["C_CONTIGUOUS"]
    assert restored.flags.writeable
    # The result must not alias the transient source buffer.
    assert not np.shares_memory(restored, np.frombuffer(data, dtype=np.uint8))


def test_to_int16_mono_scales_float_to_int16_range() -> None:
    out = to_int16_mono(np.array([[0.5, -0.5, 1.0]], dtype=np.float32))
    assert out.dtype == np.int16
    assert out.tolist() == [16383, -16383, 32767]


def test_to_int16_mono_mixes_multichannel_to_mono() -> None:
    stereo = np.array([[100, 200], [300, 400]], dtype=np.int16)  # (channels, samples)
    out = to_int16_mono(stereo)
    assert out.tolist() == [200, 300]


def test_to_int16_mono_passes_through_mono_row() -> None:
    out = to_int16_mono(np.array([[1, 2, 3]], dtype=np.int16))
    assert out.tolist() == [1, 2, 3]


def test_to_int16_mono_rejects_3d() -> None:
    with pytest.raises(ValueError, match="1-D or 2-D"):
        to_int16_mono(np.zeros((1, 1, 1), dtype=np.int16))


def test_inbound_audio_to_mono_shapes_stereo_pcm_down() -> None:
    pcm = np.array([100, 200, 300, 400], dtype=np.int16).tobytes()
    out = inbound_audio_to_mono(pcm, channels=2)
    assert out.shape == (1, 2)
    assert out.tolist() == [[150, 350]]


def test_inbound_audio_to_mono_keeps_mono() -> None:
    pcm = np.array([1, 2, 3], dtype=np.int16).tobytes()
    out = inbound_audio_to_mono(pcm, channels=1)
    assert out.shape == (1, 3)
    assert out.tolist() == [[1, 2, 3]]
