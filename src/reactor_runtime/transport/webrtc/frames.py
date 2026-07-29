"""Frame conversion between the runtime's NumPy media and libwebrtc's buffers.

libwebrtc exchanges video as tightly packed BGRA and audio as interleaved signed
16-bit PCM. These helpers translate the runtime's ``(H, W, 3)`` uint8 RGB frames
and ``(1, M)`` int16 mono audio to that wire form on the way out, and back to the
same NumPy shapes on the way in.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

_INT16_MAX = 32767.0
_ALPHA_OPAQUE = 255


def rgb_to_bgra(frame: npt.NDArray[Any]) -> tuple[bytes, int, int]:
    """Pack an ``(H, W, 3)`` RGB frame into opaque BGRA bytes for libwebrtc.

    Args:
        frame: A ``(H, W, 3)`` uint8 RGB frame, or an ``(N, H, W, 3)`` batch of
            which only the last frame is taken.

    Returns:
        The BGRA bytes together with the frame's ``(width, height)``.

    Raises:
        ValueError: If the input is not a 3-channel image once debatched.
    """
    if frame.ndim == 4:
        frame = frame[-1]
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB frame, got shape {frame.shape}")
    height, width = int(frame.shape[0]), int(frame.shape[1])
    bgra = np.empty((height, width, 4), dtype=np.uint8)
    bgra[:, :, 0] = frame[:, :, 2]
    bgra[:, :, 1] = frame[:, :, 1]
    bgra[:, :, 2] = frame[:, :, 0]
    bgra[:, :, 3] = _ALPHA_OPAQUE
    return bgra.tobytes(), width, height


def bgra_to_rgb(bgra: bytes, width: int, height: int) -> npt.NDArray[np.uint8]:
    """Copy a libwebrtc BGRA buffer into a tightly packed ``(H, W, 3)`` RGB array.

    The result is an owned copy: the source buffer is only valid for the duration
    of the libwebrtc callback that delivers it, so the returned array must not
    alias it.
    """
    arr = np.frombuffer(bgra, dtype=np.uint8).reshape(height, width, 4)
    return np.ascontiguousarray(arr[:, :, [2, 1, 0]])


def to_int16_mono(data: npt.NDArray[Any]) -> npt.NDArray[np.int16]:
    """Reduce an outbound audio payload to a 1-D int16 mono stream.

    Accepts ``(M,)``, ``(1, M)``, or ``(C, M)`` arrays. Float input (assumed in
    ``[-1, 1]``) is scaled to the int16 range; multi-channel input is mixed down
    to mono.

    Raises:
        ValueError: If the array is neither 1-D nor 2-D.
    """
    # Decide the scaling from the input dtype: mixing a multi-channel integer
    # array with ``mean`` promotes it to float, but those samples are still in
    # the int16 value range and must not be rescaled as if normalised.
    is_float = np.issubdtype(data.dtype, np.floating)
    if data.ndim == 1:
        samples = data
    elif data.ndim == 2:
        samples = data[0] if data.shape[0] == 1 else data.mean(axis=0)
    else:
        raise ValueError(f"expected a 1-D or 2-D audio array, got shape {data.shape}")
    if is_float:
        return (samples * _INT16_MAX).clip(-_INT16_MAX - 1, _INT16_MAX).astype(np.int16)
    return np.rint(samples).astype(np.int16)


def inbound_audio_to_mono(pcm: bytes, channels: int) -> npt.NDArray[np.int16]:
    """Decode interleaved int16 PCM from libwebrtc into ``(1, M)`` mono samples."""
    samples = np.frombuffer(pcm, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).round().astype(np.int16)
    return samples.reshape(1, -1)
