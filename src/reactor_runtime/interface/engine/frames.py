"""Normalizing an engine's video chunk for the wire — :func:`to_video_frames`.

An engine returns whatever its tensor library produces: a device tensor in
``[T, C, H, W]``, half or bfloat16, in whatever value range its decoder was
trained against. A transport wants host memory, channels last, ``uint8`` RGB.
This is the one place that gap is closed, so no engine has to know what a
Reactor track expects and no model code carries a conversion.

Layout is read off the shape, since the channel axis of a colour frame is the
only one that is three wide. The value range is not read off the data — a dark
``[0, 1]`` chunk and a bright ``[-1, 1]`` chunk are indistinguishable by their
extremes — so it is declared instead, defaulting to the ``[-1, 1]`` the
diffusion decoders these engines ship use.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

DEFAULT_VALUE_RANGE: tuple[float, float] = (-1.0, 1.0)
"""The floating-point range a decoded chunk is normalized from."""

_RGB = 3


def to_video_frames(
    chunk: Any, value_range: tuple[float, float] = DEFAULT_VALUE_RANGE
) -> npt.NDArray[np.uint8]:
    """Convert one step's decoded video into wire-ready frames.

    Args:
        chunk: What the engine's ``generate`` returned — a tensor exposing the
            usual ``detach``/``cpu``/``numpy`` methods, or a NumPy array.
            Channels may lead or trail, and a single frame may omit the time
            axis.
        value_range: The range a floating-point chunk spans, low to high.
            Ignored for an integer chunk, which is taken as ``0-255`` already.

    Returns:
        ``uint8`` RGB frames, ``(H, W, 3)`` for a single frame or
        ``(N, H, W, 3)`` for a batch.

    Raises:
        ValueError: If the chunk has no axis that could be RGB channels, or
            carries more dimensions than a batch of frames can.
    """
    array = _drop_leading_axes(_as_numpy(chunk))
    return _to_uint8(_channels_last(array), value_range)


def _as_numpy(chunk: Any) -> npt.NDArray[Any]:
    """Bring a chunk into host memory as a NumPy array.

    Duck-typed rather than importing a framework: the contract is deliberately
    free of one, and so is this. A floating-point tensor is widened to single
    precision first, because NumPy has no bfloat16 to convert into.
    """
    if isinstance(chunk, np.ndarray):
        return chunk
    tensor = chunk.detach() if hasattr(chunk, "detach") else chunk
    if getattr(getattr(tensor, "dtype", None), "is_floating_point", False) and hasattr(
        tensor, "float"
    ):
        tensor = tensor.float()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        return np.asarray(tensor.numpy())
    return np.asarray(tensor)


def _drop_leading_axes(array: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Drop the leading batch axes an engine adds for a batch shape of one.

    Raises:
        ValueError: If more than one sequence of frames is left after squeezing.
    """
    while array.ndim > 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim > 4:
        raise ValueError(
            f"a video chunk of shape {array.shape} carries more than one sequence of "
            "frames; the runtime serves one rollout, so the engine emits one."
        )
    return array


def _channels_last(array: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Move the RGB channel axis last, whichever end the engine put it on.

    Raises:
        ValueError: If neither candidate axis is three wide.
    """
    if array.ndim == 3:
        if array.shape[-1] == _RGB:
            return array
        if array.shape[0] == _RGB:
            return array.transpose(1, 2, 0)
    elif array.ndim == 4:
        if array.shape[-1] == _RGB:
            return array
        if array.shape[1] == _RGB:
            return array.transpose(0, 2, 3, 1)
    raise ValueError(
        f"a video chunk of shape {array.shape} has no RGB channel axis; expected "
        "[T, C, H, W] or [T, H, W, C], with or without the leading time axis."
    )


def _to_uint8(array: npt.NDArray[Any], value_range: tuple[float, float]) -> npt.NDArray[np.uint8]:
    """Scale a chunk into ``uint8``, clipping whatever falls outside its range."""
    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.floating):
        low, high = value_range
        span = high - low
        scaled = (array.astype(np.float32) - low) / (span if span else 1.0)
        return (np.clip(scaled, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return np.clip(array, 0, 255).astype(np.uint8)
