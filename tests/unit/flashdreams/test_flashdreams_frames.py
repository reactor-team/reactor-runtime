"""Frame conversion from the pipeline's tensor to uint8 frames. Needs torch; skips without it."""

from __future__ import annotations

import numpy as np
import pytest

from reactor_runtime.flashdreams import model as model_module


def test_to_frames_converts_a_batched_video_to_uint8_frames() -> None:
    torch = pytest.importorskip("torch")
    video = torch.linspace(-1.5, 1.5, 2 * 3 * 4 * 5).reshape(1, 2, 3, 4, 5)
    frames = model_module._to_frames(video)
    assert isinstance(frames, np.ndarray)
    assert frames.shape == (2, 4, 5, 3)
    assert frames.dtype == np.uint8
    assert frames.min() == 0
    assert frames.max() == 255


def test_to_frames_accepts_an_unbatched_video_and_keeps_three_channels() -> None:
    torch = pytest.importorskip("torch")
    video = torch.zeros(2, 4, 4, 5)
    frames = model_module._to_frames(video)
    assert frames.shape == (2, 4, 5, 3)
    assert (frames == 128).all()
