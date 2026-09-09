# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Serve multi-GPU video with the experimental managed worker lifecycle."""

from __future__ import annotations

from typing import Any

import numpy as np

from reactor_runtime import (
    InputField,
    Output,
    Video,
    event,
    session_started,
)
from reactor_runtime.distributed import DistributedVideoModel, DistributedWorker

FRAME_SHAPE = (4, 96, 128, 3)


class VideoOutput(Output):
    """One RGB video track."""

    main_video: Video


class StripeWorker(DistributedWorker):
    """Render disjoint horizontal bands; every rank contributes to each frame."""

    def setup(self, **setup_kwargs: Any) -> None:
        """Allocate coordinates once, on this rank's injected device."""
        if self.device == "cpu" and setup_kwargs.get("require_cuda", True):
            raise RuntimeError(
                "CUDA is unavailable. Use a CUDA-enabled PyTorch image and "
                "run `reactor run --gpus all` on an NVIDIA host."
            )
        self.lo = self.rank * FRAME_SHAPE[1] // self.world_size
        self.hi = (self.rank + 1) * FRAME_SHAPE[1] // self.world_size
        self.torch: Any = None
        if self.device.startswith("cuda"):
            import torch  # ty: ignore[unresolved-import] -- supplied by the model image

            self.torch = torch
            self.x = torch.arange(FRAME_SHAPE[2], device=self.device)
        else:
            self.x = np.arange(FRAME_SHAPE[2])

    def start_session(self, params: dict[str, Any]) -> None:
        """Begin a stateless render sequence."""

    def generate_chunk(self, index: int, controls: dict[str, Any]) -> int:
        """Write only this rank's band, returning the complete chunk's end row."""
        phase = index * FRAME_SHAPE[0] + np.arange(FRAME_SHAPE[0])
        if self.torch is not None:
            phase = self.torch.as_tensor(phase, device=self.device)
        pixels: Any = (self.x[None, :] + phase[:, None] + self.rank * 40) % 256
        pixels = pixels * float(controls["brightness"])
        if self.torch is not None:
            pixels = pixels.to(self.torch.uint8).cpu().numpy()
        else:
            pixels = pixels.astype(np.uint8)
        self.frames.array[:, self.lo : self.hi] = pixels[:, None, :, None]
        return FRAME_SHAPE[0]

    def end_session(self) -> None:
        """Release per-session resources (this procedural worker holds none)."""


class MultiGpuVideo(DistributedVideoModel):
    """Serve one shared video sequence, with controls applied between chunks."""

    fps = 24
    buffer_size = 8
    worker = StripeWorker
    frame_shape = FRAME_SHAPE
    brightness = 1.0

    @session_started
    def on_session_start(self) -> None:
        """Start each session with the model's default conditioning."""
        self.brightness = 1.0

    @event(name="set_brightness")
    async def set_brightness(self, brightness: float = InputField(ge=0.0, le=1.0)) -> None:
        """Set the brightness for the next chunk."""
        self.brightness = brightness

    @event(name="set_paused")
    async def set_paused(self, paused: bool) -> None:
        """Pause or resume while keeping the current worker session."""
        self.paused = paused

    @event(name="reset")
    async def reset(self) -> None:
        """Start the video sequence over, preserving current controls."""
        self.restart_generation()

    def controls(self) -> dict[str, Any]:
        """Read the latest values; the adapter snapshots them for every rank."""
        return {"brightness": self.brightness}

    def to_output(self, frames: np.ndarray) -> VideoOutput:
        """Route the assembled frames to the declared video track."""
        return VideoOutput(main_video=frames)
