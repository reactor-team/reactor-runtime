"""A GPU model half for the split echo: the effect plus a pass through weights on the GPU.

Used by the pod test's GPU check only. The weights are one large tensor, so
the GPU's memory shows whether one copy of them is loaded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch  # ty: ignore[unresolved-import]  # installed where the GPU check runs
from echo_model import EchoInput, EchoResult

from reactor_runtime.model_server import ModelServer, ServerRequest

WEIGHT_BYTES = 2 << 30


class GpuEchoModel(ModelServer[EchoInput, EchoResult]):
    """Invert each session's frame on the GPU, batched, after a matmul with the weights."""

    max_batch = 4
    batch_window = 0.002

    def load(self, config_path: Path | None) -> None:
        """Put the weights on the GPU."""
        assert self.device == "cuda", self.device
        side = int((WEIGHT_BYTES // 2) ** 0.5)
        self.weights = torch.randn(side, side, device="cuda", dtype=torch.float16)
        self.probe = torch.randn(1, side, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()

    def step(self, requests: list[ServerRequest[EchoInput]]) -> list[EchoResult]:
        """Invert every frame in one batched pass on the GPU."""
        frames = [np.asarray(r.payload.frame) for r in requests]
        if len({f.shape for f in frames}) != 1:
            return [EchoResult(frame=self._one(f)) for f in frames]
        batch = torch.from_numpy(np.stack(frames)).to("cuda", non_blocking=True)
        (self.probe @ self.weights).sum().item()
        out = (255 - batch).cpu().numpy()
        return [EchoResult(frame=np.ascontiguousarray(frame)) for frame in out]

    def _one(self, frame: np.ndarray) -> np.ndarray:
        return (255 - torch.from_numpy(frame.copy()).to("cuda")).cpu().numpy()
