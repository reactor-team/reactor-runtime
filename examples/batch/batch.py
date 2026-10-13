"""Batch: one small network painting a video for every session, in one forward pass.

The example of :class:`~reactor_runtime.BatchedReactorApp`. Each session holds
a latent, seeded from the client's ``seed``, that a small recurrent
convolutional network advances one step per frame and decodes to a picture.
The runtime hosts up to ``max_sessions`` sessions in this one process and hands
``generate_batch`` the input of every session that takes the step, so the
network runs once for all of them: the latents are stacked into one batch, and
the frames come back split per session.

Everything per session lives where a single-session model keeps it.
``self.state`` is the calling session's state in a handler, a hook, and
``process_input``; the session's latent is kept by its id, made in
``@session_started`` and dropped in ``@session_ended``.

The network runs on PyTorch when it is installed, on the GPU when there is
one, and on NumPy otherwise, so the example also runs where neither PyTorch
nor a GPU is present. The weights are drawn from a fixed seed, so a session's
frames depend only on its own ``seed`` and ``tint``: the same session yields
the same frames whether it runs alone or batched with others.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from reactor_runtime import (
    BatchedReactorApp,
    InputField,
    InputState,
    Output,
    StepCompleted,
    StepOutcome,
    Video,
    session_ended,
    session_started,
)

logger = logging.getLogger(__name__)

# The seed the network's weights are drawn from. Fixed, so every process
# serves the same network.
WEIGHT_SEED = 1234


class BatchOutput(Output):
    """The painted video, one frame per step."""

    main_video: Video


class BatchState(InputState):
    """What a client sets. Each public field is a ``set_<field>`` command."""

    seed: int = InputField(
        default=0,
        ge=0,
        le=2**31 - 1,
        description="Seeds the session's latent. Setting it restarts the picture.",
    )
    tint: float = InputField(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Pulls the colours from cold (0) to warm (1).",
    )


class Network(Protocol):
    """The batched network: one call advances every latent in the batch."""

    def latent(self, seed: int) -> Any:
        """Return the starting latent for *seed*."""
        ...

    def step(
        self, latents: list[Any], tints: list[float], steps: list[int]
    ) -> tuple[list[Any], np.ndarray]:
        """Advance each latent one step and decode it.

        Returns:
            The next latents, and the frames as one ``(B, H, W, 3)`` uint8 array.
        """
        ...


class Batch(BatchedReactorApp):
    """Serve up to four sessions with one network and one forward pass per step."""

    state: BatchState
    max_sessions = 4

    def load(self, config_path: Path | None) -> None:
        """Build the network the config describes.

        Args:
            config_path: ``config.yaml``: the frame size, the latent's size,
                channel count, and depth, and the backend (``auto``,
                ``torch``, or ``numpy``).
        """
        config: dict[str, Any] = {}
        if config_path is not None:
            config = yaml.safe_load(config_path.read_text()) or {}
        self.net = build_network(config)
        self.latents: dict[str, Any] = {}
        self.steps: dict[str, int] = {}
        self.seeds: dict[str, int] = {}
        logger.info("batch network built: %s", type(self.net).__name__)

    @session_started
    def begin(self) -> None:
        """Give the new session its own latent, from its seed."""
        assert self.session_id is not None
        self._restart(self.session_id, self.state.seed)

    @session_ended
    def end(self) -> None:
        """Drop the ended session's latent."""
        assert self.session_id is not None
        self.latents.pop(self.session_id, None)
        self.steps.pop(self.session_id, None)
        self.seeds.pop(self.session_id, None)

    async def process_input(self) -> BatchState:
        """Restart the picture when the client changed the seed, then step as usual."""
        assert self.session_id is not None
        if self.seeds.get(self.session_id) != self.state.seed:
            self._restart(self.session_id, self.state.seed)
        return self.state

    def generate_batch(self, inputs: dict[str, BatchState]) -> dict[str, BatchOutput]:
        """Advance every session's latent in one forward pass and decode their frames."""
        ids = list(inputs)
        latents, frames = self.net.step(
            [self.latents[sid] for sid in ids],
            [inputs[sid].tint for sid in ids],
            [self.steps[sid] for sid in ids],
        )
        for sid, latent in zip(ids, latents, strict=True):
            self.latents[sid] = latent
            self.steps[sid] += 1
        return {sid: BatchOutput(main_video=frame) for sid, frame in zip(ids, frames, strict=True)}

    async def process_output(self, outcome: StepOutcome) -> StepCompleted:
        """Emit the frame, and keep its fingerprint with the step.

        ``frame.sha256`` lets a client check that a session's frames are the
        ones it gets when it runs alone.
        """
        if outcome.error is not None:
            raise outcome.error
        output = outcome.result
        digest = hashlib.sha256(np.ascontiguousarray(output.main_video).tobytes()).hexdigest()
        return StepCompleted(output=output, files={"frame.sha256": digest.encode()})

    def _restart(self, session_id: str, seed: int) -> None:
        self.latents[session_id] = self.net.latent(seed)
        self.steps[session_id] = 0
        self.seeds[session_id] = seed


# -- The network ---------------------------------------------------------------


def build_network(config: dict[str, Any]) -> Network:
    """Build the network *config* describes, on the backend it names or the best present."""
    shape = {
        "width": int(config.get("width", 256)),
        "height": int(config.get("height", 256)),
        "latent": int(config.get("latent", 32)),
        "channels": int(config.get("channels", 16)),
        "layers": int(config.get("layers", 2)),
    }
    backend = str(config.get("backend", "auto"))
    if backend in ("auto", "torch"):
        try:
            importlib.import_module("torch")
        except ImportError:
            if backend == "torch":
                raise
        else:
            return TorchNetwork(**shape, device=config.get("device"))
    return NumpyNetwork(**shape)


def draw_weights(channels: int, layers: int) -> tuple[list[np.ndarray], np.ndarray]:
    """Draw the network's weights from :data:`WEIGHT_SEED`.

    Each hidden layer is a 3x3 convolution scaled so the recurrence contracts:
    a latent settles into a slowly moving picture instead of blowing up.
    """
    rng = np.random.default_rng(WEIGHT_SEED)
    scale = 0.9 / np.sqrt(channels * 9)
    hidden = [
        (rng.standard_normal((channels, channels, 3, 3)) * scale).astype(np.float32)
        for _ in range(layers)
    ]
    out = rng.standard_normal((3, channels)).astype(np.float32) / np.sqrt(channels)
    return hidden, out


def seed_latent(seed: int, channels: int, latent: int) -> np.ndarray:
    """Return the starting latent for *seed*, as ``(C, h, w)`` float32."""
    return (
        np.random.default_rng(seed).standard_normal((channels, latent, latent)).astype(np.float32)
    )


class NumpyNetwork:
    """The network on NumPy: every layer is one batched contraction over the whole batch."""

    def __init__(self, *, width: int, height: int, latent: int, channels: int, layers: int) -> None:
        self.width, self.height, self.size, self.channels = width, height, latent, channels
        self.hidden, self.out = draw_weights(channels, layers)

    def latent(self, seed: int) -> np.ndarray:
        """Return the starting latent for *seed*."""
        return seed_latent(seed, self.channels, self.size)

    def step(
        self, latents: list[np.ndarray], tints: list[float], steps: list[int]
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Advance each latent one step and decode it, as one batch."""
        z = np.stack(latents)
        drive = np.sin(0.1 * np.asarray(steps, dtype=np.float32))[:, None, None, None]
        for weight in self.hidden:
            z = np.tanh(_conv3x3(z, weight) + 0.5 * drive)
        rgb = 1.0 / (1.0 + np.exp(-np.einsum("oc,bchw->bohw", self.out, z)))
        frames = _to_frames(rgb.transpose(0, 2, 3, 1), np.asarray(tints, dtype=np.float32))
        return list(z), _upsample(frames, self.height, self.width)


def _conv3x3(z: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Convolve a ``(B, C, h, w)`` batch with a ``(O, C, 3, 3)`` kernel, zero-padded."""
    padded = np.pad(z, ((0, 0), (0, 0), (1, 1), (1, 1)))
    h, w = z.shape[2:]
    out = np.zeros((z.shape[0], weight.shape[0], h, w), dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            window = padded[:, :, dy : dy + h, dx : dx + w]
            out += np.einsum("oc,bchw->bohw", weight[:, :, dy, dx], window)
    return out


def _to_frames(rgb: np.ndarray, tints: np.ndarray) -> np.ndarray:
    """Tint ``(B, h, w, 3)`` values in [0, 1] and quantise them to uint8."""
    warm = np.array([1.0, 0.75, 0.5], dtype=np.float32)
    cold = np.array([0.5, 0.75, 1.0], dtype=np.float32)
    tint = tints[:, None, None, None]
    colour = cold + (warm - cold) * tint
    frames: np.ndarray = np.clip(np.rint(rgb * colour * 255.0), 0, 255).astype(np.uint8)
    return frames


def _upsample(frames: np.ndarray, height: int, width: int) -> np.ndarray:
    """Scale ``(B, h, w, 3)`` frames to ``(B, height, width, 3)``, nearest neighbour."""
    rows = np.arange(height) * frames.shape[1] // height
    cols = np.arange(width) * frames.shape[2] // width
    upsampled: np.ndarray = frames[:, rows][:, :, cols]
    return upsampled


class TorchNetwork:
    """The network on PyTorch: one batched convolution per layer, on the GPU when present."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        latent: int,
        channels: int,
        layers: int,
        device: str | None = None,
    ) -> None:
        torch: Any = importlib.import_module("torch")
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.width, self.height, self.size, self.channels = width, height, latent, channels
        hidden, out = draw_weights(channels, layers)
        self.hidden = [torch.from_numpy(w).to(self.device) for w in hidden]
        self.out = torch.from_numpy(out)[:, :, None, None].to(self.device)
        self.warm = torch.tensor([1.0, 0.75, 0.5], device=self.device)
        self.cold = torch.tensor([0.5, 0.75, 1.0], device=self.device)

    def latent(self, seed: int) -> Any:
        """Return the starting latent for *seed*, on the network's device."""
        return self.torch.from_numpy(seed_latent(seed, self.channels, self.size)).to(self.device)

    def step(
        self, latents: list[Any], tints: list[float], steps: list[int]
    ) -> tuple[list[Any], np.ndarray]:
        """Advance each latent one step and decode it, as one batch."""
        torch = self.torch
        functional = torch.nn.functional
        with torch.inference_mode():
            z = torch.stack(latents)
            drive = torch.sin(0.1 * torch.tensor(steps, dtype=torch.float32, device=self.device))
            drive = drive[:, None, None, None]
            for weight in self.hidden:
                z = torch.tanh(functional.conv2d(z, weight, padding=1) + 0.5 * drive)
            rgb = torch.sigmoid(functional.conv2d(z, self.out))
            tint = torch.tensor(tints, device=self.device)[:, None]
            colour = (self.cold + (self.warm - self.cold) * tint)[:, :, None, None]
            frames = torch.clamp(torch.round(rgb * colour * 255.0), 0, 255).to(torch.uint8)
            frames = functional.interpolate(
                frames.float(), size=(self.height, self.width), mode="nearest"
            )
            frames = frames.to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
        return list(z), frames
