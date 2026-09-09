# Experimental multi-GPU video

Write per-GPU computation; let `DistributedVideoModel` handle startup, sessions,
control snapshots, pause/resume, restarts, and cleanup. It extends `ReactorModel`,
so the same `@event`, lifecycle hooks, and typed `Output` work. No engine host
is required.

**Scope:** one node, one session, uint8 video. Several viewers share the same
sequence. This is not an action-output API or a multi-session scheduler. The
adapter and raw primitives remain experimental and off the package root.

## What you write

Once your `DistributedWorker` implements the per-rank computation, the serving
side is small. This is the minimal shape; [model.py](model.py) adds session
defaults, pause, and reset commands:

```python
from reactor_runtime import InputField, Output, Video, event
from reactor_runtime.distributed import DistributedVideoModel


class VideoOutput(Output):
    main_video: Video


class MyVideo(DistributedVideoModel):
    worker = MyWorker
    frame_shape = (4, 96, 128, 3)
    fps = 24
    brightness = 1.0

    @event(name="set_brightness")
    async def set_brightness(self, brightness: float = InputField(ge=0, le=1)) -> None:
        self.brightness = brightness

    def controls(self):
        return {"brightness": self.brightness}

    def to_output(self, frames):
        return VideoOutput(main_video=frames)
```

`MyWorker` is a module-level `DistributedWorker` subclass. The runtime sets each
worker's `rank`, `world_size`, `device`, and `frames` before calling `setup()`.
A rank may return a complete uint8 array (usually only the leader does), or write its
disjoint slice and return the exclusive end row. The complete example renders
horizontal stripes, with every rank contributing to every frame.

There is no `WorkerGroup`, executor, epoch counter, or custom `run()` to copy.
Use `WorkerGroup` directly only for custom orchestration.

## Configure once

In [reactor.yaml](reactor.yaml), `model.resources.gpu.count` is the worker count.
Set it to `1`, `2`, or the number your parallel layout requires. No Python code
or second config value needs changing. Extra visible GPUs do not change the
layout. One worker runs in-process without spawning, shared memory, or a process
group. Guard distributed-only calls in the worker with `self.world_size > 1`.

Missing or zero GPU counts also mean one worker; they do not force a CPU device.
This example requires CUDA. Its CPU tests explicitly disable that check and
process-group initialization.

## Run this checkout

Prerequisites: the [Reactor CLI](https://docs.reactor.inc/deploy/platform/installation),
Docker with NVIDIA GPU access, and a Linux NVIDIA host with the requested GPUs
and a driver compatible with CUDA 12.8. A source checkout also uses the
repository's [mise toolchain](../../README.md#development).

This API is not in the released 3.3.0 wheel. The task below packages **this
checkout** and stages the exact wheel for the image; no manual wheel copying or
direct package install on your host is needed. From the repository root:

```sh
mise run example:multi-gpu
cd examples/multi_gpu
reactor run --gpus all
```

The initial CUDA image downloads several GB and can take several minutes.
`reactor run` reuses the built local image; rebuilding is only needed after
changing code. Rebuild speed depends on the CLI/builder preserving its layer
cache, so a quick rebuild is not guaranteed.

Connect with the [JS SDK](https://docs.reactor.inc/deploy/overview) to stream the
animated horizontal bands in your application. Check liveness with
`curl http://localhost:8080/health`. Re-run the build task after code changes.

| Command | Arguments | Effect |
| --- | --- | --- |
| `set_brightness` | `{"brightness": 0.5}` | Apply to the next computation. |
| `set_paused` | `{"paused": true}` / `false` | Pause/resume without reloading weights or resetting worker state. |
| `reset` | `{}` | Restart at chunk zero, preserving current controls. |

This procedural renderer demonstrates lifecycle and sharding, not a multi-GPU
speed benchmark.

## Plug in a real model

- Load weights in the worker's `setup()`, using `self.device`.
- Override `worker_setup(config_path)` on the model to read `runtime.config`
  and return kwargs such as `{"weights_path": "/weights/model"}`. Override
  `session_params()` for per-sequence initialization. Leave `load()` and
  `run()` to the adapter.
- Keep `@event` handlers in the model. `controls()` is copied once per chunk,
  including nested values. Return picklable CPU data, paths, and scalars, not
  GPU objects or live handles.
- Use `@session_started` to reset your controls for each session. User hooks
  cannot bypass the adapter's own lifecycle bookkeeping.
- Pin `fps`, or set `adaptive_fps = True` to derive playout rate from compute
  throughput. Time spent paused is excluded.

## Lifecycle and troubleshooting

Control changes apply at computation boundaries. Pausing flushes queued playout
and holds one in-flight result until resume. Resetting discards that result and
reinitializes the worker session once the current call finishes. Last-viewer
disconnects also restart the sequence; new sessions reset the example's
brightness. Weights remain loaded. Collectives are never interrupted for a
newer control value.

Cancellation drains the active call before releasing buffers. Worker crashes,
timeouts, and inference errors fail the model and shut down its group; the
adapter does not silently retry a partial world. Multi-worker deadlines are
configurable via `startup_timeout`, `command_timeout`, and `shutdown_timeout`.
One-worker hooks run inline and cannot be preempted by those timeouts. Idle rank
death is detected at the next command; there is no standing watchdog.

- **CUDA unavailable:** use the supplied CUDA-enabled image and expose GPUs with
  `reactor run --gpus all` on an NVIDIA host.
- **Too few visible devices:** lower the manifest count or expose the requested
  devices; do not add a second worker-count setting.
- **Frame-shape error:** return uint8 frames fitting `frame_shape`. Ranks must
  cover all returned pixels without overlapping writes. Size container shared
  memory for the buffer plus NCCL's own allocations.
- **Pickling error:** pass host-side values from `controls()`, `session_params()`,
  and `worker_setup()`. These errors are reported before queue dispatch.

The framework initializes the process group and performs a clean-shutdown
barrier; compute collectives and sharding belong to the worker. Every rank must
enter collectives in the same order. The video buffer does not carry arbitrary
actions, tokens, or tensors; do not present this as the DreamZero action-output
contract. Future execution adapters may reuse the worker hooks, but experimental
signature compatibility is not promised.
