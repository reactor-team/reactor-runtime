# Shared worker example

A drifting color field, served to several sessions at once from one copy of
its "weights". It runs on a CPU and needs nothing but NumPy.

This is the reference for a model split into a model half that serves every
session and an application half per session, the split
[`skills/application-model-isolation`](../../skills/application-model-isolation/SKILL.md)
describes with the model half in its own process:

- `plasma_types.py` holds what the halves exchange, `Tick` and `Frame`,
  frozen dataclasses both halves import.
- `plasma_model.py` is the model half, a `ModelServer`. The model server
  process loads it once; it renders the frames of every waiting session in
  one `step()` call, up to `max_batch` at a time.
- `plasma.py` is the application half, an ordinary `ReactorApp`. Each session
  runs it in its own runtime process, and it resets its slot when a session
  starts. Its three step hooks split as the skill says:
  - `process_input()`, application code, builds the session's `Tick` from
    the hue the client set and the tick it keeps;
  - `generate()` is the one-line model call, `self.model.step(input)`;
  - `process_output()`, application code, puts the `Frame` on the video
    track, or re-raises what went wrong.

`reactor.yaml` names both halves:

```yaml
runtime:
  import: plasma:Plasma
  worker:
    import: plasma_model:PlasmaModel
    max_batch: 4
```

## When the model server goes away

A session runtime never runs on a dead link. Mid-session, the step raises
`WorkerLost` and `process_output()` re-raises it; between sessions, the
client's `on_lost` calls `self.crash()`:

```python
self.model = ModelClient.connect(on_lost=self.crash)
```

Either way the session ends with the error, the process exits as a failure,
and its restart waits in `load()` until the model server welcomes it again.

## Run

The model server and the sessions are separate processes on one host. They
share the model server's socket, and each slot has a shared-memory directory
of its own:

```sh
cd examples/shared_worker
mkdir -p /tmp/plasma /dev/shm/plasma/s0 /dev/shm/plasma/s1
export REACTOR_MODEL_WORKER_SOCKET=/tmp/plasma/sock
REACTOR_MAX_SESSIONS=2 REACTOR_SHM_SLOT_DIR='/dev/shm/plasma/s{slot}' \
  python -m reactor_runtime.model_server &
PORT=8080 REACTOR_SESSION_SLOT=0 REACTOR_SHM_DIR=/dev/shm/plasma/s0 \
  python -m reactor_runtime.serve &
PORT=8081 REACTOR_SESSION_SLOT=1 REACTOR_SHM_DIR=/dev/shm/plasma/s1 \
  python -m reactor_runtime.serve &
```

Each session runtime serves the usual API on its own port, and the model
server answers `GET /health` on port 8090. `deploy/pod.yaml` is the same
layout as a Kubernetes pod, with the model server as a native sidecar, and
`deploy/compose.yaml` as a Docker Compose project.

## Commands

- `set_speed` palette steps per frame, `0` to `16`.
- `set_hue` palette offset, `0` to `255`.
