# Shared worker example

A drifting color field, served to several sessions at once from one copy of
its "weights". It runs on a CPU and needs nothing but NumPy.

This is the reference for a model split into a worker half and a session half:

- `plasma_worker.py` is the worker half, a `ModelWorker`. The launcher loads it
  once, in its own process; it renders the frames of every waiting session in
  one `step()` call, up to `max_batch` at a time.
- `plasma.py` is the session half, an ordinary `ReactorApp`. Each session runs
  it in its own runtime process; its `generate()` sends the session's tick and
  hue through a `WorkerClient` and returns the frame.

`reactor.yaml` names both halves:

```yaml
runtime:
  import: plasma:Plasma
  worker: plasma_worker:PlasmaWorker
```

## Run

The worker and the sessions are separate processes on one host, sharing a
directory for the worker's socket and `/dev/shm` for the frames:

```sh
cd examples/shared_worker
export REACTOR_MODEL_WORKER_SOCKET=/tmp/plasma/sock
reactor-runtime serve-worker &
PORT=8080 REACTOR_SESSION_SLOT=0 reactor-runtime serve &
PORT=8081 REACTOR_SESSION_SLOT=1 reactor-runtime serve &
```

Each session runtime serves the usual API on its own port. `deploy/pod.yaml`
is the same layout as a Kubernetes pod, with the worker as a native sidecar,
and `deploy/compose.yaml` as a Docker Compose project.

## Commands

- `set_speed` palette steps per frame, `0` to `16`.
- `set_hue` palette offset, `0` to `255`.
