# Shared worker example

A drifting color field, served to several sessions at once from one copy of
its "weights". It runs on a CPU and needs nothing but NumPy.

This is the reference for a model split into a model half that serves every
session and an application half per session, the split
[`skills/application-model-isolation`](../../skills/application-model-isolation/SKILL.md)
describes with the model half in its own process:

- `plasma_contract.py` declares what the halves exchange, `Tick` and `Frame`,
  frozen dataclasses both halves import.
- `plasma_worker.py` is the model half, a `ModelWorker`. The worker process
  loads it once; it renders the frames of every waiting session in one
  `step()` call, up to `max_batch` at a time.
- `plasma.py` is the application half, an ordinary `ReactorApp`. Each session
  runs it in its own runtime process; it resets its slot on the worker when a
  session starts, and its `generate()` sends the session's `Tick` through a
  `WorkerClient` and returns the frame. When the worker is lost or stuck it
  ends the session with the reason and stays up, refusing sessions until its
  watcher thread has reconnected (option 2 of `reactor_runtime.model_worker`).

`reactor.yaml` names both halves:

```yaml
runtime:
  import: plasma:Plasma
  worker:
    import: plasma_worker:PlasmaWorker
    max_batch: 4
```

## Run

The worker and the sessions are separate processes on one host, sharing a
directory for the worker's socket and `/dev/shm` for the frames:

```sh
cd examples/shared_worker
export REACTOR_MODEL_WORKER_SOCKET=/tmp/plasma/sock
python -m reactor_runtime.model_worker &
PORT=8080 REACTOR_SESSION_SLOT=0 python -m reactor_runtime.serve &
PORT=8081 REACTOR_SESSION_SLOT=1 python -m reactor_runtime.serve &
```

Each session runtime serves the usual API on its own port. `deploy/pod.yaml`
is the same layout as a Kubernetes pod, with the worker as a native sidecar,
and `deploy/compose.yaml` as a Docker Compose project.

## Commands

- `set_speed` palette steps per frame, `0` to `16`.
- `set_hue` palette offset, `0` to `255`.
