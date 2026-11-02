# Serving the port

Sections 1 and 2 (build and serve) are also how Phase 5's replay gets a served
port. The rest of this file is Phase 6: it starts with a ported workspace whose
replay already matches the baseline, and ends with the model built into an
image, served on the GPU node, healthy, with every declared command proven to
work through the real dispatch, and reachable from the
[Reactor Sandbox](https://reactor-sandbox.vercel.app/) so a person can play it.

Before reporting, the agent proves the served path works. The person's job is
to judge the output. Finding broken wiring is not their job.

## 1. Build

Build from the port's workspace, `<slug>-<you>-<YYYYMMDD>` (SKILL.md Phase 0);
its directory name is the image tag:

```sh
cd <slug>-<you>-<YYYYMMDD>
reactor build          # tags reactor-local/<slug>-<you>-<YYYYMMDD>:dev
```

Building needs no GPU; a CUDA extension compiled during the build needs
`TORCH_CUDA_ARCH_LIST` in `build.build_env`
([`source-and-weights.md`](source-and-weights.md) §4). Rebuild after every
source change to the workspace: `reactor run` builds only when the tag is
missing, and otherwise it boots whatever image already holds that tag.

## 2. Serve

```sh
reactor run --gpus device=<N> --weights <weights-root> --shm-size 8g --port 8080
```

- **`--gpus`.** Without this flag no GPU is attached. `device=N` pins one
  host GPU, which appears as device 0 inside the container. Take `N` from the
  fail-closed process probe in [`source-and-weights.md`](source-and-weights.md)
  §1, run again right before launch.
- **`--weights`.** The weights root staged in Phase 1. The CLI takes
  `--weights`, else `$REACTOR_WEIGHTS_PATH` in your shell, else
  `runtime.weights_path` in `reactor.yaml`; it mounts the root at the same
  absolute path inside the container, read-write, and exports it as
  `REACTOR_WEIGHTS_PATH`. Passing `--weights` explicitly makes the run
  independent of the shell.
- **`--shm-size`.** The runtime passes step inputs and results between
  processes through shared memory. The 2 GB default is too small for large
  frame chunks.
- Environment variables go in with `-e KEY` or `-e KEY=VALUE`. Arguments after
  `--` are forwarded to the runtime entrypoint unchanged.

Run the server in a detached terminal on the node (`tmux new -d -s serve '…'`)
and send its log to a file. The checks below read that file.

### Ready means healthy and available

Poll `/health` until the response says the model has loaded:

```sh
until curl -fsS localhost:8080/health \
  | jq -e '.status == "healthy" and (.state == "available" or .state == "serving")'
do sleep 5; done
```

`state` is one of `loading`, `available`, `serving` and `terminated`. The
endpoint answers as soon as its socket binds, before the model has loaded, so
a 200 response with `state: loading` means the model is not ready yet. An
unhealthy process answers 503. Measure the time from launch to `available` and
record it as the startup time.

### The contract

```sh
curl -s localhost:8080/schema > schema.json
```

`/schema` returns the served model's OpenAPI document: its commands, their
arguments and descriptions, its messages, and its input and output tracks.
The checks in §3 drive exactly what this document declares, and the Sandbox
builds its controls from the same document.

## 3. Prove the served path

The replay stepped the model one request at a time. This step drives the same
code the way a client does: a live, streaming session. The client is the
Python SDK on the node, the same one the replay uses
([`baseline-and-replay.md`](baseline-and-replay.md) §5.1).

1. **Drive every declared command once,** with a valid value, inside a live
   session: one connection, each command from `schema.json` in turn, then one
   input for each refusal code the model declares, to prove the refusal
   arrives as a coded failure ([`port-on-reactor-app.md`](port-on-reactor-app.md)
   §7 has what each SDK sees).
2. **Open a second session on the same server** and repeat. Its first step
   must behave like the first session's: the same latency band, no leftover
   image, prompt or cache. A process that serves only once is not ready to
   hand over.
3. **Read the log for failures that never reached the client:**

   ```sh
   grep -nE 'error in command handler|model run\(\) crashed' serve.log
   ```

   `error in command handler` is an exception a handler raised other than a
   `CommandError`. The client saw only a generic `internal_error`, and the
   traceback is in the log. `model run() crashed` means the step loop died.
   Any hit is a porting bug, and it is fixed before the port is handed over.
   (A step the application refused is logged as `step refused` at DEBUG
   level only; serve with `-e REACTOR_LOG_LEVEL=DEBUG` to see those.)

### The timing numbers, and where they come from

The report and the checklist need a few numbers about speed, and they come
from two places:

| Number | Used for | Where it comes from |
|---|---|---|
| Every step's generation time | the slowest step and any stall (checklist A1); generation fps, steady and lowest, set against the trained fps (the report) | `StepOutcome.elapsed`, logged per step ([`checklist.md`](checklist.md) §1) |
| How each step's time splits between the model (`generate`) and the application code around it (`process_input`, `process_output`), and how long the loop waited for input (`input_wait`) | whether the application half is eating the budget: decoding an upload, copying frames on the host (checklist B1, B3) | `runtime_stats` |
| The fps actually delivered, and the longest gap between two frames a viewer received | whether a viewer would see a stall the step times do not show (pacing, a slow send) | `runtime_stats` |
| Input frames dropped | a model that reads a video in and falls behind it | `runtime_stats` |

`runtime_stats` is a reading the runtime takes itself, roughly every five
seconds, and streams with its other journal facts as server-sent events:

```sh
curl -sN localhost:8080/events > events.sse
```

Each reading carries `model_output` (per output track: `frames_emitted`,
`frames_per_second`, `max_emit_interval_ms`), `model_input` (per input
track: `frames_received`, `frames_dropped`) and, per connection, one
`{name, total_ms, frames}` entry per stage (`input_wait`, `process_input`,
`generate`, `process_output`, `output_pacing`). A reading sums a five-second
window, so it shows where time goes and whether a gap occurred, not which
single step was slowest; that comes from the per-step times. Save the
readings of one full session for the checklist and the report.

## 4. Reaching the served port from a browser

A person plays the port from its play page ([play-page.md](play-page.md)), or
from the [Reactor Sandbox](https://reactor-sandbox.vercel.app/), a public page
that connects to a running `reactor run` and builds its controls from the
model's schema. Both are the same kind of client. It decides nothing
about correctness: the replay (Phase 5) and the checks in §3 already ran on
the node, through the same runtime, with the client on `localhost`.

Two kinds of traffic go from the person's browser to the node:

- **Signalling** is plain HTTP on the serve port. When the browser can reach
  the node, point the Sandbox at `http://<node>:8080`; otherwise an SSH
  tunnel carries it (`ssh -N -L 8080:localhost:8080 <node>`, then
  `http://localhost:8080`).
- **Media** is WebRTC over UDP. When the browser and the node are on the same
  network or VPN and UDP between them is open, it flows directly and nothing
  else is needed. When a firewall sits between them, fix the UDP range the
  runtime uses (`WEBRTC_PORT_RANGE=min:max`), run the image with
  `--network host` so that range is the node's own, and open it. When only SSH
  reaches the node, UDP cannot cross the tunnel: media then needs a TURN
  server both ends can reach (`TURN_SERVERS`, and `ICE_TRANSPORT_POLICY=relay`
  to force it), or the person plays from a machine on the node's network.

If the Sandbox connects but the video stays black, signalling works and media
does not; check the media path. A person who cannot play from where they sit
does not make the port unfinished: say so in the report, with which of the
cases above applies.

## Exit

All of the following hold before this phase is done:

- the server is healthy and `available`, and the startup time is recorded;
- every command was driven, two sessions ran, and the log shows no handler
  error or crash;
- the `runtime_stats` readings from a full session are saved for the
  checklist;
- the report gives the person the serve command, the tunnel or network
  command that makes the node reachable, and the address to enter in the
  Sandbox.

After that, a person plays the model.
