# Serving the port

Sections 1 and 2 (build and serve) are also how Phase 4's replay gets a served
port. The rest of this file is Phase 5: it starts with a ported workspace whose
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
([`baseline-and-replay.md`](baseline-and-replay.md) §4.1).

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

### Where the time goes, from the runtime itself

During a session the runtime records a stage reading roughly every five
seconds. It streams these readings, together with every other journal fact,
as server-sent events:

```sh
curl -sN localhost:8080/events > events.sse
```

A `runtime_stats` fact contains:

- `model_output`: per output track, `frames_emitted`, `frames_per_second` and
  `max_emit_interval_ms`;
- `model_input`: per input track, `frames_received` and `frames_dropped`;
- `connection_stats[].track_stats[].frame_stages`: one `{name, total_ms,
  frames}` entry per stage, including `input_wait`, `process_input`,
  `generate`, `process_output` and `output_pacing`.

Use these readings for the checklist's timing items. They are measured on the
path a client actually uses, they split each step into the application's work
and the model's work, and the port needs no added timers to get them.
`max_emit_interval_ms` is the longest gap the viewer saw in that window, and
that gap is the stall a person notices.

## 4. Play it from the Reactor Sandbox

The [Reactor Sandbox](https://reactor-sandbox.vercel.app/) is a public page
that connects to a local `reactor run` and builds its controls from the
model's schema. A person opens it on a laptop and points it at the served
port. Two kinds of traffic have to get from the laptop to the node, and they
travel differently.

- **Signalling** is plain HTTP on the serve port. An SSH tunnel carries it:

  ```sh
  ssh -N -L 8080:localhost:8080 <node>
  ```

  The Sandbox then reaches the model at `http://localhost:8080`.

- **Media.** WebRTC carries media over UDP, and an SSH tunnel cannot carry
  UDP. `reactor run` publishes only the HTTP port. The runtime reads three
  variables that decide how media gets through:
  - `WEBRTC_PORT_RANGE` (`min:max`) fixes the UDP ports it uses;
  - `STUN_SERVERS` and `TURN_SERVERS` (`username;credential;turn:host:port`,
    comma-separated) set its ICE servers. With neither set, a public STUN
    server is used;
  - `ICE_TRANSPORT_POLICY=relay` forces all media through TURN.

  Pick whichever case applies:
  - **The laptop can reach the node's address directly** (same network or a
    VPN). Run the image with host networking so the UDP range is the node's
    own: `docker run --network host` on the same `reactor-local/<dir>:dev`
    tag, with the GPU, weights mount, `--shm-size` and `PORT` that
    `reactor run --help` lists, plus `-e WEBRTC_PORT_RANGE=…`. Allow that
    range in the node's firewall.
  - **Only SSH reaches the node.** Use a TURN server both ends can reach: set
    `TURN_SERVERS` and `ICE_TRANSPORT_POLICY=relay` on the runtime. With no
    TURN server available, this is a stop-and-report item (SKILL.md, "When a
    phase cannot finish"): the Sandbox connects, and the person cannot see
    the video.

If the Sandbox connects but the video stays black, the signalling path works
and the media path does not. Check the media path first.

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
