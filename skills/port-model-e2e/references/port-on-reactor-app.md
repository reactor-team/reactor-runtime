# Phase 4 — the port: a research repo becomes a `ReactorApp`

The input to this phase is a research repo whose own example already ran in
your image (Phase 1), a baseline recorded from it (Phase 2,
[`baseline-and-replay.md`](baseline-and-replay.md)), a source that already
generates one chunk per call (Phase 3, after the streaming refactor when it was
needed), and a weights root on the
node with every hub dependency removed
([`source-and-weights.md`](source-and-weights.md)). The output is a model
directory that serves the same computation through the step loop, ready for
the replay in Phase 5.

The rules the finished class must satisfy, which half a line belongs to, and
how the two halves meet are in
[`application-model-isolation`](../../application-model-isolation/SKILL.md).
Read it first; this file does not restate it. What follows is what a repo with
no runtime in it needs on top of those rules, and the facts of the current runtime release that a port depends on.

All runtime facts below were read from the `v3.10.2` tag. When the release
SKILL.md Phase 0 looked up is newer, re-read the facts you rely on from that
tag's source.

## Start from the reference example

Copy the shape of [`examples/waypoint/`](../../../examples/waypoint/README.md):
a GPU model with weights, two modules (`waypoint.py` is the application half,
`waypoint_model.py` the model half), a config file the runtime hands to
`load()`, a `reactor.yaml` with no Dockerfile, and a GPU-free unit test in
`tests/unit/examples/test_waypoint.py`. Its model half reads a hub id from
config; a ported model reads a local path instead (below).

```
<model>/
  reactor.yaml          # $schema, model:, runtime:, build:
  requirements.txt      # the frozen closure from Phase 1
  config.yml            # read by the model half's load(); the runtime only passes the path
  <app>.py              # application half: ReactorApp, state, commands, hooks
  <model>_model.py      # model half: plain class, load / generate / reset
  <vendored source>/    # the research code, whole, at the baseline commit
  tests/                # GPU-free tests
```

## 1. Vendor the source whole, then edit in place

Copy the research repo into the model directory **whole**, at the exact commit
Phase 2 recorded plus the Phase 3 refactor commits when there were any, with
its licence file. Edit it in place and keep the edits
small enough to read as a diff against that commit. Do not cherry-pick the
modules the example imports: research code reaches helpers through dynamic
imports, config-named classes and relative paths, and a partial copy fails at
the first code path the example did not exercise.

The model half wraps the vendored code. It imports nothing from
`reactor_runtime`, so the replay driver and the unit tests can construct it
without a server.

## 2. Classify every argument of the original

List every argument the source's entry point takes — CLI flags, config keys,
function parameters, constants buried in the example script — and give each
exactly one kind: serving config, session setup or live control, as defined
in SKILL.md ("Serving config, session setup, live control"), or delete
it when it exists only for the offline script (output
paths, `--num_frames` / `--max_chunks` caps, video writers, progress bars). The
split for client inputs is the one proposed at the start of the run, and
confirmed or taken as default. Settle it before writing any command, because it
fixes the request API, what `reset` must clear, and which shapes the warm-up
must cover.

Then write the input table of [patterns/README.md](../patterns/README.md): one row per client
input with its modality, role, kind and command. The standard there fixes the lifecycle
(`start`, `pause`, `resume`, `stop`, `reset`), the events (`state_update`, `chunk_complete`) and the
command for each concept; follow it rather than the source's own names, so a client that drives one
ported model drives the next.

The model half and the config file hold no default for any client input; values
the source's example hard-codes go into the baseline's conditions file.

Keep arguments low-level: each says how an input enters the model, never what
it is for. A flag that rewrites the prompt and picks a clip length for a "use
case" makes `prompt` and the length mean something else whenever it is on. The
prompt reaches the model exactly as the client sent it.

## 3. Weights: one root, local paths, the checkpoint asserted

`load(self, config_path: Path | None)` receives the path `runtime.config`
names; the runtime parses nothing in it. Resolve every weight file as a path
under `reactor_runtime.get_weights_path()`, which returns
`$REACTOR_WEIGHTS_PATH` when it is set and `~/.cache/reactor_registry`
otherwise (`src/reactor_runtime/paths.py:23`). The config names each weight as a
path **relative to that root**, in the exact layout Phase 1 staged, so the tree
you serve from is the tree the config expects.

- Every hub load was already rewritten to a local path in Phase 1
  ([`source-and-weights.md`](source-and-weights.md) §7); here the root becomes
  `get_weights_path()`, so `from_pretrained(f"{root}/<subdir>")` reads
  `from_pretrained(get_weights_path() / "<subdir>")`.
- Assert the checkpoint you loaded is the one the baseline used: compare a
  recorded fact (file size, a tensor shape, a config hash, a step counter in
  the checkpoint) and raise from `load()` on a mismatch. Two releases of one
  model often share file names and differ in contents, and a mismatched
  checkpoint loads cleanly and produces plausible, wrong output.
- Do the heavy imports (`torch`, the vendored package) inside `load()` or the
  model half's module, so the application module imports on a machine without
  a GPU — the schema render and the unit tests depend on it.

## 4. Multi-GPU: `DistributedRunner`

When the source needs more than one GPU, keep the model half one class and run
it on every rank with `reactor_runtime.distributed.DistributedRunner`
(`DistributedRunner`, `DistributedWorker`, `RankDesync`, `WorkerCrashed`,
`WorkerTimeout` are its exports). Call `start()` from `load()`; it waits up to
`start_timeout` (default 3600 s) for every rank to load, and each `generate()`
or `reset()` call may take up to `call_timeout` (default 30 s)
(`src/reactor_runtime/distributed/runner.py:74-100`). Set `call_timeout` above
the slowest step the model ever takes, first step included. The GPU count is
`model.resources.gpu.count` in `reactor.yaml`.

## 5. Session setup, live controls, and where encoding happens

Session setup inputs travel once, before the first step, and the application hands them
to the model inside the step input on the first step after they change (the
`seed_id` / `applied_seed_id` handshake in `examples/waypoint/waypoint.py` is the
pattern). Live controls ride every step input.

- **A setter stores; it does not compute.** A command handler validates,
  decodes and stores the newest value. Text encoding, VAE encoding of an image
  and every other GPU cost happens in the model half, once, at the step boundary
  that applies the value (or before streaming starts, when that encode measured
  alone would stall the first step). A value superseded before then is never
  encoded. A user can send several prompt changes inside one
  step interval; encoding in the setter pays for every one of them on the loop
  that is also serving the stream.
- **Encode only what is new.** Cache an encoded condition by value for the life
  of the session; a repeated value is not encoded again.
- **A session setup input that implies one-time work** (encoding a first frame) runs
  that work once, on the step that applies it. If that work draws random
  numbers, it must run exactly as often as in the source: calling it per step
  advances the generator and the output diverges for no visible reason.
- **Actions enter the model in the form it was trained on.** Movement takes the commands of
  [patterns/movement.md](../patterns/movement.md) and a pointer those of
  [patterns/pointer.md](../patterns/pointer.md), whatever the model was trained on; the port converts
  them to that form. Read how the source turns an
  action into a tensor before writing the conversion: held keys as a one-hot or a direction
  vector, camera or mouse look as continuous angles or as direction plus speed, a click as
  coordinates normalised to the output frame, a drag as a sequence of points. A drag, or several
  fingers at once, gives a variable number of points; bring it to the fixed shape the model takes
  (a fixed point count, resampled or padded with a mask) on arrival, so it adds no compile shape
  ([§6](#6-warm-up-covers-every-shape-a-client-can-produce)). Coordinates are relative to the frame
  the client sees, so state in the command's description which frame and which corner is the origin.
  A point or a drag that spans several chunks is split by chunk the way the source consumes it.
- **`reset` and session end clear every session setup input on every rank.** A stale
  reference image raises nothing; it silently conditions the next session.

## 6. Warm-up covers every shape a client can produce

Everything that compiles, autotunes or captures runs inside `load()`, before
the server reports ready. Warm the real path end to end:

- **Both shapes of an optional input** — with and without the image condition,
  with and without a reference. Presence changes which graphs are traced, and
  a shape first seen by a user pays the compile in front of them. One measured
  case: an image-conditioned warm-up moved startup from about one minute to
  about five, all of it before ready, which is where it belongs.
- **Client-sized inputs**, not target-sized ones. Synthesize the warm-up frame
  at a size a client sends, so the resize actually runs. A warm-up frame
  already at the target size made the resize a no-op, and the first real step
  recompiled for 44.6 s against a 1.9 s steady state.
- **Fixed shapes on arrival.** An input that would change a tensor shape — a
  prompt of any length, an image of any size — is padded or fitted to one shape
  in the handler, so the warm-up set is finite.

## 7. Refusals raise `CommandError`

A command that cannot be applied raises `CommandError(code, message)`. `code` is
a short, stable `snake_case` token a client branches on; `message` is prose for
a person. Never send an error message and return.

What each caller sees, verified against the current SDKs:

| Caller | On a raised `CommandError` |
|---|---|
| Python SDK `send_command` | raises `ReactorError` carrying the code and message |
| JS SDK 3.x `sendCommand` | never rejects: resolves `undefined` and emits `error` with a `ReactorError` whose `operation` is `"sendCommand"` and `command_code` is your code; also readable from `getLastError()` |
| A session started with a step count | a starting command that raises fails the session's remaining steps (`Runner._fails_remaining_steps`, `src/reactor_runtime/runner/runner.py:2396`, and `_emit_handler_failure` below it) |

Catch what a check raises (a decoder's `ValueError`, Pillow's
`DecompressionBombError`) and re-raise it as a `CommandError` with its own code,
as `set_image` in `examples/waypoint/waypoint.py` does. Any other exception
reaches the client as a generic `internal_error`, with the detail kept in the
log.

## 8. Descriptions are the product surface

The schema is generated from your code, and the descriptions are what a
client author reads.

- Every **command** description answers four things: what it does, when it is
  valid, what it answers with, and which failure codes it raises
  (`"Fails with unsupported_media when …"`).
- Every **field** description answers three: what it means and in what units,
  when a change takes effect, and how it interacts with other fields.
- Wrap an `UploadedFile` parameter in `InputField(description=…)`.
- Name `ModelMessage` classes for what they carry, with no model-name prefix,
  and start each docstring with "Emitted …".
- Send one state snapshot message: from `@connected` and as the reply to every
  command that changes state. A client that joins late or misses a reply
  recovers from the next snapshot.

## 9. The manifest and the runtime pin

```yaml
$schema: reactor/v2
model:
  name: <slug>
  version: v0.1.0
  description: |
    One paragraph a client author reads.
  resources:
    gpu:
      count: <GPUs the source needs>
runtime:
  import: <app module>:<Class>
  config: config.yml
build:
  runtime_version: "<latest release, looked up>"
  cuda_version: "<from Phase 1>"
  system_packages: [git]
  python_requirements: requirements.txt
```

`build.runtime_version` is the one exact pin of the runtime (SKILL.md Phase 0
has the lookup and the rule for `requirements.txt`). Take `build:` from Phase 1 unchanged —
the source and the port run in one environment, which is what makes the
Phase 5 comparison clean — and add the offline switches
(`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`) the weights phase established.

Render the schema to check the contract, inside the image:

```sh
python -m reactor_runtime.schema --path . --version v0.1.0 --out schema.json
```

## 10. Runtime facts a port relies on

Each checked in the `v3.10.2` source.

- **The step loop decides replay.** Do not override `run()` or
  `submit_command()`. A class that keeps both can serve a stepped session —
  one step per request, with state, commands and frames in the request
  (`ReactorApp.supports_stepped`, `src/reactor_runtime/interface/app/reactor_app.py:446-455`).
  Phase 5 replays the baseline through exactly that.
- **`process_output()` returns `Output | StepCompleted | None`**
  (`reactor_app.py:233`). Return `StepCompleted` when a step's report to a
  step-count session should carry more than the frames.
- **`self.output.flush(hold=True)`** drops queued frames and keeps the last one
  on screen instead of cutting to black; `flush()` cuts.
- **The system client.** A session started with a starting input or a step
  count has one `ClientInfo` with `system=True`; it fires `@connected` like any
  client, and nothing sent to it is delivered
  (`src/reactor_runtime/interface/client.py:27-36`). Count viewers with
  `not client.system`; a model that only steps when a non-system client is
  present never runs a stepped replay.
- **One hook per kind.** A class has one method per lifecycle hook; a
  subclass extends an inherited hook by overriding the method and calling
  `super()`. A second method marked with the same hook fails when the class is
  defined (`src/reactor_runtime/interface/events/decorators.py:18-22`).
- **Posted order.** Commands, step requests and lifecycle events are handled
  one at a time in the order posted, so a handler or hook must never wait for
  a later event — it is queued behind the wait
  (`reactor_app.py:556-563`). `await self.connected.wait()` inside a hook is
  the shape this rules out.
- **Exit status.** A crashed step loop exits the process with status 1; a stop
  that was asked for, and a model that refuses to load, exit 0
  (`src/reactor_runtime/service.py:37-40`).
- **Step count cap.** A session started with a step count takes at most
  10,000 steps (`src/reactor_runtime/runner/session_start.py:26`).
- **Logging** goes through `reactor_runtime.get_logger`, never `print()`.

## 11. Grow it one proven change at a time

The model arrives from Phase 3 already generating one chunk per call. What is
left to change here is smaller but of the same kind: state lifted out of a
function, a dependency replaced, a feature wired through. Each is a change to
the code that produces the output.

- **Prove each restructure bit-identical before building on it.** Run the old
  and new form on the same input and compare: `max|diff| == 0.0`. The same
  kernels on the same data give the same bits; anything else means the
  restructure changed what runs ([`baseline-and-replay.md`](baseline-and-replay.md)
  §5.2 helps find where). When the restructure legitimately moves one
  thing — how random numbers are drawn, most often — that move is made in the
  source and becomes the baseline, so the port itself stays bit-identical to
  it; a change to the draw plan here means the baseline is re-recorded
  ([`baseline-and-replay.md`](baseline-and-replay.md) §2.0 and §2.2).
- **Add features one at a time, in this order:** into the model half → proved
  by the offline driver on a fixed input → wired into the application half and
  served. A feature that misbehaves in the driver is the feature; reached
  through serving first, the same failure hides behind a transport, a session
  and a client.

## 12. GPU-free tests

`tests/` runs without a GPU, torch or weights, the way
`tests/unit/examples/test_waypoint.py` does: a fake engine stands in for the
vendored model, the workspace directory goes on `sys.path` as the runtime puts
it (`tests/unit/examples/conftest.py`), and the application half is driven
through its hooks. At minimum: the contract assembles, every command validates
and refuses with its documented codes, `reset` and session end clear every session setup input, and a joining client receives the snapshot. Run them inside the
image.

## Exit criterion

The model directory is complete; `tests/` pass inside the image; the schema
renders from the image; the server built from the `build:` block starts, warms
up and reports ready; and the class keeps `run()` and `submit_command()`
unoverridden, so Phase 5 can replay the baseline through a stepped session.
