---
name: porting-models-to-standalone-runtime
description: "Port a model written against an older Reactor authoring API onto this runtime. Use when an existing model fails to import, load, or run here. This is a migration guide: it lists what changed or was removed and what to do about each break, not a from-scratch tutorial."
---

# Port-over: what changed moving a model onto this runtime

Use this when bringing a model that ran on an older Reactor authoring API onto
`reactor_runtime` as shipped in this repository. It is a list of breaks and the
fix for each, in rough order of how hard they bite. It is not a tutorial for
writing a model from scratch.

Ground rule: the supported surface is what the **top-level package**
re-exports. If a name is not importable from `reactor_runtime`, it is not part
of the surface here — do not reach into submodules to find a replacement.

## Your model is a `ReactorPipeline`

`ReactorPipeline`, `InputState`, and `Idle` are supported and re-exported from
`reactor_runtime`. A pipeline ports as-is in shape: declare `state: MyState`
(an `InputState` subclass), implement `load()` + an `inference()` generator
(sync or async) that reads `self.state`, yields an `Output` per frame, and
yields `Idle` (or `None`) to skip a turn. Public `InputState` fields still
become `set_<field>` commands automatically; underscore-prefixed fields stay
private; an `UploadedFile`-typed field is a public upload slot. A hand-written
`@event` of the same `set_<field>` name overrides the generated one.

Two breaks to clear when porting a pipeline:

- **`load()` takes a config path, not a dict** — the same change as for any
  model (see below).
- **State is declared by annotation only.** Use `state: MyState`; the legacy
  `state_class = MyState` fallback is gone.

Two conveniences from older runtimes are deliberately absent: yielding a raw
`np.ndarray` (yield a typed `Output` instead) and the headless `PipelineExecutor`
step driver.

## Imports move to the package root

Update every authoring import to come from `reactor_runtime`, not
`reactor_runtime.interface`:

```python
# before
from reactor_runtime.interface import InputField, ReactorModel, ReadMode, event
# after
from reactor_runtime import InputField, ReactorModel, ReadMode, event
```

`reactor_runtime.interface` still resolves, so this is not a hard break — but the
package root is the canonical path now, and new names (e.g. `get_weights_path`)
are exported only there. Move imports over so the model tracks the new pattern.

## `load()` receives a config path, not a parsed dict

This is the main break. The signature and contract changed:

```python
# before — the runtime parsed reactor.yaml's config into a dict and passed it
def load(self, config):
    self.steps = config["steps"]

# after — you get the path to the config file (or None) and parse it yourself
def load(self, config_path: Path | None):
    config = yaml.safe_load(config_path.read_text()) if config_path else {}
    self.steps = int(config.get("steps", 4))
```

The runtime no longer reads your config. `runtime.config` in `reactor.yaml` only
*names* the file; the reactor CLI is what makes that file available inside the
container at run time. The runtime resolves it to an absolute path and hands
`load` that path (or `None` when none is named) — nothing more. A `load(config)`
body that indexed a dict needs the parse line above prepended; the rest is
unchanged.

## Connection counting breaks — use `@session_started` for per-session init

Two lifecycle semantics differ on this runtime, and together they silently
break a pattern common in models written against older runtimes: gating
"first client of a session" initialization on a connection counter.

```python
# broken here
@connected
async def on_connect(self) -> None:
    if self._connected_count == 0:
        self.state._prompt_schedule = {}
    self._connected_count += 1

@disconnected
async def on_disconnect(self) -> None:
    self._connected_count -= 1
```

Why it fails: when the session itself ends (for example, closed through the
API while clients are attached), the runtime tears the connections down
wholesale — the model hears `@session_ended` once, **not** one
`@disconnected` per client. `@disconnected` fires only when a client itself
drops mid-session. So after any server-side session end, the counter never
returns to zero. Meanwhile `self.state` is rebuilt fresh for every session,
its private fields back at their class defaults. From the second session
onward the init branch never runs and the model works against default state —
typically a `None` where the first session had a dict, surfacing as a
`TypeError` in a handler or, worse, inside the inference loop. Single-session
testing passes; only a second session on the same process exposes it.

Hook the session, not the clients:

```python
@session_started
async def on_session_started(self) -> None:
    self.state._prompt_schedule = {}
```

`@session_started` fires exactly once per session, before any client
connects, and for a `ReactorPipeline` it runs with the session's fresh
`self.state` already built: the runtime constructs `state` when the session
starts and clears it only after `@session_ended`, so a private field
initialized here stays alive across client disconnects and rejoins within
the session. If first-vs-later-client logic *within* one session is
genuinely needed, keep the flag on `self.state`: it is session-scoped, so
it cannot leak into the next session the way an attribute on the model
instance does. Audit every use of a connection counter when porting —
teardown logic hung off `@disconnected` (dropping caches, releasing a
sub-session) has the same blind spot and belongs in `@session_ended`.

## Profiler imports are gone — strip them

`get_profiler` and the whole `reactor_runtime.profiling` module
(`BucketPreset`, `ChunkRangeProfiler`, `NoOpProfiler`) do not exist here, so
they fail at module load — the model never even imports. Remove the imports and
every call site. Keep per-stage timing in plain locals if you want the log
lines; there is no profiler to feed.

## Other internal imports are gone

Anything imported from an internal submodule (in-process metrics helpers and the
like) is not part of this runtime. Remove those imports — only the package-root
surface is supported, and reaching past it is exactly what breaks on the next
change.

## `UploadedFile` is `name` / `mime_type` / `data`

Uploads work as they did: a field or `@event` parameter typed `UploadedFile` is
an upload slot, and the runtime fetches the bytes and hands the handler a file.
What changed is the shape of that file. It carries the name, the mime type, and
the bytes — the upload id the client addressed it by stays inside the runtime and
never crosses the model boundary.

`size` is a property derived from the bytes (`len(data)`), not a field the client
fills in. The two agree by construction: an upload is admitted only when its
bytes match the length the client announced, so a handler that read `file.size`
before keeps reading the same number, and it is now measured rather than
asserted. `len(file.data)` says the same thing if you prefer it explicit.

## Drop the `output: MyOutput` class annotation — `self.output` is a handle

Models written for earlier cuts of this runtime often carried an inert class
annotation naming their `Output` subclass:

```python
class MyModel(ReactorPipeline):
    state: MyState
    output: MyOutput   # remove this line
```

Outbound tracks register when the `Output` subclass is *defined*; the
annotation was never read. On this runtime the name is taken: the base class
declares `output: OutputStream` and binds the model's playout handle there
(see the next section), so a subclass re-annotating it with an `Output` type
contradicts the real attribute and fails a type check. Delete the line —
track registration is unaffected.

## Pacing: `emit()` backpressures, `self.output` controls playout

`emit()` hands the whole batch of frames downstream, tagged with the rate
they should play out at (measured from `compute_time` when given, else the
class `fps`); each connection paces itself. By default **emit waits while
downstream is full**, throttling the model to the playout rate — the same
backpressure older runtimes applied through their blocking output buffer, so
a fast model needs no rate limiter of its own. The wait runs off the model
loop; commands and lifecycle events keep dispatching. A producer that would
rather skip frames than wait (a camera-driven model) passes
`emit(..., drop=True)` and the overflow is discarded downstream.

Never tag a chunk with a doctored rate. Playout follows the tag, so a
"debuffed" or pinned `compute_time` makes the model permanently outrun its
own playout — pass the honest measured time, or none at all and let the
declared `fps` stand.

Two pieces of the older authoring surface are back, one renamed:

- **`buffer_size`** (class attribute) declares how many frames may queue
  between the model and each wire — the buffered-latency bound. It is never
  applied below one emitted chunk, so a batching model always fits a whole
  chunk. Leave it undeclared to accept the default.
- **`self.output`** is the model's handle onto its outbound stream — the
  successor of the old `output_buffer`, named for what it is: there is no
  single buffer, and the operations fan out to every connection's own queue
  (including connections that join later).

```python
self.output.fps = 24      # re-pace queued frames now; tag emits from here on
self.output.flush()       # drop queued frames, cut playout to black
await self.output.emit(x) # emit() on the model is an alias of this
```

One caveat on `self.output.fps`: the assignment holds only for a model
that declares a class-level `fps` or emits without `compute_time`. A
pipeline that declares no `fps` is driven with the measured throughput on
every yield, and each chunk's own tag supersedes the assignment — so on an
unpinned pipeline it lasts one chunk. A `set_target_fps`-style command
therefore belongs on a model that pins `fps`.

Call `flush()` when generation resets or restarts, so the client cuts to
black instead of holding the last frame of the old content. The session
recording is not flushed — a playout cut is not an archive boundary. A probe
that read `self.output_buffer._q` / `_queue` still has nothing to read; the
queues live per connection, downstream.

## What did not change

`ReactorModel` itself is the same shape: `load()` + `async def run()` driving
`await self.emit(...)`, `@event` / `@connected` / `@disconnected` handlers,
`self.connected` to gate the loop, `fps` as a class attribute, typed
`ModelMessage` returns/`self.send(...)`, and inbound media via
`self.input.<track>.try_read(n, mode=ReadMode.LATEST)` / `.read(...)` /
`.reset()`. Weights are still located with `get_weights_path()` (now imported
from `reactor_runtime`); it returns `$REACTOR_WEIGHTS_PATH` or
`~/.cache/reactor_registry`.

Recording needs nothing from the model either: the `recording:` block in
`reactor.yaml` configures the recorder, and clip requests are answered off the
runtime's own surface. Keep the block as it is.

Once the breaks above are cleared the model should import, `load`, and run; a
`reactor.yaml` naming the `ReactorModel` via `runtime.import` is all the runtime
needs to serve it.
