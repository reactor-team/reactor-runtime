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

## Recording and clips do nothing

A `recording:` block in `reactor.yaml` is inert, `requestClip` /
`requestRecording` are unanswered, and there is no clip endpoint. If the model
or its client assumed any of these, drop that assumption — recording is not
wired here.

## Uploads are gone

There is no upload store, and an `@event` cannot take an uploaded-file argument.
A command that consumed an upload must be reworked to take the data over an
input track instead.

## Pacing moved out of the model — `buffer_size` and `output_buffer` are gone

The model no longer paces its own output. `emit()` hands the whole batch of
frames straight downstream, tagged with the rate they should play out at
(measured from `compute_time` when given, else the class `fps`); the transport
paces each connection itself. There is no `buffer_size` class attribute and no
`self.output_buffer` — drop any reference to either. A model that set
`buffer_size` for latency simply removes it; a probe that read
`self.output_buffer._q` / `_queue` has nothing to read and should be dropped. The
model's only output concern is emitting media chunks at whatever rate it
produces them.

## What did not change

`ReactorModel` itself is the same shape: `load()` + `async def run()` driving
`await self.emit(...)`, `@event` / `@connected` / `@disconnected` handlers,
`self.connected` to gate the loop, `fps` as a class attribute, typed
`ModelMessage` returns/`self.send(...)`, and inbound media via
`self.input.<track>.try_read(n, mode=ReadMode.LATEST)` / `.read(...)` /
`.reset()`. Weights are still located with `get_weights_path()` (now imported
from `reactor_runtime`); it returns `$REACTOR_WEIGHTS_PATH` or
`~/.cache/reactor_registry`.

Once the breaks above are cleared the model should import, `load`, and run; a
`reactor.yaml` naming the `ReactorModel` via `runtime.import` is all the runtime
needs to serve it.
