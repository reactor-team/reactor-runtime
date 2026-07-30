# Engine example — serving an inference engine

A worked example of the engine/runtime contract: `engine.py` is an inference
engine that imports only `reactor_runtime.engine_contract`, and `app.py` is the
application that serves it.

```sh
cd examples/engine
uv run python -m reactor_runtime.serve
```

The "inference" is a brush painting on a canvas, so it runs anywhere — no
weights, no GPU.

## What the engine declares

Four kinds of declaration, and no serving code:

| Declaration | Becomes |
|---|---|
| `Move`, `Brush` (`UserInput`) | the `move` and `brush` commands |
| `Camera` (`VideoInput`, `chunk_size = 4`) | an inbound video track, delivered four frames at a time |
| `PaintInit` (`Init`) | the `init` command, and the rollout's starting state |
| `PaintStepInput` (`ModelInput`) | the conditioning of one step, engine-facing only |

`PaintPipeline` implements the four calls a runtime makes: `initialize_cache`,
`map_inputs`, `generate`, `finalize`. It satisfies the protocol structurally, so
it inherits from nothing.

## The window

Every input is stamped as it arrives and queued. Each step hands `map_inputs`
everything received since the previous step, in arrival order, events and media
in one list. There are no per-parameter aggregation rules: the engine folds the
window itself, which is why holding down a movement key travels further than a
single tap, while a burst of colour changes collapses to the last one.

Anything that has to survive between steps — the cursor, the current colour, the
canvas — is rollout state on the cache, and disappears with it.

## Initialization

There is no `start` command. The rollout is created at the first step: a client
that sends nothing gets the canvas `PaintInit` declares as its defaults, and a
client that wants its own sends `init` first. A later `init` reaches the
mapping, which resets the rollout in place and skips that step — starting a new
sequence is what initializing again means.

Give a `PaintInit` field no default and the model waits for the client instead of
starting on its own.

## The application

`PaintApp` binds the engine and adds what the engine had no business knowing:

- **An override.** `@override_input(Move)` takes the `move` command's place. The
  handler's signature is the new payload and its return value is what reaches
  the window, so returning `None` drops the input and the model never sees it.
- **A new event.** `dash` is not an engine declaration at all. It pushes into
  the same store, so the engine's fold handles it without knowing it exists.

Neither touches the mapping or the loop. To replace the fold, implement
`map_inputs` on the application; to replace the loop, implement `run` and call
`self.step()`.

## Serving the engine alone

An application that only binds an engine and declares one video track is
boilerplate, so `runtime.import` may name the engine class directly:

```yaml
runtime:
  import: engine:PaintPipeline
```

The runtime builds the application, and the engine emits on a single `main_video`
track. Anything else — a second track, audio, an override — means writing the
application.

## Stepping

`runtime.stepping: triggered` parks the loop and advances the model once per
`step` command, so the same application streams under one deployment and is
stepped by its caller under another.
