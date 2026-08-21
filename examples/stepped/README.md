# Stepped example

The runtime drives the model, and the application only describes the wire. Two
files, and the split between them is the point:

- **`generator.py`** — the model. It imports no `reactor_runtime`, takes plain
  arrays and plain values, and keeps every value that persists between steps
  (the step counter, the tint it has drifted to, the conditioning it was last
  given) private to itself.
- **`stepped.py`** — the application. It imports nothing that computes. It
  declares the state a client may set, the tracks that flow, and three small
  methods that bridge the two.

There is no `run()` in either file. The runtime steps the model while a client
is connected and paces playout from how long each step took.

## The step

```
map_step(state, input)  ->  model.generate(**inputs)  ->  to_output(...)  ->  wire
```

- `map_step` drains four webcam frames and returns the arguments the model
  wants. When fewer than four have arrived it raises `NotReady`, and the runtime
  holds the stream and tries again.
- `generate` produces the block. The conditioning (`mirror`, `drift`) rides along
  with every step, so the *model* notices when it changed and restarts its own
  rollout — the application never says when.
- `to_output` returns a `StepDone` message and a `SteppedOutput` together. The
  message goes out first, so a client tracking progress never waits behind the
  frames. Every frame is tagged with the hue that produced it, using
  `TrackPayload` metadata.

The output also states its own rate (`fps=CAMERA_FPS`). By default the runtime
paces playout from how long the step took, which is right for a model whose
speed *is* its frame rate. This one transforms a webcam and tinting is nearly
free, so its frames play at the cadence they arrived at instead.

`restart` shows where effects live: cutting playout is the application's
(`self.output.flush()`), and what a restart means to the rollout is the model's
own method (`self.model.restart()`).

## Two counts, two owners

`StepDone` carries both, because they answer different questions:

- **`chunk`** comes from the model. It is where the model is in its current
  rollout, and `restart` sets it back to zero — the model owns its rollout, so
  it owns the number that describes it.
- **`step`** comes from `stats.step`. It is how many steps the runtime has
  driven, and nothing resets it: not a restart, not a client leaving, not a new
  session. It is the runtime's own tally.

If you want a number that restarts, take it from the model, as this example
does.

## Run

This directory is a `reactor` workspace: `reactor.yaml` names the model, the
`Dockerfile` builds the image, and `requirements.txt` pins the runtime. The
model computes with NumPy alone, so it runs anywhere — no GPU.

```sh
cd examples/stepped
reactor build
reactor run
```

`reactor run` serves WebRTC signaling on `http://localhost:8080`. Connect a
client with a webcam, then send `set_mirror` or `set_drift` — commands generated
from the state fields, with no handler written for either.

## Owning the loop instead

A model whose loop is not one step per emit — one that renders long windows,
blocks waiting for input, or drives worker processes — overrides `run()`:

```python
async def run(self) -> None:
    while True:
        await self.connected.wait()
        while self.connected.is_set():
            ...
```

Everything else on the page keeps working: the state and its generated
commands, the `@event` handlers, the tracks, and the model binding. What an
override gives up is only the default loop.
