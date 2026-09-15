---
name: application-model-isolation
description: "Split a Reactor model into an application half and a model half and keep them apart. Use when writing a new ReactorApp, porting a model onto the step loop, reviewing one, or deciding which half a piece of code belongs to. Covers the three hooks, the inner contract between the halves, refusal versus failure, what the model must never do on its own, and how the application reads the model. Nothing enforces these rules; this skill is where they are written down."
---

# Application and model, two halves of one class

A `ReactorApp` is driven by the runtime one step at a time. A step is three
calls in a fixed order, and the middle one is a different kind of code from the
other two.

- `prepare_step(state, media)` is **application** code. It knows about the
  client: the state a client set, the frames a client sent, whether a step
  should happen at all.
- `generate(step_input)` is **model** code. It knows about weights, a cache, a
  rollout, and nothing else.
- `collect_step(outcome)` is **application** code again. It knows what the
  client should receive: which track the frames go on, which message to send,
  what to do when the model could not step.

Both application hooks have defaults, so the simplest model writes only
`generate()`. Keep the two kinds of code in two places. The worked example is
[`examples/waypoint/`](../../examples/waypoint/README.md):
[`waypoint_model.py`](../../examples/waypoint/waypoint_model.py) is the model
half and [`waypoint.py`](../../examples/waypoint/waypoint.py) is the
application half. Every rule below points at a line in one of those two files.

## The rules

Nothing in the runtime checks these. An application that breaks one still
runs. Follow them anyway; they are what makes the model half testable in a
notebook and the application half readable without opening the model.

### 1. The model half is a plain class with three methods

`load(config_path)`, `generate(step_input)`, `reset()`. It imports nothing
from `reactor_runtime`. It does not know clients, tracks, commands, or the
loop exist.

```python
# waypoint_model.py
class WaypointModel:
    def load(self, config_path: Path | None) -> None: ...
    def generate(self, step: WaypointStepInput) -> WaypointStepResult: ...
    def reset(self) -> None: ...
```

The application constructs it in its own `load()` and holds it under an
ordinary attribute name. The example uses `self.engine`. Do not name it
`model` or `state`: `state` is the typed state the runtime owns, and a later
release reserves `model`.

```python
# waypoint.py
def load(self, config_path: Path | None) -> None:
    self.engine = WaypointModel()
    self.engine.load(config_path)
```

### 2. `generate()` on the app is one line, written by hand

The app's `generate()` forwards to the model half and does nothing else. No
branch, no state read, no message.

```python
def generate(self, step: WaypointStepInput) -> WaypointStepResult:
    return self.engine.generate(step)
```

A `generate()` that reads `self.state`, calls `self.send()`, or decides
whether to run has application code inside the model call. Move it to
`prepare_step()` or `collect_step()`.

### 3. The two halves meet on two dataclasses the author owns

The step input is what one step needs. The step result is what one step
produced. Both are plain dataclasses; the runtime never reads their fields.

```python
@dataclass(frozen=True)
class WaypointStepInput:
    buttons: frozenset[int]
    mouse: tuple[float, float]
    scroll_wheel: int
    seed: np.ndarray | None
    seed_id: int

@dataclass(frozen=True)
class WaypointStepResult:
    frames: np.ndarray
    index: int
```

Type them. Six lines give `prepare_step()` and `collect_step()` a signature a
reader can check without opening the model. A tuple works and says nothing.

### 4. New information reaches the model inside the step input

The application never writes the model's attributes. A prompt, a seed frame,
a control value: all of it goes into the step input, and the model notices
what changed. In the example, a new seed is a new `seed_id`; the model
compares it with the one it applied and starts a new world when they differ.
The step input carries what changed: the frame rides on it only for the step
that applies it, and the id alone continues the world after that.

```python
# waypoint.py
new_seed = state._seed_id != state._applied_seed_id
return WaypointStepInput(
    ...,
    seed=state._seed if new_seed else None,
    seed_id=state._seed_id,
)

# waypoint_model.py
if step.seed_id != self.seed_id:
    if step.seed is None:
        raise NotSeeded("no seed frame to start a world from")
    self.engine.reset()
    self.engine.append_frame(seed_x4)
    self.seed_id = step.seed_id
```

If a handler needs the model to change, it calls a method the model wrote.
`reset()` is that method. Handlers run between steps, so the call is safe.

### 5. The model's state is read-only from the application

The application reads what it needs off the step result. It does not reach
into the model for a cache, a counter, or a flag. In the example the app reads
`result.index` to tag the frames and to report progress, and `result.seed_id`
to know which seed the model holds; it never reads `self.engine.index` or
`self.engine.seed_id`.

Design the result as the model's public face. Whatever the application should
know about a step, the model puts in the result.

### 6. Refusing is the application's; failing is the model's

`prepare_step()` refuses a step by raising `ApplicationError` with the reason.
The model is not called, the reason is logged, and the loop asks again. A
refusal is a fact about the client: paused, no seed yet, waiting for frames.

```python
async def prepare_step(self, state: WaypointState, media: None) -> WaypointStepInput:
    if state.paused:
        raise ApplicationError("paused")
    if state._seed is None:
        raise ApplicationError("no seed image")
    ...
```

`generate()` fails a step by raising the model's own exception. A failure is
a fact about the model: it cannot step from the state it holds. The runtime
wraps the exception into `outcome.error` and hands it to `collect_step()`.

```python
# waypoint_model.py
class NotSeeded(Exception):
    """The model holds no world to step and the step input carries no seed."""
```

Do not raise `ApplicationError` from the model half. The model half does not
import `reactor_runtime`, and a refusal is not its decision.

Write `raise ApplicationError("reason")`. Subclass it only when code or a log
filter has to tell one reason from another, and then give the subclass its
message so the call site is just `raise WaitingForCamera()`:

```python
class WaitingForCamera(ApplicationError):
    def __init__(self) -> None:
        super().__init__("waiting for 4 webcam frames")
```

### 7. Anything a client can set is a public `InputState` field

Pause is `paused: bool` on the state, checked in `prepare_step()`. The client
gets `set_paused` for free, validated from the field. Do not write a `pause`
command that flips a private flag; that is a second door into a fact the
state already holds.

```python
class WaypointState(InputState):
    paused: bool = InputField(default=False, description="...")
    action: str = InputField(default="idle", choices=[...], description="...")
    ...
```

A hand-written command is for a decision the state cannot carry: `set_image`
in the example decodes and fits the upload before it becomes the seed, and
`reset` restarts the world from the seed it already has. Do not also declare
the value as a public field the hand-written setter never writes; a dead
attribute the schema advertises is worse than one command. Private fields
(leading underscore) are session scratch the application owns, such as the
fitted seed, its id, and the id the model last reported.

### 8. The model does nothing unprompted

Three things the model must not do, each with its correct form:

- **A default prompt or seed.** If the step input has no seed and a new world
  is asked for, the model raises `NotSeeded`. It does not invent one. The
  application refuses the step earlier, so the raise is a guard, not the
  normal path.
- **A silent reset that returns a success.** When the model cannot continue,
  it raises. The application decides whether to reset, tell the client, or
  stop. A model that resets itself and returns frames has hidden a
  discontinuity from everyone.
- **Relying on the application to re-send everything every step.** The model
  holds its world. The step input carries what changed.

### 9. `reset()` takes no arguments and the runtime never calls it

`reset()` returns the model to its default state. The inputs for the new world
arrive in the first `generate()` after it, the same way they arrive in every
other step. A `reset(seed=...)` is a second door into the model that the step
input already is.

The application calls `reset()`, from a handler or from `collect_step()`, and
from `@session_ended` so the next session begins from a clean model. The
runtime never resets a model.

```python
@session_ended
def on_session_ended(self) -> None:
    self.engine.reset()
    self.output.flush()
```

`reset()` and `self.output.flush()` stay separate calls. Whether playout is
cut is an application decision.

### 10. What the client receives is written down in `collect_step()`

The return value is the media. Every message is an `await self.send()` the
author typed, and it goes on the wire before the step's media. The runtime
infers nothing from a step result: only an `Output` is media, and the mapping
from any other result type to tracks is a line in `collect_step()`.

```python
async def collect_step(self, outcome: StepOutcome) -> WaypointOutput | None:
    if outcome.error is not None:
        raise outcome.error
    result: WaypointStepResult = outcome.result
    self.state._applied_seed_id = result.seed_id
    if result.index % self.progress_interval == 0:
        await self.send(WaypointStatus.of(self.state, result.index))
    metadata = [{"step": result.index}] * result.frames.shape[0]
    return WaypointOutput(main_video=TrackPayload(result.frames, metadata=metadata))
```

A reader of `collect_step()` sees the whole client-facing effect of a step in
one place, in order.

A model failure is decided here too, and nowhere else. `generate()` fails by
raising; the runtime catches the exception and hands it to `collect_step()`
as `outcome.error` (`outcome.result` is `None` then). Two choices:

- **Recover** an error the model is known to raise. Check its type, put the
  model back into a valid state with `self.engine.reset()`, cut playout with
  `self.output.flush()` if a stale frame must not follow, send the client a
  message, and return `None` (or an `Output`). The loop continues with the
  next step.

  ```python
  async def collect_step(self, outcome: StepOutcome) -> HeliosOutput | None:
      if isinstance(outcome.error, RolloutExhausted):
          self.engine.reset()
          self.output.flush()
          await self.send(WorldRestarted(reason="rollout window reached"))
          return None
      if outcome.error is not None:
          raise outcome.error
      ...
  ```

- **Re-raise** anything you did not expect. A raise out of `collect_step()`
  is a crash of the model, not of the step: the runtime logs the traceback,
  stops dispatching commands and lifecycle hooks, ends the session with an
  error the client sees, and does not restart the loop. The process is left
  for its supervisor to recycle. This is the same outcome an uncaught
  exception in a hand-written `run()` has, and it is better than serving a
  dead model in silence.

The default `collect_step()` re-raises. The example re-raises on purpose:
`NotSeeded` cannot arrive because `prepare_step()` refuses before a step
without a seed reaches the model, so anything that does arrive is a bug or a
GPU failure, and neither is repaired by a reset. Write the reason down at the
`raise`, as the example does, so a reader knows the choice was made.

A refusal is not a failure. `ApplicationError` from `prepare_step()` is
caught by the loop before the model runs and never reaches `collect_step()`.

### 11. `generate()` never sees `StepOutcome`

`generate()` returns the step result or raises. The runtime builds the
`StepOutcome` from whichever happened. A `generate()` that returns a
`StepOutcome`, or catches its own exception to return an error value, has
taken over a job the runtime does.

## Which loop you are on

The default `run()` of `ReactorApp` is the loop that drives the three hooks.
It takes the step lock around each step, so a handler runs between steps and
never during one, and it paces playout from the measured `generate()` time
unless the class declares `fps`.

Override `run()` to write your own loop. That replaces the loop and only the
loop: the typed state, the generated setters, the step lock on handlers and
hooks, `emit()`, `send()`, `self.connected`, and the tracks are all still
there, and `prepare_step()`, `generate()`, and `collect_step()` are never
called for that class. No error, no warning. Do that for a loop that is not
one step per emit: a renderer that emits several times per step, or a model
that must block on an input. Do not declare `state:` next to a hand-written
`run()` as a pattern: the setters write it, but nothing reads it for you and
nothing bounds when a write lands relative to your loop's reads. Keep your own
values in your own attributes and write `@event` handlers for them.

Everything else in this skill is about the default loop.

## Which half does this line belong to

Ask one question: could a client observe it?

| The line | Half | Where |
| --- | --- | --- |
| `if state.paused` | application | `prepare_step()` |
| `if self.index >= self.WINDOW` | model | `generate()` |
| reading four frames off a track | application | `prepare_step()` |
| encoding those frames into latents | model | `generate()` |
| `await self.send(Status(...))` | application | `collect_step()` |
| deciding which track a result goes on | application | `collect_step()` |
| clearing the KV cache | model | `reset()` or `generate()` |
| deciding to clear it after an error | application | `collect_step()` or a handler |
| decoding an uploaded image | application | a hand-written command |
| holding the decoded image | application | a private state field |
| knowing which seed the model holds | application | a private state field, read off the result |
| the frame counter within a world | model | an attribute, exposed on the result |

## Review checklist

1. The model half imports nothing from `reactor_runtime`.
2. The app's `generate()` is one line and forwards to the model half.
3. The step input and the step result are dataclasses; the app never reads a
   model attribute.
4. Every client-settable fact is a public `InputState` field; hand-written
   commands exist only for decisions the state cannot carry.
5. `prepare_step()` refuses with `ApplicationError("reason")`; the model half
   raises its own exception type and never `ApplicationError`.
6. The model invents no default input and never resets itself.
7. `reset()` takes no arguments; the application calls it, from a handler,
   from `collect_step()`, and from `@session_ended`.
8. Every message the client receives is a `self.send()` in `collect_step()` or
   a handler; the mapping from result to `Output` is explicit.
9. `generate()` does not build, return, or catch into a `StepOutcome`.
10. `collect_step()` recovers each error the model is known to raise by type
    and re-raises the rest; a bare re-raise carries the reason in a comment.
    No blanket `except` that resets and continues on every error.
11. The model half has a test with a fake engine, and the app half has tests
    for each refusal and for `collect_step()`, including its error branch. See
    [`tests/unit/examples/test_waypoint.py`](../../tests/unit/examples/test_waypoint.py).

## Prose

This repository is public. Write every docstring and comment for an outside
reader with no other context: active voice, one topic per sentence, one name
per thing. The rules in [AGENTS.md](../../AGENTS.md) apply here as everywhere.
