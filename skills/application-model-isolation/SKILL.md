---
name: application-model-isolation
description: "Split a Reactor model into an application half and a model half and keep them apart. For reactor-runtime 3.5 and later, where ReactorApp and the step loop exist. Use when writing a new ReactorApp, finishing a port onto the step loop (porting-to-reactor-app is the how; this is the what), reviewing one, or deciding which half a piece of code belongs to. Covers the three hooks, the inner contract between the halves, refusal versus failure, what the model must never do on its own, how the application reads the model, and what a model failure does. Nothing enforces these rules; this skill is where they are written down."
---

# Application and model, two halves of one class

A `ReactorApp` is driven by the runtime one step at a time. A step is three
calls in a fixed order, and the middle one is a different kind of code from the
other two.

- `process_input()` is **application** code. It reads `self.state` and the
  media tracks, and it knows about the client: the state a client set, the
  frames a client sent, whether a step should happen at all.
- `generate(input)` is **model** code. It knows about weights, a cache, a
  rollout, and nothing else.
- `process_output(outcome)` is **application** code again. It knows what the
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

`load(config_path)`, `generate(input)`, `reset()`. It imports nothing
from `reactor_runtime`. It does not know clients, tracks, commands, or the
loop exist.

```python
# waypoint_model.py
class WaypointModel:
    def load(self, config_path: Path | None) -> None: ...
    def generate(self, input: WaypointInput) -> WaypointResult: ...
    def reset(self) -> None: ...
```

The application constructs it in its own `load()` and holds it under an
ordinary attribute name. The example uses `self.engine`. Do not name it
`state`, which is the typed state the runtime owns. The same attribute holds
a `DistributedRunner` around the class when the model needs its own process
or one process per GPU; `generate()` does not change.

```python
# waypoint.py
def load(self, config_path: Path | None) -> None:
    self.engine = WaypointModel()
    self.engine.load(config_path)
```

Four things a larger model adds to this rule:

- **Helpers that read the pipeline object are model code.** A codebase often
  has a module of functions written as `helper(pipe, ...)` that read
  `pipe.generator`, `pipe.vae`, `pipe.device`, and so on off the old class.
  Every one of those attributes is a model attribute, so the functions move
  to the model half unchanged: the model class carries the same attribute
  names, and `pipe` becomes the model instance. Nothing in such a module
  needs a rewrite, and nothing in the application may call it.
- **Both halves may read the config file, each its own keys.** `load()` on
  both sides receives `config_path`. The application takes what a client can
  observe (a default prompt, a message cadence); the model takes checkpoints,
  optimization flags, GPU count, seed, caps. One file, two readers, no key
  read by both.
- **The weights root arrives through `load()`.** `get_weights_path()` is a
  `reactor_runtime` name, so a model half never calls it and never reads
  `REACTOR_WEIGHTS_PATH`. A model whose checkpoints live under the
  deployment's weights directory takes that directory as `load()`'s second
  argument and joins the relative paths its config names onto it; the
  application resolves it once and passes it:

  ```python
  def load(self, config_path: Path | None) -> None:
      self.engine = MyModel()
      self.engine.load(config_path, get_weights_path())
  ```

  The constructor takes nothing. A model that fetches its weights elsewhere,
  as Waypoint does from Hugging Face, leaves the parameter out. Keep
  `load()`'s arguments to values that pickle, paths and scalars: the
  runtime's `DistributedRunner` constructs the model half in its own process
  and calls `load(**load_kwargs)` with exactly what the application passed,
  so `self.engine = DistributedRunner(MyModel, load_kwargs={"config_path":
  config_path, "weights_root": get_weights_path()})` is the same call made
  from another process. Any other file the config names by a relative path
  resolves against `config_path.parent`, not the working directory. The same
  rule covers anything else only the runtime knows: it enters the model half
  as an argument, never as an import.
- **The model half spawns nothing.** A model that needs its own process, or
  one process per GPU, is handed to `DistributedRunner` by the application;
  the runner owns the processes, the wire between them, and the logger in
  each child. A model half that spawns its own workers has taken on the
  runner's job, and the one runtime import that would need (the logger, at
  the top of each worker) is the sign it has.

### 2. `generate()` on the app is one line, written by hand

The app's `generate()` forwards to the model half and does nothing else. No
branch, no state read, no message.

```python
def generate(self, input: WaypointInput) -> WaypointResult:
    return self.engine.generate(input)
```

A `generate()` that reads `self.state`, calls `self.send()`, or decides
whether to run has application code inside the model call. Move it to
`process_input()` or `process_output()`.

**One `generate()` on the model half too, whatever the hardware.** A model
that has one code path per GPU topology (one GPU denoising locally, two with
a worker one denoising step behind, three over a shared channel) does not
expose three entry points. Its `load()` looks at the devices and binds the
chunk step it will use; its `generate()` does the run bookkeeping once, for
every topology, and calls that step:

```python
def load(self, config_path):
    ...
    self._step = self._chunk_step_2gpu if self.num_gpus == 2 else self._chunk_step_1gpu

def generate(self, input):
    if input.reference_id != self.reference_id:   # a new run
        self._end_run(); self._run = self._start_run(input); ...
    if self._run.frames_wanted == 1:
        self._first_frame_step(self._run, input.frames[0])   # the run's first output frame
    else:
        frames = self._step(self._run, input.frames)          # one chunk, topology-specific
    ...
```

The application has one `generate()` and does not know how many GPUs there
are. The old shape, `load()` assigning one of several `inference()`
generators to `self.inference`, becomes `load()` assigning one of several
chunk steps to a private attribute; the dispatch moved down, not out.

**The run is an object, not generator locals.** Everything a step of a run
shares (the conditioning, the caches, the counters, the frame held back to
prepend to the first chunk, whether the worker's streaming loop is open, which
index the worker waits on) lives on one object the model creates at run start
and drops at run end. Making it explicit is what surfaces the rules a
generator kept implicit: what to tell the worker when a run is interrupted
after a submit and before a collect, and how to drain the chunk still in
flight when the run reaches its cap.

### 3. The two halves meet on two dataclasses the author owns

The step input is what one step needs. The step result is what one step
produced. Both are plain dataclasses; the runtime never reads their fields.

```python
@dataclass(frozen=True)
class WaypointInput:
    buttons: frozenset[int]
    mouse: tuple[float, float]
    scroll_wheel: int
    seed: np.ndarray | None
    seed_id: int

@dataclass(frozen=True)
class WaypointResult:
    frames: np.ndarray
    index: int
    seed_id: int
```

Type them. A few lines give `process_input()` and `process_output()` a signature a
reader can check without opening the model. A tuple works and says nothing.

The result is also how the model tells the application what the **next** step
needs, so the application never has to know the model's phases. A model whose
first step of a run consumes one webcam frame and every later step four does
not make the application count: the result carries `frames_wanted`, the
application stores it on a private state field, and the next `process_input()`
reads that many frames. Anything the application must know to build the next
input, or to label a metric (`num_gpus`), rides on the result the same way.

A result may carry no media. On a pipelined model the chunk submitted on one
step comes back on the next, so the first chunk step returns `frames=None`,
and `process_output()` emits nothing for it. A `None` where media is expected is
a fact about the step, not an error.

### 4. New information reaches the model inside the step input

The application never writes the model's attributes. A prompt, a seed frame,
a control value: all of it goes into the step input, and the model notices
what changed. In the example, a new seed is a new `seed_id`; the model
compares it with the one it applied and starts a new world when they differ.
The step input carries what changed: the frame rides on it only for the step
that applies it, and the id alone continues the world after that.

```python
# waypoint.py
new_seed = self.state._seed_id != self.state._applied_seed_id
return WaypointInput(
    ...,
    seed=self.state._seed if new_seed else None,
    seed_id=self.state._seed_id,
)

# waypoint_model.py
if input.seed_id != self.seed_id:
    if input.seed is None:
        raise NotSeeded("no seed frame to start a world from")
    self.engine.reset()
    self.engine.append_frame(seed_x4)
    self.seed_id = input.seed_id
```

If a handler needs the model to change, it calls a method the model wrote.
`reset()` is that method. Handlers run between steps, so the call is safe.

**Encoding is model work and happens on the step, not in the handler.** A
handler that runs a text encoder on the prompt, or CLIP and a VAE on an
uploaded image, has model code in the application. The handler decodes and
stages: it letterboxes the upload to a uint8 array, stores the prompt text,
bumps the id. The model encodes on the step that applies the new id, as part
of starting the run, and caches what is worth caching (a prompt embedding by
its text). GPU tensors never live on the state object; the state holds
arrays, strings, and ids.

**A run is one identity, and a new id during a run is a new run.** When the
step input carries an id the model has not applied while a run is live, the
model ends that run itself (stops a worker's streaming loop, clears its
caches) and starts the next from the frame the step carries. That is not the
"silent reset" rule 8 forbids: the step asked for it, and the result reports
the id the new run holds.

**What the model applies at run start stays fixed for the run.** A prompt
the client changes mid-run takes effect on the next run, because the caches
were prefilled with the old one. Say so in the field's description; do not
try to re-encode inside a live run.

### 5. The model's state is read-only from the application

The application reads what it needs off the step result. It does not reach
into the model for a cache, a counter, or a flag. In the example the app reads
`result.index` to tag the frames and to report progress, and `result.seed_id`
to know which seed the model holds; it never reads `self.engine.index` or
`self.engine.seed_id`.

Design the result as the model's public face. Whatever the application should
know about a step, the model puts in the result.

### 6. Refusing is the application's; failing is the model's

`process_input()` refuses a step by raising `ApplicationError` with the reason.
The model is not called, the reason is logged, and the loop asks again. A
refusal is a fact about the client: paused, no seed yet, waiting for frames.

```python
async def process_input(self) -> WaypointInput:
    if self.state.paused:
        raise ApplicationError("paused")
    if self.state._seed is None:
        raise ApplicationError("no seed image")
    ...
```

`generate()` fails a step by raising the model's own exception. A failure is
a fact about the model: it cannot step from the state it holds. The runtime
wraps the exception into `outcome.error` and hands it to `process_output()`.

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

Pause is `paused: bool` on the state, checked in `process_input()`. The client
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

The application calls `reset()`, from a handler or from `process_output()`, and
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

### 10. What the client receives is written down in `process_output()`

The return value is the media. Every message is an `await self.send()` the
author typed, and it goes on the wire before the step's media. The runtime
infers nothing from a step result: only an `Output` is media, and the mapping
from any other result type to tracks is a line in `process_output()`.

```python
async def process_output(self, outcome: StepOutcome) -> WaypointOutput | None:
    if outcome.error is not None:
        raise outcome.error
    result: WaypointResult = outcome.result
    self.state._applied_seed_id = result.seed_id
    if result.index % self.progress_interval == 0:
        await self.send(WaypointStatus.of(self.state, result.index))
    metadata = [{"step": result.index}] * result.frames.shape[0]
    return WaypointOutput(main_video=TrackPayload(result.frames, metadata=metadata))
```

A reader of `process_output()` sees the whole client-facing effect of a step in
one place, in order.

A model failure is decided here too, and nowhere else. `generate()` fails by
raising; the runtime catches the exception and hands it to `process_output()`
as `outcome.error` (`outcome.result` is `None` then). Two choices:

- **Recover** an error the model is known to raise. Check its type, put the
  model back into a valid state with `self.engine.reset()`, cut playout with
  `self.output.flush()` if a stale frame must not follow, send the client a
  message, and return `None` (or an `Output`). The loop continues with the
  next step.

  ```python
  async def process_output(self, outcome: StepOutcome) -> MyOutput | None:
      if isinstance(outcome.error, RolloutExhausted):
          self.engine.reset()
          self.output.flush()
          await self.send(WorldRestarted(reason="rollout window reached"))
          return None
      if outcome.error is not None:
          raise outcome.error
      ...
  ```

- **Re-raise** anything you did not expect. A raise out of `process_output()`
  is a crash of the model, not of the step: the runtime logs the traceback,
  stops dispatching commands and lifecycle hooks, ends the session with an
  error the client sees, and does not restart the loop. Whatever runs the
  process decides whether to restart it. This is the same outcome an
  uncaught exception in a hand-written `run()` has, and it is better than
  serving a dead model in silence.

The default `process_output()` re-raises. The example re-raises on purpose:
`NotSeeded` cannot arrive because `process_input()` refuses before a step
without a seed reaches the model, so anything that does arrive is a bug or a
GPU failure, and neither is repaired by a reset. Write the reason down at the
`raise`, as the example does, so a reader knows the choice was made.

A refusal is not a failure. `ApplicationError` from `process_input()` is
caught by the loop before the model runs and never reaches `process_output()`.

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
there, and `process_input()`, `generate()`, and `process_output()` are never
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
| `if self.state.paused` | application | `process_input()` |
| `if self.index >= self.WINDOW` | model | `generate()` |
| reading four frames off a track | application | `process_input()` |
| encoding those frames into latents | model | `generate()` |
| `await self.send(Status(...))` | application | `process_output()` |
| deciding which track a result goes on | application | `process_output()` |
| clearing the KV cache | model | `reset()` or `generate()` |
| deciding to clear it after an error | application | `process_output()` or a handler |
| decoding an uploaded image | application | a hand-written command |
| holding the decoded image | application | a private state field |
| knowing which seed the model holds | application | a private state field, read off the result |
| the frame counter within a world | model | an attribute, exposed on the result |
| how many frames the next step needs | model decides, application reads | `frames_wanted` on the result, then a private state field |
| running the text encoder or the VAE on a new prompt or image | model | the step that starts a run |
| storing the prompt text and the letterboxed upload | application | a public field, a private field, a hand-written command |
| how many GPUs, which pipelined path | model | `load()` |
| telling a worker process where the run stopped | model | `reset()` / the run's end |
| what happens when the run reaches its cap | application | `process_output()`, on `result.complete` |
| a metric label such as `num_gpus` | model reports, application logs | the result, then `process_output()` |

## Review checklist

1. The model half imports nothing from `reactor_runtime`.
2. The app's `generate()` is one line and forwards to the model half.
3. The step input and the step result are dataclasses; the app never reads a
   model attribute.
4. Every client-settable fact is a public `InputState` field; hand-written
   commands exist only for decisions the state cannot carry.
5. `process_input()` refuses with `ApplicationError("reason")`; the model half
   raises its own exception type and never `ApplicationError`.
6. The model invents no default input and never resets itself.
7. `reset()` takes no arguments; the application calls it, from a handler,
   from `process_output()`, and from `@session_ended`.
8. Every message the client receives is a `self.send()` in `process_output()` or
   a handler; the mapping from result to `Output` is explicit.
9. `generate()` does not build, return, or catch into a `StepOutcome`.
10. `process_output()` recovers each error the model is known to raise by type
    and re-raises the rest; a bare re-raise carries the reason in a comment.
    No blanket `except` that resets and continues on every error.
11. The model half has a test with a fake engine, and the app half has tests
    for each refusal and for `process_output()`, including its error branch. See
    [`tests/unit/examples/test_waypoint.py`](../../tests/unit/examples/test_waypoint.py).
    When the model half's imports (torch, the model's own source tree) are
    not installable where the tests run, a `conftest.py` stubs them only
    when absent, so the same tests run without a GPU and, inside the image,
    against the real imports. The app tests drive the hooks with a fake model
    half that records steps and returns scripted results; the model tests
    drive `generate()`'s bookkeeping with the encoders and the chunk step
    replaced.
12. No encoder runs inside a command handler; the state holds arrays,
    strings, and ids, never GPU tensors.
13. The rendered schema is compared with the model it replaces. Render both
    (`python -m reactor_runtime.schema`, or `ModelContract.of(cls)` with the
    heavy imports stubbed) and diff. A class docstring on the app is
    published as the document's description, so an app that replaces one
    without a docstring carries none, or the schema moves.
14. The model half's constructor takes nothing and the class reads no
    environment variable. A weights root arrives as `load()`'s second
    argument from the application, and every relative path a config names
    resolves against the config file, not the working directory.

## Prose

This repository is public. Write every docstring and comment for an outside
reader with no other context: active voice, one topic per sentence, one name
per thing. The rules in [AGENTS.md](../../AGENTS.md) apply here as everywhere.
