---
name: porting-to-reactor-app
description: "Move a model that already runs on the standalone 3.x runtime (3.0 to 3.4: a ReactorModel with a hand-written run() loop, or a ReactorPipeline with an inference() generator) onto reactor-runtime 3.5's ReactorApp and the step loop. This is the 3.x to 3.5 port. Use when porting such a model, when a model still yields Idle or flips private flags the loop consumes, or when reviewing such a port. A 2.x model first goes through porting-models-to-standalone-runtime. The method is a split, not a translation: find the application's bounds, find the model's real constraints, and write each half where it belongs so the loop's decisions become explicit."
---

# Porting a model onto `ReactorApp`

This is the **3.x to 3.5** port. The model already runs on the standalone
runtime installed as a package (reactor-runtime 3.0 to 3.4); what changes is
its shape. A model still on 2.x (the pre-baked base image, imports from
`reactor_runtime.interface`, `load(config: dict)`) goes through
[`porting-models-to-standalone-runtime`](../porting-models-to-standalone-runtime/SKILL.md)
first and arrives here running unchanged on 3.x.

A model written before the step loop owns its own loop. On `ReactorModel` it
is a `run()` with `while self.connected.is_set()`. On `ReactorPipeline` it is
an `inference()` generator that yields an `Output` or `Idle`. In both, one
function holds three kinds of decision at once: whether to step, how to step,
and what to send. The port takes that function apart and puts each decision
in the hook that owns it.

Read [`application-model-isolation`](../application-model-isolation/SKILL.md)
first. It states the rules the finished port must satisfy; this skill is how
to get there from existing code. The worked example is
[`examples/waypoint/`](../../examples/waypoint/README.md), a small model with
one loop. The steps below also carry what a larger port taught: a model with
one loop per GPU topology, a pipelined chunk that lags the step that submitted
it, encoders that ran inside command handlers, and a run whose first step
consumes a different input than the rest.

## The method is a split, not a translation

Do not rewrite `inference()` line by line into `generate()`. A generator that
yields `Idle` while paused, checks a `_do_reset` flag, calls the engine, sends
a progress message, and yields a frame is application code and model code
braided together. Unbraid it. For every line, ask two questions:

1. **Could a client observe or cause this?** Then it is application code. It
   belongs in `process_input()`, `process_output()`, a handler, or a state field.
2. **Is this a fact about the weights, the cache, or the rollout?** Then it is
   model code. It belongs in the model class, behind `generate()` or `reset()`.

A line that answers neither is loop mechanics (`yield Idle`, a sleep, a
generator restart, `await self.connected.wait()`), and it goes away: the
runtime's loop does that now.

## Step 1: inventory the old loop

Before writing anything, read `run()` or `inference()` and sort what you find
into four lists.

**Client facts.** Conditions the loop checks that a client set or caused:
`_paused`, `_start_requested`, `not self.state.prompt`, a buffer with too few
frames. Each becomes a refusal in `process_input()`, and the value it reads
becomes a public state field where a client sets it.

**Model facts.** Conditions about the model's own state: a cache that is
empty, an index past a window, a seed not yet applied. Each becomes a check
inside the model class, and the failing case becomes the model's own
exception.

**Effects.** Everything the loop sends or cuts: `await self.send(...)`,
`self.output.flush()`, a progress counter. Each becomes a line in
`process_output()`, or in the handler that owns the decision.

**Mechanics.** `yield Idle`, `yield None`, `asyncio.sleep`, `continue` after a
flag check, the `finally:` that closes the generator, the outer
`while True: await self.connected.wait()`. Delete all of it.

Two more lists for a loop that is bigger than one function:

**Generator locals that outlive one turn.** The conditioning snapshot, the
caches, the counters, a first frame held back to prepend to the first chunk,
whether a worker's streaming loop is open. These become fields of one run
object the model half creates when a run starts and drops when it ends. List
them now; the port is done when none is a local.

**Phases.** Places where the loop consumed a different input in different
turns: one webcam frame to make the first output frame, four per chunk after
that. Each phase boundary becomes a fact the model reports on its result
(`frames_wanted`) and the application reads for the next `process_input()`.
The application never learns the phases; it reads the result.

If `load()` assigns one of several `inference()` generators to
`self.inference` (one per GPU topology), inventory each generator with the
same lists. They share a preamble and differ in the chunk step; the port
keeps one run bookkeeping and several chunk steps (step 3).

## Step 2: draw the application's bounds in `process_input()`

Every client fact from the inventory is one `if` at the top of
`process_input()`, refusing with `ApplicationError` and the reason as the
message. What the loop used to skip a turn on, the application now refuses a
step on.

```python
# before, inside inference()
while not self.state._start_requested:
    yield Idle
...
while self.state._paused:
    yield Idle

# after
async def process_input(self, state: WaypointState, media: None) -> WaypointInput:
    if state.paused:
        raise ApplicationError("paused")
    if state._seed is None:
        raise ApplicationError("no seed image")
    return WaypointInput(...)
```

Then build the step input. It is a dataclass you write, carrying exactly what
one step needs: the controls, the prompt, the frames read off a track with
`try_read()`, the seed. Reading a track belongs here and nowhere else: a
`media.webcam.try_read(4)` that returns `None` is another refusal.

Two things change shape on the way:

- **A private flag a client flips becomes a public field.** `_paused: bool`
  set by a hand-written `pause` command is `paused: bool = InputField(...)`,
  and the client gets `set_paused` generated. Delete the `pause` and `resume`
  handlers. Keep a field private only when the client cannot set it: a
  decoded image, a seed id, a counter the app maintains.
- **A flag the loop consumes goes away.** `_start_requested` and `_do_reset`
  existed because a handler could not touch the generator; it left a note
  and the next turn read it. Handlers run between steps now, so a handler
  does the thing: it calls `self.engine.reset()`, or bumps the seed id the
  step input carries. There is no note to leave and nothing to consume.

## Step 3: find the model's real constraints and give them a class

Take the engine calls out of the loop and put them in a plain class with
`load()`, `generate()`, and `reset()`. It imports nothing from
`reactor_runtime`. The app constructs it in `load()` and holds it under an
ordinary attribute name, `self.engine` in the example. Do not name it
`state`, which is the typed state the runtime owns. The same attribute later
holds a `DistributedRunner` around the class when the model needs its own
process, and nothing else on the app changes.

If the old pipeline resolved its checkpoint paths under `get_weights_path()`,
that call moves to the application's `load()` and its result becomes the
model half's second argument: `self.engine.load(config_path,
get_weights_path())`. Inside the model half, the old `_resolve()` helper
becomes `weights_root / relative_path`. The constructor takes nothing. The
model half imports `get_weights_path` no more than it imports anything else
from `reactor_runtime`, and it resolves any other file the config names
against `config_path.parent`, not the working directory. A model that fetches
its weights from elsewhere leaves the parameter out. See rule 1 of
[`application-model-isolation`](../application-model-isolation/SKILL.md).

Then ask what makes a step invalid for the model itself, with no client in
the picture. A window that is full. A world that was never seeded. A cache
that was reset and needs a first frame. Each is a check inside `generate()`
and its own exception type. This is the one place a raise is the right
answer: the model says "I cannot step from where I am" and the application
decides what to do about it in `process_output()`.

```python
# waypoint_model.py
class NotSeeded(Exception):
    """The model holds no world to step and the step input carries no seed."""

class WaypointModel:
    def generate(self, input: WaypointInput) -> WaypointResult:
        if input.seed_id != self.seed_id:
            if input.seed is None:
                raise NotSeeded("no seed frame to start a world from")
            ...
```

Notice what the model does **not** check: `paused`, `start_requested`, "is a
client connected". Those are client facts and they were refused before
`generate()` was called. If a check in the old loop cannot be classified as
either a client fact or a model fact, it is usually two checks in one; split
it.

The step result is the second dataclass. Put in it whatever the application
needs to know about the step: the frames, the model's own index, anything the
old loop used to read off `self.engine.*` directly, and anything the
application needs to build the next input (`frames_wanted`) or to label a
metric (`num_gpus`). After the port the application reads the result and
never the engine.

**Helpers written against the pipeline object move with the model.** A
module of `helper(pipe, ...)` functions that read `pipe.generator`,
`pipe.vae`, `pipe.executor`, `pipe.device` is model code. Give the model
class the same attribute names and pass it where the pipeline went; the
module needs no edit. Grep for every attribute those helpers read and make
sure `load()` sets each on the model class.

**Several loops become one `generate()` and several chunk steps.** Where the
old class chose an `inference()` generator per GPU topology, the model half's
`load()` chooses a chunk step (`self._step = self._chunk_step_2gpu`), and one
`generate()` does the shared work around it: start a run on a new id, the
first-frame step, the chunk step, the drain at the cap. The old generators'
shared preamble (encode the reference, build caches, prefill, first frame)
becomes `_start_run()` and `_first_frame_step()`; their bodies become the
chunk steps; the block after each loop (stop the worker, collect the last
result) becomes `_drain()`; the `finally:` that told a worker where the loop
stopped becomes `_end_run()`, which `reset()` and a new id both call.

**A pipelined chunk step returns the previous chunk.** On a topology where a
worker denoises one step behind, the step that submits chunk `k` collects
chunk `k-1`, so the first chunk step returns `None` and the run's last chunk
is collected by the drain. Keep that on the result (`frames=None`); the
application emits nothing for it. Keep the worker's stop index on the run
object too: after a completed step the worker waits for `chunk_index + 1`,
after a step that raised between submit and collect for `chunk_index + 2`.
The old `finally:` knew this by a local flag; the run object carries it as
`submitted`.

**Encoders leave the handlers.** Where `set_prompt` ran the text encoder and
`set_image` ran CLIP and the VAE, the handlers now store text and a
letterboxed uint8 array and bump an id; `_start_run()` encodes them on the
step that applies the id. The first step of a run pays what the handler used
to pay, and the state object holds no tensors.

The app's `generate()` is then one line:

```python
def generate(self, input: WaypointInput) -> WaypointResult:
    return self.engine.generate(input)
```

## Step 4: put the effects in `process_output()`

Every `send()` the loop made, every progress counter, every mapping from a
result to an `Output`, lives in `process_output()`. It receives the outcome,
result or error, and returns the media to emit or `None`.

```python
# before, inside inference()
frames = self.engine.gen_frame(ctrl=ctrl).cpu().numpy()
self.state._step_idx += 1
if self.state._step_idx % self.progress_interval == 0:
    await self.send(GenerationProgress(step_index=self.state._step_idx))
yield WaypointOutput(main_video=frames)

# after
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

A message sent here goes on the wire before the step's media, so the
ordering the old loop got by sending before yielding is kept without effort.

Recovery from a model error lives here too. In the old loop an engine
exception either escaped `inference()` and killed the model loop, or was
caught in a `try` around the forward pass. On the step loop the runtime
catches it for you and hands it to `process_output()` as `outcome.error`, so
the `try` becomes an `if`:

```python
# before, inside inference()
try:
    frames = self.engine.step(...)
except RolloutExhausted:
    self.engine.reset()
    await self.send(WorldRestarted(...))
    continue

# after
async def process_output(self, outcome: StepOutcome) -> MyOutput | None:
    if isinstance(outcome.error, RolloutExhausted):
        self.engine.reset()
        self.output.flush()
        await self.send(WorldRestarted(...))
        return None
    if outcome.error is not None:
        raise outcome.error
    ...
```

Port every `except` the old loop had into such an `if`, and re-raise the
rest. A re-raise out of `process_output()` has the same effect an escaped
exception had in `inference()`: the runtime logs it, ends the session with
an error, and does not restart the model loop. Do not turn that into a
blanket recovery; a model that resets itself on every error hides the bug
that caused it. Write the reason for a bare re-raise at the `raise`.

## Step 5: rewrite the handlers to act, not to flag

With the loop gone, a handler no longer coordinates with a generator. Go
through each `@event` and each lifecycle hook:

- A handler that set `_do_reset = True` calls `self.engine.reset()` and
  `self.output.flush()` directly, and forgets what the model held
  (`self.state._applied_seed_id = None` in the example), so the next step
  input carries the seed again.
- A handler that set `_start_requested = True` bumps the id the step input
  carries, so the model sees a new world on the next step. Or it disappears,
  if uploading the seed is itself the start, as in the example.
- `pause` and `resume` disappear: `set_paused` is generated from the field.
- A `@session_started` that reset the engine so a new session begins clean
  moves to `@session_ended`. The runtime never resets a model; the
  application does, when the session that used it ends.
- Any handler that read `self.engine.<attribute>` reads it off the last step
  result instead, or asks the model class for a method that returns it.

Counters the old loop kept on the state (`_step_idx`, `_frames_generated`)
usually belong to the model, which counts its own steps and exposes the count
on the result. Delete the state fields.

Two behaviours of the old driver need a home, because nothing restarts a
generator any more:

- **Natural completion.** When the old generator returned at its cap, the
  pipeline driver restarted it, and with the conditions still met a new run
  began at once. `process_output()` does that on the step whose result says
  `complete`: send the completion message, call `self.engine.reset()`, clear
  the applied id so the next step carries the reference again, and
  `self.output.flush()` as the restart used to. The model raises its own
  `RunComplete` if stepped past the cap without a reset, so the application's
  call is the only way forward.
- **A new image during a run.** Where `set_image` set `_do_reset` and the
  generator returned and restarted, the handler now bumps the id and flushes;
  the model ends the live run and starts the next when the id reaches it on
  the step input. No handler calls into the model for this.

## Step 6: what the port deletes

If any of these survive, the port is not done:

- `Idle`, `yield`, and `inference()` itself.
- `_start_requested`, `_do_reset`, or any flag a handler sets for the loop.
- Hand-written `pause` / `resume` over a private flag.
- `await self.connected.wait()` and `while self.connected.is_set()`.
- `asyncio.sleep` used to yield the loop.
- `compute_time=` passed to `emit()` by hand; the loop paces playout from the
  measured `generate()` time, and a declared `fps` pins it.
- A `finally:` block that released what the session held; do it in
  `@session_ended`.
- Private state fields that hold tensors (`_prompt_cond`, `_clip_fea`,
  `_initial_latent`); the model half holds them on its run object.
- Host-side timers around the chunk; `outcome.elapsed` is the runtime's
  measurement of `generate()`, and the metrics line in `process_output()` uses
  it. Time-to-first-frame starts from the step whose result says
  `run_started`.

One thing a port on the step loop cannot keep: an overlap the old loop
arranged across turns, such as encoding the next chunk's frames on one CUDA
stream while decoding the current chunk on another. On the step loop the
next chunk's frames arrive with the next step, so the two serialize. Measure
the cost and write it in the port notes; on the topology that had the
overlap it is one encode of a few frames per chunk.

Keep `run()` overridden only when the loop is truly not one step per emit: a
renderer that emits several times per step, or a model that must block on an
input. That is the escape hatch, not the target of a port. Overriding `run()`
replaces the default loop and only the loop: the dispatch layer stays,
handlers and hooks still run under the step lock, `emit()`, `send()`,
`self.connected`, and the tracks are all there, and `process_input()`,
`generate()`, and `process_output()` are never called for that class. No error,
no warning. A port that keeps `run()` does not declare `state:`: the setters
would write it, but nothing reads it for the loop and nothing bounds when a
write lands relative to the loop's reads. It keeps its own values in its own
attributes and its own `@event` handlers, the way a 3.4.0 model did.

## A port, start to finish

A `ReactorPipeline` with a paused flag, a start command, and a progress
message:

```python
class OldState(InputState):
    prompt: str = InputField(default="")
    _paused: bool = False
    _started: bool = False
    _index: int = 0

class Old(ReactorPipeline):
    state: OldState

    def load(self, config_path):
        self.pipe = load_pipe(config_path)

    @event(name="start", description="Begin generating.")
    async def start(self) -> None:
        self.pipe.reset()
        self.state._started = True

    @event(name="pause", description="Hold generation.")
    async def pause(self) -> None:
        self.state._paused = True

    async def inference(self):
        while not self.state._started:
            yield Idle
        while True:
            if self.state._paused or not self.state.prompt:
                yield Idle
                continue
            frame = self.pipe.step(self.state.prompt)
            self.state._index += 1
            if self.state._index % 50 == 0:
                await self.send(Progress(index=self.state._index))
            yield Frame(main_video=frame)
```

The same model on `ReactorApp`, split in two files:

```python
# old_model.py: the model half, no reactor_runtime import
@dataclass(frozen=True)
class OldInput:
    prompt: str

@dataclass(frozen=True)
class OldResult:
    frame: np.ndarray
    index: int

class OldModel:
    def load(self, config_path):
        self.pipe = load_pipe(config_path)
        self.reset()

    def generate(self, input: OldInput) -> OldResult:
        frame = self.pipe.step(input.prompt)
        self.index += 1                     # the old loop counted this frame before it reported
        return OldResult(frame=frame, index=self.index)

    def reset(self) -> None:
        self.pipe.reset()
        self.index = 0


# old.py: the application half
class NewState(InputState):
    prompt: str = InputField(default="", description="Scene to render.")
    paused: bool = InputField(default=False, description="Hold generation.")

class New(ReactorApp):
    state: NewState

    def load(self, config_path):
        self.engine = OldModel()
        self.engine.load(config_path)

    async def process_input(self, state: NewState, media: None) -> OldInput:
        if state.paused:
            raise ApplicationError("paused")
        if not state.prompt:
            raise ApplicationError("no prompt set")
        return OldInput(prompt=state.prompt)

    def generate(self, input: OldInput) -> OldResult:
        return self.engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> Frame | None:
        if outcome.error is not None:
            raise outcome.error
        result: OldResult = outcome.result
        if result.index % 50 == 0:
            await self.send(Progress(index=result.index))
        return Frame(main_video=result.frame)

    @event(name="reset", description="Start over from the current prompt.")
    def reset(self) -> None:
        self.engine.reset()
        self.output.flush()

    @session_ended
    def end(self) -> None:
        self.engine.reset()
```

What moved where: `_paused` became a public field and `pause` disappeared;
`start` disappeared because a prompt being set is the start, and `reset`
took over the one job `start` still had; the `Idle` loop became two
refusals; the index moved into the model and rides on the result, counted
the way the old loop counted it, so `Progress` still goes out after frames
50, 100, and so on; the progress message moved to `process_output()`. The
client contract gained `set_paused` and `reset` and lost `start` and
`pause`. Say so in the change.

## Verify the port

1. Render the schema before and after (`python -m reactor_runtime.schema`)
   and diff the two documents. Every command that changed did so on purpose,
   and the change is written in the PR. When the model's imports need a GPU
   image, render both with those modules stubbed: the schema is built from
   the class, not from the weights. A class docstring on the app is
   published as the document's description; if the old class had none, the
   new one carries none.
2. Run the review checklist in
   [`application-model-isolation`](../application-model-isolation/SKILL.md).
3. The model half has a test with a fake engine; the app half has a test for
   each refusal and for `process_output()`, including its error branch. Where
   torch and the model's source tree are not installable in the test
   environment, a `conftest.py` stubs them when absent, so the tests run
   without a GPU and, inside the image, against the real imports. The model
   half's run bookkeeping (new id starts a run, first-frame step, pipelined
   lag, drain at the cap, worker stop indices on reset) is testable with the
   chunk step and the encoders replaced.
4. Serve it and drive it from a client: pause, resume, reset, and every
   command the old model had. Frames reach the client and the messages arrive
   before the frames they describe. Serve the model it replaces the same way
   and compare the message sequence and the frame cadence.
5. Write the port notes: every decision that was not a mechanical
   translation (which phases the result reports, what moved out of handlers,
   what the run object surfaced, what was lost), so the next port and this
   skill can learn from them.
