---
name: porting-to-reactor-app
description: "Move a model from a hand-written ReactorModel run() loop or a ReactorPipeline inference() generator onto ReactorApp and the step loop. Use when porting an existing model, when a model still yields Idle or flips private flags the loop consumes, or when reviewing such a port. The method is a split, not a translation: find the application's bounds, find the model's real constraints, and write each half where it belongs so the loop's decisions become explicit."
---

# Porting a model onto `ReactorApp`

A model written before the step loop owns its own loop. On `ReactorModel` it
is a `run()` with `while self.connected.is_set()`. On `ReactorPipeline` it is
an `inference()` generator that yields an `Output` or `Idle`. In both, one
function holds three kinds of decision at once: whether to step, how to step,
and what to send. The port takes that function apart and puts each decision
in the hook that owns it.

Read [`application-model-isolation`](../application-model-isolation/SKILL.md)
first. It states the rules the finished port must satisfy; this skill is how
to get there from existing code. The worked example is
[`examples/waypoint/`](../../examples/waypoint/README.md).

## The method is a split, not a translation

Do not rewrite `inference()` line by line into `generate()`. A generator that
yields `Idle` while paused, checks a `_do_reset` flag, calls the engine, sends
a progress message, and yields a frame is application code and model code
braided together. Unbraid it. For every line, ask two questions:

1. **Could a client observe or cause this?** Then it is application code. It
   belongs in `prepare_step()`, `collect_step()`, a handler, or a state field.
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
frames. Each becomes a refusal in `prepare_step()`, and the value it reads
becomes a public state field where a client sets it.

**Model facts.** Conditions about the model's own state: a cache that is
empty, an index past a window, a seed not yet applied. Each becomes a check
inside the model class, and the failing case becomes the model's own
exception.

**Effects.** Everything the loop sends or cuts: `await self.send(...)`,
`self.output.flush()`, a progress counter. Each becomes a line in
`collect_step()`, or in the handler that owns the decision.

**Mechanics.** `yield Idle`, `yield None`, `asyncio.sleep`, `continue` after a
flag check, the `finally:` that closes the generator, the outer
`while True: await self.connected.wait()`. Delete all of it.

## Step 2: draw the application's bounds in `prepare_step()`

Every client fact from the inventory is one `if` at the top of
`prepare_step()`, refusing with `ApplicationError` and the reason as the
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
async def prepare_step(self, state: WaypointState, media: None) -> WaypointStepInput:
    if state.paused:
        raise ApplicationError("paused")
    if state._seed is None:
        raise ApplicationError("no seed image")
    return WaypointStepInput(...)
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
ordinary attribute name.

Then ask what makes a step invalid for the model itself, with no client in
the picture. A window that is full. A world that was never seeded. A cache
that was reset and needs a first frame. Each is a check inside `generate()`
and its own exception type. This is the one place a raise is the right
answer: the model says "I cannot step from where I am" and the application
decides what to do about it in `collect_step()`.

```python
# waypoint_model.py
class NotSeeded(Exception):
    """The model holds no world to step and the step input carries no seed."""

class WaypointModel:
    def generate(self, step: WaypointStepInput) -> WaypointStepResult:
        if step.seed_id != self.seed_id:
            if step.seed is None:
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
old loop used to read off `self.engine.*` directly. After the port the
application reads the result and never the engine.

The app's `generate()` is then one line:

```python
def generate(self, step: WaypointStepInput) -> WaypointStepResult:
    return self.engine.generate(step)
```

## Step 4: put the effects in `collect_step()`

Every `send()` the loop made, every progress counter, every mapping from a
result to an `Output`, lives in `collect_step()`. It receives the outcome,
result or error, and returns the media to emit or `None`.

```python
# before, inside inference()
frames = self.engine.gen_frame(ctrl=ctrl).cpu().numpy()
self.state._step_idx += 1
if self.state._step_idx % self.progress_interval == 0:
    await self.send(GenerationProgress(step_index=self.state._step_idx))
yield WaypointOutput(main_video=frames)

# after
async def collect_step(self, outcome: StepOutcome) -> WaypointOutput | None:
    if outcome.error is not None:
        raise outcome.error
    result: WaypointStepResult = outcome.result
    if result.index % self.progress_interval == 0:
        await self.send(WaypointStatus.of(self.state, result.index))
    metadata = [{"step": result.index}] * result.frames.shape[0]
    return WaypointOutput(main_video=TrackPayload(result.frames, metadata=metadata))
```

A message sent here goes on the wire before the step's media, so the
ordering the old loop got by sending before yielding is kept without effort.

Recovery from a model error lives here too. Where the old loop caught an
engine exception, reset, and continued, `collect_step()` checks
`outcome.error`, calls `self.engine.reset()`, sends a message, and returns
`None`. Re-raise anything you did not expect.

## Step 5: rewrite the handlers to act, not to flag

With the loop gone, a handler no longer coordinates with a generator. Go
through each `@event` and each lifecycle hook:

- A handler that set `_do_reset = True` calls `self.engine.reset()` and
  `self.output.flush()` directly.
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

Keep `run()` overridden only when the loop is truly not one step per emit: a
renderer that emits several times per step, or a model that must block on an
input. That is the escape hatch, not the target of a port.

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
class StepInput:
    prompt: str

@dataclass(frozen=True)
class StepResult:
    frame: np.ndarray
    index: int

class OldModel:
    def load(self, config_path):
        self.pipe = load_pipe(config_path)
        self.reset()

    def generate(self, step: StepInput) -> StepResult:
        frame = self.pipe.step(step.prompt)
        index = self.index
        self.index += 1
        return StepResult(frame=frame, index=index)

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

    async def prepare_step(self, state: NewState, media: None) -> StepInput:
        if state.paused:
            raise ApplicationError("paused")
        if not state.prompt:
            raise ApplicationError("no prompt set")
        return StepInput(prompt=state.prompt)

    def generate(self, step: StepInput) -> StepResult:
        return self.engine.generate(step)

    async def collect_step(self, outcome: StepOutcome) -> Frame | None:
        if outcome.error is not None:
            raise outcome.error
        result: StepResult = outcome.result
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
refusals; the index moved into the model and rides on the result; the
progress message moved to `collect_step()`. The client contract gained
`set_paused` and `reset` and lost `start` and `pause`. Say so in the change.

## Verify the port

1. Render the schema before and after (`python -m reactor_runtime.schema`).
   Every command that changed did so on purpose, and the change is written in
   the PR.
2. Run the review checklist in
   [`application-model-isolation`](../application-model-isolation/SKILL.md).
3. The model half has a test with a fake engine; the app half has a test for
   each refusal and for `collect_step()`.
4. Serve it and drive it from a client: pause, resume, reset, and every
   command the old model had. Frames reach the client and the messages arrive
   before the frames they describe.
