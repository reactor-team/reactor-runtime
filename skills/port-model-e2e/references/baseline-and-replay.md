# Baseline and replay

Phase 2 records what the source does. Phase 5 runs the port under the same
conditions and proves it does the same thing. Together they are the reason a
port can be trusted: a playable model that is shown to agree with the code it
came from. Everything in this file serves one comparison, and the bar for it
is **bit-identical output, step by step**.

The source environment and the weights come from
[`source-and-weights.md`](source-and-weights.md); the streaming refactor, when the source needs one, from
[`streaming-refactor.md`](streaming-refactor.md) (Phase 3); the port itself from
[`port-on-reactor-app.md`](port-on-reactor-app.md) (Phase 4).

## Phase 2: record the baseline

### 2.0 Plan the steps and the random draws before recording

A baseline is recorded per step, and a step is what the port will make it: one
chunk, or one action step. Before recording anything, read the source
and write down two things:

- **Where the port's step boundaries fall.** For a source that already loops
  per chunk, they are its own iterations. For a source that renders
  offline, map its pipeline first, as step 1 of
  [`streaming-refactor.md`](streaming-refactor.md) describes, and take the
  boundaries from that map.
- **How the source draws random numbers, and how the port will.** From the same
  map: which generator, what shapes, in what order. When the two plans differ,
  the source edit in 2.2 is needed before recording.

The plan decides whether Phase 3 has work to do. If the refactor there arrives
at a different plan, the baseline is re-recorded under the new plan before the
refactor continues.

### 2.1 Freeze nondeterminism, then prove the source agrees with itself

A baseline is only a reference if running it again gives the same bytes. Before
recording, remove every source of run-to-run variation the source carries:

| Source of variation | Freeze it with |
|---|---|
| Kernel selection by timing (`torch.compile` max-autotune, Triton autotune, TensorRT tactics) | turn autotuning off, or pin the chosen configs to a file both runs load |
| cuDNN algorithm search | `torch.backends.cudnn.benchmark = False`, `torch.backends.cudnn.deterministic = True` |
| Nondeterministic kernels (atomic adds in scatter, index_add, some backward-style ops used in inference) | `torch.use_deterministic_algorithms(True)` with `CUBLAS_WORKSPACE_CONFIG=:4096:8`; a kernel that has no deterministic version raises, which names it |
| Unseeded draws | seed Python `random`, NumPy, and every torch generator (CPU and each CUDA device) from the conditions file; find generators the source creates itself with `grep -rn "Generator(\|manual_seed\|randn\|rand(\|randint\|normal_" <src>` |
| Data-dependent order | sorted file listings, a single data-loader worker, fixed thread counts |

Then run the source twice, **in two separate processes**, under the same
conditions, and diff every step's output. Two calls inside one process share
warm caches and autotune results, so they agree more often than two processes
do; the comparison the port will face is cross-process.

- **Identical:** the bar for the refactor (Phase 3) and the replay (Phase 5)
  is bit-identical.
- **Not identical after everything above:** name the operation that still
  varies (a kernel with no deterministic version, say). Record the source's own
  per-step `max|diff|` between its two runs; that series is the **floor**, and
  the run continues. In Phases 3 and 5 the per-step diff against the baseline
  is held within the floor at every step. For an autoregressive model the floor
  grows with the step index, because each step's difference feeds the next;
  the floor recorded from the source's own two runs already carries that growth.
- **Not identical, and no operation accounts for it:** the baseline has no
  anchor. Report what varies between the runs and stop (SKILL.md, "When a phase
  cannot finish").

Freezing goes into the source as the smallest edit that does it, and the same
settings are carried into the port. Record the edit with the baseline.

### 2.2 Make the source draw random numbers the way the port will

A fixed seed fixes the **sequence** of random numbers, not which number lands
where. If the port draws them differently from the source, the two disagree
from the first step, with the same seed, and no diff can tell a porting bug
from that.

The usual cause is a restructure the port needs. A source that renders a whole
clip offline often draws all its noise at once; a port that generates step by
step draws each step's noise when it needs it. With seed 42, suppose the
generator's sequence is `a1 a2 a3 a4 a5 a6`:

- The source draws a `[channel, time]` tensor of shape `[2, 3]` in one call.
  Row-major, channel 0 is `a1 a2 a3` and channel 1 is `a4 a5 a6`, so time
  step 0 gets `(a1, a4)`.
- The port draws a `[channel]` tensor of shape `[2]` per step, so step 0 gets
  `(a1, a2)`.

Same seed, different numbers, from step 0 on. A CUDA generator also advances
by a call-size-dependent offset, so one large draw and several small ones
differ even without a layout change.

This section is the one place the rule is written; the other files of this skill
link here.

**Make that one change in the source, before recording:**

1. Edit the source so it draws exactly as the port will: the same generator
   object, the same shapes, the same order, the same device. Change nothing
   else. Keep the edit as a patch next to the baseline.
2. Run the untouched source and the edited source under the same conditions
   and render them side by side. The content differs; the quality must not.
   This is the one comparison in the workflow that only a person can judge,
   and the only one that is not bit-for-bit.
3. Record the baseline from the edited source.

The port then draws exactly as the baseline does and is held to bit-identical
output. The served model carries no hook for test noise; its random numbers
come from its own generator, seeded the way the conditions file says.

If the source already draws the way the port will (a model that steps
internally usually does), skip this section and record from the untouched
source.

### 2.3 One conditions file, read by both sides

Write the conditions down **before** the source runs, as `baseline/conditions.yaml`,
and make the source's runner read it instead of carrying its own flags. The
port's replay reads the same file, so "same conditions" is a property of the
setup rather than something retyped.

| In `conditions.yaml` | Why it is fixed |
|---|---|
| source repo URL and commit; the image reference | what produced the reference |
| every seed (global, noise, sampler) and the freezing settings from 2.1 | an unpinned draw diverges |
| the full config: resolution, steps, dtype, sampler, guidance scale, and every other value the example sets | a default that differs between the two sides is a difference you will chase as a bug |
| the session inputs: first frame or reference image (as a file under `baseline/inputs/`), the initial prompt | the conditioning the port must follow |
| the per-step input schedule (below) | the output of an interactive model depends on when an input lands, not only on what it is |
| the number of steps | both trajectories end at the same step |

Inputs the source does not expose as parameters (a hard-coded seed, a prompt
inside a script) are lifted into this file with the smallest edit to the
source, recorded with the baseline.

The schedule's shape follows the step shape read in SKILL.md Phase 0:

| Step shape | The schedule is |
|---|---|
| A chunk of a continuous stream, driven by text or by actions | a timeline: for each chunk index, the inputs that arrive before it (a prompt change, the keys held, a click with its coordinates, the points of a drag that fall in that chunk) |
| A chunk of output per chunk of an input video | a recorded input video under `baseline/inputs/`, plus any prompt or reference changes by chunk index |

### 2.4 The trajectory

The source run writes `baseline/source/`:

- per step, the output in its native form: decoded frames as arrays (`.npy`),
  and latents where they are cheap to save;
- per step, the wall-clock time;
- one rendered clip of the whole run.

Save frames as arrays, not as encoded video. An encoded file is lossy, and a
diff of two decodes measures the encoder.

**Record what the baseline cost, next to what it produced.** The per-step time
and the frame rate it works out to are the "before" number for every later
speed claim, and this is the only point at which they are cheap to get. Take
them from timing the source itself emits where it emits any; where a field
looks like per-step time, check whether it is cumulative before dividing by it.

### 2.5 Exit

- The source ran twice in two processes with the same output, or the floor is
  recorded with the operation that causes it.
- Any random-number edit is applied, its patch kept, and its side-by-side
  against the untouched source rendered.
- `baseline/conditions.yaml`, `baseline/inputs/`, and `baseline/source/` are
  written, with per-step time.

## Phase 5: replay the baseline on the port

### 5.1 Pick the route

All three routes run the port's own code inside the port's image, under the
same `conditions.yaml`, and write `baseline/port/` in the same layout as
`baseline/source/`.

The two served routes need the port built and serving on the node: build,
`reactor run` and the `/health` wait are in
[`serving.md`](serving.md) §1–2. The replay client runs on the
node too, so nothing crosses a network: the Python client SDK
(`reactor-sdk` on PyPI; the client, not the runtime) in a small virtual
environment, talking to `localhost`.

| Route | What it exercises | Use it when |
|---|---|---|
| **Stepped session** (default) | the served path: the runtime's dispatch, the setters, all three hooks, one step per request | the app leaves `run()` and `submit_command()` to the runtime, which is the normal case |
| **Model-half driver** | `load()` and `generate()` of the model half only | the first comparison after a restructure, to isolate the model from the application |
| **Starting input with a step count** | a streamed session that applies its inputs before any client connects and closes after N steps | the inputs are fixed at the start and the comparison only needs per-step files |

**Stepped session.** A session started with `"generation": "stepped"` runs
one step per client request. A request carries `state` values, `commands`,
and `media_input` frames; the state and commands reach the model as ordinary
commands ahead of the step, and when the request reaches the front of the
inbound queue its frames are written to the input tracks and the three hooks
run once, with nothing else dispatched in between. Nothing streams; each
step's result is returned to the requester
(`src/reactor_runtime/interface/app/reactor_app.py`, module docstring;
`src/reactor_runtime/runner/session_start.py`, `Generation`). That makes it
an exact replay of a schedule: put each step's inputs from `conditions.yaml`
into that step's request.

- Result frames travel uncompressed: the raw frame encoding carries video as
  `uint8` RGB `(height, width, 3)` and audio as `int16` PCM
  (`src/reactor_runtime/protocol/frames.py`). Compare those arrays.
- Live input tracks are ignored in a stepped session; a step reads only the
  frames its request carries (`src/reactor_runtime/runner/runner.py`,
  `Runner.media_received`). A video-to-video replay sends each chunk's input frames in
  that step's request.
- A model that overrides `run()` or `submit_command()` cannot be stepped; the
  start is refused with a message naming the model
  (`ReactorApp.supports_stepped`, `Runner._check_generation`). A port leaves
  `run()` and `submit_command()` to the runtime.
- `examples/echo` is the reference: its README shows a stepped session from
  the Python SDK (`Reactor(name, local=True, generation="stepped")`, then
  `reactor.step(state=...)`), and `tests/unit/examples/test_echo.py` drives
  the same thing in-process, posting commands and a step envelope onto the
  app's inbound queue.

**Model-half driver.** The model half is a plain class with `load(config_path)`,
`generate(input)`, and `reset()` that imports nothing from the runtime
(`skills/application-model-isolation`, rule 1). The driver loads it, builds
each step's input dataclass from `conditions.yaml`, calls `generate()` per
step, and saves the result. It runs inside the port's image
(`docker run --entrypoint python …`) on the same GPUs. It skips the
application half, so a model-half match followed by a stepped-session
mismatch locates the bug in the application half.

**Starting input with a step count.** `POST /start_session` accepts
`starting_input` (`state` values, then `commands`, applied before any client
connects) and `steps`, after which the runtime closes the session
(`src/reactor_runtime/runner/session_start.py`, at most 10,000 steps). With a
`runtime.step_results` block in `reactor.yaml`, each finished step is kept and
served at `GET /sessions/{id}/steps/{n}` and `/sessions/{id}/steps/{n}/{name}`.
The saved `output.mp4` is H.264 or H.265 (`src/reactor_runtime/step_results/mp4.py`),
so it is not a comparison input; compare raw arrays the step writes as extra
files instead (`StepCompleted.files`, returned from `process_output()`), or
use one of the other two routes. Inputs that change mid-session cannot be
scheduled this way.

### 5.2 Compare, step by step

For every step, both sides' outputs as arrays:

| Column | Meaning |
|---|---|
| `max\|diff\|` | the largest absolute difference anywhere in the step's output |
| `mean\|diff\|` | the average absolute difference |
| max relative | the largest difference relative to the baseline's magnitude |
| cosine | cosine similarity of the two flattened outputs |
| source time, port time | each side's wall clock for that step |

**The bar is bit-identical: `max|diff| = 0` on every step**, or within the
source's own floor (§2.1) when one was recorded. A port moves code; it changes
no numerics. Precision changes, kernel swaps, fused ops and compilation are
acceleration work and come after the port.

A tolerance is allowed only for a change the port could not avoid (a
dependency that cannot be installed, replaced by an equivalent). Name the
change and state the tolerance **before** the run, in this form:

- max absolute difference,
- max relative difference,
- cosine similarity of the flattened outputs,

each with its threshold. A tolerance chosen after seeing the numbers is not a
bar. The report lists each such change with its declared tolerance.

**When a step does not match, find where it diverged.** The other columns are
the diagnostic:

- A difference that is small and spread like noise across the output (no
  region, frame or channel stands out, the side-by-side shows the same
  content) points at numerics that changed without anyone meaning them to: a
  different dtype resolved, a different kernel picked, an autotune result.
  The resolved-configuration log below usually names it.
- A difference that is large or structured (wrong content, a frame offset, a
  channel swapped, a conditioning input ignored, a different random draw) is a
  logic difference: the port does something the source does not.
- Compare against the floor first: a difference near it is what the source
  does to itself.
- For an autoregressive model a difference grows over steps, because each
  step's difference feeds the next; the first step that differs is the one to
  debug.

Log the configuration each side actually resolved (dtype, attention backend,
compile mode, the seeds) at the start of each run, and diff the two logs. A
setting that never applied on one side shows up there, not in the output diff.

### 5.3 One side-by-side artifact

Render one video: source on the left, port on the right, the same step aligned, with labels burned in: which side is which, the
step index, that step's `max|diff|` and cosine, and each side's step time. One file for
the whole trajectory, not one per step.

### 5.4 Read the result

- **Bit-identical on every step** (or within the floor): the port reproduces
  the source.
- **Diverges on the port, and the source run was good:** a porting bug, by
  construction. The first diverging step and the first diverging part of that
  step (compare intermediate tensors through the model-half driver) say where.
- **Wrong on both sides:** the problem predates the port. Debug the source.
- **The environments differ** (the source could only run in its own image, not
  in the port's): open the environment deviation record from
  [`source-and-weights.md`](source-and-weights.md) first. A library version
  change can move numerics by itself; rerun the source in the port's image
  where it installs, and compare that to the original baseline before
  touching the port.

### 5.5 Exit

`baseline/port/` is written under the same conditions; the per-step table
shows `max|diff| = 0` on every step (or stays within the floor from 2.1), or
stays inside a tolerance declared before the run for a named, unavoidable
change; the side-by-side artifact is rendered.

## Who looks, and when

The workflow runs to the end without waiting for a person. Three things are
made for a person to look at, and all three go into the final report, in this
order:

1. the random-number edit side by side against the untouched source, when one
   was made: if that edit hurt quality, everything after it is measured
   against the wrong reference;
2. the source's own output: whether the model itself is any good;
3. the replay side by side with its per-step table.

One switch, chosen at the start: **stop after the source example.** The
workflow runs the untouched source's own example, renders it, and stops there,
before any edit to the source (the random-number edit included), so a person
can decide whether the model is worth porting before any more time is spent.
Off by default.
