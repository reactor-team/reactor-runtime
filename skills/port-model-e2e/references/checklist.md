# Phase 7 — the served-model checklist

A port is not finished when the model serves. It is finished when the served model **behaves well**:
it keeps up with what it was measured to do, leaves nothing behind between sessions, holds its
memory steady, and fails visibly. This file holds that bar.

The checklist judges the port, not the model. It covers what changes when a model leaves its
research source and enters the runtime:

- how work is split across load, session and step;
- how inputs arrive;
- how state lives across sessions;
- how memory is held;
- how failures surface.

It does not judge picture quality or long-session drift, which belong to the model. It does not judge
transport, delivered fps or frame pacing either, which belong to the runtime. Phase 5's replay has
already shown that the port computes what the source computes. This phase shows that it does so in
a form that holds for the whole life of a server.

The bar is four principles. Each one asks one question; its checks are the ways that question has
been answered so far, and its examples are what went wrong when it was not.

| Principle | The question it asks |
|---|---|
| **A** · Pay every one-time cost before ready | Does any user ever wait for a compile, a warm-up or a new shape? |
| **B** · Do only the work needed, when it is needed | Is anything computed, copied or stored that nobody reads, or computed before it is used? |
| **C** · Every session starts clean and memory stays flat | After many sessions, is the server in the state, and at the memory, it was after the first? |
| **D** · Failures are visible and land at the right level | When something breaks, does the right party find out, and does the process stop when it can no longer serve? |

## 1. How the verdict is written

**One table, one row per check, then one row per check made beyond the list.**

| Column | Content |
|---|---|
| Check | its label and short name (`A1 No one-time cost lands in a session`); an extra check is named by what it checked and the principle it belongs to |
| Verdict | `pass` · `fail` · `does not apply` (with why) · `not checked` (with why) |
| Evidence | what decided the verdict: the log line, the timing with its step index, the `file:line`, the peak |
| How | what was driven or read to get it: the session, the command burst, the grep, the timer |

There are exactly four verdicts. A problem read in the code but not yet seen on the served model is
either confirmed by driving the model, or reported as `not checked` together with what the code
shows.

**Each principle is the boundary; its checks are not.** Apply the principle's question to whatever
the model has: an audio track, a variable number of references, a multi-stage pipeline. Record
anything you checked beyond the list as an extra row under the principle it tests.

**A failing row is a porting bug.** It is fixed before the human judges the output, or it is put to
the human with its evidence and decided by them. The checklist returns verdicts; it does not decide.

### The one measurement most checks read

The timing checks read one measurement, taken on the served model.

**The session.** Run one full session from a fresh start: from its first step to its end (its hard
limit, or several minutes for an endless model). On the same server, add the first step of a second
session. Keep the server log from start to finish.

**Per-step time.** Record every step's generation time from `StepOutcome.elapsed` (the wall-clock
seconds `generate()` took), which `process_output()` receives; in a stepped session each step's
result carries it as `elapsed_seconds`. The slowest step and any stall are read from these. The fps a viewer actually received, and the
longest gap between two frames, come from the runtime's own readings ([`serving.md`](serving.md) §3),
which also say where the time went when delivery falls behind the step times. Add timers inside `generate()` only where
one step needs splitting further, into denoise, decode and encode.

**Load the same session.** Drive the condition changes, bursts and odd-shaped inputs that other
checks need inside this session or a second one on the same server. One warmed process then answers
all of them.

Report beside the table:

- every step's time, for both sessions;
- the slowest step, with its index;
- the measured startup time.

## 2. A · Pay every one-time cost before ready

The server reports ready only after everything expensive that happens once has already happened.
No user is ever the first to trigger a compile, a capture or a new shape.

**A1. No one-time cost lands in a session.** Compile, autotune, graph capture, allocator growth,
engine builds, collective setup, the first pass through a branch, a new shape partway through: all of
it is paid inside `load()`, before ready.
**Pass:**
- no step anywhere in the session lies outside the session's own band (the steady band, or the
  fitted trend for a model whose time grows), and this holds for the first step of a later session
  as well as the first step after a fresh start;
- no compile or autotune line appears in the log after ready.

**A2. Warm-up runs the real path end to end**, through the real dispatch, over every shape a session
can take; what it must cover is in [`port-on-reactor-app.md`](port-on-reactor-app.md) §6.
**Check:** on a freshly started server, send the first step with each optional setup input present
and absent, and with inputs at a client's size; no compile line appears and no step leaves the band.

**A3. Inputs that would change a tensor's shape are fixed to one shape on arrival.** A prompt is
padded to one token length; an image is resized and cropped to the configured resolution; a drag or a multi-point touch is brought to a
fixed point count; any other variable input is padded or bucketed. A shape that must vary is a short, known list, and each entry
is warmed in `load()`.
**Pass:** list every user input that reaches a tensor, with the shape it is brought to. Then, on the
warmed server, send inputs that differ on every one of those axes (a short and a long prompt, a
portrait and a landscape image, a one-point and a many-point drag). No compile line may appear, and no step may leave the band.

**A4. Startup time is stated.** Measure from launch to `/health` reporting `available` (load plus
warm-up) and report it. Neither `reactor.yaml` nor `reactor run` sets a startup timeout; when the
person names the timeout of the place the model will be deployed, hold the time against it.
Otherwise the fit is `not checked`, with that reason, and the time is still reported.

**Examples.**
- A warm-up frame already at the target size made the resize a no-op; the first real image was a
  different size, and its step recompiled for 44.6 s against a 1.9 s steady state.
- Warming the image-conditioned shapes moved startup from about one minute to about five. That cost
  belongs before ready, and it is reported so the host's timeout can be checked against it.

## 3. B · Do only the work needed, when it is needed

Every computation and every copy on the serving path has a reader. Work is done once, at the moment
its result is used.

**B1. Setters do no compute**; the rule is in [`port-on-reactor-app.md`](port-on-reactor-app.md) §5.
Refusals still happen at once, in the setter.
**Check:** send a burst of three commands within one step interval and count the encodes: one, for
the newest value.

**B2. Free text is stripped once, in the handler.** That stripped value is the only one stored,
encoded and echoed. A prompt that strips to empty is a value the client chose, encoded as the
source encodes the empty string ([`SKILL.md`](../SKILL.md), "An empty prompt is a value").
**Check:** send a prompt of only spaces, then a padded prompt. The first is stored and echoed as
empty and generation runs on it; the second is stored and echoed stripped. Starting before any
prompt was sent is refused.

**B3. Every host-side array operation earns its place.** List each array operation between the
generator and the transport (`np.stack`, `np.concatenate`, `.astype`, `np.ascontiguousarray`,
`.copy()`, `.numpy()` after a permute). For each one, ask in order:
1. Can it be dropped?
2. Can it run on the GPU?
3. If it must stay on the CPU, why, how many bytes it touches per step, and does it allocate?

**Check:** time each one over a full session and read the maximum, not the mean.

**B4. The serving path computes and keeps nothing that nobody reads.** Compare what each library call
on the per-step path computes and allocates against what the port actually reads from it. Turn off
any unread output where the library has a switch. Otherwise, report its per-step cost.

**Examples.**
- When each setter encoded, a three-command burst pushed the step that followed from 1.013 s to
  1.238 s, because each encode shared the GPU with generation; two of the three encodes were thrown
  away.
- Stacking a clip's frames into one new 1.1 GB host array usually took under a second, but took
  16–60 s on a host whose free memory was fragmented, because the kernel compacted memory for huge
  pages on first touch. The frames had been one array on the GPU; laying them out there and copying
  once removes the stack.
- Asked only for frames, one generation library also built a full-precision copy of the whole clip in
  pinned memory, 4.27 GB for 345 frames, which nothing read.

## 4. C · Every session starts clean and memory stays flat

The hundredth session on a server behaves like the first: the same output for the same input, at the
same memory. Nothing from one session survives into the next, and the load holds only what serving
needs.

**C1. A reset or a new session leaves nothing behind.** No image, cache, counter or conditioning
carries over from the previous session. The first step of a second session matches the first step
of a fresh process. A stale setup input does not raise an error; it silently conditions the next
session, so compare outputs, not just logs.

**C2. A restart reproduces.** The same inputs and seed give the same output after the server
restarts. Autotune and algorithm choices are pinned, and seeds are owned by the session.

**C3. The load wastes no host memory in large blocks.** That means:
- no module built in a higher precision than its weights;
- no build on the CPU followed by a move to the GPU;
- no whole copy of the weights per rank;
- no host copy kept after the move.

**C4. Peak memory is recorded, and fits a stated limit.** Record the host and GPU peak over load plus
a full session. Neither `reactor.yaml` nor `reactor run` sets a host memory limit; when the person
names the limit of the place the model will be deployed, serve under that same limit
(`docker run --memory`, on the port's image) and record the headroom. Otherwise the fit is
`not checked`, with that reason, and the peak is still reported. Do this after C3.

**C5. Memory converges as sessions accumulate, on the GPU and the host alike.** Run twenty or more
sessions back to back. In each, a client connects, receives output and disconnects. At every session
end, sample:
- GPU memory per device;
- host RSS of every process;
- `/dev/shm` used bytes inside the container (`df -B1 /dev/shm`).

Each series must settle to a constant level. A rising floor or a steady climb fails. When one grows,
force `gc.collect()` and sample again. Memory that comes back was held by a reference cycle; memory
that stays is held by a live reference, and a referrer dump names the holder.

**C6. Pinned host memory is bounded and recorded.** Record each process's page-locked total after
warm-up and after a full session, and name what holds it. **Pass:** the total is flat across
sessions, and inside C4's limit when one is stated.

**Examples.**
- One multi-GPU load peaked at 368–387 GiB of host memory when 20 GiB was enough.
- A model that fit on an unrestricted host was killed for running out of memory 13 times in a row
  once a host limit was enforced.
- A sender that kept a one-frame view of the last clip it played pinned that whole clip in
  `/dev/shm`, about 0.4 GiB per session, until the cap filled and every request failed.
- A caching pinned allocator keeps one buffer per size class. After a few clips of different
  lengths, one process held 8 + 4 + 2 GiB that the kernel could neither reclaim nor move.

## 5. D · Failures are visible and land at the right level

A bad request is the client's problem and is told to the client. A broken model is the operator's
problem and stops the process. Nothing logs and carries on while the problem remains.

**D1. Each failure lands at the right level, and none is hidden.**
- **A worker or rank that breaks ends the service.** The process exits with an error so the
  platform restarts it. The other ranks never keep serving, and never hang on a collective. A
  failed load or warm-up never reports ready.
- **A command that fails does not take effect.** It raises `CommandError(code, message)`, the
  issuing client receives the code, the session keeps the previous value, and the next valid command
  works.
- **A catch that logs and carries on while the problem remains is a fail.**

**Check:**
- send a command that must fail, then a valid one;
- with several workers, kill one and confirm the process exits.

**D2. The exit status says what happened.** A step loop that crashes brings the process down with
status 1. A model that refuses to load exits 0, and so does a stop that was asked for. Read the exit
status and the last log lines, never only the container state. A model that raises inside
`generate()` must surface as a crash. It must not keep the process alive with nothing generating.

## 6. How the checklist grows

A new incident or finding does not become a new item. First decide which principle it belongs to:
which of the four questions above it answers.

- **A principle covers it.** Add it there, merged with what is already written: a new check when it
  needs its own pass criterion, otherwise one sentence in an existing check, or one example under
  the principle. Prefer widening an existing check to adding a sibling.
- **No principle covers it.** Add a new principle, stated as a principle with the one question it
  asks, and put the finding under it as its first check or example.

An example describes what happened, why, and the numbers that show it, in a few lines, without
naming the model it happened on.
