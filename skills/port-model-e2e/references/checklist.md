# Phase 6 — the served-model checklist

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
transport, delivered fps or frame pacing either, which belong to the runtime. Phase 4's replay has
already shown that the port computes what the source computes. This phase shows that it does so in
a form that holds for the whole life of a server.

## 1. How the verdict is written

**One table, one row per item, then one row per check made beyond the list.**

| Column | Content |
|---|---|
| Item | number and short name; an extra check is named by what it checked |
| Verdict | `pass` · `fail` · `does not apply` (with why) · `not checked` (with why) |
| Evidence | what decided the verdict: the log line, the timing with its step index, the `file:line`, the peak |
| How | what was driven or read to get it: the session, the command burst, the grep, the timer |

There are exactly four verdicts. A problem read in the code but not yet seen on the served model is
either confirmed by driving the model, or reported as `not checked` together with what the code
shows.

**Each item is a principle.** Its examples illustrate it; they are not its boundary. Apply the
principle to whatever the model has: an audio track, a variable number of references, a multi-stage
pipeline. Ask of every part one question: *does the port make it do only the work it needs, at the
moment it needs it, in a form that holds for the whole life of the server?* Record anything you
checked beyond the list as an extra row.

**A failing row is a porting bug.** It is fixed before the human judges the output, or it is put to
the human with its evidence and decided by them. The checklist returns verdicts; it does not decide.

### The one measurement most items read

The timing items read one measurement, taken on the served model.

**The session.** Run one full session from a fresh start: from its first step to its end (its hard
limit, or several minutes for an endless model). On the same server, add the first step of a second
session. Keep the server log from start to finish.

**Per-step time.** Record every step's generation time. The runtime already times each step's
`process_input`, `generate` and `process_output`, and each frame's `input_wait`. Each
`runtime_stats` fact on `GET /events` carries those stage readings, one per window (every 5 s by
default), so per-stage time is read off the served path itself. Add synchronized part timers inside
`generate()` only where a stage needs splitting further, into denoise, decode and encode.

**Load the same session.** Drive the condition changes, bursts and odd-shaped inputs that other items
need inside this session or a second one on the same server. One warmed process then answers all of
them.

Report beside the table:

- every step's time, for both sessions;
- the slowest step, with its index;
- the measured startup time.

## 2. The items

### Load and startup

1. **No one-time cost lands in a session.** Compile, autotune, graph capture, allocator growth, engine
   builds, collective setup, the first pass through a branch, a new shape partway through: all of it is
   paid inside `load()`, before ready. Nothing compiles, autotunes or captures after the ready line.
   **Pass:**
   - no step anywhere in the session lies outside the session's own band (the steady band, or the
     fitted trend for a model whose time grows), and this holds for the first step of a later session
     as well as the first step after a fresh start;
   - no compile or autotune line appears in the log after ready.
2. **Warm-up runs the real path end to end**, through the real dispatch, over every shape a session
   can take; what it must cover is in [`port-on-reactor-app.md`](port-on-reactor-app.md) §6.
   **Check:** on a freshly started server, send the first step with each optional setup input present
   and absent, and with inputs at a client's size; no compile line appears and no step leaves the band.
3. **Startup time is stated.** Measure from launch to `/health` reporting `available` (load plus
   warm-up) and report it. Neither `reactor.yaml` nor `reactor run` sets a startup timeout; when the
   person names the timeout of the place the model will be deployed, hold the time against it.
   Otherwise the fit is `not checked`, with that reason, and the time is still reported.

### Inputs

4. **Setters do no compute**; the rule is in [`port-on-reactor-app.md`](port-on-reactor-app.md) §5.
   Refusals still happen at once, in the setter.
   **Check:** send a burst of three commands within one step interval and count the encodes: one, for
   the newest value.
   *Why:* when each setter encoded, a three-command burst pushed the step that followed from 1.013 s to
   1.238 s, because each encode shared the GPU with generation.
5. **Inputs that would change a tensor's shape are fixed to one shape on arrival.** A prompt is padded
   to one token length; an image is resized and cropped to the configured resolution; any other
   variable input is padded or bucketed. A shape that must vary is a short, known list, and each entry
   is warmed in `load()`.
   **Pass:** list every user input that reaches a tensor, with the shape it is brought to. Then, on the
   warmed server, send inputs that differ on every one of those axes (a short and a long prompt, a
   portrait and a landscape image). No compile line may appear, and no step may leave the band.
6. **Free text is stripped once, in the handler.** That stripped value is the only one stored,
   encoded and echoed, and "blank" is decided on it.

### Session lifecycle

7. **A reset or a new session leaves nothing behind.** No image, cache, counter or conditioning
   carries over from the previous session. The first step of a second session matches the first step
   of a fresh process. A stale setup input does not raise an error; it silently conditions the next
   session, so compare outputs, not just logs.
8. **A restart reproduces.** The same inputs and seed give the same output after the server restarts.
   Autotune and algorithm choices are pinned, and seeds are owned by the session.

### Memory

9. **The load wastes no host memory in large blocks.** That means:
   - no module built in a higher precision than its weights;
   - no build on the CPU followed by a move to the GPU;
   - no whole copy of the weights per rank;
   - no host copy kept after the move.

   *Why:* one multi-GPU load peaked at 368–387 GiB of host memory when 20 GiB was enough.
10. **Peak memory is recorded, and fits a stated limit.** Record the host and GPU peak over load plus
    a full session. Neither `reactor.yaml` nor `reactor run` sets a host memory limit; when the person
    names the limit of the place the model will be deployed, serve under that same limit
    (`docker run --memory`, on the port's image) and record the headroom. Otherwise the fit is
    `not checked`, with that reason, and the peak is still reported. Do this after item 9.
    *Why:* a model that fit on an unrestricted host was killed for running out of memory 13 times in a
    row once a host limit was enforced.
11. **Memory converges as sessions accumulate, on the GPU and the host alike.** Run twenty or more
    sessions back to back. In each, a client connects, receives output and disconnects. At every
    session end, sample:
    - GPU memory per device;
    - host RSS of every process;
    - `/dev/shm` used bytes inside the container (`df -B1 /dev/shm`).

    Each series must settle to a constant level. A rising floor or a steady climb fails. When one
    grows, force `gc.collect()` and sample again. Memory that comes back was held by a reference cycle;
    memory that stays is held by a live reference, and a referrer dump names the holder.
    *Why:* a sender that kept a one-frame view of the last clip it played pinned that whole clip in
    `/dev/shm`, about 0.4 GiB per session, until the cap filled and every request failed.
12. **Pinned host memory is bounded and recorded.** Record each process's page-locked total after
    warm-up and after a full session, and name what holds it. **Pass:** the total is flat across
    sessions, and inside item 10's limit when one is stated.
    *Why:* a caching pinned allocator keeps one buffer per size class. After a few clips of different
    lengths, one process held 8 + 4 + 2 GiB that the kernel could neither reclaim nor move.

### Host-side data path

13. **Every host-side array operation earns its place.** List each array operation between the
    generator and the transport (`np.stack`, `np.concatenate`, `.astype`, `np.ascontiguousarray`,
    `.copy()`, `.numpy()` after a permute). For each one, ask in order:
    1. Can it be dropped?
    2. Can it run on the GPU?
    3. If it must stay on the CPU, why, how many bytes it touches per step, and does it allocate?

    Time each one over a full session and read the maximum, not the mean.
    *Why:* stacking a clip's frames into one new 1.1 GB host array usually took under a second, but
    took 16–60 s on a host whose free memory was fragmented, because the kernel compacted memory for
    huge pages on first touch. The frames had been one array on the GPU; laying them out there and
    copying once removes the stack.
14. **The serving path computes and keeps nothing that nobody reads.** Compare what each library call
    on the per-step path computes and allocates against what the port actually reads from it. Turn off
    any unread output where the library has a switch. Otherwise, report its per-step cost.
    *Why:* asked only for frames, one generation library also built a full-precision copy of the whole
    clip in pinned memory, 4.27 GB for 345 frames, which nothing read.

### Failures surface

15. **Each failure lands at the right level, and none is hidden.**
    - **A worker or rank that breaks ends the service.** The process exits with an error so the
      platform restarts it. The other ranks never keep serving, and never hang on a collective. A
      failed load or warm-up never reports ready.
    - **A command that fails does not take effect.** It raises `CommandError(code, message)`, the
      issuing client receives the code, the session keeps the previous value, and the next valid
      command works.
    - **A catch that logs and carries on while the problem remains is a fail.**

    **Check:**
    - send a command that must fail, then a valid one;
    - with several workers, kill one and confirm the process exits.
16. **The exit status says what happened.** A step loop that crashes brings the process down with
    status 1. A model that refuses to load exits 0, and so does a stop that was asked for. Read the exit
    status and the last log lines, never only the container state. A model that raises inside
    `generate()` must surface as a crash. It must not keep the process alive with nothing generating.

### Runtime

17. **The port is on the latest runtime release.** `build.runtime_version` equals the release
    SKILL.md Phase 0 looked up at the time of the port.

