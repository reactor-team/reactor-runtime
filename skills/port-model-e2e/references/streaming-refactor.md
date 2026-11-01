# Turning an offline generator into one chunk per step

Phase 3 reference. Use it when the research code can produce a video but plans
it in advance. In that case it does one of three things:

- allocates noise for the whole video,
- pre-computes every condition for the whole trajectory,
- staggers its stages across "ticks" in a pipelined scheduler.

The step loop needs the opposite. Each step generates one chunk, and it knows
only the input that is current when that step starts. The user may change that
input before the next chunk, and may stop at any time.

**Scope.** This applies when one step of the port is a chunk of a stream:
text-driven streams, action-driven worlds, video-to-video. A model whose step is
one clip per request usually keeps its offline generator, called once per clip,
and skips this file.

## The target shape

```
for each step (driven live, one input at a time):
    cond   = build_conditions(current_input, running_state)   # this chunk only
    noise  = draw_noise(this chunk's latent frames)            # this chunk only
    z      = denoiser.denoise_chunk(noise, cond)               # bounded KV cache
    z      = refiner.refine_chunk(z)                           # if the model has one
    pixels = decoder.decode_chunk(z)                           # causal, per-chunk cache
    emit(pixels)
    running_state = advance(running_state, current_input)
```

What a chunk may carry across to the next one:

- **Allowed: bounded state.** A KV cache with a sink and a window, a running
  pose, the global anchor frame, a decoder's tail cache. Leave a bounded KV cache
  inside the stage that owns it; splitting it out adds complexity and buys
  nothing for interaction.
- **Must go: unbounded state.** Anything sized to the whole session: a
  full-length noise volume, `*_full` buffers of clean or refined frames, a
  pre-baked trajectory of conditions.

### One input governs one chunk

A chunk holds K latent frames, and its condition format could often express K
different inputs. Live, only one input exists when the chunk starts, and most
models cannot take a new input in the middle of a chunk. So broadcast the
current input across the chunk. That makes the chunk the unit of interactive
control.

## A pipelined scheduler buys no latency

Some offline streamers stagger their stages. At tick t the denoiser works on
chunk t while the decoder works on chunk t−1.

- **On one GPU.** The stages run one after another anyway, so the scheduler
  overlaps nothing. Every input still passes through every stage before its
  first frame appears; the extra machinery only changes how often frames are
  emitted.
- **Across GPUs.** Pipelining raises throughput, but each input still waits
  behind every chunk already in flight. Each pipeline stage adds roughly one
  chunk of delay between an input and the first frame that shows it.

The lever for latency is parallelism *inside* a stage. Collapse the scheduler
into a plain sequential loop.

The step loop forces part of this anyway. Suppose the old code encoded the
next chunk's inputs on one CUDA stream while decoding the current chunk on
another. On the step loop, the next chunk's inputs only arrive with the next
step, so those two now run one after the other. Measure what that costs and
write it in the port notes.

## The refactor, in order

Do these steps **in order, one commit each**. Run the source's own example
after every step and compare it with the previous step's output. The bar is
bit-identical: `max|diff| = 0.0` on the same input. A step that is not
bit-identical changed what runs; find out why before starting the next step.
The baseline Phase 2 recorded is the reference that all of these steps are
held to.

1. **Map the pipeline.** Name every stage and the tensor shape where each pair
   of stages connects. Label every tensor bounded or full-sequence. Find the
   scheduler, where noise is drawn, where conditions are built, and each
   stage's chunk grid; stages often use different grids.
2. **Collapse the scheduler** into a sequential per-chunk loop. Delete the tick,
   lag and "next index" bookkeeping, and keep the exact operations in the same
   order. On one GPU the output is unchanged.
3. **Replace the full-sequence buffers** with two things: the global anchor
   frame, and a bounded rolling buffer of frames waiting for the next stage,
   evicted as they are consumed. The data is the same; it is just never held at
   full length.
4. **Extract the per-chunk denoiser.** Write `denoise_chunk(noise, chunk_idx)`.
   It takes only this chunk's noise and updates the bounded KV cache. Keep the
   original generator as a thin wrapper over it, so the old entry point stays a
   regression check.
5. **Build conditions per chunk.** Carry the running control state (pose,
   orientation, the current prompt) across chunks. Build the conditions for
   *this* chunk only, from the single current input. The input source is a list
   when replaying the baseline and a live queue when serving; the code path is
   the same for both.
6. **Find the length ceiling and apply it** (next section).
7. **Take debug work out of the hot path.** No unconditional dumps, and no
   f-strings built inside the per-chunk loop. Put them behind a flag, so a
   disabled debug path costs nothing.

### When a step changes how random numbers are drawn

Step 4 often moves the noise draw. One full-volume `randn` up front becomes one
draw per chunk. With the same seed, the per-chunk draws hand different numbers
to the same positions, so the output differs from the untouched source.

The model gets no hook for feeding noise in. The draw change is made in the
source and the baseline is recorded from the edited source, as
[baseline-and-replay.md](baseline-and-replay.md) §2.2 lays out; the stage map
of step 1 is what Phase 2 reads to know the change is coming (§2.0 there). If
this refactor arrives at a draw plan different from the one the baseline was
recorded under, re-record the baseline before Phase 4. Every other step stays
held to bit-identity.

## Find the length ceiling — the session is endless by default

An interactive session runs for as long as the client keeps it open. There are
only two reasons to stop after N chunks:

- **The network has a hard limit.**
- **A person asked for a cap.**

A budget the port picks for itself is neither. That means no default of 64
chunks, and no `max_chunks` field on a command.

| What the source has | Hard limit? | What the port does |
|---|---|---|
| a positional table (RoPE or learned) precomputed for `0..max_seq_len-1` and indexed by **absolute** frame position | **yes** | stop at that limit, not before it |
| frequencies computed from coordinates on the fly | no | run endlessly; precision degrades slowly at extreme coordinates |
| a causal decoder holding only a tail cache and a first-chunk flag | no | run endlessly |
| `num_frames` / `max_chunks` validated against a range, with a short clip as the default | no | delete the field |
| noise or conditions allocated for the whole volume up front | no | allocate per chunk (refactor steps 3–5) |
| a KV cache or context that only grows | no, but VRAM may end the session first | measure VRAM at the hard limit; a window or compression is a lossy change and needs a person's decision |

A bounded KV window does not lift a table-based limit. The position index is
still absolute, so the session crashes the first time it passes the table.

**How to find the limit:**

- Grep for `rope`, `rotary`, `max_seq_len`, `positional`, `freqs`,
  `register_buffer`, `arange`, and any hard-coded frame or sequence length.
- Decide, stage by stage, whether its limit is table-based or continuous.
- The **smallest** ceiling across all stages is the limit for the whole model.

**How to apply it:**

- Read the limit from the model at load time, for example from the table's own
  length. Never write a literal.
- Stop generation exactly there.
- Name the mechanism in the code and in the message the client receives: "the
  temporal position table has 1024 latent slots", not "the model's maximum".

Extending or rolling the table changes trained behaviour, so it is a lossy
change and outside a port.

## The details that decide whether it works

- **Chunk 0 is special.** Keep its differences in small `if idx == 0` branches,
  not in a parallel code path. Three causes stack up:
  - the anchor or sink frame is given, not generated;
  - when the total frame count does not divide evenly, the remainder is packed
    into the first chunk;
  - a causal decoder maps the first latent frame to fewer pixel frames.
- **Never re-chunk a trained stage.** Suppose the denoiser segments as "sink,
  then remainder first" and the decoder takes uniform blocks. Do not change
  either grid to make them line up: a stage's sink and cache semantics can
  depend on its grid. Bridge the two grids with the bounded rolling buffer:
  pull producer chunks until the consumer's block exists, then consume it.
- **Anchor normalisation stays global.** If conditions are expressed relative
  to the first frame (`inverse(pose[0])`), compute that anchor once, from the
  session's frame 0. Re-anchoring every chunk makes the trajectory drift.
- **Check the RNG when one-time work moves.** Session-level work such as
  encoding a reference frame may draw from the generator. Moving it into the
  loop, or running it once per chunk, shifts every later noise draw. Run it
  where the source ran it, the same number of times.
- **Stress-test with tens of chunks.** A run of one to three chunks never
  reaches the steady-state grid alignment or the buffer eviction, and those are
  where these bugs live.
- **The objective checks run on the node:**
  - exit code;
  - a control-flow log with chunk and stage counts;
  - the pixel-frame count;
  - peak VRAM that stays flat as the session gets longer.
- **Judging quality by eye is a person's job.** The refactor proves equality,
  and it does not judge how good the output looks.

## Done when

- Generation is one sequential loop with one chunk per step, and holds no
  tensor sized to the session.
- Every refactor step is its own commit, bit-identical to the previous step
  (or to the edited source, for a draw change).
- Conditions are built per chunk from the current input.
- The length limit is read from the model, applied exactly, and named in
  messages; with no table-based limit, the loop is endless.
- A run of tens of chunks completes, with flat VRAM.
