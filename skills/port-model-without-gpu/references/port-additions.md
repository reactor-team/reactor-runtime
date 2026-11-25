# What a port adds when the platform is the only place it runs

The port itself is written as [`port-model-e2e`](../../port-model-e2e/SKILL.md) teaches it, and that skill is
not restated here. This file holds only what changes when no GPU is at hand: where its guidance assumes a local
run, what a port must carry so that the platform's logs are enough to debug it, the multi-GPU facts learned from
remote runs, and the record of what could not be proven.

## 1. Where the porting content lives

| Topic | Read | What differs without a GPU |
|---|---|---|
| Workspace layout, the reference example | [`port-on-reactor-app.md`](../../port-model-e2e/references/port-on-reactor-app.md), "Start from the reference example" | the workspace comes from `reactor init` (SKILL.md Phase 2); fix its defaults per [`reactor-yaml.md`](reactor-yaml.md) |
| The application/model split, hooks, refusal vs failure | [`application-model-isolation`](../../application-model-isolation/SKILL.md) | none |
| Vendor the source whole | `port-on-reactor-app.md` §1 | none |
| Kinds of input, the input table, command names, lifecycle, events | `port-on-reactor-app.md` §2 and [`patterns/README.md`](../../port-model-e2e/patterns/README.md) | no person is asked at the start; take the proposed kinds and write them down (§3 below) |
| Weights: one root, relative paths, checkpoint asserted | `port-on-reactor-app.md` §3; hub loads removed as in [`source-and-weights.md`](../../port-model-e2e/references/source-and-weights.md) §2 (inventory) and §7 (rewrite, offline switches) | §7's `--network none` gate cannot run; `HF_HUB_OFFLINE=1` in `build.runtime_env` makes a missed hub load fail at the first deploy instead |
| One chunk per step | [`streaming-refactor.md`](../../port-model-e2e/references/streaming-refactor.md) | each step goes on the unproven list (§3) |
| Session setup, live controls, where encoding happens | `port-on-reactor-app.md` §5 | none |
| Warm-up covers every shape | `port-on-reactor-app.md` §6 | it is also the only run of the per-chunk code before a user (§2 below) |
| Refusals, descriptions, moderation | `port-on-reactor-app.md` §7 and §8 | none |
| The manifest | `port-on-reactor-app.md` §9 | the project + environment CLI does not apply `model.version` or `model.resources`; the GPU is a deploy flag ([`reactor-yaml.md`](reactor-yaml.md)) |
| Runtime facts a port relies on | `port-on-reactor-app.md` §10 | none |
| Multi-GPU | `port-on-reactor-app.md` §4 | the GPU count comes from `--gpu TYPExN`, not `model.resources.gpu.count`; the rest is in §4 below |
| GPU-free tests of the application half | `port-on-reactor-app.md` §12 | run them in the image `reactor build` makes, next to [`local-checks.md`](local-checks.md) |
| The served-model checklist | [`checklist.md`](../../port-model-e2e/references/checklist.md) | each check runs against the deployed environment through `scripts/session_check.py` and the serve log, or goes on the unproven list |

## 2. Make the platform's answer enough to debug from

A traceback in `reactor logs` is the only debugger, and each fix loop costs about 13-14 minutes (SKILL.md Phase 6).

- **Shape asserts at every module boundary, with the shapes in the message**, so one traceback names both sides:

  ```python
  assert lat.shape[2] == cond.shape[2], f"latent frames {tuple(lat.shape)} vs cond {tuple(cond.shape)}"
  ```

- **Warm-up runs at the largest input size** (`port-on-reactor-app.md` §6). Without a local GPU it is the only
  run of the per-chunk code before a user: a planted shape error in the chunk code failed at deploy only because
  the warm-up ran it, and a GPU memory shortfall at the largest size fails there instead of in a session.
- **A missing weight file raises from `load()` with the full path it looked for**, so the instance's MESSAGE
  column names it.

## 3. Write down the choices no run has checked

Keep a list, in the workspace README or the report, of every choice the port made that no run has checked, each
marked **unproven**, so the person who plays the model knows what to look at:

- how random noise is drawn, and every step of a streaming refactor;
- a value the source never defined (a frame rate, a chunk size picked for streaming);
- a dependency pin relaxed or swapped to resolve against the runtime ([`local-checks.md`](local-checks.md));
- an attention or kernel path chosen for the platform (a prebuilt wheel, an SDPA fallback);
- the kinds proposed for each input, which no person confirmed.

The person who plays it reads this list to know what to look at.

## 4. Multi-GPU, as learned from remote runs

When to add GPUs, and how many, is SKILL.md Phase 0. The runner itself is
`reactor_runtime.distributed.DistributedRunner` (`port-on-reactor-app.md` §4 and its docstring). What the
remote ports added:

```python
# in the application's load()
import torch
world_size = torch.cuda.device_count()    # the GPUs the environment gives this instance
if world_size == 1:
    self.engine = Engine(); self.engine.load(config_path, get_weights_path())
else:
    self.engine = DistributedRunner(Engine, world_size=world_size,
        load_kwargs={"config_path": str(config_path), "weights_root": str(get_weights_path())},
        call_timeout=float(os.environ.get("PORT_CALL_TIMEOUT_S", "120")))
    self.engine.start()
```

- **`world_size` from the visible GPUs.** The runner never reads the GPU count from the platform. Deriving it from
  `torch.cuda.device_count()` lets one image run on 1 or N GPUs, and `reactor environment update <env> --gpu
  TYPExN` changes it with no build. `call_timeout` read from a variable changes the same way.
- **Only paths and scalars in `load_kwargs`**, and only `generate()` and `reset()` cross the runner. Anything else
  the application read from the engine (an example input, a load report) moves into a torch-free helper, or into a
  log line on each rank.
- **Reuse the source's sequence-parallel path** (functions that patch attention or the forward); it decides how
  the work splits, and is far less to prove than a new split. If ranks produce separate parts of the output,
  gather them onto rank 0 inside `generate()`.
- **Every rank sees identical inputs.** Resolve a random seed (`-1`) once in the application and pass the number
  in the input; then rank 0 draws each random tensor and broadcasts it, or every rank seeds the same generator.
  Randomness drawn per rank makes the ranks diverge with no error.
- **Warm-up runs on every rank in lockstep**, since the collectives are inside the chunk.
- **Every rank fails together at load.** If one rank raises while another already waits in a collective, the
  waiting rank hangs until NCCL's own timeout. Agree before the first collective of warm-up:

  ```python
  err = None
  try:
      self._load_weights(weights_root)
  except Exception as e:                    # keep it; raise after every rank knows
      err = e
  flag = torch.tensor([1 if err else 0], device=self.device)
  torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
  if err: raise err
  if flag.item(): raise RuntimeError(f"rank {self.rank}: another rank failed to load; see its traceback")
  ```

  An error on one rank inside a chunk still becomes a `WorkerTimeout` after `call_timeout`.
- **Ranks other than 0 log at WARNING and above only**, so their INFO lines never reach `reactor logs`. Log a
  rank's load report, resource line and watchdog at WARNING when `rank > 0`.
- **`TORCH_DISTRIBUTED_DEBUG=DETAIL`** (`--var`) while bringing the port up: a mismatched collective then fails
  with each rank's shape instead of hanging. On a 2-GPU sequence-parallel port it cost nothing measurable per
  chunk. The failures and their messages are in [`errors.md`](errors.md), Resource failures.

## 5. Observability (required, a stopgap)

The platform shows an instance's phase and the lines your code logs. It does not show how long a chunk took,
whether a chunk is stuck, or what a process holds. Every port logs three things until the platform reports them
itself; when it does, drop the matching part. These are log lines in the model, not the runtime's `/metrics`
([`instrumenting-observability`](../../instrumenting-observability/SKILL.md) covers that). Python `logging` lines
and `faulthandler`'s plain-text stderr both reach `reactor logs`.

| What | Why: what was observed without it |
|---|---|
| **(a) A timing line per chunk and per warm-up chunk**: generate seconds, frames, fps. In a session from `StepOutcome.elapsed` in `process_output()`; in warm-up from `time.perf_counter()` around the call | The runtime measures `elapsed` but does not log it. Without the line there is no way to tell whether the model keeps up with playout, or that chunk time grows as a cache fills (a 14B port: 2.7 s to 4.1 s per chunk) |
| **(b) A stall watchdog**: `faulthandler.dump_traceback_later(bound, repeat=True)` before `generate()` and before each warm-up chunk, cancelled after. The bound for warm-up chunk 0 is longer, because it compiles | A chunk planted to hang 180 s: the instance stayed `in_session`, and the client still received about 13 frames per second. Nothing on the platform or the client noticed. Only the stack dump, every 60 s, named the line. A warm-up that never ends kept the instance in `loading` for 25+ minutes (no startup timeout exists), holding its GPU, with no other log line |
| **(c) A resource line**: GPU allocated / reserved, host RSS, `/dev/shm` used of total. At a session's first chunk, every 10 chunks, and at session end | A planted `/dev/shm` leak showed in the line (`/dev/shm 26.56 of 32.0 GiB used` at chunk 25) before the crash at chunk 31. A host-memory kill leaves no traceback; a rising RSS in this line is the only trace |

The model half (it imports nothing from the runtime):

```python
import faulthandler, logging, os, time
log = logging.getLogger(__name__)
STALL_S = float(os.environ.get("PORT_STALL_S", "60"))                 # above the slowest normal chunk
WARMUP_STALL_S = float(os.environ.get("PORT_WARMUP_STALL_S", "300"))  # chunk 0 compiles

def chunk_line(label, seconds, frames):
    return f"{label}: generate {seconds:.2f} s, {frames} frames, {frames / max(seconds, 1e-9):.1f} fps"

def resource_line(tag):
    gpu = rss = shm = "n/a"
    try:
        import torch
        if torch.cuda.is_available():
            gpu = (f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated / "
                   f"{torch.cuda.memory_reserved() / 2**30:.1f} GiB reserved")
    except Exception:
        pass                                   # a resource line must never break the model
    try:
        with open("/proc/self/status") as f:
            rss = next(f"{int(l.split()[1]) / 2**20:.1f} GiB" for l in f if l.startswith("VmRSS:"))
    except Exception:
        pass
    try:
        st = os.statvfs("/dev/shm")
        total, free = st.f_blocks * st.f_frsize, st.f_bavail * st.f_frsize
        shm = f"{(total - free) / 2**30:.2f} of {total / 2**30:.1f} GiB used"
    except OSError:
        pass
    log.info("resources %s: gpu %s; host rss %s; /dev/shm %s", tag, gpu, rss, shm)

# in Engine.load(), the warm-up loop:
for i, inp in enumerate(warmup_inputs):
    faulthandler.dump_traceback_later(WARMUP_STALL_S if i == 0 else STALL_S, repeat=True)
    try:
        t = time.perf_counter()
        out = self.generate(inp)               # ends with the copy to the CPU, so the GPU is done
    finally:
        faulthandler.cancel_dump_traceback_later()
    log.info("%s", chunk_line(f"warm-up chunk {i}", time.perf_counter() - t, out.frames.shape[0]))
```

The application half places the other two, around the one-line `generate()` and in `process_output()`:

```python
def generate(self, inp):
    self._chunk += 1
    faulthandler.dump_traceback_later(STALL_S, repeat=True)   # stack to stderr if the chunk hangs
    try:
        out = self.engine.generate(inp)
    finally:
        faulthandler.cancel_dump_traceback_later()
    if self._chunk == 1 or self._chunk % 10 == 0:
        resource_line(f"session chunk {self._chunk}")
    return out

async def process_output(self, outcome):
    if outcome.error is not None:
        log.error("chunk %d failed after %.2f s: %r", self._chunk, outcome.elapsed, outcome.error)
        raise outcome.error
    log.info("%s", chunk_line(f"chunk {self._chunk}", outcome.elapsed, outcome.result.frames.shape[0]))
    ...                                        # the chunk_complete message and the frames, as usual
```

Reset the chunk counter in `@session_started`, and log a resource line in `@session_ended`. Log through
`reactor_runtime.get_logger` in the application half.

**On multi-GPU, the watchdog and the resource line go inside `Engine.generate()`**, which runs in every rank's
process. With a `DistributedRunner` the application process only waits on a queue, so a watchdog there dumps a
stack that names no rank and no line. Log both at WARNING on ranks other than 0 (§4).

Set the bounds from the warm-up's timing lines once you have them: the session bound above the slowest chunk, the
warm-up bound above chunk 0, changed with `--var` and no build. A dump is not an error by itself; it repeats every
bound while the call is still running, and the stack shows where.
