# Errors: where to read, how to fix

Every failure below was seen on the platform. Times were measured on one endpoint with B200 GPUs; the
multi-GPU rows ran on RTX 6000 Pro. The load-failure, per-chunk, hang and resource rows come from faults planted
on purpose in a working port (a wrong weight file name, a wrong tensor shape, an allocation past the limit, a
sleep), so the message and the delay are what a real bug of that kind produces.

The phases, reasons and messages below were seen on the current CLI for GPU and host memory, a hung chunk, a
hung warm-up and a restart, and on the previous CLI for the rest; they come from the platform and the runtime. The
commands below are the project + environment ones. `<env>` is the environment
(`production`, `staging`); the project comes from the nearest `reactor.yaml`, or `-p <org>/<name>`.

## Where each kind of message appears

| Source | Command | What it holds | Arrives |
|---|---|---|---|
| Build output | printed by `reactor deploy <env>` (or `reactor build --remote`) | the image build: dependency resolution, `build.run` steps | the build took ~9-10 min; its log arrives in one block at the end, so silence during the build is normal |
| Deploy result | the same command, after the build | the plan against the account's allocation and regions; the rollout | seconds for a refusal; the rollout after that |
| Environment status | `reactor status <env>` | the revision's state: `deploying`, `ready`, `degraded`, `failed`, `stopped`, with instances by phase per region | live; `--wait-for ready --timeout 30m` exits non-zero if the revision fails |
| Instance table | `reactor environment instances <env>` | per instance slot: PHASE, time in it, RESTARTS, REASON and a one-line MESSAGE | a load failure showed ~2-3 min after the image was ready |
| Serve log | `reactor logs <env> --since 30m --raw` (`--follow` to tail) | the runtime's JSON records, the full traceback in `exc_info`, your timing, resource and watchdog lines (`port-additions.md` §5) | with the instance table |
| Session log | `reactor logs --session <session-id>` | one session's records | during the session |
| Revisions | `reactor environment history <env>`, `reactor environment show <env> [-o json]` | what each revision deployed: image and weights digests, GPU, variables | seconds |
| Client messages | the client's `message` / `error` events (`messages.jsonl` from `session_check.py`) | command refusals, the model's own messages | live |

The serve log is JSON lines mixed with plain-text lines. Pull the errors and their tracebacks with (field names
as seen on the current CLI; check one raw line first):

```sh
reactor logs <env> --since 30m --raw \
  | jq -Rr 'fromjson? | select(.level == "error")
            | "\(.ts) \(.attributes.instance_id) \(.msg)\n\(.exc_info // "")"'
```

Add `--instance <id>` (from `reactor environment instances`) to read one instance.

## Watching instances

`reactor deploy` and `reactor status --wait-for` wait on the revision's state. Watch the instances as well: a
watcher must stop on `load_failed`, `crash_loop` and phase `failed`, not only on `ready`, or it waits forever on a
load failure. Two states never end on their own and need a limit of your own: `loading` past the expected warm-up,
and `pending` with `waiting_for_capacity`.

```sh
ENV=production; start=$(date +%s)
while sleep 30; do
  out="$(reactor environment instances "$ENV")"; date -u +%T; printf '%s\n' "$out"
  # image_pull_failed cleared on its own within ~2 min of every new image: allow 5 min.
  if (( $(date +%s) - start < 300 )); then rest="$(grep -v image_pull_failed <<<"$out")"; else rest="$out"; fi
  grep -Eq 'load_failed|crash_loop|[[:space:]]failed[[:space:]]' <<<"$rest" && { echo PROBLEM; break; }
  reactor status "$ENV" | grep -Eq '(^|[[:space:]])ready([[:space:]]|$)' && { echo READY; break; }
done
```

The table's exact layout in the new CLI (one row per region with counts per phase, then one row per slot) is
from its help, not from a run; adjust the patterns to what it prints.

## The table

| What failed | What you see | Where the detail is | How to fix and retry |
|---|---|---|---|
| No org slug | `reactor whoami` shows `Account Slug: not set`; a project create is refused | `whoami`'s last line says how to set it | finish the account's Serve setup or ask Reactor support; then `model.name: <org>/<name>` |
| Remote build | `reactor deploy` fails after ~9 min | the build output | fix `requirements.txt` / `build:` / `.dockerignore`; deploy again. Phase 3's import test in the local image catches most of these first |
| Build context too large | the deploy refuses before building | the CLI's message | limits: 4096 entries, 32 MiB per file, 256 MiB total, 64 MiB compressed; move large files to `.dockerignore` |
| Deploy refused | "no '<GPU>' allocation", or a region error | the deploy output | pick `--gpu TYPExN` and `--require-regions` from `check_account.sh`; deploy again |
| Image pull right after a new image | PHASE `failed`, REASON `image_pull_failed`, "The image of a container of the model's pod cannot be pulled" | the instance table | wait. It cleared on its own within about 2 min and the instance became `ready`, on every new image so far. `reactor deploy` may exit with "revision N failed" on it; check the instance table before treating that as a failure. Change nothing |
| Long `syncing_weights` | the phase lasts minutes | the instance table | normal for an `hf://` set: 86 GB took about 10 min. A machine that already holds the set skips most of it. If it never advances, check the set: `reactor environment show <env>` for its digest, `reactor weights show <tag-or-digest>` |
| Stuck in `loading` | PHASE `loading` far past the warm-up time; no REASON. Seen for 25+ min with a warm-up that never returned. There is no startup timeout, and the instance holds its GPU | the watchdog's stack dumps in the serve log, repeated every bound, name the line | fix the hang; meanwhile scale to 0 or roll back. Without the watchdog the log shows nothing after the last warm-up line |
| Load failure | PHASE `failed`, REASON `load_failed`, MESSAGE = the exception's message, ~2-3 min after the image was ready. RESTARTS climbs | the full traceback is the `exc_info` of the `model failed to load; terminating the session` record in the serve log (jq filter above) | fix the code or config; `reactor deploy <env>`. Meanwhile scale to 0 or roll back (below) |
| Bug in the per-chunk code | the same as a load failure, because warm-up in `load()` runs one chunk. Planted shape error: MESSAGE `Sizes of tensors must match except in dimension 0. Expected size 5 but got size 4 for tensor number 1 in the list.` | `exc_info`; the traceback ended inside the vendored model at a `torch.cat` | the shape asserts at module boundaries name the tensor and its shapes. Without warm-up this bug would surface only in a user's session |
| Session errors or no frames | the client gets an `error` event, or no frames after `start` | `reactor logs --session <session-id>`, and the client's messages | fix the code; deploy again |
| A chunk hangs in a session | nothing. A chunk planted to sleep 180 s: the instance stayed `in_session`, the client kept receiving about 13 frames per second, and no runtime, platform or client message appeared | only the watchdog's stack dump in the serve log (every 60 s) named the line | fix the hang. The per-chunk timing line shows the gap afterwards; the stack dump shows it while it happens |
| Model gone later | 0 instances, no revision or reason recorded (seen once, two days after a good deploy, previous CLI) | `reactor environment history <env>` | `reactor scale <env> --desired 1`; check `reactor status <env>` before handing a link out |

### Lines to ignore

- After every failed start, the serve log has a plain-text uvicorn traceback ending in
  `asyncio.exceptions.CancelledError: Task cancelled, timeout graceful shutdown exceeded`. It is the shutdown
  of the failed process, not the cause. The cause is the `model failed to load` record before it.

### A failing instance keeps its GPU

A load failure restarts in a loop and holds its GPU the whole time. While a fix is being built, stop it or go back
to the last good revision:

```sh
reactor scale <env> --desired 0
reactor environment history <env>                 # find the last good revision
reactor rollback <env> --to <revision> --dry-run  # then without --dry-run
```

Rollback restores only the image and weights; it keeps the GPU, count, regions, variables and secrets.

### Restart

`reactor restart <env>` replaces every instance of the environment with the same settings. There is no restart of
one instance. A hung instance stayed in `terminating` for about 5-6 min before its replacement could take the GPU.

### The cost of one fix loop

About 10 min of remote build, about 1.3 min of promotion ("Promoting the image to project"), then 2-3 min from
the image to the error, so about 13-14 min per attempt. A one-comment change measured 9.8 min of build, 1.3 min
of promotion and 3.5 min to `ready` on a machine that already held the weights. Remote builds share no layer cache: a one-line fix reinstalls every dependency. Read the
whole traceback and fix everything it implies before rebuilding. A bug the warm-up does not reach costs the same
loop plus a session.

## Changing an environment without a build

Each command writes a new revision and, unless `--no-wait`, waits for the rollout. A setting it does not name keeps
its value.

```sh
reactor environment update <env> --var KEY=VALUE           # a variable; replaces the running instances
reactor environment update <env> --unset KEY
reactor environment update <env> --gpu NVIDIA_B200x2       # GPU type and count per instance
reactor environment update <env> --weights <tag|digest|dir|hf://…@sha>
reactor environment update <env> --image <tag|digest>      # an image already pushed (reactor image list)
reactor scale <env> --desired 0                            # instance count; 0 stops every instance
```

Use it to flip a switch the code reads from the environment (a debug flag, a stall bound, `call_timeout`), change
the GPU count of a port that derives `world_size` from the visible GPUs, or scale, without a 9-minute rebuild.
Every command takes `--dry-run`.

## Resource failures (GPU and host memory, /dev/shm, multi-GPU, placement)

Each row was produced by a fault planted in a working 1.3B video model (one B200; two RTX 6000 Pro for the
multi-GPU rows), and records what the instance table, the serve log and the client showed.

| What failed | What you see | Where the detail is | How to fix |
|---|---|---|---|
| GPU memory at load or warm-up | PHASE `failed`, REASON `load_failed`, MESSAGE the CUDA out-of-memory text (`CUDA out of memory. Tried to allocate 16.33 GiB. GPU 0 has a total capacity of 178.35 GiB of which 15.53 GiB is free …`). Clear | `exc_info` of the `model failed to load` record | a larger GPU, a second GPU, lower precision, or offloading. Warm-up at the largest input size makes this fail here, not in a session |
| GPU memory during a session | the client sees only DISCONNECTED; no error event names the cause | the serve log: the `OutOfMemoryError` traceback | find what grows per chunk (the resource line's GPU numbers rise); free it, or bound the cache |
| Host memory over the instance's limit | PHASE `crash_loop`, MESSAGE "The model crashed. Its logs have the cause." But the log just stops: the kernel kills the process, so there is no traceback | nothing in the log names it. The last lines are a load step | **Rule:** a `crash_loop` with no traceback whose last lines are a load step is host memory until shown otherwise. Lower the load's peak host use (the resource line's RSS): load shards one at a time straight to the GPU, `safetensors` with `device="cuda"`, `mmap=True` for `torch.load`. Host memory is managed by Reactor and not settable in `reactor.yaml`; ask Reactor if the model needs more |
| `/dev/shm` full | the failing call raises `[Errno 28] No space left on device` (at load: `load_failed`; in a session: the chunk fails with `OSError` and the instance goes to `crash_loop`) | the traceback in the serve log; the resource line shows `/dev/shm` rising before it (a planted leak: 26.56 of 32.0 GiB used at chunk 25, crash at chunk 31) | `/dev/shm` is 32 GiB on the platform and not configurable. Keep large tensors out of shared memory, and remove what you write there |
| Multi-GPU: a collective hangs | the step ends as `WorkerTimeout: generate: no answer from every rank within 120.0s` after `call_timeout` (120 s set; the default is 30 s) | the serve log | find the rank that skipped or reordered a collective. Set `call_timeout` above the slowest normal step, first step included |
| Multi-GPU: ranks run mismatched collectives | with `TORCH_DISTRIBUTED_DEBUG=DETAIL`: `load_failed`, and `Detected mismatch between collectives on ranks. Rank 0 is running collective: CollectiveFingerPrint(… TensorShape=[1024] …)` with each rank's shape. Without it, likely a hang (NCCL does not compare shapes) | the serve log | set `TORCH_DISTRIBUTED_DEBUG=DETAIL` while bringing a multi-GPU port up (it cost nothing measurable per chunk on a 2-GPU sequence-parallel port); fix the rank whose shape differs |
| Multi-GPU: one rank fails before a collective | the other ranks wait in the collective; the load does not fail until NCCL's own timeout (untested; torch's default is 10 min) | the failing rank's traceback, logged at WARNING or above (ranks other than 0 drop INFO) | make every rank fail together at load (`port-additions.md` §4) |
| Multi-GPU: no node holds N GPUs | PHASE `pending`, REASON `waiting_for_capacity`, MESSAGE "No node has N free <GPU> in <region>." Indefinitely (15+ min observed), while the quota stays held | the instance table | the account's N GPUs may be on different machines. Use fewer GPUs per instance or another type (`reactor environment update <env> --gpu …`) |
