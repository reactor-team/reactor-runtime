# Endurance tests

Long runs that check the runtime lets go of everything a client held once the
client is gone. A unit test cannot see a leak like this: it takes a real
runtime serving real clients, cycle after cycle.

## How a run works

- The runtime runs from source in its own process, started by
  `endurance/launcher.py`. The launcher boots it the way `reactor-runtime` does and adds
  one test-only route, `GET /_endurance/probe`. The published package is not
  changed.
- The model is `model/tracked_echo.py`: `examples/echo` with a fixed burst
  (`ENDURANCE_BURST`, default 30 frames), so every emit is a clip-sized batch.
  It keeps a weak reference to each batch it emits, so the probe can tell
  whether anything still holds one.
- The client is a bare `reactor_webrtc` peer (`endurance/harness.py`). It does the
  runtime's HTTP signaling and control messages itself, over localhost, with
  no client SDK in between: `BareClient` in v0 JSON, `V1Client` in v1
  protobuf. `abrupt_disconnect` runs it in a process of its own, to kill it.
- After every cycle the test waits for the runtime to drop the connection and
  finish tearing it down, then reads the probe and samples the runtime's
  process. Each reading is one sample.

## What is checked

Exact counts, which must be 0 on every cycle:

| Count | What it means |
|---|---|
| `short_cycles` | A cycle whose client got fewer than its 60 frames back: it did not exercise what the scenario claims to. |
| `unreleased_connections` | A connection still in the runtime's registry 30 s after its client left. The detail reports how long the runtime took to drop the others, which is how fast it notices a client is gone. |
| `unanswered_requests` | v1 only: a cycle where the publish, the schema request or a command was refused or never answered, or a notice the client was owed (a stop's `session_ended` or `moderation`) never arrived. |
| `failed_stops` | `session_churn` only: a cycle whose stop was refused, did not drop the client, came to a session that was not orphaned when it should have been, or left the session not ready again. Each names the stop it happened on. |
| `pacers_holding_output` | A media pacer that still reaches model output after it stopped. Checked the moment it stops, so it does not depend on when garbage collection runs. |
| `pinned_batches` | Model output the cyclic garbage collector freed: nothing but reference-cycle garbage held it. |
| `live_connections`, `live_peers`, `live_pacers` | Transport objects still alive after a collection, with no client connected. |
| `live_client_stats` | Connections the runtime still keeps client-stats state for, with no client connected. |
| `live_participants` | Room participant connections still alive after a collection, with no participant in the room. |
| `live_step_requests` | Step requests whose order the runtime still keeps for a connection, or whose failure the model still keeps, with no client connected. |
| `tasks` | asyncio tasks on the runtime's loop; must never rise above the reading taken before the first client. |
| `child_processes` | Processes the runtime started, at any depth, an exited one nobody reaped included; must never rise above the reading taken before the first client. |
| `shm_blocks`, `shm_bytes` | The distributed package's shared-memory blocks in `/dev/shm` (`rr*`); must never rise above the first reading. Judged only for a runtime with worker processes, and Linux only: elsewhere they are not files, and the run says they were not measured. The count covers the whole machine, which a CI runner gives to one scenario; run locally, it assumes nothing else on the machine uses the distributed package meanwhile. |

Trends of the runtime process (RSS, CPU per cycle, threads, file
descriptors, and the RSS of its child processes, with its slope, when it has
any): the last third of the run against the middle third, after a
warm-up of `ENDURANCE_WARMUP_SECONDS` (default 120). RSS is also judged by
its slope over the last two thirds, in MB per hour (`rss_slope`), judged
only when those samples span 75 minutes or more, as a two-hour run's do; a
shorter run reports the rate as ⚪ Not judged. A run too short to judge a
trend ends in ERROR rather than in a verdict; a failed count still fails it.
The runtime runs with `MALLOC_ARENA_MAX=2` unless the environment sets it:
with glibc's default of up to 8 arenas per core, each keeping the most memory
it ever held, RSS climbs to a plateau set by the core count, not by a leak.

## Scenarios

| Scenario | What it does |
|---|---|
| `connection_churn` | One session. Each cycle a client joins, streams a webcam both ways until 60 frames come back, and leaves. The client speaks the v0 JSON control protocol. |
| `connection_churn_v1` | The same churn with a client that speaks v1 protobuf, as current SDKs do: it also requests the schema and, once a second, pings, sends client stats and a `set_effect` or `set_intensity` command. Every request must be answered without an error (`unanswered_requests`). |
| `room_participant_churn` | One session connected to a LiveKit room, the room connection open throughout. Each cycle a participant joins the room, requests the schema and runs a command over the Reactor protocol, calls an RPC method that runs a command, sends a data packet on a topic that runs one, publishes a `webcam` track the session reads by name, sends a microphone byte stream on the topic mapped to `mic`, and leaves. Every request that has an answer must get it, and the topic command must get none; the runtime's own counts must also show that the model's handler ran the topic command, which the runtime did not reject, and that frames arrived on both inputs, since LiveKit accepts a packet, a track, or a stream whether or not the runtime uses it (`unanswered_requests`). Needs a LiveKit server (`LIVEKIT_URL`), and skips without one or against a runtime that serves no LiveKit transport. |
| `session_churn` | Each cycle starts a session under a fresh id, a v1 client streams until its 60 frames come back, and the session is stopped: in turn a plain stop, the client leaving first and the orphaned session stopped after, a stop with a close reason, and a moderated stop. The last two owe the client its `session_ended` or `moderation` notice before the connection closes. The runtime is read once the session is ready again. |
| `multi_connection` | One session shared by several clients. Each cycle a v1 client publishes the webcam, and three v1 viewers that publish nothing, receive-only with no webcam track negotiated, join and each get 60 frames of the same output. They leave in two steps, in turn the viewers first or the publisher first; when the publisher leaves first, a new publisher takes over the webcam while the viewers stay, and must be granted it. |
| `steady_stream` | One v1 client stays connected for the whole run, streaming while it pings, sends client stats and a command once a second. The runtime is read every 15 s as it streams: what the live connection holds (`live_*`, `tasks`) must never rise above the first reading, `pacers_holding_output` and `pinned_batches` stay 0, and every interval gets 60 frames back. Once the client leaves, the reading after teardown must be 0 on every count (`*_after_close`), with the tasks back to the reading before it joined. The trends judge the streaming readings. |
| `distributed_echo` | The v1 connection churn on `model-distributed/`, where the echo's engine runs in a worker process behind a `DistributedRunner`, so every frame crosses to the worker and back through shared memory. |
| `shared_worker_echo` | The v1 connection churn on `model-shared-worker/`, the echo split in two: the runtime serves the session half, and the scenario starts a model worker (`python -m reactor_runtime.model_worker`) beside it, so every frame crosses its socket and shared memory. The session's one connection to the worker lasts the run. |
| `abrupt_disconnect` | The v1 churn, with each client in its own process (`endurance/abrupt.py`), killed with `SIGKILL` a moment after its frames came back instead of closing. The runtime learns the client is gone only from the wire going silent, as it does when a client crashes or loses its network. |
| `stepped_session_churn` | One long-lived stepped session. Each cycle a v1 client requests 10 steps one after another, each setting the effect through `state` and carrying one webcam frame in `media_input`, waits for every result, and closes. Every step must run (`short_cycles`) and every request must get its result (`unanswered_requests`). |

## Running it

```sh
ENDURANCE_DURATION_SECONDS=600 mise run test-endurance
# One scenario:
mise run test-endurance endurance-tests/tests/test_connection_churn.py
# A smoke run, as PR CI does: two cycles, or one per stop for session_churn,
# judged only on what each cycle shows on its own.
ENDURANCE_SMOKE_CYCLES=2 mise run test-endurance
# The room churn, against a local dev LiveKit server.
LIVEKIT_URL=ws://127.0.0.1:7880 mise run test-endurance endurance-tests/tests/test_room_participant_churn.py
```

Each run writes `endurance-results/<scenario>.json`, `-report.md` and
`-summary.txt`, plus the runtime's log. In CI, the **Endurance Tests**
workflow runs every scenario on demand: `duration_minutes` sets the length,
and `runtime_ref` tests another tag or commit's runtime source with the
current suite. It swaps in only `src/reactor_runtime`: the dependencies
(`uv.lock`, so `reactor-webrtc` among them) and the wire bindings stay the
current ref's. A `v3.8.2` run, for example, uses this ref's `reactor-webrtc`,
not the one 3.8.2 shipped with. That measures a leak in the runtime's own
code; a leak in a dependency can pass or fail differently from the release
itself, so read such a result as the runtime's code, not as what shipped.

The checks and the reports (`endurance/trends.py`, `endurance/report.py`)
need no runtime, and their unit tests under `unit/` run with the rest of the
test suite. So does `unit/test_launcher_targets.py`, which checks that what the
launcher hooks still has the shape it expects.

## Adding a scenario

Add `tests/test_<name>.py`. A scenario is a loop of cycles: `run_cycles(...)`
(`endurance/churn.py`) runs its cycle until the deadline, reads the runtime
once the cycle's client is gone, and checks and reports the run. A churn of a
different client is one call to `run_churn(...)` with a `BareClient`
subclass, or with any client that has its `run()` and `unanswered`
(`ChurnClient`). A scenario that does more per cycle passes `run_cycles` its
own cycle, as `session_churn` does. CI picks the file up as its own job.

## When a change needs this suite

The suite keeps its value only if new features join it. Check this list on
every PR that touches the runtime:

- **State held per connection or per session** (objects, tasks, threads,
  buffers, caches, file descriptors, processes, shared memory): say which
  existing count covers it (`live_connections`, `live_peers`, `live_pacers`,
  `tasks`, `child_processes`, `shm_blocks`, the trends), or
  extend the probe in `endurance/launcher.py` with a count that does.
- **A new client-facing path** (commands, uploads, client stats, clips and
  recordings, several clients on one session, a control protocol version):
  add a scenario that drives it, or extend an existing one. Today's scenarios
  cover the v0 and v1 control protocols, commands and client stats, a client
  that dies without closing, sessions started and stopped, with and without a
  close reason or a moderation verdict, several clients on one session, one
  connection streaming for hours, a model run in worker processes, stepped
  sessions and their step requests, and room participants with their RPC and
  data-topic commands and media inputs; uploads, clips and recordings have
  none yet.
- **A change to what the launcher hooks** (`MediaPacer.stop`,
  `http_server.build_app`, `Runner._connections`, the transport classes the
  probe counts): update `endurance/launcher.py` in the same PR.
  `unit/test_launcher_targets.py` and the smoke job fail if you do not.
- **A new check**: show it fails on the commit before the fix it guards and
  passes after it, the way `pacers_holding_output` was validated against the
  pacer fix.
