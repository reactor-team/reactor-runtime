# Endurance tests

Long runs that check the runtime lets go of everything a client held once the
client is gone. A unit test cannot see a leak like this: it takes a real
runtime serving real clients, cycle after cycle.

## How a run works

- The runtime runs from source in its own process, started by
  `launcher.py`. The launcher boots it the way `reactor-runtime` does and adds
  one test-only route, `GET /_endurance/probe`. The published package is not
  changed.
- The model is `model/tracked_echo.py`: `examples/echo` with a fixed burst
  (`ENDURANCE_BURST`, default 30 frames), so every emit is a clip-sized batch.
  It keeps a weak reference to each batch it emits, so the probe can tell
  whether anything still holds one.
- The client is a bare `reactor_webrtc` peer (`harness.py`). It does the
  runtime's HTTP signaling and control messages itself, over localhost, with
  no client SDK in between.
- After every cycle the test waits for the runtime to drop the connection and
  finish tearing it down, then reads the probe and samples the runtime's
  process. Each reading is one sample.

## What is checked

Exact counts, which must be 0 on every cycle:

| Count | What it means |
|---|---|
| `pacers_holding_output` | A media pacer that still reaches model output after it stopped. Checked the moment it stops, so it does not depend on when garbage collection runs. |
| `pinned_batches` | Model output the cyclic garbage collector freed: nothing but reference-cycle garbage held it. |
| `live_connections`, `live_peers`, `live_pacers` | Transport objects still alive after a collection, with no client connected. |
| `tasks` | asyncio tasks on the runtime's loop; must never rise above the first reading. |

Trends of the runtime process (RSS, CPU per cycle, threads, file
descriptors): the last third of the run against the middle third, after a
warm-up of `ENDURANCE_WARMUP_SECONDS` (default 120). A run too short to judge
a trend ends in ERROR rather than in a verdict; a failed count still fails it.

## Scenarios

| Scenario | What it does |
|---|---|
| `connection_churn` | One session. Each cycle a client joins, streams a webcam both ways until 60 frames come back, and leaves. |

## Running it

```sh
mise run //proto:gen && uv sync --locked
ENDURANCE_DURATION_SECONDS=600 uv run pytest endurance-tests/tests/test_connection_churn.py -s
```

Each run writes `endurance-results/<scenario>.json`, `-report.md` and
`-summary.txt`, plus the runtime's log.

The checks and the reports (`trends.py`, `report.py`) need no runtime, and
their unit tests under `unit/` run with the rest of the test suite.

## Adding a scenario

Add `tests/test_<name>.py`. Drive the runtime through the `runtime` fixture
and `BareClient`, take one `ResourceSampler.sample()` per cycle from
`runtime.settled_probe()`, and end in a `finally` block with
`finish_and_check(...)` over `exact_metrics(...) + trend_metrics(...)`.
