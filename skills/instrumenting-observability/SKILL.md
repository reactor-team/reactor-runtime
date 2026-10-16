---
name: instrumenting-observability
description: "Add or change a metric or a stage reading in this runtime. Use when a new fact about the process is worth measuring, when an existing instrument needs a label or a bucket, when a change touches the /metrics surface, or when a change adds a stage to a frame's or a command's trip or adds a path that frames or commands take. Covers the registry, the naming and label rules, where the call belongs, stage readings, and what a metrics PR must carry."
---

# Instrumenting the runtime

The runtime keeps one Prometheus registry, renders it on `GET /metrics`, and
waits. Everything below follows from that: the process is passive, a scraper
reads it on its own schedule, and a process nobody scrapes pays only the memory
of its own counters.

## The rule

Measure a fact when it aggregates across time or across sessions. Counts,
durations, and distributions all aggregate. A fact that is only interesting for
one specific session — an id, a message, a stack trace — belongs in the journal
or in a log line, never in a label.

One process hosts one model and serves one session at a time, so a per-process
series aggregates cleanly once the scraper attaches the identity of where the
process runs. That is why the instruments here carry no identity of their own.

Two things stay out:

- **Pushing.** The runtime never opens a connection to report telemetry. There
  is no exporter, no endpoint to configure, and no work at all when nobody
  scrapes.
- **Model compute.** How long a model takes to produce a frame is the model
  author's business, so `/metrics` measures that a frame rate fell, not why.
  A session's stage readings do time the parts of each step (see
  [Stage readings](#stage-readings)): they are per-session facts in the
  journal, read next to the rest of that session's trip, not a series a
  scraper aggregates.

## Where the registry lives

`RuntimeMetrics` in [`src/reactor_runtime/metrics.py`](../../src/reactor_runtime/metrics.py)
owns the `CollectorRegistry` and publishes the identity of the process as a
single `runtime_info` series. `serve._assemble` builds one holder and
passes it to every component that observes on it.

No module holds a registry of its own. A component takes the holder as an
argument and never reaches for a global. This is what keeps one test from
reading another test's numbers, and it is why a test can build a holder, drive a
code path, and read the result back with no cleanup.

Instruments are grouped into a small class per domain — `MetricsRecorder`,
`CommandMetrics`, `ModelMetrics`, `WebRtcMetrics` — that declares its
instruments in `__init__` and exposes one method per fact:

```python
class UploadMetrics:
    """Records what the store did with the uploads a client sent."""

    def __init__(self, metrics: RuntimeMetrics) -> None:
        """Declare the upload instruments on the registry of *metrics*."""
        self._uploads = Counter(
            "runtime_uploads_total",
            "Uploads a client sent, by what the store did with them.",
            ["outcome"],
            registry=metrics.registry,
        )

    def stored(self) -> None:
        """Count an upload the store accepted."""
        self._uploads.labels(outcome="stored").inc()
```

The method names the fact, not the label. A call site reads as the thing that
just happened, and no label value is spelled out anywhere but here. Build the
group once, where its owner is built — a group built per request registers on
the registry twice and raises.

## Naming and units

- Every name starts with `runtime_`, snake case, one name per thing. This is the
  application prefix
  [Prometheus asks for](https://prometheus.io/docs/practices/naming/), and it is
  load-bearing rather than decorative: a metric name is global to the store that
  ingests it, so a bare `sessions_total` merges with whatever else in a deployment
  counts sessions, under one name, with two meanings and two label sets. An
  operator who finds the prefix verbose can rewrite `__name__` at scrape time; no
  operator can separate two producers that already share a name.
- Base units only: seconds, bytes. Never milliseconds, never a unit in a label.
- A counter ends in `_total`. A histogram of a duration ends in `_seconds`.
- One name holds one quantity. If neither `sum()` nor `avg()` across a metric's
  labels means anything, it is two metrics wearing one name: connections that
  opened and connections that closed are counted apart, because their sum is a
  number nobody wants while their difference — the connections teardown reaped —
  is one somebody reads.
- The help string says what the number means to somebody who has never read this
  repository. It is the only documentation a dashboard author gets.
- A gauge holds a level that goes up and down, such as the live connections. A
  count that only rises is a counter, so a `rate()` over it is meaningful.

## Labels

Every label value comes from a set that is small and known before the process
starts: the members of an enum, the commands the model declares in its schema,
the output tracks the model declares. Nothing else.

A client controls what it puts on the wire, so a name off the wire is not
bounded. A command the model does not declare is recorded under
`UNKNOWN_COMMAND`, which holds a client that spells a command wrong in a loop to
one series instead of one series per attempt. Apply the same treatment to any
new label fed from client input.

Never label with an id, an error message, a file name, an SDP, or any other free
text. A series lives for as long as the storage keeps it, so one unbounded label
value is not one bad sample — it is a series that stays forever.

Identity is not a label either. A per-observation `machine_id` or `model_name`
multiplies every series in the process by the size of the fleet and buys nothing
a scraper does not already attach. The static identity rides `runtime_info`
and stops there.

Two shapes to reject in review:

- **A multiplexed gauge.** One `runtime_gauges{name=...}` collector that
  accepts whatever the caller passes is an unbounded label with extra steps. It
  also loses the type: a rate over it means nothing when half its children are
  levels.
- **A metric per session.** A registry that grows with the sessions the process
  served leaks. The number of series a process can hold is fixed at the moment
  it starts.

Declare the children a query reads before the first event of their kind, so a
fresh process answers a rate with zero rather than with nothing:

```python
for reason in EndReason:
    self._sessions.labels(reason=reason.value)
```

Seed a set that is only known at run time as soon as it is known. The command
counter and the frame counter seed the commands and the outbound tracks the model
declares, the moment the bridge loads, so a scrape lists a command nobody sent
and a track nothing came out of with a zero. Without it, an unused command and a
command the model does not have read exactly the same, and so do a silent track
and a track that was never declared.

## Where the call belongs

One fact, one choke point. Find the single place the runtime already decides the
fact, and observe there:

| Fact | Choke point |
| --- | --- |
| Session lifecycle, connections, errors | `MetricsRecorder.observe`, on the state machine's transitions |
| Command outcome and ingress | `Runner._submit_command` |
| Handshake timings | `WebRTCAcceptor._negotiate` and `_opened` |
| Model load, emitted media | `Runner.start` and `Runner._emit_media` |

The session instruments are the pattern to copy. The state machine already
carries every session fact, so one listener subscribed to it is the whole
session surface and the runner holds no inline metric call at all.

If a new fact needs the same instrument called from two places, the call is at
the wrong level. Move up to where the fact is decided once.

## Durations

- Read a monotonic clock. A wall clock moves under an adjustment and turns a
  duration negative.
- Take the clock through an injected `clock` argument when the instrument reads
  both ends of the interval, so a test drives time instead of sleeping. When the
  start is stamped somewhere else and handed in, leave the clock alone: a fake
  end against a real start measures nothing, and the test controls the interval
  by choosing the start.
- Stamp the start at the moment the wait truly began, not where it is convenient
  to write. The handshake runs from the offer, not from the answer, because the
  offer is when the client started waiting. A command runs from the moment its
  frame arrived, so the measurement includes the time it spent queued.
- Leave out a wait that belongs to somebody else. Command ingress skips the wait
  for the bytes of an upload, because that is the client's own speed, and a
  histogram that counts it reports the client where a reader looks for the
  runtime.
- An interval between two repeats of the same event starts over when the process
  had nothing to do in between. The gap between emissions resets at a session,
  because a model waiting for its next client is not a model that stalled, and the
  one sample that spanned the pause would be the worst the histogram ever took.
- Choose buckets from the shape the number really has, and write a comment that
  says what shape that is. The buckets are the resolution of every quantile
  anybody will ever read.
- An operation that never finishes is not a slow observation. Record nothing.
  The gap between the count that started and the count that finished is the
  reading, and folding a timeout into the histogram moves its tail to a constant.

## Threads

The `prometheus_client` collectors take a lock per operation and are safe to
call from any thread. Media frames are counted on the model thread at the frame
rate of the model, which is cheap next to producing the frame. Nothing here
needs a queue or a hand-off to the event loop.

## Stage readings

A stage reading says where a session's frames and commands spent their time,
stage by stage, over the window since the previous reading. The runtime
journals its readings as the `runtime_stats` fact, and a client sends its own
in `client_stats`. A reading is a per-session fact, so it lives in the journal
and never in `/metrics`. The wire comments on `FrameStage` and `MessageStage`
in `proto/reactor_wire/v1/platform.proto` are the reference for every stage
name and what it covers.

### Who times which stage

Each stage is timed by the side that sees it, and that side alone reports it:

- **The transport** times what happens on the wire and around it: for WebRTC,
  libwebrtc's own stages, the chunked data channel's, `delivery` and its parts,
  `emit`, `output_pacing` and `output_queue` and its parts. It holds them in
  its own window and hands them over through `FrameStageSource` and
  `MessageStageSource`.
- **The model loop** times what happens inside the model's process: a frame's
  `input_wait`, each step's `process_input`, `generate` and `process_output`,
  and the points a command passes on its way to and through its handler. It
  reports them up through the frame timing sink, the step report and the
  command timing sink, and names no command stage itself.
- **The runner** owns what the model loop reports. `RunnerStages` keeps it per
  connection, cleared when a session starts, names the command stages from
  their times, and adds them to each connection's reading beside the
  transport's when the runner builds a report.

A transport never holds a stage the runtime times, and the runner never holds
one a transport times. That keeps the runtime's own stages on every
connection's reading, whatever transport carries it.

### Adding a stage

1. Time it where the side that owns it already handles the frame or the
   command, with `time.perf_counter()` or `time.monotonic()`, as the
   neighbouring stages do.
2. Add it to that side's window: a transport's `STAGES` tuple, or
   `FRAME_STAGES` or `MESSAGE_STAGES` in `runner/stages.py`, in the order a
   frame or a command goes through it.
3. A stage inside another one is a part: list it in that side's parts table,
   so its reading carries `part_of`. A reader nests or skips a part by
   `part_of` alone, so a new part needs no change in any reader.
4. Describe it in the wire comment on `FrameStage` or `MessageStage`, with the
   side that reports it, and in the docstring of the window that holds it.
5. Add a test that drives the code path and reads the stage back off the
   reading.

A path that frames or commands take keeps its timings intact only if it
carries what the next stage needs: a command keeps the `received_at` the
runner stamped when it arrived, a frame keeps the connection that sent it,
and a step reports its stages on its step report. A new path that drops one
of them drops the stages after it, silently.

### A new transport

A transport that times its own stages implements `FrameStageSource`,
`MessageStageSource`, or both, next to `TransportStatsSource`. Keep the totals
in a `StageTotals` from `core/stages.py`, which a report takes from the event
loop while the transport adds from its own threads. A transport that times
nothing implements neither and still gets the runtime's stages on its
reading.

### The wire

A change under `proto/` ships in a wire release, which merging it to main
cuts. The runtime vendors the pinned release, so the next runtime release
needs `[tool.reactor-wire] version` in `pyproject.toml` moved to that
release; `wire-check-release` stops the release until then. `wire-check`
reports the same drift on a pull request without failing it.

Until the pin moves, the runtime decodes with bindings older than `proto/`,
so code that reads a new field off a decoded message reads it with a
default, as `getattr(stat, "message_stages", ())` does, rather than failing
on bindings that lack it.

## Checklist for a metrics change

1. The fact aggregates, and it is not model compute.
2. It is observed at one choke point, in an instrument group built by that
   component's owner.
3. Every label value is bounded; children a query needs are pre-declared.
4. Durations use the injected monotonic clock, with buckets that carry a comment.
5. Tests land in the same pull request. Drive the real code path and read the
   result back off the registry:

   ```python
   assert metrics.registry.get_sample_value(
       "runtime_commands_total", {"command": "set_prompt", "outcome": "accepted"}
   ) == 1.0
   ```

   A child that was never touched reads `None`, not `0.0`. Assert `is None` when
   the point of the test is that nothing was recorded.

6. A change to the HTTP surface regenerates the committed contract with
   `mise run http-spec`. The commit hook runs it for you and stages the result,
   and `http-spec-check` fails the build on drift.
7. `mise run lint`, `mise run typecheck`, and `mise run test` pass.

## Prose

This repository is public. Write every help string, docstring, and comment for
an outside reader who has no other context. Keep to active voice, one topic per
sentence, and one name per thing — the rules in
[AGENTS.md](../../AGENTS.md) apply here as they do everywhere else.
