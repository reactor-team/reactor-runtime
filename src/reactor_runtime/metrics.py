"""The runtime's metrics registry.

Metrics leave this process only when somebody asks for them. The runtime holds
one Prometheus registry, renders it on ``GET /metrics``, and waits. It opens no
connection of its own to report telemetry, so a deployment scrapes the endpoint
on whatever schedule it likes, and a runtime nobody scrapes pays nothing beyond
the memory of its own counters.

The assembly creates the registry and injects it into the components that observe
on it. No module here holds a registry of its own, so one process renders exactly
the instruments its own components registered, and a test builds a holder, reads
it back, and never sees another test's numbers.

An observation carries only labels with a small, fixed set of values, such as the
name of a command in the model's schema or the name of a member of an enum. The
identity of the process is not one of them: it rides a single ``runtime_info``
series, and the scraper attaches whatever else it knows about where the process
runs.

Every instrument is named ``runtime_`` followed by what it measures. A metric name
is global to the store that ingests it, so the prefix is what keeps a series about
a session in this process distinct from a series about a session in whatever else
a deployment scrapes.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

from reactor_runtime.core import (
    JOURNAL_EVENTS,
    EndReason,
    SessionEvent,
    SessionState,
    Transition,
)

CONTENT_TYPE = CONTENT_TYPE_LATEST
"""The media type of a rendered registry, which is the Prometheus text format."""

# A session runs for seconds when a client fails to arrive and for hours when one
# stays, so the buckets span both and stay coarse in between.
_SESSION_DURATION_BUCKETS = (1.0, 5.0, 15.0, 30.0, 60.0, 300.0, 600.0, 1800.0, 3600.0, 7200.0)
# A client that already holds an offer connects in under a second. The orphan
# timeout ends a client-less session at a minute, so the last boundary sits
# there. A session no client ever joined observes nothing at all.
_FIRST_CLIENT_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0)
# Teardown closes the wires and stops the recorder. It is fast, and the tail is
# the interesting part, so the buckets sit below the grace period.
_TEARDOWN_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
# Ingress is the runtime's own work, and it reaches the model in under a
# millisecond while the loop is free. The upper buckets are a loop that is
# starved, which is the condition the measurement exists to expose.
_COMMAND_INGRESS_BUCKETS = (0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
# Building the answer is local work and takes milliseconds. Reaching a connected
# wire adds the round trips of ICE and DTLS, and a client behind a hostile
# network takes seconds or never arrives.
_HANDSHAKE_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
# A model reads its weights once. Small models load in seconds and large ones
# hold the process for minutes, which is the whole cold start a client waits on.
_MODEL_LOAD_BUCKETS = (1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)
# The boundaries around a frame period cover a model that emits one frame at a
# time: 33ms is 30fps, and 67ms is half of it. The higher ones are a model that
# emits a batch at a time, whose gaps are the play-out duration of a batch, and
# the top of the range is a stall either of them would feel as a freeze.
_EMIT_INTERVAL_BUCKETS = (0.005, 0.01, 0.02, 0.033, 0.067, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)

UNKNOWN_COMMAND = "unknown"
"""The command label for a name the model does not declare.

A client sends whatever name it likes, so the name on the wire is not a bounded
value and cannot be a label. Only the commands in the model's schema are bounded.
Every other name shares this one series, which keeps a client that spells a
command wrong in a loop from minting a series for each attempt.
"""


class RuntimeMetrics:
    """The registry one process observes on, and the identity it publishes.

    Every instrument in the process registers against :attr:`registry`, and
    :meth:`render` serves it. The process states its own version and model once,
    as a ``runtime_info`` series, so no other instrument needs to repeat
    that identity on each observation.
    """

    def __init__(self, *, version: str, model: str) -> None:
        """Create an empty registry and publish the identity of the process on it.

        Args:
            version: The version of the runtime that runs in this process.
            model: The reference of the model this process hosts.
        """
        self.registry = CollectorRegistry()
        Info(
            "runtime",
            "The version of the runtime and the model this process hosts.",
            registry=self.registry,
        ).info({"version": version, "model": model})

    def render(self) -> bytes:
        """Render the registry in the Prometheus text format.

        The registry exists before the model starts to load, so a scrape during a
        slow load answers with the identity of the process and every observation
        made so far.
        """
        return generate_latest(self.registry)


def _reason_label(detail: Mapping[str, Any]) -> str:
    """Return the end reason a move carries, as a bounded label value.

    The runtime authors every reason, and a move that names none is a plain stop —
    the same reading the runner's own dispatch takes. The type guard holds the
    label to the five :class:`EndReason` values whatever a caller puts in the
    detail, because one unbounded label value costs a series forever.
    """
    reason = detail.get("reason", EndReason.STOPPED)
    return reason.value if isinstance(reason, EndReason) else EndReason.STOPPED.value


class MetricsRecorder:
    """Records the session lifecycle on the registry, one transition at a time.

    Subscribes to the session state machine and reads the moves that pass. The
    machine already carries every session fact — a start, each connection, an
    error, a teardown, an eviction — so one listener over it is the whole session
    surface, and no component of the runtime calls an instrument inline.

    The recorder keeps the little state a duration needs: when the session
    started, whether a client has arrived yet, and when teardown began. It reads
    a monotonic clock rather than the wall-clock stamp on the transition, so a
    duration survives a clock adjustment.
    """

    def __init__(
        self,
        metrics: RuntimeMetrics,
        *,
        state: SessionState = SessionState.CREATED,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Declare the session instruments on the registry of *metrics*.

        Args:
            metrics: The holder whose registry the instruments register against.
            state: The state the session is in now, published at once so a scrape
                before the first move still reports where the session sits.
            clock: The monotonic source durations are measured against.
        """
        registry = metrics.registry
        self._clock = clock
        self._sessions = Counter(
            "runtime_sessions_total",
            "Sessions that ended, by the reason they ended.",
            ["reason"],
            registry=registry,
        )
        self._duration = Histogram(
            "runtime_session_duration_seconds",
            "How long a session ran, from its start to the end of its teardown.",
            ["reason"],
            buckets=_SESSION_DURATION_BUCKETS,
            registry=registry,
        )
        self._first_client = Histogram(
            "runtime_session_time_to_first_client_seconds",
            "How long a session waited for its first client to connect.",
            buckets=_FIRST_CLIENT_BUCKETS,
            registry=registry,
        )
        self._teardown = Histogram(
            "runtime_session_teardown_seconds",
            "How long a session took to unwind, from the start of teardown to a ready model.",
            buckets=_TEARDOWN_BUCKETS,
            registry=registry,
        )
        self._opened = Counter(
            "runtime_connections_opened_total",
            "Client connections that opened.",
            registry=registry,
        )
        self._closed = Counter(
            "runtime_connections_closed_total",
            "Client connections a client itself closed.",
            registry=registry,
        )
        self._active = Gauge(
            "runtime_connections_active",
            "Client connections attached to the session right now.",
            registry=registry,
        )
        self._session_state = Gauge(
            "runtime_session_state",
            "The state the session is in, as one series per state holding 1 or 0.",
            ["state"],
            registry=registry,
        )
        self._errors = Counter(
            "runtime_session_errors_total",
            "Errors the session recorded. The message stays in the journal.",
            registry=registry,
        )
        self._started_at: float | None = None
        self._client_seen = False
        self._closing_at: float | None = None
        self._live = 0
        # Declare the series a query reads before the first event of its kind, so
        # a fresh process answers a rate with zero rather than with nothing.
        for reason in EndReason:
            self._sessions.labels(reason=reason.value)
        self._active.set(0)
        self._publish_state(state)

    def observe(self, transition: Transition) -> None:
        """Fold one session move into the instruments.

        Runs on every legal move, including the journal self-loops. A move that
        changes no state records only what its event says, so the state gauge and
        the state-entry durations stay true while a segment or an error rides out
        during teardown.
        """
        now = self._clock()
        event = transition.event
        entered = transition.from_state is not transition.to_state
        self._fold_connections(event)
        if entered:
            self._publish_state(transition.to_state)
        if event is SessionEvent.ERROR:
            self._errors.inc()
        if event is SessionEvent.CONNECTION_OPENED:
            self._note_first_client(now)
        if transition.is_session_start:
            self._started_at = now
            self._client_seen = False
        if entered and transition.to_state is SessionState.CLOSING:
            self._closing_at = now
        if transition.is_session_end:
            if self._closing_at is not None:
                self._teardown.observe(now - self._closing_at)
            self._end_session(_reason_label(transition.detail), now)
        if entered and event is SessionEvent.EVICTION:
            # An eviction is terminal from anywhere and skips teardown, so the
            # session it interrupted ends here rather than on a cleanup move.
            self._end_session(_reason_label(transition.detail), now)

    def _note_first_client(self, now: float) -> None:
        """Measure the wait for the first client of the session, once per session."""
        if self._started_at is None or self._client_seen:
            return
        self._first_client.observe(now - self._started_at)
        self._client_seen = True

    def _end_session(self, reason: str, now: float) -> None:
        """Count a session that ended and measure how long it ran.

        A move that ends no session — an eviction while the model sat idle, or a
        model that failed to load — counts nothing, because no session ran.
        """
        if self._started_at is not None:
            self._duration.labels(reason=reason).observe(now - self._started_at)
            self._sessions.labels(reason=reason).inc()
        self._started_at = None
        self._client_seen = False
        self._closing_at = None

    def _fold_connections(self, event: SessionEvent) -> None:
        """Count each connection fact and hold the gauge at the live count.

        Teardown closes every wire wholesale and reports no per-connection loss,
        so a move that is neither a connection fact nor a self-loop clears the
        count. That is the state machine's own rule for its live connections, and
        following it keeps the gauge from holding the connections a finished
        session left behind.

        The counters follow the same rule, so every connection that opened counts
        on one and only the ones a client closed itself count on the other. Their
        difference is the number teardown reaped, and it grows with the sessions
        the process has served rather than showing a leak.
        """
        if event is SessionEvent.CONNECTION_OPENED:
            self._opened.inc()
            self._live += 1
        elif event is SessionEvent.CONNECTION_CLOSED:
            self._closed.inc()
            self._live = max(0, self._live - 1)
        elif event is SessionEvent.CONNECTION_ANSWERED or event in JOURNAL_EVENTS:
            return
        else:
            self._live = 0
        self._active.set(self._live)

    def _publish_state(self, state: SessionState) -> None:
        """Raise the series of the current state and drop every other one."""
        for member in SessionState:
            self._session_state.labels(state=member.name.lower()).set(
                1.0 if member is state else 0.0
            )


class CommandMetrics:
    """Records what a client asked the model to do, and how long the ask took.

    Every command the runtime admits passes one choke point, which already
    branches on the three outcomes a command can have: the model accepted it, the
    contract rejected it, or it referenced an upload the store could not produce.
    This class names those three outcomes as three methods, so the choke point
    reads as the outcome it just decided and no label value appears at the call
    site.

    Ingress covers what the runtime does with a command before the model sees it:
    the wait for the event loop, the decode, the contract validation, and the
    enqueue. It stops when the command is enqueued, so it measures the runtime and
    not the model — a handler that runs for a minute does not appear here. It also
    excludes the wait for the bytes of an upload the command references, which is
    the client's own latency and would otherwise bury the runtime's.
    """

    def __init__(self, metrics: RuntimeMetrics) -> None:
        """Declare the command instruments on the registry of *metrics*.

        Args:
            metrics: The holder whose registry the instruments register against.
        """
        self._commands = Counter(
            "runtime_commands_total",
            "Commands a client sent, by command and by what the runtime did with it.",
            ["command", "outcome"],
            registry=metrics.registry,
        )
        self._ingress = Histogram(
            "runtime_command_ingress_seconds",
            "How long the runtime took to carry a command from the wire to the model.",
            ["command"],
            buckets=_COMMAND_INGRESS_BUCKETS,
            registry=metrics.registry,
        )

    def declare(self, commands: Iterable[str]) -> None:
        """Seed the series of every command the model declares.

        A command nobody has sent yet has no series at all, which reads the same
        as a command the model does not have. Seeding the declared names answers
        "which of my commands do clients use" off one scrape, with a zero for the
        ones nobody sends.

        Only the two outcomes an ordinary command has are seeded. An unresolved
        upload is a fault, and a row of zeroes for a fault that never happened
        says nothing a missing series does not.
        """
        for command in commands:
            for outcome in ("accepted", "rejected"):
                self._commands.labels(command=command, outcome=outcome)

    def accepted(self, command: str, *, since: float) -> None:
        """Count a command the model took, and measure how long it waited."""
        self._commands.labels(command=command, outcome="accepted").inc()
        self._ingress.labels(command=command).observe(time.monotonic() - since)

    def rejected(self, command: str, *, since: float) -> None:
        """Count a command the contract refused, and measure how long that took.

        A rejection reaches the same choke point as an acceptance and costs the
        same work, so it belongs in the ingress measurement.
        """
        self._commands.labels(command=command, outcome="rejected").inc()
        self._ingress.labels(command=command).observe(time.monotonic() - since)

    def unresolved_upload(self, command: str) -> None:
        """Count a command dropped because an upload it references never arrived.

        This one records no ingress. Ingress measures a command the runtime
        carried to the model, and this one never got there.
        """
        self._commands.labels(command=command, outcome="unresolved_upload").inc()


class ModelMetrics:
    """Records how long the model took to load and what it emits.

    The facts the runtime knows about a model without looking inside it. The load
    is the cold start a client waits through before the process can serve
    anything. The emitted media is the output the model produces, counted in
    frames, so the rate of the counter is the frame rate the model sustains, and
    timed between emissions, so a stall the average frame rate would absorb is
    still visible.

    Nothing here measures the model's compute. A frame rate that falls is
    visible, and why it fell belongs to the model author's own tooling.
    """

    def __init__(
        self,
        metrics: RuntimeMetrics,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Declare the model instruments on the registry of *metrics*."""
        self._clock = clock
        self._load = Histogram(
            "runtime_model_load_seconds",
            "How long the model took to come up, from the import to a running model.",
            ["outcome"],
            buckets=_MODEL_LOAD_BUCKETS,
            registry=metrics.registry,
        )
        self._frames = Counter(
            "runtime_media_frames_total",
            "Frames the model emitted, by output track.",
            ["track"],
            registry=metrics.registry,
        )
        self._interval = Histogram(
            "runtime_media_emit_interval_seconds",
            "Wall-clock time between one emission on an output track and the next.",
            ["track"],
            buckets=_EMIT_INTERVAL_BUCKETS,
            registry=metrics.registry,
        )
        self._last_emit: dict[str, float] = {}

    def declare(self, tracks: Iterable[str]) -> None:
        """Seed the frame count of every output track the model declares.

        A track the model has emitted nothing on reads zero rather than being
        absent, which is what tells a silent track apart from a track this model
        does not have.

        Args:
            tracks: The names of the model's outbound media tracks.
        """
        for track in tracks:
            self._frames.labels(track=track)

    def session_started(self) -> None:
        """Start the emission timing over for a new session.

        The span between the last frame one session emitted and the first frame
        of the next is a model waiting for a client, not a model that stalled, so
        no interval crosses a session boundary.
        """
        self._last_emit.clear()

    def loaded(self, *, since: float) -> None:
        """Measure a model that came up and is ready to serve."""
        self._load.labels(outcome="ok").observe(self._clock() - since)

    def load_failed(self, *, since: float) -> None:
        """Measure a model that failed to come up.

        A failed load is terminal for the process, so this is observed at most
        once and a scrape that catches it reports how long the process spent
        before it gave up.
        """
        self._load.labels(outcome="failed").observe(self._clock() - since)

    def emitted(self, track: str, frames: int) -> None:
        """Count the frames one emission carried on *track* and time the gap to it.

        Counted in frames rather than in emissions because the model batches: one
        emission can carry a whole batch of video frames, and a counter of
        emissions would report a rate lower than the true frame rate by the size
        of the batch.

        The gap to the previous emission is measured as it stands, undivided by
        the batch, so a model that emits a batch at a time has a baseline of the
        play-out duration of one batch and a stall reads as an excursion above it.
        The rate of the counter gives the frame rate the model averages; a rate
        cannot show that half a minute of it arrived in one burst, and the gaps
        can.

        Called on the model thread at the frame rate of the model. Each
        instrument takes a lock per call, which is cheap next to producing the
        frame.
        """
        now = self._clock()
        previous = self._last_emit.get(track)
        if previous is not None:
            self._interval.labels(track=track).observe(now - previous)
        self._last_emit[track] = now
        self._frames.labels(track=track).inc(frames)


class WebRtcMetrics:
    """Records how long a WebRTC handshake took and whether it finished.

    A handshake has two legs that fail for different reasons and are worth
    telling apart. Building an answer is local work: it reads the offer, sets up
    the peer, and takes milliseconds unless the runtime itself is in trouble.
    Reaching a connected wire is the client's network doing ICE and DTLS, which
    takes round trips and, behind a hostile network, never finishes at all.

    Both are measured from the moment the offer arrived, because that is when the
    client starts waiting.
    """

    def __init__(
        self,
        metrics: RuntimeMetrics,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Declare the handshake instruments on the registry of *metrics*."""
        self._clock = clock
        self._negotiation = Histogram(
            "runtime_webrtc_negotiation_seconds",
            "How long the runtime took to answer an offer.",
            ["outcome"],
            buckets=_HANDSHAKE_BUCKETS,
            registry=metrics.registry,
        )
        self._connect = Histogram(
            "runtime_webrtc_connect_seconds",
            "How long a client took to reach a live wire, from its offer to a connected peer.",
            buckets=_HANDSHAKE_BUCKETS,
            registry=metrics.registry,
        )

    def answered(self, *, since: float) -> None:
        """Measure an offer the runtime answered."""
        self._negotiation.labels(outcome="ok").observe(self._clock() - since)

    def negotiation_failed(self, *, since: float) -> None:
        """Measure an offer the runtime could not answer."""
        self._negotiation.labels(outcome="failed").observe(self._clock() - since)

    def connected(self, *, since: float) -> None:
        """Measure a client that reached a live wire.

        An offer that never connects contributes nothing here. It is not a slow
        connection, it is an absent one, and it already shows as a negotiation
        that was answered with no connection to follow it.
        """
        self._connect.observe(self._clock() - since)
