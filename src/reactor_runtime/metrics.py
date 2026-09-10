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
from typing import TYPE_CHECKING, Any

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
    TrackDirection,
    Transition,
)

if TYPE_CHECKING:
    # The transport package imports this module, so the type of the sample it
    # hands in is imported for type checking only: taking it at run time would
    # close an import cycle.
    from reactor_runtime.transport.webrtc.stats import PeerStats, TrackStat

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
# A client on the same continent is tens of milliseconds away and one across an
# ocean is a few hundred. The lower boundaries resolve the good paths, where a
# regression is a doubling nobody would see on a coarser scale, and the top of
# the range is a path bad enough that interaction has already broken down.
_RTT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6)
# Jitter is the spread in arrival times of a stream that is playing normally,
# and on a healthy path it stays inside a frame period. The boundaries climb
# through the range a receiver's buffer absorbs to the one where it cannot, so a
# stream that has started to stutter separates from one that still plays.
_JITTER_BUCKETS = (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5)
# Bytes per second, so the boundaries read as 64kbit, 256kbit, 1Mbit, 3Mbit,
# 5Mbit, 10Mbit, 20Mbit and 40Mbit. The low end is a path that has collapsed to
# voice-call capacity, the middle is where a video model has to start shedding
# quality, and the top is a path that was never the constraint.
_BANDWIDTH_BUCKETS = (8e3, 32e3, 125e3, 375e3, 625e3, 1.25e6, 2.5e6, 5e6)
# A fraction, so the boundaries read as a tenth of a percent through to half of
# the stream. Video survives a percent and starts to show artefacts by a few,
# which is where the resolution sits; above a tenth the picture is breaking up
# whatever the exact figure.
_LOSS_RATIO_BUCKETS = (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)

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
    """Records how a WebRTC wire was established and how well it then carried.

    A handshake has two legs that fail for different reasons and are worth
    telling apart. Building an answer is local work: it reads the offer, sets up
    the peer, and takes milliseconds unless the runtime itself is in trouble.
    Reaching a connected wire is the client's network doing ICE and DTLS, which
    takes round trips and, behind a hostile network, never finishes at all.

    Both are measured from the moment the offer arrived, because that is when the
    client starts waiting.

    Once a wire is live, the peer samples it on a fixed cadence, and those
    samples are how a viewer's complaint about the picture becomes a number.
    They are folded in per connection through :meth:`sampler`, because the peer
    reports its packet counts as running totals and a total only becomes a rate
    once it is differenced against the sample before it.

    What the samples can and cannot show is worth stating plainly, because it
    decides which half of a media problem this process can answer. Outbound, the
    runtime sees what it put on the wire and what it discarded before that, and
    it learns what arrived only when the receiver reports back. Inbound, the loss
    and the jitter are its own measurements, because it is the receiver.

    Two round trips are recorded and they are not the same number. ICE measures
    one with its connectivity checks, which keep succeeding on a path whose media
    queue has grown; the receiver measures the other on the stream itself, which
    is the delay media actually took. A gap between them is a queue building
    somewhere the checks do not travel through.

    The repair traffic is what turns bad early, before loss reaches the picture,
    because a retransmission that arrives in time hides the loss that prompted
    it. Both directions report it: inbound this process is the one asking, and
    outbound the client asks and the counters are what it asked for, so the two
    share an instrument and the track's direction is what tells them apart.

    Requests and repairs are counted separately on purpose. A retransmission
    went out in answer to a request, so the two track each other while a path is
    merely lossy and diverge when requests start going unanswered — which is the
    reading that says repair is no longer keeping up. The keyframe requests are
    the level past that: a client sends one when its decoder can no longer
    continue at all, and answering it costs a whole keyframe.
    """

    def __init__(
        self,
        metrics: RuntimeMetrics,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Declare the handshake and transport instruments on *metrics*."""
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
        self._rtt = Histogram(
            "runtime_webrtc_rtt_seconds",
            "Round trip of the ICE connectivity checks on the nominated candidate pair.",
            buckets=_RTT_BUCKETS,
            registry=metrics.registry,
        )
        self._media_rtt = Histogram(
            "runtime_webrtc_media_rtt_seconds",
            "Round trip of an outbound track, as the receiver measured it, by track.",
            ["track"],
            buckets=_RTT_BUCKETS,
            registry=metrics.registry,
        )
        self._loss_ratio = Histogram(
            "runtime_webrtc_loss_ratio",
            "Fraction of an outbound track the receiver reports as lost, by track.",
            ["track"],
            buckets=_LOSS_RATIO_BUCKETS,
            registry=metrics.registry,
        )
        self._bandwidth = Histogram(
            "runtime_webrtc_bandwidth_estimate_bytes_per_second",
            "What congestion control believes the path to the client will carry.",
            buckets=_BANDWIDTH_BUCKETS,
            registry=metrics.registry,
        )
        self._jitter = Histogram(
            "runtime_webrtc_jitter_seconds",
            "Spread in arrival times of an inbound track, by track.",
            ["track"],
            buckets=_JITTER_BUCKETS,
            registry=metrics.registry,
        )
        self._packets_sent = Counter(
            "runtime_webrtc_packets_sent_total",
            "Packets the runtime put on the wire for an outbound track, by track.",
            ["track"],
            registry=metrics.registry,
        )
        self._packets_received = Counter(
            "runtime_webrtc_packets_received_total",
            "Packets the runtime took off the wire for an inbound track, by track.",
            ["track"],
            registry=metrics.registry,
        )
        self._packets_lost = Counter(
            "runtime_webrtc_packets_lost_total",
            "Packets of a track that never arrived, by track and by which way it flowed.",
            ["track", "direction"],
            registry=metrics.registry,
        )
        self._packets_retransmitted = Counter(
            "runtime_webrtc_packets_retransmitted_total",
            "Packets sent again to repair a loss the receiver reported, by track.",
            ["track"],
            registry=metrics.registry,
        )
        self._bytes_sent = Counter(
            "runtime_webrtc_bytes_sent_total",
            "Payload bytes put on the wire for an outbound track, by track.",
            ["track"],
            registry=metrics.registry,
        )
        self._bytes_received = Counter(
            "runtime_webrtc_bytes_received_total",
            "Payload bytes taken off the wire for an inbound track, by track.",
            ["track"],
            registry=metrics.registry,
        )
        self._frames_sent = Counter(
            "runtime_webrtc_frames_sent_total",
            "Video frames encoded and sent for an outbound track, by track.",
            ["track"],
            registry=metrics.registry,
        )
        self._frames_decoded = Counter(
            "runtime_webrtc_frames_decoded_total",
            "Video frames decoded from an inbound track, by track.",
            ["track"],
            registry=metrics.registry,
        )
        self._frames_dropped = Counter(
            "runtime_webrtc_frames_dropped_total",
            "Video frames the decoder discarded from an inbound track, by track.",
            ["track"],
            registry=metrics.registry,
        )
        self._nacks = Counter(
            "runtime_webrtc_nacks_total",
            "Retransmissions asked for on a track, by track and by which way it flowed.",
            ["track", "direction"],
            registry=metrics.registry,
        )
        self._keyframe_requests = Counter(
            "runtime_webrtc_keyframe_requests_total",
            "Requests to restart a track from a fresh keyframe, by track and direction. "
            "Picture Loss Indications and Full Intra Refresh requests together.",
            ["track", "direction"],
            registry=metrics.registry,
        )
        self._dropped_frames = Counter(
            "runtime_media_dropped_frames_total",
            "Outbound video frames the pacer discarded because its queue was full.",
            registry=metrics.registry,
        )
        self._dropped_bundles = Counter(
            "runtime_media_dropped_bundles_total",
            "Outbound media bundles discarded because the peer's frame queue was full.",
            registry=metrics.registry,
        )
        self._dropped_samples = Counter(
            "runtime_media_dropped_samples_total",
            "Outbound audio samples discarded to cap the send buffer.",
            registry=metrics.registry,
        )
        self._silence_frames = Counter(
            "runtime_media_silence_frames_total",
            "Ten-millisecond audio frames sent as silence because the model produced none.",
            registry=metrics.registry,
        )

    def sampler(
        self, *, outbound: Iterable[str] = (), inbound: Iterable[str] = ()
    ) -> ConnectionStatsRecorder:
        """Return a recorder that folds one connection's samples in.

        Seeds the per-track children for the tracks this connection carries, so
        a track that stayed silent reads zero instead of being absent, which is
        what tells it apart from a track the model never declared.

        Args:
            outbound: Names of the tracks flowing to the client.
            inbound: Names of the tracks flowing from the client.
        """
        for track in outbound:
            self._packets_sent.labels(track=track)
            self._packets_lost.labels(track=track, direction=TrackDirection.OUT.value)
            self._packets_retransmitted.labels(track=track)
            self._bytes_sent.labels(track=track)
            self._frames_sent.labels(track=track)
            self._media_rtt.labels(track=track)
            self._loss_ratio.labels(track=track)
            self._nacks.labels(track=track, direction=TrackDirection.OUT.value)
            self._keyframe_requests.labels(track=track, direction=TrackDirection.OUT.value)
        for track in inbound:
            self._packets_received.labels(track=track)
            self._packets_lost.labels(track=track, direction=TrackDirection.IN.value)
            self._bytes_received.labels(track=track)
            self._frames_decoded.labels(track=track)
            self._frames_dropped.labels(track=track)
            self._nacks.labels(track=track, direction=TrackDirection.IN.value)
            self._keyframe_requests.labels(track=track, direction=TrackDirection.IN.value)
            self._jitter.labels(track=track)
        return ConnectionStatsRecorder(self)

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


class ConnectionStatsRecorder:
    """Folds the stat samples of one connection into the shared instruments.

    Built only by :meth:`WebRtcMetrics.sampler`, one per connection, and reads
    the instruments of the group that built it.

    The peer reports its packet counts as totals for the life of the wire, and a
    counter here has to move by what the last window cost instead. That takes
    the previous sample, which is a fact about one connection rather than about
    the process, so it is held here and released with the connection. The
    instruments stay on the shared registry, so the number of series a process
    holds is fixed however many connections it goes on to serve.

    A total that comes back lower than the one before it is treated as a fresh
    start rather than as a negative increment, which keeps a peer that reset its
    own counters from rejecting the sample outright.
    """

    def __init__(self, metrics: WebRtcMetrics) -> None:
        """Start the recorder with no previous sample to difference against.

        Args:
            metrics: The group whose instruments each sample is folded into.
        """
        self._metrics = metrics
        self._totals: dict[tuple[str, str], int] = {}
        self._silence_frames = 0
        self._dropped_samples = 0
        self._dropped_bundles = 0
        self._dropped_frames = 0

    def observe(self, stats: PeerStats) -> None:
        """Fold one sample of a live wire into the transport instruments.

        Runs on the event loop at the peer's sampling cadence. A field the peer
        could not measure arrives as ``None`` and records nothing, because an
        absent reading is not a reading of zero.
        """
        if stats.rtt_seconds is not None:
            self._metrics._rtt.observe(stats.rtt_seconds)
        if stats.available_outgoing_bitrate_bps is not None:
            # Held in bytes per second, which is the base unit every other size
            # in this registry is reported in.
            self._metrics._bandwidth.observe(stats.available_outgoing_bitrate_bps / 8.0)
        for track in stats.tracks:
            self._fold_track(track)
        self._fold_media(stats)

    def _fold_track(self, track: TrackStat) -> None:
        """Move each of one track's counters by what the last window cost.

        Which readings a track carries follows from its direction, and a field
        the peer left unset is skipped rather than counted as no movement.
        """
        name = track.name
        group = self._metrics
        self._advance(group._packets_sent, "packets_sent", name, track.packets_sent)
        self._advance(group._packets_received, "packets_received", name, track.packets_received)
        self._advance(
            group._packets_retransmitted,
            "packets_retransmitted",
            name,
            track.retransmitted_packets_sent,
        )
        self._advance(group._bytes_sent, "bytes_sent", name, track.bytes_sent)
        self._advance(group._bytes_received, "bytes_received", name, track.bytes_received)
        self._advance(group._frames_sent, "frames_sent", name, track.frames_sent)
        self._advance(group._frames_decoded, "frames_decoded", name, track.frames_decoded)
        self._advance(group._frames_dropped, "frames_dropped", name, track.frames_dropped)
        self._advance(group._nacks, "nacks", name, track.nacks, direction=track.direction.value)
        self._advance(
            group._keyframe_requests,
            "keyframe_requests",
            name,
            track.keyframe_requests,
            direction=track.direction.value,
        )
        # Loss is reported for both directions and shares one instrument, so the
        # way the track flowed is what tells the two apart.
        self._advance(
            group._packets_lost,
            "packets_lost",
            name,
            track.packets_lost,
            direction=track.direction.value,
        )
        if track.jitter is not None:
            group._jitter.labels(track=name).observe(track.jitter)
        # Both of these are the receiver's own measurements of what this side
        # sent, so they are absent until its first report arrives.
        if track.rtt_seconds is not None:
            group._media_rtt.labels(track=name).observe(track.rtt_seconds)
        if track.loss_ratio is not None:
            group._loss_ratio.labels(track=name).observe(track.loss_ratio)

    def _advance(
        self, counter: Counter, field: str, track: str, total: int | None, **labels: str
    ) -> None:
        """Move *counter* by how far *total* went past the last one for *track*.

        A total below the one before it is read as a counter that started over,
        and the whole of it counts as the increment — which keeps a peer that
        reset its own counters from handing a counter here a negative move it
        would refuse outright.
        """
        if total is None:
            return
        key = (field, track)
        previous = self._totals.get(key, 0)
        self._totals[key] = total
        moved = total - previous if total >= previous else total
        counter.labels(track=track, **labels).inc(moved)

    def _fold_media(self, stats: PeerStats) -> None:
        """Count the outbound media this window manufactured or discarded.

        None of it appears in transport statistics: these are the frames and
        samples the runtime dropped or invented on its own side, before anything
        reached the wire, and they are the only outbound quality signal this
        process can see by itself.
        """
        media = stats.media
        silence = media.silence_frames - self._silence_frames
        samples = media.dropped_samples - self._dropped_samples
        bundles = media.dropped_bundles - self._dropped_bundles
        frames = media.dropped_frames - self._dropped_frames
        self._silence_frames = media.silence_frames
        self._dropped_samples = media.dropped_samples
        self._dropped_bundles = media.dropped_bundles
        self._dropped_frames = media.dropped_frames
        self._metrics._silence_frames.inc(max(0, silence))
        self._metrics._dropped_samples.inc(max(0, samples))
        self._metrics._dropped_bundles.inc(max(0, bundles))
        self._metrics._dropped_frames.inc(max(0, frames))
