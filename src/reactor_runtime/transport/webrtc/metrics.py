"""Record WebRTC handshake timings and live transport statistics."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable

from prometheus_client import Counter, Histogram

from reactor_runtime.core import TrackDirection
from reactor_runtime.runtime_metrics import RuntimeMetrics
from reactor_runtime.transport.webrtc.stats import PeerStats, TrackStat

# Building the answer is local work and takes milliseconds. Reaching a connected
# wire adds the round trips of ICE and DTLS, and a client behind a hostile
# network takes seconds or never arrives.
_HANDSHAKE_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
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
        self,
        *,
        outbound: Iterable[str] = (),
        inbound: Iterable[str] = (),
        allowed_tracks: Iterable[str] = (),
    ) -> ConnectionStatsRecorder:
        """Return a recorder that folds one connection's samples in.

        Seeds the per-track children for the tracks this connection carries, so
        a track that stayed silent reads zero instead of being absent, which is
        what tells it apart from a track the model never declared.

        Args:
            outbound: Names of the tracks flowing to the client.
            inbound: Names of the tracks flowing from the client.
            allowed_tracks: Trusted model track names. All other names use the
                fixed ``unknown`` label, during seeding and observation.
        """
        recorder = ConnectionStatsRecorder(self, frozenset(allowed_tracks))
        for name in outbound:
            track = recorder.label(name)
            self._packets_sent.labels(track=track)
            self._packets_lost.labels(track=track, direction=TrackDirection.OUT.value)
            self._packets_retransmitted.labels(track=track)
            self._bytes_sent.labels(track=track)
            self._frames_sent.labels(track=track)
            self._media_rtt.labels(track=track)
            self._loss_ratio.labels(track=track)
            self._nacks.labels(track=track, direction=TrackDirection.OUT.value)
            self._keyframe_requests.labels(track=track, direction=TrackDirection.OUT.value)
        for name in inbound:
            track = recorder.label(name)
            self._packets_received.labels(track=track)
            self._packets_lost.labels(track=track, direction=TrackDirection.IN.value)
            self._bytes_received.labels(track=track)
            self._frames_decoded.labels(track=track)
            self._frames_dropped.labels(track=track)
            self._nacks.labels(track=track, direction=TrackDirection.IN.value)
            self._keyframe_requests.labels(track=track, direction=TrackDirection.IN.value)
            self._jitter.labels(track=track)
        return recorder

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

    def __init__(self, metrics: WebRtcMetrics, allowed_tracks: frozenset[str]) -> None:
        """Start the recorder with no previous sample to difference against.

        Args:
            metrics: The group whose instruments each sample is folded into.
            allowed_tracks: Trusted model names permitted as metric labels.
        """
        self._metrics = metrics
        self._allowed_tracks = allowed_tracks
        self._totals: dict[tuple[str, str, tuple[tuple[str, str], ...]], int] = {}
        self._silence_frames = 0
        self._dropped_samples = 0
        self._dropped_bundles = 0
        self._dropped_frames = 0

    def label(self, name: str) -> str:
        """Return the bounded metric label for a peer's track name."""
        return name if name in self._allowed_tracks else "unknown"

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
            group._jitter.labels(track=self.label(name)).observe(track.jitter)
        # Both of these are the receiver's own measurements of what this side
        # sent, so they are absent until its first report arrives.
        if track.rtt_seconds is not None:
            group._media_rtt.labels(track=self.label(name)).observe(track.rtt_seconds)
        if track.loss_ratio is not None:
            group._loss_ratio.labels(track=self.label(name)).observe(track.loss_ratio)

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
        key = (field, track, tuple(sorted(labels.items())))
        previous = self._totals.get(key, 0)
        self._totals[key] = total
        moved = total - previous if total >= previous else total
        counter.labels(track=self.label(track), **labels).inc(moved)

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
