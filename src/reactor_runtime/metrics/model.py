# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Record model load times and media output."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable

from prometheus_client import Counter, Histogram

from reactor_runtime.metrics.registry import RuntimeMetrics

# A model reads its weights once. Small models load in seconds and large ones
# hold the process for minutes, which is the whole cold start a client waits on.
_MODEL_LOAD_BUCKETS = (1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)
# The boundaries around a frame period cover a model that emits one frame at a
# time: 33ms is 30fps, and 67ms is half of it. The higher ones are a model that
# emits a batch at a time, whose gaps are the play-out duration of a batch, and
# the top of the range is a stall either of them would feel as a freeze.
_EMIT_INTERVAL_BUCKETS = (0.005, 0.01, 0.02, 0.033, 0.067, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)


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
