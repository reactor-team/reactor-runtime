import threading
import time

import numpy as np

from reactor_runtime.core.values import (
    MediaBundle,
    MediaChunk,
    TrackData,
    TrackDirection,
    TrackInfo,
    TrackKind,
)
from reactor_runtime.transport.livekit.pacer import RoomPacer

VIDEO = TrackInfo(name="main_video", kind=TrackKind.VIDEO, direction=TrackDirection.OUT)
AUDIO = TrackInfo(
    name="main_audio", kind=TrackKind.AUDIO, rate=24_000, direction=TrackDirection.OUT
)


def av_chunk(values: list[int], *, fps: float = 25.0, wait: bool = False) -> MediaChunk:
    """A batch of frames whose pixels and 40 ms audio slices both encode the frame's value."""
    n = len(values)
    video = np.stack([np.full((4, 4, 3), v, dtype=np.uint8) for v in values])
    audio = np.concatenate([np.full(960, v, dtype=np.int16) for v in values]).reshape(1, -1)
    bundle = MediaBundle(
        tracks={
            "main_video": TrackData(info=VIDEO, data=video),
            "main_audio": TrackData(info=AUDIO, data=audio),
        }
    )
    return MediaChunk(bundle=bundle, fps=fps, n_frames=n, wait=wait)


class Sink:
    def __init__(self) -> None:
        self.seen: list[MediaBundle] = []

    def __call__(self, bundle: MediaBundle) -> None:
        self.seen.append(bundle)

    def pixels(self) -> list[int]:
        return [int(b.tracks["main_video"].data[0, 0, 0]) for b in self.seen]


def make_pacer(depth: int = 10) -> tuple[RoomPacer, Sink]:
    sink = Sink()
    return RoomPacer({"main_video": VIDEO}, sink, queue_depth=depth), sink


def test_the_first_frame_the_room_sees_is_the_models() -> None:
    pacer, sink = make_pacer()
    pacer._emit_one_tick()  # nothing queued: nothing sent, not even black

    pacer.submit(av_chunk([7]))
    pacer._emit_one_tick()

    assert sink.pixels() == [7]


def test_each_frame_goes_out_with_its_own_audio_slice_in_order() -> None:
    pacer, sink = make_pacer()
    pacer.submit(av_chunk([1, 2, 3]))

    for _ in range(3):
        pacer._emit_one_tick()

    assert sink.pixels() == [1, 2, 3]
    slices = [b.tracks["main_audio"].data for b in sink.seen]
    assert [s.shape for s in slices] == [(1, 960)] * 3
    assert [int(s[0, 0]) for s in slices] == [1, 2, 3]
    assert all((s == s[0, 0]).all() for s in slices)


def test_an_empty_queue_sends_nothing_after_real_frames() -> None:
    # A subscriber keeps the frame it decoded; nothing is repeated to fill a gap.
    pacer, sink = make_pacer()
    pacer.submit(av_chunk([1]))
    pacer._emit_one_tick()
    pacer._emit_one_tick()

    assert len(sink.seen) == 1


def test_a_plain_flush_cuts_to_one_black_frame_at_the_last_size() -> None:
    pacer, sink = make_pacer()
    pacer.submit(av_chunk([9, 9, 9]))
    pacer._emit_one_tick()

    pacer.flush()
    pacer._emit_one_tick()
    pacer._emit_one_tick()

    assert sink.pixels() == [9, 0]
    black = sink.seen[1].tracks
    assert set(black) == {"main_video"}  # no audio rides the black frame
    assert black["main_video"].data.shape == (4, 4, 3)


def test_a_hold_flush_drops_the_queue_and_sends_nothing() -> None:
    pacer, sink = make_pacer()
    pacer.submit(av_chunk([5, 5, 5]))
    pacer._emit_one_tick()

    pacer.flush(hold=True)
    pacer._emit_one_tick()
    pacer.submit(av_chunk([6]))
    pacer._emit_one_tick()

    assert sink.pixels() == [5, 6]


def test_a_flush_before_any_frame_sends_no_black() -> None:
    pacer, sink = make_pacer()
    pacer.flush()
    pacer._emit_one_tick()

    assert sink.seen == []


def test_a_full_queue_drops_from_the_tail_and_counts_it() -> None:
    pacer, _ = make_pacer(depth=2)

    assert pacer.submit(av_chunk([1, 2])) == 2
    assert pacer.submit(av_chunk([3])) == 0
    assert pacer.dropped_frames == 1


def test_a_batch_larger_than_the_depth_fits_whole() -> None:
    pacer, _ = make_pacer(depth=2)

    assert pacer.submit(av_chunk([1, 2, 3, 4])) == 4


def test_a_flush_releases_a_producer_waiting_for_room() -> None:
    pacer, _ = make_pacer(depth=1)
    pacer.submit(av_chunk([1]))
    pacer._stop.clear()  # a running pacer, so a backpressure submit waits
    enqueued: list[int] = []
    producer = threading.Thread(
        target=lambda: enqueued.append(pacer.submit(av_chunk([2], wait=True)))
    )
    producer.start()
    time.sleep(0.05)

    pacer.flush(hold=True)
    producer.join(timeout=1.0)

    assert not producer.is_alive()
    assert enqueued == [0]


def test_the_pacing_thread_paces_at_the_chunk_rate() -> None:
    pacer, sink = make_pacer(depth=50)
    pacer.submit(av_chunk(list(range(1, 11)), fps=100.0))

    pacer.start()
    time.sleep(0.2)
    pacer.stop()

    assert sink.pixels() == list(range(1, 11))


def test_a_stopped_pacer_takes_no_more_media() -> None:
    pacer, _ = make_pacer()
    pacer.stop()

    assert pacer.submit(av_chunk([1])) == 0
