"""The InputSynchronizer joins the tracks of one capture by exact stamp."""

from __future__ import annotations

import pytest

from reactor_runtime.apps.robot.sync import (
    SYNC_BUFFER_SIZE,
    InputSet,
    InputSynchronizer,
    SyncStats,
)


def _sync(buffer: int = SYNC_BUFFER_SIZE) -> InputSynchronizer:
    return InputSynchronizer(("scene", "wrist"), ("state",), buffer=buffer)


def _capture(sync: InputSynchronizer, stamp: int) -> None:
    sync.push("scene", stamp, f"scene@{stamp}")
    sync.push("wrist", stamp, f"wrist@{stamp}")
    sync.push("state", stamp, {"t": stamp})


class TestConstruction:
    def test_defaults(self) -> None:
        sync = InputSynchronizer(("scene",))
        assert sync.tracks == ("scene",)
        assert sync.data_tracks == ()
        assert sync.reference == "scene"
        assert sync.buffer == SYNC_BUFFER_SIZE
        assert sync.stats() == SyncStats(buffered={"scene": 0})

    def test_reference_defaults_to_first_data_track_without_cameras(self) -> None:
        assert InputSynchronizer((), ("state",)).reference == "state"

    def test_reference_may_be_a_data_track(self) -> None:
        sync = InputSynchronizer(("scene",), ("state",), reference="state")
        assert sync.reference == "state"

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"tracks": ()}, "at least one track"),
            ({"tracks": ("scene", "scene")}, "distinct"),
            ({"tracks": ("scene",), "data_tracks": ("scene",)}, "distinct"),
            ({"tracks": ("scene",), "reference": "wrist"}, "reference track"),
            ({"tracks": ("scene",), "buffer": 0}, "at least 1 item"),
            ({"tracks": "scene"}, "not one string"),
        ],
    )
    def test_refuses_bad_arguments(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            InputSynchronizer(**kwargs)

    def test_push_refuses_unknown_track(self) -> None:
        with pytest.raises(KeyError, match="unknown track 'top'"):
            _sync().push("top", 1, "frame")
        assert _sync().stats().pushed == 0


class TestExactJoin:
    def test_takes_a_complete_set(self) -> None:
        sync = _sync()
        _capture(sync, 1_000)
        taken = sync.take()
        assert taken == InputSet(
            capture_time_us=1_000,
            frames={"scene": "scene@1000", "wrist": "wrist@1000"},
            data={"state": {"t": 1_000}},
        )

    def test_set_is_read_only(self) -> None:
        sync = _sync()
        _capture(sync, 1)
        taken = sync.take()
        assert taken is not None
        with pytest.raises(TypeError):
            taken.frames["scene"] = "other"  # type: ignore[ty:invalid-assignment] - proving it is read-only

    def test_incomplete_set_is_not_taken(self) -> None:
        sync = _sync()
        sync.push("scene", 1_000, "s")
        sync.push("state", 1_000, {})
        assert sync.take() is None
        sync.push("wrist", 1_000, "w")
        taken = sync.take()
        assert taken is not None
        assert taken.capture_time_us == 1_000

    def test_stamps_one_microsecond_apart_do_not_join(self) -> None:
        sync = _sync()
        sync.push("scene", 1_000, "s")
        sync.push("wrist", 1_001, "w")
        sync.push("state", 1_000, {})
        assert sync.take() is None

    def test_arrival_order_does_not_matter(self) -> None:
        sync = _sync()
        sync.push("state", 2_000, {"t": 2})
        sync.push("wrist", 2_000, "w2")
        sync.push("scene", 1_000, "s1")
        sync.push("scene", 2_000, "s2")
        taken = sync.take()
        assert taken is not None
        assert taken.capture_time_us == 2_000
        assert taken.frames == {"scene": "s2", "wrist": "w2"}

    def test_reference_track_defines_the_captures(self) -> None:
        sync = InputSynchronizer(("scene",), ("state",), reference="state")
        sync.push("scene", 1_000, "s1")
        sync.push("scene", 2_000, "s2")
        sync.push("state", 1_000, {})
        taken = sync.take()
        assert taken is not None
        assert taken.capture_time_us == 1_000

    def test_same_stamp_replaces_the_item(self) -> None:
        sync = _sync()
        _capture(sync, 1_000)
        sync.push("scene", 1_000, "again")
        taken = sync.take()
        assert taken is not None
        assert taken.frames["scene"] == "again"
        assert sync.stats().dropped == 0

    def test_an_empty_take_changes_nothing(self) -> None:
        sync = _sync()
        sync.push("scene", 1_000, "s")
        before = sync.stats()
        assert sync.take() is None
        assert sync.stats() == before


class TestNewestWins:
    def test_takes_the_newest_complete_set_and_skips_older_ones(self) -> None:
        sync = _sync()
        _capture(sync, 1_000)
        _capture(sync, 2_000)
        _capture(sync, 3_000)
        sync.push("scene", 4_000, "incomplete")
        taken = sync.take()
        assert taken is not None
        assert taken.capture_time_us == 3_000
        stats = sync.stats()
        assert stats.taken == 1
        assert stats.skipped == 2

    def test_never_hands_out_a_capture_twice(self) -> None:
        sync = _sync()
        _capture(sync, 1_000)
        assert sync.take() is not None
        assert sync.take() is None
        _capture(sync, 1_000)
        assert sync.take() is None

    def test_never_hands_out_an_older_capture_after_a_newer_one(self) -> None:
        sync = _sync()
        _capture(sync, 2_000)
        assert sync.take() is not None
        _capture(sync, 1_500)
        assert sync.take() is None
        _capture(sync, 2_500)
        taken = sync.take()
        assert taken is not None
        assert taken.capture_time_us == 2_500
        assert sync.stats().skipped == 0

    def test_takes_an_older_set_while_the_newer_one_is_incomplete(self) -> None:
        sync = _sync()
        _capture(sync, 1_000)
        sync.push("scene", 2_000, "s2")
        sync.push("state", 2_000, {})
        taken = sync.take()
        assert taken is not None
        assert taken.capture_time_us == 1_000
        sync.push("wrist", 2_000, "w2")
        taken = sync.take()
        assert taken is not None
        assert taken.capture_time_us == 2_000


class TestTrimming:
    def test_take_discards_items_at_or_before_the_set(self) -> None:
        sync = _sync()
        _capture(sync, 1_000)
        sync.push("scene", 2_000, "s2")
        sync.push("wrist", 500, "w0")
        sync.push("wrist", 3_000, "w3")
        assert sync.take() is not None
        stats = sync.stats()
        assert stats.buffered == {"scene": 1, "wrist": 1, "state": 0}
        assert stats.trimmed == 4

    def test_items_newer_than_the_set_survive_for_the_next_take(self) -> None:
        sync = _sync()
        _capture(sync, 1_000)
        sync.push("scene", 2_000, "s2")
        sync.push("state", 2_000, {})
        assert sync.take() is not None
        sync.push("wrist", 2_000, "w2")
        taken = sync.take()
        assert taken is not None
        assert taken.frames == {"scene": "s2", "wrist": "w2"}


class TestFullRing:
    def test_drops_the_oldest_item(self) -> None:
        sync = _sync(buffer=2)
        for stamp in (1_000, 2_000, 3_000):
            sync.push("scene", stamp, f"s{stamp}")
        sync.push("wrist", 1_000, "w1")
        sync.push("state", 1_000, {})
        assert sync.take() is None
        stats = sync.stats()
        assert stats.dropped == 1
        assert stats.buffered["scene"] == 2

    def test_drops_an_item_older_than_everything_held(self) -> None:
        sync = _sync(buffer=2)
        sync.push("scene", 2_000, "s2")
        sync.push("scene", 3_000, "s3")
        sync.push("scene", 1_000, "s1")
        sync.push("wrist", 1_000, "w1")
        sync.push("state", 1_000, {})
        assert sync.take() is None
        assert sync.stats().dropped == 1

    def test_keeps_stamp_order_when_an_item_lands_in_the_middle(self) -> None:
        sync = _sync(buffer=2)
        sync.push("scene", 1_000, "s1")
        sync.push("scene", 3_000, "s3")
        sync.push("scene", 2_000, "s2")
        sync.push("wrist", 2_000, "w2")
        sync.push("state", 2_000, {})
        taken = sync.take()
        assert taken is not None
        assert taken.capture_time_us == 2_000
        assert sync.stats().buffered["scene"] == 1


class TestReset:
    def test_drops_every_item_and_keeps_the_counts(self) -> None:
        sync = _sync()
        _capture(sync, 1_000)
        assert sync.take() is not None
        _capture(sync, 2_000)
        sync.reset()
        assert sync.take() is None
        stats = sync.stats()
        assert stats.buffered == {"scene": 0, "wrist": 0, "state": 0}
        assert stats.resets == 1
        assert stats.pushed == 6
        assert stats.taken == 1

    def test_forgets_the_last_set_taken(self) -> None:
        sync = _sync()
        _capture(sync, 5_000)
        assert sync.take() is not None
        sync.reset()
        _capture(sync, 1_000)
        taken = sync.take()
        assert taken is not None
        assert taken.capture_time_us == 1_000
