"""Tests for the stage accumulator a transport and the runner share."""

from __future__ import annotations

import math

from reactor_runtime.core.stages import StageTotals


def test_adds_up_each_keys_stages_and_returns_them_in_order() -> None:
    totals = StageTotals(("decode", "delivery"))
    totals.add("video", "delivery", 2.0)
    totals.add("video", "decode", 1.0, 2)
    totals.add("video", "delivery", 3.0)
    totals.add(None, "send", 0.5)

    taken = totals.take()

    assert taken == {
        "video": [("decode", 1.0, 2), ("delivery", 5.0, 2)],
        None: [("send", 0.5, 1)],
    }
    assert totals.take() == {}


def test_a_stage_outside_the_order_comes_after_it() -> None:
    totals = StageTotals(("decode",))
    totals.add("video", "unknown", 1.0)
    totals.add("video", "decode", 1.0)

    assert [name for name, _, _ in totals.take()["video"]] == ["decode", "unknown"]


def test_a_take_of_some_keys_keeps_the_others() -> None:
    totals = StageTotals()
    totals.add("a", "decode", 1.0)
    totals.add("b", "decode", 2.0)

    assert totals.take(["a"]) == {"a": [("decode", 1.0, 1)]}
    assert totals.take() == {"b": [("decode", 2.0, 1)]}


def test_no_measurement_is_left_out() -> None:
    totals = StageTotals()
    totals.add("a", "decode", -1.0)
    totals.add("a", "decode", math.nan)
    totals.add("a", "decode", math.inf)
    totals.add("a", "decode", 1.0, 0)

    assert totals.take() == {}


def test_clear_drops_every_total() -> None:
    totals = StageTotals()
    totals.add("a", "decode", 1.0)
    totals.clear()

    assert totals.take() == {}
