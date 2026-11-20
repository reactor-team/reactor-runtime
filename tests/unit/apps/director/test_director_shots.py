"""The generation queue, the playout queue and the history, as plain bookkeeping."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from reactor_runtime.apps.director.shots import (
    History,
    ShotEntry,
    ShotQueue,
    ShotStatus,
    dependents,
)


def _shot(shot_id: str, **sources: str) -> ShotEntry:
    return ShotEntry(id=shot_id, inputs={}, sources=dict(sources), continues=None, metadata="")


def _ids(queue: ShotQueue | History) -> list[str]:
    return [entry.id for entry in queue]


def test_a_shot_without_a_position_joins_the_back() -> None:
    queue = ShotQueue(capacity=3)
    queue.add(_shot("a"))
    queue.add(_shot("b"))
    assert _ids(queue) == ["a", "b"]


def test_a_position_inserts_and_one_past_the_back_appends() -> None:
    queue = ShotQueue(capacity=4)
    queue.add(_shot("a"))
    queue.add(_shot("b"))
    assert queue.add(_shot("c"), position=0) == 0
    assert queue.add(_shot("d"), position=99) == 3
    assert _ids(queue) == ["c", "a", "b", "d"]


def test_a_full_queue_refuses_another_shot() -> None:
    queue = ShotQueue(capacity=1)
    queue.add(_shot("a"))
    assert queue.full
    with pytest.raises(ValueError, match="full"):
        queue.add(_shot("b"))


def test_move_repositions_within_the_queue_and_clamps() -> None:
    queue = ShotQueue(capacity=3)
    a, b, c = _shot("a"), _shot("b"), _shot("c")
    for entry in (a, b, c):
        queue.add(entry)
    assert queue.move(c, 0) == 0
    assert queue.move(a, 10) == 2
    assert _ids(queue) == ["c", "b", "a"]


def test_next_to_build_skips_a_shot_that_is_not_ready_without_blocking_the_rest() -> None:
    queue = ShotQueue(capacity=3)
    waiting, free = _shot("waiting", continue_from="x"), _shot("free")
    queue.add(waiting)
    queue.add(free)
    assert queue.next_to_build(lambda entry: not entry.sources) is free


def test_next_to_build_skips_a_shot_already_building() -> None:
    queue = ShotQueue(capacity=2)
    building, queued = _shot("building"), _shot("queued")
    building.status = ShotStatus.BUILDING
    queue.add(building)
    queue.add(queued)
    assert queue.next_to_build(lambda entry: True) is queued


def test_the_history_evicts_the_oldest_first_but_never_a_protected_shot() -> None:
    history = History(capacity=2)
    for shot_id in ("a", "b", "c"):
        history.put(_shot(shot_id))
    evicted = history.trim(protected={"a"})
    assert [entry.id for entry in evicted] == ["b"]
    assert _ids(history) == ["a", "c"]


def test_putting_a_shot_again_makes_it_the_newest() -> None:
    history = History(capacity=2)
    a = _shot("a")
    history.put(a)
    history.put(_shot("b"))
    history.put(a)
    history.put(_shot("c"))
    assert [entry.id for entry in history.trim(protected=set())] == ["b"]


def test_dependents_are_found_through_other_dependents() -> None:
    queue = ShotQueue(capacity=4)
    child = _shot("child", continue_from="root")
    grandchild = _shot("grandchild", ending_from="child")
    unrelated = _shot("unrelated")
    for entry in (unrelated, grandchild, child):
        queue.add(entry)
    assert [entry.id for entry in dependents(queue, {"root"})] == ["child", "grandchild"]


def test_a_shot_hands_generate_a_read_only_view() -> None:
    entry = ShotEntry(id="a", inputs={"prompt": "x"}, sources={}, continues=None, metadata="")
    shot = entry.shot()
    assert shot.id == "a"
    assert shot.inputs["prompt"] == "x"
    assert isinstance(shot.inputs, MappingProxyType)
    entry.inputs["prompt"] = "changed"
    assert shot.inputs["prompt"] == "x"
