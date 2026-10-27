"""The author-facing contract: declared types, the worker class, its settings, and its health.

What a model author writes is checked here without a socket: which declarations
the schema accepts and how strictly it dumps and loads them, what a
``ModelServer`` subclass may declare, how ``reactor.yaml`` may adjust its
serving settings, and the health hooks the worker's ``/health`` reads.
"""

from __future__ import annotations

import dataclasses
import threading
import time
import types
from dataclasses import InitVar, dataclass, field
from typing import Any

import numpy as np
import pytest

from reactor_runtime import ReactorApp
from reactor_runtime.model_server import (
    ModelServer,
    SchemaError,
    ServerRequest,
    ServerSettings,
    cuda_health,
    schema,
)

# -- what a declaration may hold ------------------------------------------------------------


@dataclass(frozen=True)
class Inner:
    label: str
    weight: float


@dataclass(frozen=True)
class Everything:
    flag: bool
    count: int
    ratio: float
    name: str
    raw: bytes
    frame: np.ndarray
    nothing: None
    maybe: int | None
    inner: Inner
    items: list[Inner]
    pair: tuple[int, str]
    many: tuple[float, ...]
    by_name: dict[str, int]
    by_index: dict[int, Inner | None]


def _everything() -> Everything:
    return Everything(
        flag=True,
        count=3,
        ratio=0.5,
        name="x",
        raw=b"\x00\x01",
        frame=np.arange(4, dtype=np.uint8),
        nothing=None,
        maybe=None,
        inner=Inner("a", 1.0),
        items=[Inner("b", 2.0)],
        pair=(1, "one"),
        many=(1.0, 2.0),
        by_name={"k": 1},
        by_index={0: Inner("c", 3.0), 1: None},
    )


def test_a_declaration_of_every_allowed_field_type_validates_and_round_trips() -> None:
    value = _everything()
    out = schema.load(schema.dump(value, Everything), Everything)
    assert type(out) is Everything
    assert type(out.inner) is Inner
    np.testing.assert_array_equal(out.frame, value.frame)
    # Python 3.13 compares dataclass fields one by one, and an array has no
    # single truth value: compare every field but the array (checked above).
    blank: Any = None
    assert dataclasses.replace(out, frame=blank) == dataclasses.replace(value, frame=blank)


class NotADataclass:
    x: int


@dataclass
class NotFrozen:
    x: int


@dataclass(frozen=True)
class PostInit:
    x: int

    def __post_init__(self) -> None:
        raise AssertionError("ran")


@dataclass(frozen=True, init=False)
class OwnInit:
    x: int

    def __init__(self, x: int) -> None:
        object.__setattr__(self, "x", x)


@dataclass(frozen=True)
class OwnNew:
    x: int

    def __new__(cls, *args: Any, **kwargs: Any) -> OwnNew:
        return super().__new__(cls)


@dataclass(frozen=True)
class Defaulted:
    x: int = 1


@dataclass(frozen=True)
class Factory:
    x: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class NotInInit:
    x: int = field(init=False)


@dataclass(frozen=True)
class WithInitVar:
    x: int
    y: InitVar[int]


@dataclass(frozen=True)
class AnyField:
    x: Any


@dataclass(frozen=True)
class SetField:
    x: set[int]


@dataclass(frozen=True)
class ObjectField:
    x: object


@dataclass(frozen=True)
class TwoTypes:
    x: int | str


@dataclass(frozen=True)
class FloatKeys:
    x: dict[float, int]


@dataclass(frozen=True)
class Recursive:
    x: list[Recursive]


class PlainBase:
    pass


@dataclass(frozen=True)
class FromPlainBase(PlainBase):
    x: int


@pytest.mark.parametrize(
    ("declared", "reason"),
    [
        (NotADataclass, "is not a dataclass"),
        (NotADataclass(), "is not a class"),
        (NotFrozen, "is not frozen"),
        (PostInit, "defines __post_init__"),
        (OwnInit, "defines __init__"),
        (OwnNew, "defines __new__"),
        (Defaulted, "has a default"),
        (Factory, "has a default"),
        (NotInInit, "is not in __init__"),
        (WithInitVar, "InitVar"),
        (AnyField, "is not a type that travels"),
        (SetField, "is not a type that travels"),
        (ObjectField, "is not a type that travels"),
        (TwoTypes, "only `T | None`"),
        (FloatKeys, "only str or int keys"),
        (Recursive, "contains itself"),
        (FromPlainBase, "inherits from PlainBase"),
    ],
)
def test_a_declaration_that_is_not_data_only_is_refused(declared: Any, reason: str) -> None:
    with pytest.raises(SchemaError, match=reason):
        schema.validate(declared)


def test_a_nested_declaration_is_checked_with_its_parent() -> None:
    @dataclass(frozen=True)
    class Holds:
        x: NotFrozen

    with pytest.raises(SchemaError, match=r"Holds.x: .*NotFrozen is not frozen"):
        schema.validate(Holds)


def test_a_dataclass_base_contributes_its_fields() -> None:
    @dataclass(frozen=True)
    class Child(Inner):
        extra: int

    assert [f.name for f in schema.validate(Child).fields] == ["label", "weight", "extra"]


# -- dump and load, exactly -----------------------------------------------------------------


@dataclass(frozen=True)
class Scalars:
    flag: bool
    count: int
    ratio: float


# The wrong field types are what these test.
@pytest.mark.parametrize(
    ("value", "reason"),
    [
        (Scalars(flag=1, count=1, ratio=1.0), "flag must be bool"),  # ty: ignore[invalid-argument-type]
        (Scalars(flag=True, count=True, ratio=1.0), "count must be int"),
        (Scalars(flag=True, count=1.0, ratio=1.0), "count must be int"),  # ty: ignore[invalid-argument-type]
        (Scalars(flag=True, count=1, ratio=True), "ratio must be float"),
        (Scalars(flag=True, count=1, ratio="1"), "ratio must be float"),  # ty: ignore[invalid-argument-type]
    ],
)
def test_dump_checks_every_field_exactly(value: Scalars, reason: str) -> None:
    with pytest.raises(SchemaError, match=reason):
        schema.dump(value, Scalars)


def test_dump_takes_an_int_for_a_float_and_a_numpy_scalar_for_its_value() -> None:
    # numpy scalars and an int where the declaration says bool, int, and float.
    value: Any = Scalars(flag=np.bool_(True), count=np.int64(2), ratio=3)  # ty: ignore[invalid-argument-type]
    fields = schema.dump(value, Scalars)
    assert fields == {"flag": True, "count": 2, "ratio": 3.0}
    assert [type(v) for v in fields.values()] == [bool, int, float]


def test_dump_refuses_an_instance_of_another_type() -> None:
    with pytest.raises(SchemaError, match=r"expected a .*Scalars, got a Inner"):
        schema.dump(Inner("a", 1.0), Scalars)
    with pytest.raises(SchemaError, match=r"expected a .*Scalars, got a dict"):
        schema.dump({"flag": True, "count": 1, "ratio": 1.0}, Scalars)


@pytest.mark.parametrize(
    ("data", "reason"),
    [
        ({"flag": True, "count": 1}, "missing ratio"),
        ({"flag": True, "count": 1, "ratio": 1.0, "more": 1}, "unexpected more"),
        ({"flag": 1, "count": 1, "ratio": 1.0}, "flag must be bool"),
        ({"flag": True, "count": True, "ratio": 1.0}, "count must be int"),
        ({"flag": True, "count": 1, "ratio": 1}, "ratio must be float"),
        ([True, 1, 1.0], "the fields of"),
    ],
)
def test_load_checks_every_field_exactly(data: Any, reason: str) -> None:
    with pytest.raises(SchemaError, match=reason):
        schema.load(data, Scalars)


def test_load_takes_a_list_for_a_tuple_and_none_for_an_optional() -> None:
    value = _everything()
    fields = schema.dump(value, Everything)
    fields["pair"] = [1, "one"]
    fields["many"] = [1.0, 2.0]
    fields["maybe"] = 4
    out = schema.load(fields, Everything)
    assert out.pair == (1, "one")
    assert out.many == (1.0, 2.0)
    assert out.maybe == 4
    fields["items"] = ({"label": "b", "weight": 2.0},)
    with pytest.raises(SchemaError, match="items must be list"):
        schema.load(fields, Everything)


@pytest.mark.parametrize("key", ["__class__", "__reduce__", "__module__", "__init__", "type"])
def test_a_hostile_field_name_is_refused_and_instantiates_nothing(key: str) -> None:
    with pytest.raises(SchemaError, match=f"unexpected {key}"):
        schema.load({"label": "a", "weight": 1.0, key: "os.system"}, Inner)


def test_load_builds_the_instance_without_running_the_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema.validate(Inner)

    def refuse(self: Any, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("__init__ ran")

    monkeypatch.setattr(Inner, "__init__", refuse)
    out = schema.load({"label": "a", "weight": 1.0}, Inner)
    assert (out.label, out.weight) == ("a", 1.0)


def test_the_fingerprint_follows_every_declared_field_and_the_signal_order() -> None:
    @dataclass(frozen=True)
    class A:
        x: int

    @dataclass(frozen=True)
    class B:
        y: str

    def fp(request: Any = A, result: Any = B, signals: tuple[Any, ...] = (A, B)) -> str:
        return schema.fingerprint(request_type=request, result_type=result, signals=signals)

    assert fp() == fp()
    assert fp(signals=(B, A)) != fp()
    assert fp(request=None) != fp()
    assert fp(result=None, request=None, signals=()) == fp(result=None, request=None, signals=())

    @dataclass(frozen=True)
    class A2:
        x: float

    A2.__qualname__ = A.__qualname__
    A2.__module__ = A.__module__
    assert fp(request=A2) != fp()


# -- the worker class -----------------------------------------------------------------------


@dataclass(frozen=True)
class Preempt:
    reason: str


@dataclass(frozen=True)
class WithArray:
    frame: np.ndarray


@pytest.mark.parametrize(
    ("attrs", "error", "reason"),
    [
        ({"request_type": NotFrozen}, TypeError, "request_type: .*is not frozen"),
        ({"result_type": AnyField}, TypeError, "result_type: .*is not a type that travels"),
        ({"signals": frozenset({"preempt"})}, TypeError, "signals must be a tuple"),
        ({"signals": ("preempt",)}, TypeError, "is not a class"),
        ({"signals": (WithArray,)}, TypeError, "holds arrays or bytes"),
        ({"signals": (Preempt, Preempt)}, TypeError, "names a type twice"),
        ({"max_batch": 0}, ValueError, "max_batch must be a positive integer"),
        ({"max_batch": 2.0}, ValueError, "max_batch must be a positive integer"),
        ({"batch_window": -1.0}, ValueError, "batch_window must be a number"),
        ({"max_request_bytes": 0}, ValueError, "max_request_bytes"),
        ({"shm_input_bytes": True}, ValueError, "shm_input_bytes"),
    ],
)
def test_a_worker_class_is_checked_as_it_is_defined(
    attrs: dict[str, Any], error: type[Exception], reason: str
) -> None:
    with pytest.raises(error, match=reason):
        type("Bad", (ModelServer,), attrs)


def test_a_worker_class_with_typed_declarations_defines() -> None:
    class Good(ModelServer):
        request_type = Everything
        result_type = Inner
        signals = (Preempt, Scalars)

        def step(self, requests: list[ServerRequest[Everything]]) -> list[Inner]:
            return [Inner(r.payload.name, 1.0) for r in requests]

    assert Good.signals == (Preempt, Scalars)


def test_the_default_hooks_report_healthy_and_take_signals_quietly() -> None:
    worker = ModelServer()
    assert worker.health() is None
    assert worker.unhealthy is None
    worker.signal(0, Preempt("x"))
    with pytest.raises(NotImplementedError):
        worker.step([])


def test_mark_unhealthy_latches_its_first_reason_from_any_thread() -> None:
    worker = ModelServer()
    threads = [
        threading.Thread(target=worker.mark_unhealthy, args=(f"reason {i}",)) for i in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    first = worker.unhealthy
    assert first is not None
    assert first.startswith("reason ")
    worker.mark_unhealthy("later")
    assert worker.unhealthy == first


def test_cuda_health_without_torch_reports_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(__import__("sys").modules, "torch", raising=False)
    assert cuda_health() is None


def _fake_torch(*, available: bool, fail: bool) -> types.ModuleType:
    torch = types.ModuleType("torch")

    def synchronize() -> None:
        if fail:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

    torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
        is_available=lambda: available, synchronize=synchronize
    )
    torch.ones = lambda *args, **kwargs: None  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
    return torch


@pytest.mark.parametrize(
    ("available", "fail", "reason"),
    [
        (True, False, None),
        (False, False, "the GPU is not visible"),
        (True, True, "CUDA error: RuntimeError: CUDA error: an illegal memory access"),
    ],
)
def test_cuda_health_reads_the_torch_the_process_imported(
    monkeypatch: pytest.MonkeyPatch, available: bool, fail: bool, reason: str | None
) -> None:
    monkeypatch.setitem(
        __import__("sys").modules, "torch", _fake_torch(available=available, fail=fail)
    )
    got = cuda_health()
    if reason is None:
        assert got is None
    else:
        assert got is not None
        assert got.startswith(reason)


# -- the serving settings -------------------------------------------------------------------


class Served(ModelServer):
    max_batch = 8
    batch_window = 0.01
    max_request_bytes = 1 << 20
    shm_input_bytes = 1 << 16
    shm_output_bytes = 1 << 18


def test_the_class_settings_apply_without_overrides() -> None:
    settings = ServerSettings.resolve(Served, {})
    assert settings == ServerSettings(8, 0.01, 1 << 20, 1 << 16, 1 << 18)
    assert settings.shm_slot_bytes == (1 << 16) + (1 << 18)


def test_the_manifest_may_lower_max_batch_raise_shm_and_set_the_rest() -> None:
    settings = ServerSettings.resolve(
        Served,
        {
            "max_batch": 2,
            "batch_window": 0,
            "max_request_bytes": 1 << 30,
            "shm_input_bytes": 1 << 20,
            "shm_output_bytes": 1 << 22,
        },
    )
    assert settings == ServerSettings(2, 0.0, 1 << 30, 1 << 20, 1 << 22)
    assert type(settings.batch_window) is float


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"max_batch": 9}, "above Served.max_batch = 8"),
        ({"shm_input_bytes": 1}, "below Served.shm_input_bytes"),
        ({"shm_output_bytes": 1 << 10}, "below Served.shm_output_bytes"),
        ({"max_batch": 0}, "positive integer"),
        ({"max_batch": "2"}, "positive integer"),
        ({"max_request_bytes": 1.5}, "positive integer"),
        ({"batch_window": -1}, "at least 0"),
        ({"batch_window": "fast"}, "at least 0"),
        ({"batch_window": float("inf")}, "at least 0"),
        ({"sessions": 4}, r"unknown settings \['sessions'\]"),
    ],
)
def test_the_manifest_may_not_ask_for_what_the_class_cannot_serve(
    overrides: dict[str, Any], reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        ServerSettings.resolve(Served, overrides)


# -- the application's availability report --------------------------------------------------


def test_the_availability_replay_at_bind_comes_before_any_later_report() -> None:
    class App(ReactorApp):
        def generate(self, input: Any) -> Any:
            return None

    app = App()
    app.report_unavailable("worker_unavailable: not up")
    assert app.unavailable == "worker_unavailable: not up"
    heard: list[str | None] = []
    reporter: list[threading.Thread] = []

    def sink(reason: str | None) -> None:
        if not reporter:
            reporter.append(threading.Thread(target=app.report_available))
            reporter[0].start()
            time.sleep(0.1)
        heard.append(reason)

    app._bind_availability(sink)
    reporter[0].join(5)
    assert heard == ["worker_unavailable: not up", None]
    assert app.unavailable is None
