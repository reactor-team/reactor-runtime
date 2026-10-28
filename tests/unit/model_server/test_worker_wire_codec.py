"""The wire and the codec: frames, typed control messages, and the one payload encoding.

These run without a worker: each check is on the bytes one side writes and the
other reads, so a hostile or broken peer is a test that writes the bytes itself.
"""

from __future__ import annotations

import json
import pickle
import shutil
import socket
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from reactor_runtime.distributed.ipc import SharedSlot
from reactor_runtime.model_server import codec, schema, wire
from reactor_runtime.model_server.wire import Kind


def _pair() -> tuple[socket.socket, socket.socket]:
    return socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)


# -- frames ---------------------------------------------------------------------------------


def test_a_frame_round_trips_its_kind_id_and_payload() -> None:
    a, b = _pair()
    with a, b:
        wire.send(a, Kind.STEP, 7, b"body")
        assert wire.recv(b, 16) == (Kind.STEP, 7, b"body")


def test_a_frame_past_its_limit_is_refused_before_its_payload_is_read() -> None:
    a, b = _pair()
    with a, b:
        # Announce 100 MB and send none of it: the reader must not wait for it.
        a.sendall(wire.HEADER.pack(100 << 20, Kind.SIGNAL, 0))
        b.settimeout(2)
        with pytest.raises(wire.ProtocolError, match="passes the 1024 limit"):
            wire.recv(b, 1024)


def test_the_limit_can_depend_on_the_frame_kind() -> None:
    a, b = _pair()
    with a, b:
        wire.send(a, Kind.STEP, 1, b"x" * 64)
        wire.send(a, Kind.SIGNAL, 2, b"x" * 64)
        limits = {Kind.STEP: 64, Kind.SIGNAL: 8}
        assert wire.recv(b, limits.__getitem__)[0] is Kind.STEP
        with pytest.raises(wire.ProtocolError, match="SIGNAL frame"):
            wire.recv(b, limits.__getitem__)


def test_an_unknown_frame_kind_is_a_protocol_error() -> None:
    a, b = _pair()
    with a, b:
        a.sendall(wire.HEADER.pack(0, 99, 0))
        with pytest.raises(wire.ProtocolError, match="unknown frame kind 99"):
            wire.recv(b, 16)


def test_a_peer_that_closes_mid_frame_is_closed_not_a_short_read() -> None:
    a, b = _pair()
    with b:
        a.sendall(wire.HEADER.pack(10, Kind.STEP, 0) + b"abc")
        a.close()
        with pytest.raises(wire.Closed):
            wire.recv(b, 64)


# -- typed control messages -----------------------------------------------------------------


_MESSAGES: list[Any] = [
    wire.Hello(slot=None, pid=1, token="abcd1234", protocol=wire.PROTOCOL, schema="00ff"),
    wire.Hello(slot=3, pid=1, token="abcd1234", protocol=wire.PROTOCOL, schema="00ff"),
    wire.Welcome(
        slot=0,
        generation=1,
        max_batch=4,
        max_request_bytes=1 << 20,
        shm_slot_bytes=9 << 20,
        signals=("Preempt",),
        request_prefix="rr_s0_",
        transport={"input_block": "rr_s0_in", "probe": "rr_probe"},
    ),
    wire.Failure(type="m.E", message="m", traceback="t", worker_lost=False, batch_failed=True),
    wire.ResetAck(generation=2),
    wire.Refusal(reason="no"),
]


@pytest.mark.parametrize("message", _MESSAGES, ids=lambda m: type(m).__name__)
def test_a_control_message_round_trips_as_a_json_object_of_its_fields(message: Any) -> None:
    payload = wire.encode_message(message)
    assert isinstance(json.loads(payload), dict)
    assert wire.decode_message(payload, type(message)) == message


def _fields(message: Any) -> dict[str, Any]:
    return json.loads(wire.encode_message(message))


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda f: f.pop("token"), "missing token"),
        (lambda f: f.update(extra=1), "unexpected extra"),
        (lambda f: f.update(pid="1"), "pid must be int"),
        (lambda f: f.update(pid=True), "pid must be int"),
        (lambda f: f.update(protocol=2.0), "protocol must be int"),
        (lambda f: f.update(slot="0"), "slot must be int"),
    ],
)
def test_a_hello_is_decoded_strictly(change: Any, reason: str) -> None:
    fields = _fields(_MESSAGES[1])
    change(fields)
    with pytest.raises(wire.ProtocolError, match=reason):
        wire.decode_message(json.dumps(fields).encode(), wire.Hello)


def test_a_bool_field_is_not_an_int_and_an_int_is_not_a_bool() -> None:
    fields = _fields(_MESSAGES[3])
    fields["worker_lost"] = 0
    with pytest.raises(wire.ProtocolError, match="worker_lost must be bool"):
        wire.decode_message(json.dumps(fields).encode(), wire.Failure)
    ack = {"generation": True}
    with pytest.raises(wire.ProtocolError, match="generation must be int"):
        wire.decode_message(json.dumps(ack).encode(), wire.ResetAck)


@pytest.mark.parametrize(
    "payload", [b"[1]", b'"x"', b"null", b"not json", b"\xff\xfe", pickle.dumps(shutil.rmtree)]
)
def test_a_control_payload_that_is_not_a_json_object_is_a_protocol_error(payload: bytes) -> None:
    with pytest.raises(wire.ProtocolError):
        wire.decode_message(payload, wire.Refusal)


def test_a_welcome_transport_map_holds_strings_only() -> None:
    fields = _fields(_MESSAGES[2])
    fields["transport"] = {"probe": 1}
    with pytest.raises(wire.ProtocolError, match="must be str"):
        wire.decode_message(json.dumps(fields).encode(), wire.Welcome)


def test_the_protocol_version_is_a_positive_integer() -> None:
    assert type(wire.PROTOCOL) is int
    assert wire.PROTOCOL >= 2


# -- the codec ------------------------------------------------------------------------------


def _round_trip(value: Any) -> Any:
    slot = SharedSlot(initial_bytes=4096)
    try:
        header, _ = codec.encode(value, slot)
        _, spans, tree = codec.parse(header)
        buf = slot._shm.buf
        assert buf is not None
        copies = [memoryview(bytearray(buf[o : o + n])) for o, n in spans]
        del buf
    finally:
        slot.close()
    return codec.decode(tree, copies.__getitem__)


@pytest.mark.parametrize(
    "value",
    [
        np.array(7),
        np.array(3.5, dtype=np.float32),
        np.zeros((0, 4), dtype=np.int16),
        np.arange(6, dtype=">i4").reshape(2, 3),
        np.array(["2026-10-26"], dtype="datetime64[D]"),
        np.arange(12, dtype=np.float64).reshape(3, 4)[:, ::2],
        np.array([1 + 2j], dtype=np.complex64),
        {"t": (1, "a", None, True, b"xy"), 3: [1.5, float("inf"), float("-inf")]},
    ],
)
def test_the_codec_round_trips_what_it_carries(value: Any) -> None:
    out = _round_trip(value)
    if isinstance(value, np.ndarray):
        assert out.shape == value.shape
        assert out.dtype == value.dtype
        np.testing.assert_array_equal(out, value)
    else:
        assert out == value


@dataclass(frozen=True)
class _Record:
    x: int


def _function() -> None:
    """A function, which never travels."""


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        (_function, "function does not travel"),
        (lambda: None, "function does not travel"),
        ({"on_done": print}, "does not travel"),
        (object(), "object does not travel"),
        ({1, 2}, "set does not travel"),
        (threading.Lock(), "lock does not travel"),
        (_Record(1), "_Record does not travel"),
        (np.array([object()], dtype=object), "dtype object"),
        ({(1, 2): 3}, "dict key of type tuple"),
        ({True: 1}, "dict key of type bool"),
    ],
)
def test_the_codec_refuses_anything_but_data(value: Any, reason: str) -> None:
    slot = SharedSlot(initial_bytes=4096)
    try:
        with pytest.raises(TypeError, match=reason):
            codec.encode(value, slot)
        with pytest.raises(TypeError, match=reason):
            codec.size(value)
    finally:
        slot.close()


@pytest.mark.parametrize("key", ["__class__", "__reduce__", "__module__", "type", "__init__"])
def test_a_hostile_key_is_plain_data_and_instantiates_nothing(key: str) -> None:
    value = {key: "builtins.eval", "args": ["__import__('os').system('false')"]}
    assert _round_trip(value) == value
    tree = {"d": [[key, "os.system"]]}
    assert codec.decode(tree, lambda i: memoryview(b"")) == {key: "os.system"}


def _deep(depth: int) -> Any:
    tree: Any = 1
    for _ in range(depth):
        tree = {"l": [tree]}
    return tree


@pytest.mark.parametrize(
    ("tree", "reason"),
    [
        (_deep(codec.MAX_DEPTH + 1), "deeper than"),
        ({"l": [0] * (codec.MAX_NODES + 1)}, "more than"),
        ({"o": 1}, "unknown or malformed tag"),
        ({"l": 1}, "unknown or malformed tag"),
        ({"l": [], "t": []}, "neither plain nor one tagged object"),
        ({"d": [["k"]]}, "a key and a value"),
        ({"d": [[1.5, 1]]}, "neither a string nor an integer"),
        ({"f": "1e999"}, "unknown or malformed tag"),
        ({"b": 5}, "span 5"),
        ({"b": True}, "unknown or malformed tag"),
        ({"a": ["|O", [1], 0]}, "does not travel"),
        ({"a": ["V8", [1], 0]}, "does not travel"),
        ({"a": ["x" * 17, [1], 0]}, "short string"),
        ({"a": ["<zz", [1], 0]}, "not one numpy knows"),
        ({"a": ["<f8", [3], 0]}, "needs 24 bytes"),
        ({"a": ["<f8", [-1], 0]}, "non-negative integers"),
        ({"a": ["<f8", [1] * 33, 0]}, "non-negative integers"),
        ({"a": ["<u1", [2**62, 2**62, 0], 0]}, "needs"),
        ({"a": ["<f8", [1], "0"]}, "not an index"),
    ],
)
def test_decode_refuses_what_encode_never_makes(tree: Any, reason: str) -> None:
    def buffer(index: int) -> memoryview:
        if index != 0:
            raise IndexError(index)
        return memoryview(bytes(8))

    with pytest.raises(codec.MalformedPayload, match=reason):
        codec.decode(tree, buffer)


@pytest.mark.parametrize(
    ("header", "reason"),
    [
        (b"not json", "not JSON"),
        (pickle.dumps(shutil.rmtree), "not JSON"),
        (b"[]", "version 1"),
        (json.dumps({"v": 2, "block": "x", "spans": [], "tree": 1}).encode(), "version 1"),
        (json.dumps({"v": 1, "spans": [], "tree": 1}).encode(), "no block name"),
        (json.dumps({"v": 1, "block": "x", "spans": [[-1, 4]], "tree": 1}).encode(), "two non"),
        (json.dumps({"v": 1, "block": "x", "spans": [[0, True]], "tree": 1}).encode(), "two non"),
    ],
)
def test_parse_refuses_a_header_encode_never_makes(header: bytes, reason: str) -> None:
    with pytest.raises(codec.MalformedPayload, match=reason):
        codec.parse(header)


def test_a_header_past_the_cap_is_refused_on_encode() -> None:
    slot = SharedSlot(initial_bytes=4096)
    try:
        with pytest.raises(codec.HeaderTooLarge, match="over the 1048576 limit"):
            codec.encode(list(range(codec.MAX_HEADER_BYTES // 4)), slot)
        # The same data as an array travels: its bytes sit in shared memory.
        header, _ = codec.encode(np.arange(codec.MAX_HEADER_BYTES // 4), slot)
        assert len(header) < 1024
    finally:
        slot.close()


def test_a_header_past_the_cap_is_refused_before_it_is_parsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed: list[int] = []
    real = json.loads

    def counting(data: Any, *args: Any, **kwargs: Any) -> Any:
        parsed.append(len(data))
        return real(data, *args, **kwargs)

    monkeypatch.setattr(codec.json, "loads", counting)
    header = b'{"v":1,"block":"x","spans":[],"tree":"' + b"x" * codec.MAX_HEADER_BYTES + b'"}'
    with pytest.raises(codec.MalformedPayload, match="passes the 1048576 byte limit"):
        codec.parse(header)
    assert parsed == []


def test_a_signal_tree_carries_plain_values_only() -> None:
    value = {"reason": "barge", "seq": 3, "at": [1.5, None]}
    assert codec.from_tree(json.loads(json.dumps(codec.to_tree(value)))) == value
    with pytest.raises(TypeError, match="do not travel in a signal"):
        codec.to_tree({"a": np.ones(2)})
    with pytest.raises(TypeError, match="do not travel in a signal"):
        codec.to_tree({"b": b"xy"})
    with pytest.raises(codec.MalformedPayload, match="span 0"):
        codec.from_tree({"b": 0})


def test_a_typed_payload_travels_as_its_fields() -> None:
    @dataclass(frozen=True)
    class Step:
        frame: np.ndarray
        chunk: int

    fields = schema.dump(Step(frame=np.ones(3), chunk=2), Step)
    out = schema.load(_round_trip(fields), Step)
    assert type(out) is Step
    assert out.chunk == 2
    np.testing.assert_array_equal(out.frame, np.ones(3))
