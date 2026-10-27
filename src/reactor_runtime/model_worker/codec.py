"""The one encoding a request or an output takes between a session and the worker.

Nothing on the socket is pickled: a session process must not be able to run
code in the worker, which holds every session's state. A payload is a tree of
plain values, which travels as JSON, and numpy arrays, whose bytes travel in a
shared block the header names:

* ``None``, ``bool``, ``int``, ``float``, and ``str``;
* ``list``, ``tuple``, and ``dict`` with ``str`` or ``int`` keys;
* ``bytes``;
* a numpy array of a numeric, boolean, or datetime dtype (an object or a
  structured dtype is refused); a numpy scalar travels as the Python number it
  holds.

Anything else is refused by :func:`encode` with a ``TypeError`` naming it.
:func:`decode` checks everything it reads (the shape of the tree, each array's
dtype, shape, and span) before it builds anything, and raises
:class:`MalformedPayload` for what it refuses.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

# Bounds on what one header may describe, so a malformed one cannot make the
# reader recurse or allocate without limit.
MAX_DEPTH = 64
MAX_NODES = 1 << 20
_KINDS = frozenset("biufcmM")
VERSION = 1


class MalformedPayload(ValueError):  # noqa: N818 (named for the event a caller catches)
    """A header that does not describe a payload this codec makes."""


class Writer(Protocol):
    """Where :func:`encode` writes an array's bytes."""

    @property
    def name(self) -> str:
        """The block a reader maps, read once every buffer is written."""
        ...

    def begin(self) -> None:
        """Start a payload; the next write lands at offset zero."""

    def write(self, buffer: Any) -> tuple[int, int]:
        """Append one flat buffer and return its ``(offset, length)`` span."""
        ...


def encode(payload: Any, writer: Writer) -> tuple[bytes, int]:
    """Write *payload*'s arrays and bytes through *writer* and return its header.

    Returns:
        The JSON header, and how many bytes the payload carries in all.

    Raises:
        TypeError: The payload holds a value this codec does not carry.
    """
    writer.begin()
    spans: list[list[int]] = []

    def store(buffer: Any) -> int:
        offset, length = writer.write(buffer)
        spans.append([offset, length])
        return len(spans) - 1

    tree = _to_tree(payload, store, 0)
    header = json.dumps(
        {"v": VERSION, "block": writer.name, "spans": spans, "tree": tree},
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return header, len(header) + sum(length for _, length in spans)


def size(payload: Any) -> int:
    """Return the bytes *payload*'s arrays and bytes take in a block, without writing them.

    Raises:
        TypeError: The payload holds a value this codec does not carry.
    """
    total = [0]

    def count(buffer: Any) -> int:
        total[0] += buffer if isinstance(buffer, int) else memoryview(buffer).nbytes
        return 0

    _to_tree(payload, count, 0, measure=True)
    return total[0]


def parse(header: bytes) -> tuple[str, list[tuple[int, int]], Any]:
    """Return a header's block name, spans, and tree, checking their shape.

    Raises:
        MalformedPayload: The header is not one :func:`encode` makes.
    """
    try:
        document = json.loads(header)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise MalformedPayload(f"the header is not JSON: {type(exc).__name__}") from None
    if not isinstance(document, dict) or document.get("v") != VERSION:
        raise MalformedPayload("the header is not a version 1 payload")
    block, spans = document.get("block"), document.get("spans")
    if not isinstance(block, str) or not isinstance(spans, list):
        raise MalformedPayload("the header has no block name or no spans")
    checked: list[tuple[int, int]] = []
    for span in spans:
        if not (
            isinstance(span, list)
            and len(span) == 2
            and all(type(v) is int and v >= 0 for v in span)
        ):
            raise MalformedPayload("a span is not two non-negative integers")
        checked.append((span[0], span[1]))
    return block, checked, document.get("tree")


def decode(tree: Any, buffer: Callable[[int], memoryview]) -> Any:
    """Rebuild a payload from its *tree*, taking span *i*'s bytes from ``buffer(i)``.

    Raises:
        MalformedPayload: The tree is malformed, too deep or too large, or an
            array's dtype, shape, or span does not hold together.
    """
    count = [0]
    return _from_tree(tree, buffer, 0, count)


def _to_tree(value: Any, store: Callable[[Any], int], depth: int, *, measure: bool = False) -> Any:
    if depth > MAX_DEPTH:
        raise TypeError(f"the payload nests deeper than {MAX_DEPTH}")
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"f": repr(value)}
    if isinstance(value, np.generic) and not isinstance(value, np.void | np.object_):
        return _to_tree(value.item(), store, depth, measure=measure)
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in _KINDS:
            raise TypeError(f"an array of dtype {value.dtype} does not travel; use a numeric one")
        if measure:
            return store(value.nbytes)
        array = np.ascontiguousarray(value)
        return {"a": [array.dtype.str, list(array.shape), store(array.reshape(-1).view(np.uint8))]}
    if isinstance(value, bytes | bytearray | memoryview):
        return {"b": store(memoryview(value).cast("B"))}
    if isinstance(value, list):
        return {"l": [_to_tree(item, store, depth + 1, measure=measure) for item in value]}
    if isinstance(value, tuple):
        return {"t": [_to_tree(item, store, depth + 1, measure=measure) for item in value]}
    if isinstance(value, dict):
        items = []
        for key, item in value.items():
            if not isinstance(key, str | int) or isinstance(key, bool):
                raise TypeError(f"a dict key of type {type(key).__name__} does not travel")
            items.append([key, _to_tree(item, store, depth + 1, measure=measure)])
        return {"d": items}
    raise TypeError(
        f"a {type(value).__name__} does not travel to or from the model worker: use plain "
        "values, lists, tuples, dicts, bytes, and numpy arrays"
    )


def _from_tree(node: Any, buffer: Callable[[int], memoryview], depth: int, count: list[int]) -> Any:
    count[0] += 1
    if count[0] > MAX_NODES:
        raise MalformedPayload(f"the payload has more than {MAX_NODES} values")
    if depth > MAX_DEPTH:
        raise MalformedPayload(f"the payload nests deeper than {MAX_DEPTH}")
    if node is None or isinstance(node, bool | int | float | str):
        return node
    if not isinstance(node, dict) or len(node) != 1:
        raise MalformedPayload("a value is neither plain nor one tagged object")
    ((tag, body),) = node.items()
    if tag == "l" and isinstance(body, list):
        return [_from_tree(item, buffer, depth + 1, count) for item in body]
    if tag == "t" and isinstance(body, list):
        return tuple(_from_tree(item, buffer, depth + 1, count) for item in body)
    if tag == "d" and isinstance(body, list):
        out: dict[Any, Any] = {}
        for pair in body:
            if not (isinstance(pair, list) and len(pair) == 2):
                raise MalformedPayload("a dict entry is not a key and a value")
            key, item = pair
            if not isinstance(key, str | int) or isinstance(key, bool):
                raise MalformedPayload("a dict key is neither a string nor an integer")
            out[key] = _from_tree(item, buffer, depth + 1, count)
        return out
    if tag == "f" and body in ("nan", "inf", "-inf"):
        return float(body)
    if tag == "b" and type(body) is int:
        return bytes(_span(buffer, body))
    if tag == "a" and isinstance(body, list) and len(body) == 3:
        return _array(body, buffer)
    raise MalformedPayload(f"an unknown or malformed tag {tag!r}")


def _array(body: list[Any], buffer: Callable[[int], memoryview]) -> np.ndarray:
    dtype_str, shape, index = body
    if not isinstance(dtype_str, str) or len(dtype_str) > 16:
        raise MalformedPayload("an array's dtype is not a short string")
    try:
        dtype = np.dtype(dtype_str)
    except (TypeError, ValueError):
        raise MalformedPayload(f"an array's dtype {dtype_str!r} is not one numpy knows") from None
    if dtype.kind not in _KINDS or dtype.fields is not None or dtype.subdtype is not None:
        raise MalformedPayload(f"an array of dtype {dtype_str!r} does not travel")
    if not (
        isinstance(shape, list)
        and len(shape) <= 32
        and all(type(n) is int and n >= 0 for n in shape)
    ):
        raise MalformedPayload("an array's shape is not a list of non-negative integers")
    if type(index) is not int:
        raise MalformedPayload("an array's span is not an index")
    raw = _span(buffer, index)
    if math.prod(shape) * dtype.itemsize != raw.nbytes:
        raise MalformedPayload(
            f"an array of shape {shape} and dtype {dtype_str} needs "
            f"{math.prod(shape) * dtype.itemsize} bytes, its span holds {raw.nbytes}"
        )
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def _span(buffer: Callable[[int], memoryview], index: int) -> memoryview:
    try:
        return buffer(index)
    except IndexError:
        raise MalformedPayload(f"the payload names span {index}, which is not there") from None
