import json
import math
from typing import Any

import pytest

from reactor_runtime.protocol.common import dict_to_struct, struct_to_dict


def _round_trip(payload: dict[Any, Any]) -> dict[str, Any]:
    return struct_to_dict(dict_to_struct(payload))


def test_string_keys_pass_through_unchanged() -> None:
    payload = {"prompt": "hi", "strength": 0.5, "nested": {"a": [1, 2]}}
    assert _round_trip(payload) == payload


def test_int_keys_coerce_to_strings() -> None:
    assert _round_trip({1: "a", 42: "b"}) == {"1": "a", "42": "b"}


def test_float_keys_coerce_to_strings() -> None:
    assert _round_trip({1.5: "a", -0.25: "b"}) == {"1.5": "a", "-0.25": "b"}


def test_bool_and_none_keys_coerce_like_json() -> None:
    assert _round_trip({True: "t", False: "f", None: "n"}) == {
        "true": "t",
        "false": "f",
        "null": "n",
    }


def test_non_finite_float_keys_coerce_like_json() -> None:
    assert _round_trip({math.inf: 1, -math.inf: 2, math.nan: 3}) == {
        "Infinity": 1,
        "-Infinity": 2,
        "NaN": 3,
    }


def test_nested_dict_keys_coerce() -> None:
    payload = {"schedule": {3: "sunset", 7: {"9": "dawn", 11: "noon"}}}
    assert _round_trip(payload) == {"schedule": {"3": "sunset", "7": {"9": "dawn", "11": "noon"}}}


def test_dict_keys_inside_lists_coerce() -> None:
    payload = {"steps": [{1: "a"}, [{2: "b"}], "plain"]}
    assert _round_trip(payload) == {"steps": [{"1": "a"}, [{"2": "b"}], "plain"]}


def test_colliding_keys_last_iterated_wins() -> None:
    assert _round_trip({"1": "text", 1: "int"}) == {"1": "int"}
    assert _round_trip({1: "int", "1": "text"}) == {"1": "text"}


def test_coercion_matches_json_dumps() -> None:
    payload: dict[Any, Any] = {
        "outer": {1: "a", 2.5: "b", False: "c", None: "d"},
        "list": [{10: "x"}],
    }
    assert _round_trip(payload) == json.loads(json.dumps(payload))


def test_unsupported_key_type_names_the_key() -> None:
    with pytest.raises(TypeError, match=r"tuple: \(1, 2\)"):
        dict_to_struct({"outer": {(1, 2): "a"}})
