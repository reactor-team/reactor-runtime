"""Failure-path tests for the secret loader: value rules and value hygiene.

A failure must name keys and the failure, and never carry a value, a byte of a
value, or an exception chain that does. A failed load keeps no value alive in
its traceback.
"""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import pytest

from reactor_runtime import secrets
from reactor_runtime.secrets import EXPECTED_KEYS_ENV, load_expected_secrets

_GOOD = b"synth-value-3c6d6c03"


def _env(names: list[str]) -> dict[str, str]:
    return {EXPECTED_KEYS_ENV: json.dumps(names)}


def _write(root: Path, name: str, data: bytes) -> None:
    (root / name).write_bytes(data)


def _mount(root: Path, files: dict[str, bytes]) -> None:
    for name, data in files.items():
        _write(root, name, data)


def _assert_no_value_leak(
    exc: SystemExit,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    blob: bytes,
) -> None:
    """Assert that *blob*, and every byte of it in hex form, appears on no surface."""
    captured = capsys.readouterr()
    surfaces = [str(exc), repr(exc), caplog.text, captured.out, captured.err]
    surfaces += [str(exc.__cause__), str(exc.__context__)]
    needle = blob.decode("utf-8", errors="ignore")
    for text in surfaces:
        assert not needle or needle not in text
        for byte in set(blob):
            assert f"0x{byte:02x}" not in text
            assert f"\\x{byte:02x}" not in text


def _assert_no_value_in_loader_frames(exc: SystemExit) -> None:
    """Assert that no loader frame on the traceback still holds a ``synth-`` value."""
    held: list[str] = []
    frame = exc.__traceback__
    while frame is not None:
        if frame.tb_frame.f_globals.get("__name__") == secrets.__name__:
            held += [repr(value) for value in frame.tb_frame.f_locals.values()]
        frame = frame.tb_next
    assert held
    assert [text for text in held if "synth-" in text] == []


@pytest.mark.parametrize(
    ("data", "failure"),
    [
        (b"", "is empty"),
        (b"x" * 32769, "is larger than 32768 bytes"),
        (b"synth-nul-\x00-end", "contains a NUL byte"),
        (b"synth-tail-notutf8-\xff", "is not valid UTF-8"),
        (b"\xed\xa0\x80-synth-surrogate", "is not valid UTF-8"),
        (b"\xc3", "is not valid UTF-8"),
    ],
)
def test_a_bad_value_in_a_later_file_blocks_with_a_names_only_message(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    data: bytes,
    failure: str,
) -> None:
    _mount(tmp_path, {"AAA_FIRST": _GOOD, "ZZZ_LAST": data})

    with pytest.raises(SystemExit, match=f"'ZZZ_LAST' {failure}") as raised:
        load_expected_secrets(_env(["AAA_FIRST", "ZZZ_LAST"]), tmp_path)

    assert raised.value.__cause__ is None
    assert "position" not in str(raised.value)
    assert not isinstance(raised.value.__context__, UnicodeError)
    _assert_no_value_leak(raised.value, caplog, capsys, data)


@pytest.mark.parametrize(
    "data",
    [b"synth-bad-\x00-nul", b"synth-bad-\xff-utf8", b"synth-bad-" + b"x" * 32768],
    ids=["nul", "utf8", "oversize"],
)
def test_a_failed_load_keeps_no_value_in_its_traceback(tmp_path: Path, data: bytes) -> None:
    _mount(tmp_path, {"AAA_FIRST": _GOOD, "ZZZ_LAST": data})

    with pytest.raises(SystemExit) as raised:
        load_expected_secrets(_env(["AAA_FIRST", "ZZZ_LAST"]), tmp_path)

    _assert_no_value_in_loader_frames(raised.value)


def test_a_failed_metadata_check_names_the_key_and_keeps_no_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mount(tmp_path, {"AAA_FIRST": _GOOD, "ZZZ_LAST": _GOOD})
    fstat = os.fstat
    calls: list[int] = []

    def failing_fstat(descriptor: int) -> os.stat_result:
        calls.append(descriptor)
        if len(calls) == 2:
            raise OSError(errno.EIO, "I/O error")
        return fstat(descriptor)

    monkeypatch.setattr(secrets.os, "fstat", failing_fstat)

    with pytest.raises(SystemExit, match="'ZZZ_LAST' is unreadable") as raised:
        load_expected_secrets(_env(["AAA_FIRST", "ZZZ_LAST"]), tmp_path)

    _assert_no_value_in_loader_frames(raised.value)
