"""Failure-path tests for the secret loader: value rules, value hygiene, and races.

A failure must name keys and the failure, and never carry a value, a byte of a
value, or an exception chain that does. A failed load keeps no value alive in
its traceback. A path that changes after the check never leads the read out of
the mount.
"""

from __future__ import annotations

import errno
import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from reactor_runtime import secrets
from reactor_runtime.secrets import EXPECTED_KEYS_ENV, export_secrets, load_expected_secrets

_GOOD = b"synth-value-3c6d6c03"

_POSIX_ONLY = pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="POSIX file types")


@pytest.fixture(autouse=True)
def _clean_environ() -> Iterator[None]:
    """Restore the whole process environment, because the export writes it directly."""
    snapshot = dict(os.environ)
    os.environ.pop(EXPECTED_KEYS_ENV, None)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(snapshot)


def _env(names: list[str]) -> dict[str, str]:
    return {EXPECTED_KEYS_ENV: json.dumps(names)}


def _write(root: Path, name: str, data: bytes) -> None:
    (root / name).write_bytes(data)


def _mount_atomic_writer(root: Path, files: dict[str, bytes]) -> None:
    stamp = root / "..2026_10_13_04_00_00.123456789"
    stamp.mkdir()
    for name, data in files.items():
        _write(stamp, name, data)
    (root / "..data").symlink_to(stamp.name)
    for name in files:
        (root / name).symlink_to(f"..data/{name}")


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
    _mount_atomic_writer(tmp_path, {"AAA_FIRST": _GOOD, "ZZZ_LAST": data})

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
    _mount_atomic_writer(tmp_path, {"AAA_FIRST": _GOOD, "ZZZ_LAST": data})

    with pytest.raises(SystemExit) as raised:
        load_expected_secrets(_env(["AAA_FIRST", "ZZZ_LAST"]), tmp_path)

    _assert_no_value_in_loader_frames(raised.value)


def test_a_failed_metadata_check_names_the_key_and_keeps_no_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mount_atomic_writer(tmp_path, {"AAA_FIRST": _GOOD, "ZZZ_LAST": _GOOD})
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


@_POSIX_ONLY
def test_a_symlink_swapped_after_the_check_blocks_the_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path / "..store"
    store.mkdir()
    _write(store, "payload", _GOOD)
    (tmp_path / "API_TOKEN").symlink_to("..store/payload")
    (store / "swapped").symlink_to("payload")
    realpath = os.path.realpath

    def stale(path: Any) -> str:
        # The check sees the payload. The open then meets a symlink in its place.
        resolved = realpath(path)
        return str(store / "swapped") if Path(resolved).name == "payload" else resolved

    monkeypatch.setattr(secrets.os.path, "realpath", stale)

    with pytest.raises(SystemExit, match="not a regular file"):
        load_expected_secrets(_env(["API_TOKEN"]), tmp_path)


@_POSIX_ONLY
def test_a_directory_swapped_after_the_check_blocks_the_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "secrets"
    store = root / "..store"
    store.mkdir(parents=True)
    _write(store, "payload", _GOOD)
    (root / "API_TOKEN").symlink_to("..store/payload")
    outside = tmp_path / "outside"
    outside.mkdir()
    _write(outside, "payload", b"synth-outside")
    realpath = os.path.realpath

    def swap_after_check(path: Any) -> str:
        # The check sees the mount. The read then meets a directory link to the outside.
        resolved = realpath(path)
        if Path(resolved).name == "payload" and not (root / "..old").exists():
            store.rename(root / "..old")
            store.symlink_to(outside, target_is_directory=True)
        return resolved

    monkeypatch.setattr(secrets.os.path, "realpath", swap_after_check)

    with pytest.raises(SystemExit, match="'API_TOKEN' is not a regular file inside the mount"):
        load_expected_secrets(_env(["API_TOKEN"]), root)


@pytest.mark.parametrize("names", [[], ["API_TOKEN"]])
def test_export_logs_key_names_and_a_count_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    names: list[str],
) -> None:
    for name in names:
        _write(tmp_path, name, _GOOD)
    monkeypatch.setattr(secrets, "SECRETS_ROOT", tmp_path)
    monkeypatch.setenv(EXPECTED_KEYS_ENV, json.dumps(names))
    caplog.set_level(logging.INFO)

    export_secrets()

    [record] = [record for record in caplog.records if record.name == secrets.__name__]
    assert getattr(record, "reactor_fields", None) == {"keys": names, "count": len(names)}
    assert _GOOD.decode() not in caplog.text
