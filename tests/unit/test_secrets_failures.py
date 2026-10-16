"""Failure-path tests for the secret loader: value rules and value hygiene.

A failure must name keys and the failure, and never carry a value or a byte of
a value.
"""

from __future__ import annotations

import json
import logging
import os
import traceback
from collections.abc import Iterator
from pathlib import Path

import pytest

from reactor_runtime import secrets
from reactor_runtime.secrets import EXPECTED_KEYS_ENV, export_secrets, load_expected_secrets

_GOOD = b"synth-value-3c6d6c03"


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
    """Assert that *blob* appears on no surface, the formatted traceback included."""
    captured = capsys.readouterr()
    formatted = "".join(traceback.format_exception(exc))
    needle = blob.decode("utf-8", errors="ignore")
    for text in [str(exc), repr(exc), caplog.text, captured.out, captured.err, formatted]:
        assert not needle or needle not in text


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

    # A decode error quotes the bad byte and its position, so it must not show.
    assert "position" not in "".join(traceback.format_exception(raised.value))
    _assert_no_value_leak(raised.value, caplog, capsys, data)


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
