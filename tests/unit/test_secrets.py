"""Unit tests for the secret loader.

The tests build temporary mounts, including the atomic-writer layout that the
deployment platform mounts (a timestamped directory, a ``..data`` symlink, and
one symlink per key). A failure message must name keys and the failure, and
never carry a value or a byte of a value.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from reactor_runtime import secrets
from reactor_runtime.secrets import EXPECTED_KEYS_ENV, load_expected_secrets

_GOOD = b"synth-value-3c6d6c03"
_IMAGE_DEFAULT = "synth-image-default-7712"


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


def _case_sensitive(root: Path) -> bool:
    probe = root / "CaSePrObE"
    probe.touch()
    try:
        return not (root / "caseprobe").exists()
    finally:
        probe.unlink()


def _skip_as_root() -> None:
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("permission bits do not bind root")


# -- The expected-names variable ---------------------------------------------


def test_an_absent_variable_touches_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the loader touched the file system")

    monkeypatch.setattr(secrets.os, "listdir", forbidden)
    monkeypatch.setattr(secrets.os, "lstat", forbidden)
    monkeypatch.setattr(secrets.os.path, "realpath", forbidden)

    assert load_expected_secrets({}, tmp_path) == {}


def test_the_loader_never_changes_the_environment(tmp_path: Path) -> None:
    os.environ["API_TOKEN"] = _IMAGE_DEFAULT
    snapshot = dict(os.environ)
    _write(tmp_path, "API_TOKEN", _GOOD)

    assert load_expected_secrets(_env(["API_TOKEN"]), tmp_path) == {"API_TOKEN": _GOOD.decode()}
    assert dict(os.environ) == snapshot


@pytest.mark.parametrize(
    ("raw", "failure"),
    [
        ("", "invalid JSON"),
        ("not json", "invalid JSON"),
        ('["API_TOKEN",]', "invalid JSON"),
        ("{}", "JSON array"),
        ("null", "JSON array"),
        ('"API_TOKEN"', "JSON array"),
        ("5", "JSON array"),
        ("[1]", "index 0 is not a string"),
        ('["A", null]', "index 1 is not a string"),
        ('[["A"]]', "index 0 is not a string"),
    ],
)
def test_a_malformed_variable_blocks_the_start(tmp_path: Path, raw: str, failure: str) -> None:
    with pytest.raises(SystemExit, match=failure):
        load_expected_secrets({EXPECTED_KEYS_ENV: raw}, tmp_path)


@pytest.mark.parametrize(
    "name", ["1A", "A-B", "A B", "a.b", "", "A\n", "\nA", chr(0xC4), chr(0xFF21) + "PI_TOKEN"]
)
def test_an_invalid_name_blocks_the_start(tmp_path: Path, name: str) -> None:
    with pytest.raises(SystemExit, match="is invalid") as raised:
        load_expected_secrets(_env([name]), tmp_path)

    assert "\n" not in str(raised.value)


def test_a_duplicate_name_blocks_the_start(tmp_path: Path) -> None:
    _write(tmp_path, "API_TOKEN", _GOOD)

    with pytest.raises(SystemExit, match="'API_TOKEN' more than once"):
        load_expected_secrets(_env(["API_TOKEN", "API_TOKEN"]), tmp_path)


def test_names_compare_case_exact(tmp_path: Path) -> None:
    _write(tmp_path, "api_token", _GOOD)

    with pytest.raises(SystemExit, match="missing 'API_TOKEN'; unexpected 'api_token'"):
        load_expected_secrets(_env(["API_TOKEN"]), tmp_path)


def test_case_distinct_files_are_two_secrets(tmp_path: Path) -> None:
    if not _case_sensitive(tmp_path):
        pytest.skip("the file system folds case")
    _write(tmp_path, "TOKEN", b"upper")
    _write(tmp_path, "token", b"lower")

    assert load_expected_secrets(_env(["TOKEN", "token"]), tmp_path) == {
        "TOKEN": "upper",
        "token": "lower",
    }


# -- The set contract ----------------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [
        b"synth-trailing-newline\n",
        b"synth-crlf\r\n",
        b"  synth-spaces  ",
        b"\xef\xbb\xbfsynth-bom",
        "synth-café-中".encode(),
        b"x" * 32768,
    ],
)
def test_the_exact_bytes_are_kept(tmp_path: Path, data: bytes) -> None:
    _write(tmp_path, "API_TOKEN", data)

    assert load_expected_secrets(_env(["API_TOKEN"]), tmp_path)["API_TOKEN"].encode() == data


@pytest.mark.parametrize("layout", ["absent", "empty", "bookkeeping"])
def test_an_empty_list_passes_without_payload(tmp_path: Path, layout: str) -> None:
    root = tmp_path / "secrets"
    if layout != "absent":
        root.mkdir()
    if layout == "bookkeeping":
        _mount_atomic_writer(root, {})

    assert load_expected_secrets(_env([]), root) == {}


def test_an_empty_list_with_a_payload_entry_blocks_the_start(tmp_path: Path) -> None:
    _write(tmp_path, "STRAY", _GOOD)

    with pytest.raises(SystemExit, match="unexpected 'STRAY'"):
        load_expected_secrets(_env([]), tmp_path)


def test_missing_and_unexpected_entries_are_named_together(tmp_path: Path) -> None:
    _mount_atomic_writer(tmp_path, {"B_KEY": _GOOD, "EXTRA": _GOOD, ".hidden": _GOOD})

    with pytest.raises(SystemExit) as raised:
        load_expected_secrets(_env(["A_KEY", "B_KEY", "C_KEY"]), tmp_path)

    assert str(raised.value).endswith("missing 'A_KEY', 'C_KEY'; unexpected '.hidden', 'EXTRA'")


def test_a_missing_mount_names_every_expected_key(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="missing secret files 'A_KEY', 'B_KEY'"):
        load_expected_secrets(_env(["B_KEY", "A_KEY"]), tmp_path / "absent")


def test_an_unexpected_entry_name_stays_one_line(tmp_path: Path) -> None:
    _write(tmp_path, "BAD\nlevel=error injected", _GOOD)

    with pytest.raises(SystemExit, match="unexpected") as raised:
        load_expected_secrets(_env([]), tmp_path)

    assert "\n" not in str(raised.value)


@pytest.mark.parametrize("names", [[], ["API_TOKEN"]])
def test_a_root_that_is_a_file_blocks_the_start(tmp_path: Path, names: list[str]) -> None:
    root = tmp_path / "secrets"
    root.write_bytes(_GOOD)

    with pytest.raises(SystemExit, match="is not a directory"):
        load_expected_secrets(_env(names), root)


def test_an_unreadable_root_blocks_the_start(tmp_path: Path) -> None:
    _skip_as_root()
    root = tmp_path / "secrets"
    root.mkdir()
    root.chmod(0)
    try:
        with pytest.raises(SystemExit, match="is unreadable"):
            load_expected_secrets(_env(["API_TOKEN"]), root)
    finally:
        root.chmod(0o700)


def test_a_directory_named_as_a_secret_blocks_the_start(tmp_path: Path) -> None:
    (tmp_path / "API_TOKEN").mkdir()

    with pytest.raises(SystemExit, match="'API_TOKEN' is not a regular file"):
        load_expected_secrets(_env(["API_TOKEN"]), tmp_path)
