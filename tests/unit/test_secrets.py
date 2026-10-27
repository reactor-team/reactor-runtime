"""Unit tests for the secret loader.

The tests build temporary mounts, including the atomic-writer layout that the
deployment platform mounts: a timestamped directory, a ``..data`` symlink to it,
and one symlink per key.
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import pytest

from reactor_runtime.secrets import load_expected_secrets

_GOOD = b"synth-value-3c6d6c03"


def _mount(root: Path, files: dict[str, bytes]) -> None:
    stamp = root / "..2026_10_13_04_00_00.123456789"
    stamp.mkdir(parents=True)
    (root / "..data").symlink_to(stamp.name)
    for name, data in files.items():
        (stamp / name).write_bytes(data)
        (root / name).symlink_to(f"..data/{name}")


@pytest.mark.parametrize("raw", ["not json", "{}", "[1]"])
def test_a_malformed_variable_blocks_the_start(tmp_path: Path, raw: str) -> None:
    with pytest.raises(SystemExit, match="must hold a JSON array of secret names"):
        load_expected_secrets(raw, tmp_path)


@pytest.mark.parametrize("name", ["1A", "A-B", "A\n"])
def test_an_invalid_name_blocks_the_start(tmp_path: Path, name: str) -> None:
    with pytest.raises(SystemExit, match="is invalid"):
        load_expected_secrets(json.dumps([name]), tmp_path)


def test_a_duplicate_name_blocks_the_start(tmp_path: Path) -> None:
    (tmp_path / "API_TOKEN").write_bytes(_GOOD)

    with pytest.raises(SystemExit, match="'API_TOKEN' more than once"):
        load_expected_secrets('["API_TOKEN", "API_TOKEN"]', tmp_path)


def test_the_files_must_match_the_names_exactly(tmp_path: Path) -> None:
    for name in ["B_KEY", "a_key", ".hidden"]:
        (tmp_path / name).write_bytes(_GOOD)

    with pytest.raises(SystemExit) as raised:
        load_expected_secrets('["A_KEY", "B_KEY"]', tmp_path)

    assert str(raised.value).endswith("missing ['A_KEY'], unexpected ['.hidden', 'a_key']")


@pytest.mark.parametrize("layout", ["absent", "empty", "bookkeeping"])
def test_an_empty_list_passes_without_payload(tmp_path: Path, layout: str) -> None:
    root = tmp_path / "secrets"
    if layout == "empty":
        root.mkdir()
    if layout == "bookkeeping":
        _mount(root, {})

    assert load_expected_secrets("[]", root) == {}


def test_an_empty_list_with_a_payload_file_blocks_the_start(tmp_path: Path) -> None:
    (tmp_path / "STRAY").write_bytes(_GOOD)

    with pytest.raises(SystemExit, match=r"unexpected \['STRAY'\]"):
        load_expected_secrets("[]", tmp_path)


@pytest.mark.parametrize("data", [b"  synth-spaces\r\n", b"x" * 32768])
def test_the_exact_bytes_are_kept(tmp_path: Path, data: bytes) -> None:
    _mount(tmp_path, {"API_TOKEN": data})

    assert load_expected_secrets('["API_TOKEN"]', tmp_path) == {"API_TOKEN": data.decode()}


def test_a_directory_named_as_a_secret_blocks_the_start(tmp_path: Path) -> None:
    (tmp_path / "API_TOKEN").mkdir()

    with pytest.raises(SystemExit, match="'API_TOKEN' is not a regular file"):
        load_expected_secrets('["API_TOKEN"]', tmp_path)


@pytest.mark.skipif(os.geteuid() == 0, reason="permission bits do not bind root")
@pytest.mark.parametrize("target", ["API_TOKEN", "."])
def test_an_unreadable_file_or_mount_blocks_the_start(tmp_path: Path, target: str) -> None:
    (tmp_path / "API_TOKEN").write_bytes(_GOOD)
    (tmp_path / target).chmod(0)
    try:
        with pytest.raises(SystemExit, match="unreadable"):
            load_expected_secrets('["API_TOKEN"]', tmp_path)
    finally:
        (tmp_path / target).chmod(0o700)


@pytest.mark.parametrize(
    ("data", "failure"),
    [
        (b"", "is empty"),
        (b"x" * 32769, "is larger than 32768 bytes"),
        (b"synth-nul-\x00-end", "contains a NUL byte"),
        (b"synth-notutf8-\xff", "is not valid UTF-8"),
    ],
)
def test_a_bad_value_blocks_the_start_with_a_names_only_message(
    tmp_path: Path, data: bytes, failure: str
) -> None:
    (tmp_path / "API_TOKEN").write_bytes(data)

    with pytest.raises(SystemExit, match=f"'API_TOKEN' {failure}") as raised:
        load_expected_secrets('["API_TOKEN"]', tmp_path)

    # A decode error quotes the bad byte and its position, so it must not show.
    formatted = "".join(traceback.format_exception(raised.value))
    assert "position" not in formatted
    needle = data.decode(errors="ignore")
    assert not needle or needle not in formatted


def test_a_symlink_that_escapes_the_mount_blocks_the_start(tmp_path: Path) -> None:
    root = tmp_path / "secrets"
    root.mkdir()
    # A sibling whose name starts with the mount's name is still outside it.
    (tmp_path / "secrets2").mkdir()
    (tmp_path / "secrets2" / "payload").write_bytes(_GOOD)
    (root / "API_TOKEN").symlink_to("../secrets2/payload")

    with pytest.raises(SystemExit, match="'API_TOKEN' is a symlink that escapes the mount"):
        load_expected_secrets('["API_TOKEN"]', root)
