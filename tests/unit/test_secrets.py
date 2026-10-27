"""Unit tests for the secret loader.

The tests build temporary mounts, including the atomic-writer layout that the
deployment platform mounts: a timestamped directory, a ``..data`` symlink to it,
and one symlink per key.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import traceback
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

import pytest

from reactor_runtime import secrets
from reactor_runtime.secrets import EXPECTED_KEYS_ENV, export_secrets, load_expected_secrets

_GOOD = b"synth-value-3c6d6c03"
_IMAGE_DEFAULT = "synth-image-default-7712"


@pytest.fixture(autouse=True)
def _clean_environ() -> Iterator[None]:
    """Restore the environment, because the export writes ``os.environb`` directly."""
    with mock.patch.dict(os.environ):
        os.environ.pop(EXPECTED_KEYS_ENV, None)
        yield


def _mount(root: Path, files: dict[str, bytes]) -> None:
    stamp = root / "..2026_10_13_04_00_00.123456789"
    stamp.mkdir(parents=True)
    (root / "..data").symlink_to(stamp.name)
    for name, data in files.items():
        (stamp / name).write_bytes(data)
        (root / name).symlink_to(f"..data/{name}")


def _export(root: Path, names: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(secrets, "SECRETS_ROOT", root)
    monkeypatch.setenv(EXPECTED_KEYS_ENV, json.dumps(names))
    export_secrets()


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


def test_an_absent_variable_touches_no_file(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object) -> None:
        raise AssertionError("the loader touched the file system")

    monkeypatch.setattr(os, "listdir", forbidden)
    snapshot = dict(os.environ)

    export_secrets()

    assert dict(os.environ) == snapshot


def test_export_sets_the_exact_bytes_over_an_image_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    data = "synth-café\n".encode()
    _mount(tmp_path, {"API_TOKEN": data})
    os.environ["API_TOKEN"] = _IMAGE_DEFAULT
    caplog.set_level(logging.INFO)

    _export(tmp_path, ["API_TOKEN"], monkeypatch)

    assert os.environ["API_TOKEN"] == data.decode()
    # A rank process of the distributed runner inherits the environment the same way.
    probe = "import os, sys; sys.stdout.buffer.write(os.environb[b'API_TOKEN'])"
    child = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, check=True, timeout=60
    )
    assert child.stdout == data
    [record] = [record for record in caplog.records if record.name == secrets.__name__]
    assert getattr(record, "reactor_fields", None) == {"keys": ["API_TOKEN"], "count": 1}


def test_an_image_default_never_masks_a_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    os.environ["API_TOKEN"] = _IMAGE_DEFAULT

    with pytest.raises(SystemExit, match=r"missing \['API_TOKEN'\]"):
        _export(tmp_path / "absent", ["API_TOKEN"], monkeypatch)

    assert os.environ["API_TOKEN"] == _IMAGE_DEFAULT


def test_a_bad_later_file_leaves_the_environment_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mount(tmp_path, {"AAA_FIRST": _GOOD, "ZZZ_LAST": b"synth\x00"})
    os.environ["AAA_FIRST"] = _IMAGE_DEFAULT

    with pytest.raises(SystemExit, match="'ZZZ_LAST' contains a NUL byte"):
        _export(tmp_path, ["AAA_FIRST", "ZZZ_LAST"], monkeypatch)

    assert os.environ["AAA_FIRST"] == _IMAGE_DEFAULT
    assert "ZZZ_LAST" not in os.environ
