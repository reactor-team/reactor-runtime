"""Managed secrets that a deployment platform mounts as files.

A deployment platform mounts one file per managed secret under a well-known
directory (:data:`SECRETS_ROOT`) and names the files it mounted in the reserved
environment variable ``REACTOR_SECRETS_EXPECTED_KEYS``. The value is a strict
JSON array of names, and ``[]`` when the deployment has no secrets.
:func:`load_expected_secrets` checks that the mounted files match the expected
set exactly, validates every value, and returns the values by name.

When the variable is absent, the loader does nothing and touches no file.

Every failure raises :class:`SystemExit` with a message that names the affected
keys or entries and the failure. A message never contains a value or a byte of
a value.
"""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path

EXPECTED_KEYS_ENV = "REACTOR_SECRETS_EXPECTED_KEYS"
SECRETS_ROOT = Path("/run/secrets")

_MAX_BYTES = 32768
_NAME_PATTERN = "[A-Za-z_][A-Za-z0-9_]*"
_NAME_RE = re.compile(_NAME_PATTERN)


def load_expected_secrets(env: Mapping[str, str], root: Path) -> dict[str, str]:
    """Load the managed secrets that *env* names from the files under *root*.

    This function reads files and returns the decoded values. It never
    changes ``os.environ``. When *env* has no ``REACTOR_SECRETS_EXPECTED_KEYS``,
    it returns an empty mapping and touches no file.

    Args:
        env: The environment to read ``REACTOR_SECRETS_EXPECTED_KEYS`` from.
        root: The directory the platform mounted the secret files under.

    Returns:
        The decoded value of every expected secret, keyed by name. A value is
        the exact file content: nothing is stripped or added.

    Raises:
        SystemExit: If the variable holds invalid JSON or an invalid name
            list, if the mounted entries do not match the expected set
            exactly, if an entry is not a usable regular file, or if a value
            is empty, larger than 32 KiB, not valid UTF-8, or contains a NUL
            byte. The message names the keys and the failure, never a value.
    """
    raw = env.get(EXPECTED_KEYS_ENV)
    if raw is None:
        return {}
    expected = _parse_expected(raw)
    # ``..data`` and the timestamped directory beside it are mount bookkeeping.
    logical = {name for name in _listing(root, expected) if not name.startswith("..")}
    missing = sorted(set(expected) - logical)
    unexpected = sorted(logical - set(expected))
    if missing or unexpected:
        problems = []
        if missing:
            problems.append(f"missing {', '.join(repr(name) for name in missing)}")
        if unexpected:
            problems.append(f"unexpected {', '.join(repr(name) for name in unexpected)}")
        raise SystemExit(
            f"the secret files in {root} do not match {EXPECTED_KEYS_ENV}: {'; '.join(problems)}"
        )
    return _read_secrets(root, sorted(expected))


def _parse_expected(raw: str) -> list[str]:
    """Return the expected names from the raw variable, or block the start."""
    try:
        parsed = json.loads(raw)
    except ValueError:
        raise SystemExit(
            f"{EXPECTED_KEYS_ENV} holds invalid JSON; expected a JSON array of names"
        ) from None
    if not isinstance(parsed, list):
        raise SystemExit(f"{EXPECTED_KEYS_ENV} must hold a JSON array of secret names")
    seen: set[str] = set()
    names: list[str] = []
    for index, item in enumerate(parsed):
        if not isinstance(item, str):
            raise SystemExit(f"{EXPECTED_KEYS_ENV} entry at index {index} is not a string")
        if _NAME_RE.fullmatch(item) is None:
            raise SystemExit(f"secret name {item!r} is invalid; names match {_NAME_PATTERN}")
        if item in seen:
            raise SystemExit(f"{EXPECTED_KEYS_ENV} lists {item!r} more than once")
        seen.add(item)
        names.append(item)
    return names


def _listing(root: Path, expected: list[str]) -> list[str]:
    """Return the raw entries of the mount directory, or block the start.

    An absent directory has no entries when no secret is expected.
    """
    try:
        return os.listdir(root)
    except FileNotFoundError:
        if expected:
            shown = ", ".join(repr(name) for name in sorted(expected))
            raise SystemExit(f"missing secret files {shown}: {root} does not exist") from None
        return []
    except NotADirectoryError:
        raise SystemExit(f"the secret directory {root} is not a directory") from None
    except OSError:
        raise SystemExit(f"the secret directory {root} is unreadable") from None


def _read_secrets(root: Path, names: list[str]) -> dict[str, str]:
    """Read every expected file."""
    return {name: _read_secret(root, name) for name in names}


def _read_secret(root: Path, name: str) -> str:
    """Read and validate one expected file, returning its decoded value.

    Raises:
        SystemExit: If the entry is unusable or its value fails a value rule.
    """
    path = os.path.join(root, name)
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        raise SystemExit(f"secret file {name!r} is missing") from None
    except OSError:
        raise SystemExit(f"secret file {name!r} is unreadable") from None
    if not stat.S_ISREG(info.st_mode):
        # Refused before any open, so a FIFO or a device can never block the boot.
        raise SystemExit(f"secret file {name!r} is not a regular file")
    data = _read_all(path, name)
    if b"\x00" in data:
        raise SystemExit(f"secret file {name!r} contains a NUL byte")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise SystemExit(f"secret file {name!r} is not valid UTF-8") from None


def _read_all(path: str, name: str) -> bytes:
    """Read the regular file at *path*, enforcing the size rules.

    Raises:
        SystemExit: If the file is unreadable, empty, or over the limit.
    """
    try:
        with open(path, "rb") as file:
            data = file.read(_MAX_BYTES + 1)
    except OSError:
        raise SystemExit(f"secret file {name!r} is unreadable") from None
    if len(data) > _MAX_BYTES:
        raise SystemExit(f"secret file {name!r} is larger than {_MAX_BYTES} bytes")
    if not data:
        raise SystemExit(f"secret file {name!r} is empty")
    return data
