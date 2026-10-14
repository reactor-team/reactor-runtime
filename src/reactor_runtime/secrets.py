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

import contextlib
import errno
import json
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

EXPECTED_KEYS_ENV = "REACTOR_SECRETS_EXPECTED_KEYS"
SECRETS_ROOT = Path("/run/secrets")

_MAX_BYTES = 32768
_NAME_MAX = 255
_NAME_PATTERN = "[A-Za-z_][A-Za-z0-9_]*"
_NAME_RE = re.compile(_NAME_PATTERN)
_DISPLAY_MAX = 64
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)

_FAILED = object()


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
    root_fd = _open_root(root, expected)
    if root_fd is None:
        return {}
    try:
        # ``..data`` and the timestamped directory beside it are mount bookkeeping.
        logical = {name for name in _listing(root_fd, root) if not name.startswith("..")}
        missing = sorted(set(expected) - logical)
        unexpected = sorted(logical - set(expected))
        if missing or unexpected:
            problems = []
            if missing:
                problems.append(f"missing {', '.join(_display(name) for name in missing)}")
            if unexpected:
                problems.append(f"unexpected {', '.join(_display(name) for name in unexpected)}")
            raise SystemExit(
                f"the secret files in {root} do not match {EXPECTED_KEYS_ENV}: "
                f"{'; '.join(problems)}"
            )
        return _read_secrets(root_fd, sorted(expected))
    finally:
        os.close(root_fd)


def _parse_expected(raw: str) -> list[str]:
    """Return the expected names from the raw variable, or block the start."""
    parsed: Any = _FAILED
    # ``ValueError`` covers a decode error and an over-long number.
    with contextlib.suppress(ValueError, RecursionError):
        parsed = json.loads(raw)
    if parsed is _FAILED:
        raise SystemExit(f"{EXPECTED_KEYS_ENV} holds invalid JSON; expected a JSON array of names")
    if not isinstance(parsed, list):
        raise SystemExit(f"{EXPECTED_KEYS_ENV} must hold a JSON array of secret names")
    seen: set[str] = set()
    names: list[str] = []
    for index, item in enumerate(parsed):
        if not isinstance(item, str):
            raise SystemExit(f"{EXPECTED_KEYS_ENV} entry at index {index} is not a string")
        if len(item) > _NAME_MAX:
            raise SystemExit(f"secret name {_display(item)} is longer than {_NAME_MAX} characters")
        if _NAME_RE.fullmatch(item) is None:
            raise SystemExit(
                f"secret name {_display(item)} is invalid; names match {_NAME_PATTERN}"
            )
        if item in seen:
            raise SystemExit(f"{EXPECTED_KEYS_ENV} lists {_display(item)} more than once")
        seen.add(item)
        names.append(item)
    return names


def _open_root(root: Path, expected: list[str]) -> int | None:
    """Open the mount directory, or return ``None`` when it is absent and allowed to be."""
    try:
        return os.open(root, os.O_RDONLY | _O_DIRECTORY)
    except FileNotFoundError:
        if expected:
            shown = ", ".join(_display(name) for name in sorted(expected))
            raise SystemExit(f"missing secret files {shown}: {root} does not exist") from None
        return None
    except NotADirectoryError:
        raise SystemExit(f"the secret directory {root} is not a directory") from None
    except OSError:
        raise SystemExit(f"the secret directory {root} is unreadable") from None


def _listing(root_fd: int, root: Path) -> list[str]:
    """Return the raw entries of the open mount directory, or block the start."""
    try:
        return os.listdir(root_fd)
    except OSError:
        raise SystemExit(f"the secret directory {root} is unreadable") from None


def _read_secrets(root_fd: int, names: list[str]) -> dict[str, str]:
    """Read every expected file, keeping no value alive when one of them fails."""
    values: dict[str, str] = {}
    name = ""
    try:
        for name in names:
            values[name] = _read_secret(root_fd, name)
    except OSError:
        values.clear()
        raise SystemExit(f"secret file {_display(name)} is unreadable") from None
    except BaseException:
        values.clear()
        raise
    return values


def _read_secret(root_fd: int, name: str) -> str:
    """Read and validate one expected file, returning its decoded value.

    Raises:
        SystemExit: If the entry is unusable or its value fails a value rule.
    """
    try:
        info = os.lstat(name, dir_fd=root_fd)
    except FileNotFoundError:
        raise SystemExit(f"secret file {_display(name)} is missing") from None
    except OSError:
        raise SystemExit(f"secret file {_display(name)} is unreadable") from None
    if not stat.S_ISREG(info.st_mode):
        # Refused before any open, so a FIFO or a device can never block the boot.
        raise SystemExit(f"secret file {_display(name)} is not a regular file")
    data = _open_and_read(root_fd, name)
    has_nul = b"\x00" in data
    text = None if has_nul else _decode(data)
    # A failure must not keep the value alive in this frame's traceback.
    del data
    if text is None:
        failure = "contains a NUL byte" if has_nul else "is not valid UTF-8"
        raise SystemExit(f"secret file {_display(name)} {failure}")
    return text


def _open_and_read(root_fd: int, name: str) -> bytes:
    """Open the regular file *name* in the open mount directory and read it fully.

    ``O_NOFOLLOW`` refuses an entry that became a symlink after the check.
    ``O_NONBLOCK`` keeps the open instant for a special file, and ``fstat``
    confirms a regular file.

    Raises:
        SystemExit: If the file is missing, not a regular file, unreadable,
            empty, or larger than the value limit.
    """
    try:
        descriptor = os.open(name, os.O_RDONLY | _O_NONBLOCK | _O_NOFOLLOW, dir_fd=root_fd)
    except FileNotFoundError:
        raise SystemExit(f"secret file {_display(name)} is missing") from None
    except OSError as error:
        kind = "not a regular file" if error.errno == errno.ELOOP else "unreadable"
        raise SystemExit(f"secret file {_display(name)} is {kind}") from None
    try:
        mode: int | None = None
        with contextlib.suppress(OSError):
            mode = os.fstat(descriptor).st_mode
        if mode is None:
            raise SystemExit(f"secret file {_display(name)} is unreadable")
        if not stat.S_ISREG(mode):
            raise SystemExit(f"secret file {_display(name)} is not a regular file")
        return _read_all(descriptor, name)
    finally:
        os.close(descriptor)


def _read_all(descriptor: int, name: str) -> bytes:
    """Read the descriptor to the end, enforcing the size rules.

    Raises:
        SystemExit: If the value is empty, unreadable, or over the limit.
    """
    buffer = bytearray()
    failure: str | None = None
    try:
        while len(buffer) <= _MAX_BYTES:
            chunk = os.read(descriptor, _MAX_BYTES + 1 - len(buffer))
            if not chunk:
                break
            buffer += chunk
    except OSError:
        failure = "is unreadable"
    chunk = b""
    if failure is None and len(buffer) > _MAX_BYTES:
        failure = f"is larger than {_MAX_BYTES} bytes"
    elif failure is None and not buffer:
        failure = "is empty"
    data = bytes(buffer) if failure is None else b""
    # A failure must not keep the value alive in this frame's traceback.
    buffer.clear()
    if failure is not None:
        raise SystemExit(f"secret file {_display(name)} {failure}")
    return data


def _decode(data: bytes) -> str | None:
    """Decode *data* as strict UTF-8, or return ``None`` when it is not valid.

    The caller raises outside the ``except`` block, so the ``SystemExit`` has
    no ``__cause__`` or ``__context__``. A ``UnicodeDecodeError`` quotes the
    bad byte, and a value byte must never reach a failure message.
    """
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _display(name: str) -> str:
    """Render an entry or key name as a bounded, single-line literal."""
    shown = name if len(name) <= _DISPLAY_MAX else name[:_DISPLAY_MAX] + "..."
    return repr(shown)
