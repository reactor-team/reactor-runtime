"""Managed secrets, loaded into the environment before the model module imports.

A deployment platform mounts one file per secret under :data:`SECRETS_ROOT` and
names the files in ``REACTOR_SECRETS_EXPECTED_KEYS``, a JSON array of names that
is ``[]`` when there are none. :func:`export_secrets` checks that the files match
the names exactly and validates every value before it sets any value in the
process environment. Every entry whose name starts with ``..`` is mount
bookkeeping and is skipped. A mounted value replaces a value that the image set
under the same name. A name that the runtime reads for its own configuration,
any ``REACTOR_`` name or one in :data:`RESERVED_NAMES`, is refused. When the
variable is absent, nothing happens.

Every failure raises :class:`SystemExit` with a message that names keys and the
failure, never a value.
"""

from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path

from reactor_runtime import log

EXPECTED_KEYS_ENV = "REACTOR_SECRETS_EXPECTED_KEYS"
SECRETS_ROOT = Path("/run/secrets")

RESERVED_PREFIX = "REACTOR_"
"""Names under this prefix configure the runtime and cannot be secret names."""

RESERVED_NAMES = frozenset(
    {
        "EXIT_TIMEOUT_SECONDS",
        "HOST",
        "ICE_TRANSPORT_POLICY",
        "LOCAL_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "ORPHAN_TIMEOUT_SECONDS",
        "PORT",
        "RANK",
        "SIGTERM_GRACE_PERIOD",
        "STUN_SERVERS",
        "TURN_SERVERS",
        "WEBRTC_BWE_INITIAL_KBPS",
        "WEBRTC_BWE_MAX_KBPS",
        "WEBRTC_BWE_MIN_KBPS",
        "WEBRTC_CLIENT_PING_TIMEOUT_SECONDS",
        "WEBRTC_PORT_RANGE",
        "WEBRTC_SENDER_MAX_KBPS",
        "WEBRTC_SENDER_MIN_KBPS",
        "WEBRTC_VIDEO_CODECS",
        "WORLD_SIZE",
    }
)
"""Names outside ``REACTOR_`` that the runtime reads or sets for its own configuration.

A secret under one of them would replace that setting, so the loader refuses it.
"""

_MAX_BYTES = 32768
_NAME_PATTERN = "[A-Za-z_][A-Za-z0-9_]*"
_NAME_RE = re.compile(_NAME_PATTERN)

logger = log.get_logger(__name__)


def load_expected_secrets(raw: str, root: Path) -> dict[str, str]:
    """Read and validate the secret files that *raw* names under *root*.

    Every entry under *root* whose name starts with ``..`` is mount bookkeeping
    and is skipped.

    Args:
        raw: The value of ``REACTOR_SECRETS_EXPECTED_KEYS``.
        root: The directory that holds the secret files.

    Returns:
        The exact text of every expected file, keyed by name.

    Raises:
        SystemExit: If *raw* is not a JSON array of unique valid names, if a
            name is reserved for the runtime's own configuration (see
            :data:`RESERVED_PREFIX` and :data:`RESERVED_NAMES`), if the
            files under *root* do not match it exactly, or if a file is not a
            readable regular file inside *root* that holds 1 to 32768 bytes of
            UTF-8 without a NUL byte.
    """
    try:
        expected = json.loads(raw)
    except ValueError:
        expected = None
    if not isinstance(expected, list) or not all(isinstance(name, str) for name in expected):
        raise SystemExit(f"{EXPECTED_KEYS_ENV} must hold a JSON array of secret names")
    for name in expected:
        if _NAME_RE.fullmatch(name) is None:
            raise SystemExit(f"secret name {name!r} is invalid; names match {_NAME_PATTERN}")
        if name.startswith(RESERVED_PREFIX) or name in RESERVED_NAMES:
            raise SystemExit(
                f"secret name {name!r} is reserved for the runtime's own configuration"
            )
        if expected.count(name) > 1:
            raise SystemExit(f"{EXPECTED_KEYS_ENV} lists {name!r} more than once")
    try:
        entries = os.listdir(root)
    except FileNotFoundError:
        entries = []
    except OSError:
        raise SystemExit(f"the secret directory {root} is unreadable") from None
    # Every ``..`` entry is mount bookkeeping, such as ``..data`` and its timestamped directory.
    present = {entry for entry in entries if not entry.startswith("..")}
    if present != set(expected):
        missing = sorted(set(expected) - present)
        unexpected = sorted(present - set(expected))
        raise SystemExit(
            f"the secret files in {root} do not match {EXPECTED_KEYS_ENV}: "
            f"missing {missing}, unexpected {unexpected}"
        )
    real_root = os.path.realpath(root)
    return {name: _read_secret(real_root, name) for name in sorted(expected)}


def export_secrets() -> None:
    """Load the mounted secrets and set them in the process environment.

    Does nothing when ``REACTOR_SECRETS_EXPECTED_KEYS`` is absent. Each value
    goes in through ``os.environb`` as the exact file bytes, whatever the locale.

    Raises:
        SystemExit: If a check in :func:`load_expected_secrets` fails. The
            environment is then unchanged.
    """
    raw = os.environ.get(EXPECTED_KEYS_ENV)
    if raw is None:
        return
    values = load_expected_secrets(raw, SECRETS_ROOT)
    for name, text in values.items():
        os.environb[name.encode()] = text.encode()
    logger.info("loaded secret files", keys=sorted(values), count=len(values))


def _read_secret(real_root: str, name: str) -> str:
    path = os.path.realpath(os.path.join(real_root, name))
    if os.path.commonpath((real_root, path)) != real_root:
        raise SystemExit(f"secret file {name!r} is a symlink that escapes the mount")
    try:
        info = os.stat(path)
    except OSError:
        raise SystemExit(f"secret file {name!r} is unreadable") from None
    if not stat.S_ISREG(info.st_mode):
        # Refused before any open, so a FIFO or a device can never block the boot.
        raise SystemExit(f"secret file {name!r} is not a regular file")
    try:
        with open(path, "rb") as file:
            data = file.read(_MAX_BYTES + 1)
    except OSError:
        raise SystemExit(f"secret file {name!r} is unreadable") from None
    if len(data) > _MAX_BYTES:
        raise SystemExit(f"secret file {name!r} is larger than {_MAX_BYTES} bytes")
    if not data:
        raise SystemExit(f"secret file {name!r} is empty")
    if b"\x00" in data:
        raise SystemExit(f"secret file {name!r} contains a NUL byte")
    try:
        return data.decode()
    except UnicodeDecodeError:
        raise SystemExit(f"secret file {name!r} is not valid UTF-8") from None
