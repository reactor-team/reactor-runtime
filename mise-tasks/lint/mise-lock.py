#!/usr/bin/env -S uv run --python 3.12
# fmt: off
#MISE description="[Quality] Verify mise.lock is complete against mise.toml"
# fmt: on
"""Verify mise.lock is complete against mise.toml.

For every tool in mise.toml [tools], assert that mise.lock has a matching
[[tools.<name>]] entry (version-prefix match) with every platform listed in
[settings].lockfile_platforms. When lockfile_platforms is unset, fall back to
the union of platforms actually present in the lockfile.

MISE_LOCK_CHECK_FROM selects which copy of the files to read:
    - "worktree" (default): the files on disk
    - "index": the staged version (`git show :<path>`), for pre-commit
    - "head": the committed version (`git show HEAD:<path>`), for pre-push / CI

That keeps unrelated commits and pushes from being blocked by local unstaged
edits to mise.toml. The check is skipped when the selected source matches the
merge-base with BASE_REF (default: origin/main).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

BASE_REF = os.environ.get("BASE_REF", "origin/main")
SOURCE = os.environ.get("MISE_LOCK_CHECK_FROM", "worktree")
_VALID_SOURCES = ("worktree", "index", "head")


def die(msg: str) -> None:
    """Print an error to stderr and exit with status 1."""
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def git_show(spec: str) -> bytes | None:
    """Return file content at a git ref (e.g. ':mise.toml', 'HEAD:mise.toml')."""
    r = subprocess.run(["git", "show", spec], capture_output=True, check=False)
    return r.stdout if r.returncode == 0 else None


def read_source(path: str) -> bytes:
    """Read `path` from the configured source (worktree / index / head)."""
    if SOURCE == "worktree":
        return Path(path).read_bytes()
    spec = f":{path}" if SOURCE == "index" else f"HEAD:{path}"
    content = git_show(spec)
    if content is None:
        die(f"git show {spec}: file not found or not a git repo")
    return content


def parse_toml(data: bytes, label: str) -> dict:
    """Parse TOML bytes, dying with a labelled error when decoding fails."""
    try:
        return tomllib.loads(data.decode("utf-8"))
    except tomllib.TOMLDecodeError as e:
        die(f"failed to parse {label}: {e}")


def platforms_in(entry: dict) -> set[str]:
    """Return the platform names an entry's keys declare.

    `[tools.X."platforms.Y"]` is parsed by tomllib as a flat literal key
    'platforms.Y'.
    """
    return {k.removeprefix("platforms.") for k in entry if k.startswith("platforms.")}


def version_matches(spec: str, locked: str) -> bool:
    """Return True when the locked version satisfies a prefix spec.

    '1.26' matches '1.26.2'.
    """
    return (locked + ".").startswith(spec + ".")


def main() -> int:
    """Run the drift check and return a process exit code (0 ok, 1 drift)."""
    if SOURCE not in _VALID_SOURCES:
        die(f"MISE_LOCK_CHECK_FROM must be one of {_VALID_SOURCES}, got '{SOURCE}'")

    os.chdir(os.environ.get("MISE_LOCK_CHECK_ROOT", "."))

    # Pre-flight: worktree mode needs the files on disk; git modes only
    # need to be inside a repo (checked implicitly by read_source).
    if SOURCE == "worktree":
        for f in ("mise.toml", "mise.lock"):
            if not Path(f).is_file():
                die(f"{f} not found in {Path.cwd()}")

    toml_bytes = read_source("mise.toml")
    lock_bytes = read_source("mise.lock")

    # Short-circuit: if the selected source's mise.toml and mise.lock both match
    # the merge-base exactly, nothing has changed that could affect the lockfile,
    # so skip the full parse.
    base = subprocess.run(
        ["git", "merge-base", "HEAD", BASE_REF],
        capture_output=True,
        text=True,
        check=False,
    )
    if base.returncode == 0:
        base_sha = base.stdout.strip()
        base_toml = git_show(f"{base_sha}:mise.toml")
        base_lock = git_show(f"{base_sha}:mise.lock")
        if base_toml == toml_bytes and base_lock == lock_bytes:
            print(f"✓ mise.lock check skipped: no changes vs {BASE_REF}")
            return 0

    toml_data = parse_toml(toml_bytes, "mise.toml")
    lock_data = parse_toml(lock_bytes, "mise.lock")

    declared = toml_data.get("tools", {})
    if not declared:
        die("mise.toml has no [tools] section")
    locked = lock_data.get("tools", {})

    # Expected platforms: prefer [settings].lockfile_platforms as source of truth.
    # Only fall back to the lockfile union when the setting is absent, so that
    # adding a platform to the setting without re-locking is caught.
    configured = toml_data.get("settings", {}).get("lockfile_platforms")
    if configured is not None:
        expected = set(configured)
    else:
        expected = set()
        for entries in locked.values():
            for entry in entries:
                expected |= platforms_in(entry)

    errors: list[str] = []
    for tool, spec in declared.items():
        # A tool is pinned either as a bare version string or as an inline table
        # ({ version = "...", ... }) when a backend takes extra options.
        version = spec.get("version", "") if isinstance(spec, dict) else spec
        entries = locked.get(tool, [])
        match = next(
            (e for e in entries if version_matches(str(version), str(e.get("version", "")))),
            None,
        )
        if match is None:
            found = ", ".join(e.get("version", "?") for e in entries) or "none"
            errors.append(f"{tool}@{version}: no matching entry in mise.lock (found: {found})")
            continue
        # Some backends have no per-platform binary artifacts, so `mise lock`
        # never emits platform sub-tables for them: go: is compiled from source
        # by the Go toolchain, npm: is platform-independent JavaScript, and
        # pipx: is a platform-independent Python package. Skip their coverage
        # check.
        backend = match.get("backend", "")
        if backend.startswith(("go:", "npm:", "pipx:")):
            continue
        missing = expected - platforms_in(match)
        if missing:
            errors.append(
                f"{tool}@{match.get('version')}: missing platforms: {','.join(sorted(missing))}"
            )

    if errors:
        print("mise.lock is out of date:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        print("\nRun: mise lock", file=sys.stderr)
        return 1

    print(f"✓ mise.lock OK: {len(declared)} tool(s), {len(expected)} platform(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
