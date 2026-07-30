#!/usr/bin/env python3
"""Render the body of a GitHub release or pre-release.

The release workflow calls this after it builds the distributions, so the
version in ``pyproject.toml`` is the version the artifacts carry: a release
reads ``X.Y.Z``, and a pre-release reads the ``X.Y.Z.dev<run>`` the workflow
stamped before the build.

Every body names the wire-protocol release the wheel vendors. That schema
follows its own CalVer lifecycle, so a reader cannot derive it from the package
version.

Git decides where the change list starts. A release measures from the previous
``v*`` tag, and a pre-release from the last tag of either kind, so each build
lists only the commits it adds. Both spans come from ``git describe`` on the
parent of the commit being released, which answers the same way before the new
tag exists and after it does.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = "reactor-team/reactor-runtime"
REPO_ROOT = Path(__file__).resolve().parent.parent

_START_TAGS = {"release": ["v*"], "prerelease": ["pre/v*", "v*"]}


def _git(*args: str) -> str | None:
    """Return the output of a git command, or None when git rejects it."""
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def versions() -> tuple[str, str]:
    """Return the package version and the wire-protocol version it vendors."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    return str(data["project"]["version"]), str(data["tool"]["reactor-wire"]["version"])


def start_tag(kind: str, commit: str) -> str | None:
    """Return the tag the change list starts from, or None for the first build."""
    patterns = [f"--match={pattern}" for pattern in _START_TAGS[kind]]
    return _git("describe", "--tags", "--abbrev=0", *patterns, f"{commit}^")


def changes(start: str | None, commit: str) -> list[str]:
    """Return the subject of every commit *commit* adds since *start*."""
    if start is None:
        return []
    log = _git("log", "--first-parent", "--pretty=format:%s", f"{start}..{commit}")
    return [line for line in (log or "").splitlines() if line]


def body(kind: str, commit: str, version: str | None = None) -> str:
    """Return the markdown body for a *kind* build of *commit*."""
    pyproject_version, wire = versions()
    version = version or pyproject_version
    if kind == "release":
        lead = f"```sh\npip install reactor-runtime=={version}\n```"
    else:
        lead = (
            "Development build from `main`. The wheel and the source distribution are "
            "attached to this release."
        )
    wire_url = f"https://github.com/{REPO}/releases/tag/wire/v{wire}"
    sections = [
        f"`reactor-runtime` **{version}**",
        lead,
        f"- **Wire protocol:** `reactor_wire.v1` [`{wire}`]({wire_url})\n- **Commit:** `{commit}`",
    ]
    start = start_tag(kind, commit)
    entries = changes(start, commit)
    if entries:
        listed = "\n".join(f"- {entry}" for entry in entries)
        sections.append(f"## Changes since `{start}`\n\n{listed}")
    return "\n\n".join(sections) + "\n"


def main() -> int:
    """Print the release body the arguments describe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=["release", "prerelease"])
    parser.add_argument("--commit", required=True, help="the commit the build came from")
    parser.add_argument(
        "--version",
        help="the version the artifacts carry, when pyproject.toml no longer holds it",
    )
    args = parser.parse_args()
    print(body(args.kind, args.commit, args.version), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
