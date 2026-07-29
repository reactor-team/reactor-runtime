#!/usr/bin/env python3
"""Render the body of a GitHub release or pre-release.

The release workflow calls this after it builds the distributions, so the
version in ``pyproject.toml`` is the version the artifacts carry: a release
reads ``X.Y.Z``, and a pre-release reads the ``X.Y.Z.dev<run>`` the workflow
stamped before the build.

Both bodies name the wire-protocol release the wheel vendors. That version is
the schema the package speaks, it moves on its own CalVer lifecycle, and a
reader cannot derive it from the package version, so the release states it.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

REPO = "reactor-team/reactor-runtime"
REPO_ROOT = Path(__file__).resolve().parent.parent


def versions() -> tuple[str, str]:
    """Return the package version and the wire-protocol version it vendors."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    return str(data["project"]["version"]), str(data["tool"]["reactor-wire"]["version"])


def body(kind: str, commit: str) -> str:
    """Return the markdown body for a *kind* release built from *commit*."""
    version, wire = versions()
    if kind == "release":
        install = f"Install it from PyPI:\n\n```sh\npip install reactor-runtime=={version}\n```"
    else:
        install = (
            "This build comes from a merge to `main`, and it is not on PyPI. "
            "To install it, download the wheel from the assets below."
        )
    wire_url = f"https://github.com/{REPO}/releases/tag/wire/v{wire}"
    sections = [
        f"`reactor-runtime` **{version}**",
        install,
        "**Wire protocol:** this build carries the `reactor_wire.v1` bindings from "
        f"wire release [`{wire}`]({wire_url}).",
        f"**Commit:** `{commit}`",
    ]
    return "\n\n".join(sections) + "\n"


def main() -> int:
    """Print the release body the arguments describe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=["release", "prerelease"])
    parser.add_argument("--commit", required=True, help="the commit the build came from")
    args = parser.parse_args()
    print(body(args.kind, args.commit), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
