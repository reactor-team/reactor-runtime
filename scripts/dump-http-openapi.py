#!/usr/bin/env python3
"""Write — or verify — the HTTP surface's OpenAPI contract at ``api/openapi.json``.

The committed spec is the runtime's HTTP contract artifact: reviewers see
surface changes as part of a PR's diff, and the contract gate diffs it against
``main`` (per PR) and the previous release (at release time). Run via
``mise run http-spec``; a pre-commit hook runs it automatically when a commit
touches the HTTP surface, and a unit test fails when the committed file is
stale.

With ``--check`` the script renders and compares without writing, exiting
non-zero when the committed file is missing or has drifted from the code. This
is the blocking drift gate ``mise run http-spec-check`` runs in CI: the
contract the gate diffs must be the surface the code actually serves, so a
stale file has to fail the build rather than diff a surface that no longer
exists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from reactor_runtime.http.spec import render_spec_json

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = REPO_ROOT / "api" / "openapi.json"


def _write(rendered: str, previous: str | None) -> int:
    """Write the rendered spec and report whether the committed file changed."""
    if previous == rendered:
        print(f"{SPEC_PATH.relative_to(REPO_ROOT)} is up to date")
        return 0
    SPEC_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPEC_PATH.write_text(rendered)
    action = "updated" if previous is not None else "created"
    print(f"{action} {SPEC_PATH.relative_to(REPO_ROOT)}")
    return 0


def _check(rendered: str, previous: str | None) -> int:
    """Fail when the committed spec is missing or drifted from a fresh render."""
    rel = SPEC_PATH.relative_to(REPO_ROOT)
    if previous is None:
        print(
            f"::error::{rel} is missing — run `mise run http-spec` and commit it",
            file=sys.stderr,
        )
        return 1
    if previous != rendered:
        print(
            f"::error::{rel} has drifted from the HTTP surface — run "
            "`mise run http-spec` and commit the result",
            file=sys.stderr,
        )
        return 1
    print(f"{rel} matches the HTTP surface")
    return 0


def main() -> int:
    """Render the spec, then write it or verify the committed copy is fresh."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed spec matches the code without writing (drift gate)",
    )
    args = parser.parse_args()

    rendered = render_spec_json()
    previous = SPEC_PATH.read_text() if SPEC_PATH.exists() else None
    return _check(rendered, previous) if args.check else _write(rendered, previous)


if __name__ == "__main__":
    sys.exit(main())
