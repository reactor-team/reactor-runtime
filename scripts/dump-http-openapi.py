#!/usr/bin/env python3
"""Write the HTTP surface's OpenAPI contract to ``api/openapi.json``.

The committed spec is the runtime's HTTP contract artifact: reviewers see
surface changes as part of a PR's diff, and the contract gate diffs it against
``main`` (per PR) and the previous release (at release time). Run via
``mise run http-spec``; a pre-commit hook runs it automatically when a commit
touches the HTTP surface, and a unit test fails when the committed file is
stale.
"""

from __future__ import annotations

import sys
from pathlib import Path

from reactor_runtime.http.spec import render_spec_json

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = REPO_ROOT / "api" / "openapi.json"


def main() -> int:
    """Render the spec and report whether the committed file changed."""
    rendered = render_spec_json()
    previous = SPEC_PATH.read_text() if SPEC_PATH.exists() else None
    if previous == rendered:
        print(f"{SPEC_PATH.relative_to(REPO_ROOT)} is up to date")
        return 0
    SPEC_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPEC_PATH.write_text(rendered)
    action = "updated" if previous is not None else "created"
    print(f"{action} {SPEC_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
