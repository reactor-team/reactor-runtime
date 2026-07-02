#!/usr/bin/env python3
"""Classify an OpenAPI contract diff and gate the version bump it mandates.

The runtime's HTTP surface is versioned by the package, and the policy is
deliberate: a **breaking** contract change mandates at least a **minor** bump,
any other contract change mandates at least a **patch** bump, and the mapping
is capped there — a major bump is never required. The diff is computed by
`oasdiff <https://github.com/Tufin/oasdiff>`_, pinned to an OpenAPI-3.1-capable
image because FastAPI renders 3.1 documents.

Two modes:

``report`` (per PR, informational)
    Print the diff class and the bump class the change will mandate at the
    next release. When the contract changed at all, render oasdiff's markdown
    changelog — appended to ``$GITHUB_STEP_SUMMARY`` when set, printed
    otherwise. Informational means it never blocks: a tool failure (docker
    absent, a registry pull throttled) degrades to a warning and exit ``0``.

``enforce`` (at release, gating)
    Additionally compare the actual version bump against the mandated class
    and exit ``1`` when it falls short (e.g. a breaking diff on a patch-only
    bump).

oasdiff's exit-code contract: ``0`` clean, ``1`` breaking-changes-found (valid
output on stdout), anything above is a tool error. In ``enforce`` mode a tool
error fails closed — a misfire must never read as "no contract change" where
the version gate is decided.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

OASDIFF_IMAGE = "tufin/oasdiff:v1.15.0-openapi31.beta.3"

_BUMP_RANK = {"none": 0, "patch": 1, "minor": 2, "major": 3}


class ToolError(RuntimeError):
    """Raised when oasdiff (or docker) fails outright; the gate fails closed."""


def _run_oasdiff(subcommand: str, specs_dir: Path, fmt: str) -> subprocess.CompletedProcess[str]:
    """Run one oasdiff subcommand against ``previous.json``/``current.json`` in *specs_dir*."""
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{specs_dir}:/specs:ro",
        OASDIFF_IMAGE,
        subcommand,
        "/specs/previous.json",
        "/specs/current.json",
        "--format",
        fmt,
    ]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        raise ToolError("docker is required to run the pinned oasdiff image") from None


def _oasdiff_json(subcommand: str, specs_dir: Path) -> list | dict:
    """Run *subcommand* and parse its JSON output, failing closed on tool errors."""
    result = _run_oasdiff(subcommand, specs_dir, "json")
    if result.returncode > 1:
        raise ToolError(f"oasdiff {subcommand} failed (exit {result.returncode}):\n{result.stderr}")
    stdout = result.stdout.strip()
    if not stdout:
        return {} if subcommand == "diff" else []
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ToolError(f"oasdiff {subcommand} produced invalid JSON: {exc}") from None
    if subcommand == "diff":
        return parsed if isinstance(parsed, dict) else {}
    return parsed if isinstance(parsed, list) else []


def classify_diff(previous: Path, current: Path) -> tuple[str, Path]:
    """Classify the contract diff between two spec files.

    Returns:
        A ``(diff_class, specs_dir)`` pair: ``"breaking"``, ``"changed"``, or
        ``"none"``, plus the staged temp directory (for a follow-up changelog
        render against the same copies).
    """
    specs_dir = Path(tempfile.mkdtemp(prefix="http-spec-gate-"))
    shutil.copyfile(previous, specs_dir / "previous.json")
    shutil.copyfile(current, specs_dir / "current.json")

    breaking = _oasdiff_json("breaking", specs_dir)
    if breaking:
        return "breaking", specs_dir
    raw_diff = _oasdiff_json("diff", specs_dir)
    # info.* moves with metadata (title, version), never the served contract.
    changed = isinstance(raw_diff, dict) and any(
        value for key, value in raw_diff.items() if key != "info"
    )
    return ("changed" if changed else "none"), specs_dir


def changelog_markdown(specs_dir: Path) -> str:
    """Render the human-readable changelog for the staged spec pair."""
    result = _run_oasdiff("changelog", specs_dir, "markdown")
    if result.returncode > 1:
        raise ToolError(f"oasdiff changelog failed (exit {result.returncode}):\n{result.stderr}")
    return result.stdout


def required_bump(diff_class: str) -> str:
    """Map a diff class to the version bump it mandates — capped at minor."""
    return {"breaking": "minor", "changed": "patch"}.get(diff_class, "none")


def classify_bump(previous: str, current: str) -> str:
    """Classify the actual semver bump between two versions."""
    try:
        prev = [int(part) for part in previous.split(".")[:3]]
        curr = [int(part) for part in current.split(".")[:3]]
    except ValueError:
        raise ToolError(f"versions must be numeric semver: {previous!r} -> {current!r}") from None
    if len(prev) != 3 or len(curr) != 3:
        raise ToolError(f"versions must be MAJOR.MINOR.PATCH: {previous!r} -> {current!r}")
    for name, index in (("major", 0), ("minor", 1), ("patch", 2)):
        if curr[index] != prev[index]:
            return name
    return "none"


def _emit_changelog(specs_dir: Path) -> None:
    """Surface the changelog: into the CI step summary when present, else stdout."""
    markdown = changelog_markdown(specs_dir)
    if not markdown.strip():
        return
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as summary:
            summary.write("## HTTP contract changes\n\n" + markdown + "\n")
        print("changelog written to the CI step summary")
    else:
        print(markdown)


def main() -> int:
    """Run the gate in the requested mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["report", "enforce"])
    parser.add_argument("--previous", required=True, type=Path, help="the older spec file")
    parser.add_argument("--current", required=True, type=Path, help="the newer spec file")
    parser.add_argument("--previous-version", help="released version behind --previous (enforce)")
    parser.add_argument("--current-version", help="version being released (enforce)")
    args = parser.parse_args()

    try:
        diff_class, specs_dir = classify_diff(args.previous, args.current)
    except ToolError as error:
        if args.mode == "report":
            print(f"::warning::HTTP contract check skipped: {error}", file=sys.stderr)
            return 0
        raise
    mandated = required_bump(diff_class)
    if mandated == "none":
        print("contract diff: none; no version bump mandated")
    else:
        print(f"contract diff: {diff_class}; mandates a {mandated} bump")
    if diff_class != "none":
        # The changelog is presentation, not the gate: a render failure is
        # surfaced but decides nothing in either mode.
        try:
            _emit_changelog(specs_dir)
        except ToolError as error:
            print(f"::warning::changelog render failed: {error}", file=sys.stderr)

    if args.mode == "report":
        return 0

    if not args.previous_version or not args.current_version:
        raise ToolError("enforce mode needs --previous-version and --current-version")
    actual = classify_bump(args.previous_version, args.current_version)
    if _BUMP_RANK[actual] >= _BUMP_RANK[mandated]:
        print(
            f"version bump ok: {args.previous_version} -> {args.current_version} "
            f"({actual}) satisfies the mandated {mandated}"
        )
        return 0
    print(
        f"::error::the HTTP contract diff is {diff_class}, which mandates at least a "
        f"{mandated} bump, but {args.previous_version} -> {args.current_version} is "
        f"only a {actual} bump. Re-tag with a {mandated} bump.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ToolError as error:
        print(f"::error::{error}", file=sys.stderr)
        sys.exit(2)
