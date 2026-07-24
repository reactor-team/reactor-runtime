#!/usr/bin/env python3
"""Fetch the pinned wire-protocol wheel and vendor its bindings into ``src/``.

The runtime does not commit the generated ``reactor_wire`` bindings. Instead it
pins a released wire-protocol version and vendors that artifact at build time,
so the published wheel always contains the exact bytes that were released and
breaking-change-checked, not a local regeneration.

The pin lives in ``pyproject.toml`` under ``[tool.reactor-wire] version`` as a
CalVer string (e.g. ``1.20260618.42``). This script resolves it to the
``wire/v<version>`` GitHub release on ``reactor-team/reactor-runtime``,
downloads ``reactor_wire-<version>-py3-none-any.whl``, and extracts the
``reactor_wire`` package into ``src/`` (gitignored) or into ``--into``.

Authentication is optional for the public repository; if ``GH_TOKEN`` or
``GITHUB_TOKEN`` is set it is used to lift API rate limits and read private
releases.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

REPO = "reactor-team/reactor-runtime"
REPO_ROOT = Path(__file__).resolve().parent.parent


def read_pinned_version() -> str:
    """Return the wire-protocol version pinned in pyproject.toml."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    try:
        return str(data["tool"]["reactor-wire"]["version"])
    except KeyError:
        print(
            "Error: [tool.reactor-wire] version is not set in pyproject.toml",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


def _auth_headers(accept: str) -> dict[str, str]:
    headers = {"Accept": accept}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def _get(url: str, accept: str) -> bytes:
    request = urllib.request.Request(url, headers=_auth_headers(accept))
    with urllib.request.urlopen(request) as response:
        return response.read()


def fetch_release(tag: str) -> dict:
    """Fetch the GitHub release for *tag*, or exit if it does not exist."""
    url = f"https://api.github.com/repos/{REPO}/releases/tags/{tag}"
    try:
        return json.loads(_get(url, "application/vnd.github+json"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(
                f"Error: no release found at tag {tag} on {REPO}.\n"
                "Cut a wire/v* release (push a proto change to main) or fix the "
                "pin in [tool.reactor-wire].",
                file=sys.stderr,
            )
            raise SystemExit(1) from None
        print(f"Error fetching release {tag}: {e}", file=sys.stderr)
        raise SystemExit(1) from None


def download_wheel(release: dict, wheel_name: str, dest_dir: Path) -> Path:
    """Download the wheel asset from *release* into *dest_dir*."""
    asset = next((a for a in release.get("assets", []) if a["name"] == wheel_name), None)
    if asset is None:
        available = [a["name"] for a in release.get("assets", [])]
        print(
            f"Error: asset {wheel_name} not in release {release['tag_name']}.\n"
            f"  Available: {available}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    dest_dir.mkdir(parents=True, exist_ok=True)
    wheel_path = dest_dir / wheel_name
    asset_url = f"https://api.github.com/repos/{REPO}/releases/assets/{asset['id']}"
    wheel_path.write_bytes(_get(asset_url, "application/octet-stream"))
    return wheel_path


def vendor(wheel_path: Path, into: Path) -> None:
    """Extract the reactor_wire package from the wheel into *into*."""
    target = into / "reactor_wire"
    if target.exists():
        shutil.rmtree(target)

    with zipfile.ZipFile(wheel_path) as zf:
        members = [n for n in zf.namelist() if n.startswith("reactor_wire/")]
        if not members:
            print("Error: wheel contains no reactor_wire/ package", file=sys.stderr)
            raise SystemExit(1)
        for member in members:
            if member.endswith("/"):
                continue
            out = into / member
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(zf.read(member))
    print(f"  Vendored {len(members)} files into {target}/")


def main() -> None:
    """Resolve the pin, download the released wheel, and vendor its bindings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--into",
        type=Path,
        default=REPO_ROOT / "src",
        help="directory to extract the reactor_wire package into (default: src/)",
    )
    args = parser.parse_args()

    version = read_pinned_version()
    tag = f"wire/v{version}"
    wheel_name = f"reactor_wire-{version}-py3-none-any.whl"
    print(f"Fetching wire protocol {version} ({tag})...")

    release = fetch_release(tag)
    wheel_path = download_wheel(release, wheel_name, REPO_ROOT / "build" / "wire")
    vendor(wheel_path, args.into)
    print("Done.")


if __name__ == "__main__":
    main()
