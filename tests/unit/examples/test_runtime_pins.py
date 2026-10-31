"""Every example workspace pins the runtime release this checkout is.

``reactor build`` installs the release ``build.runtime_version`` names in an
example's ``reactor.yaml``, and its ``requirements.txt`` pins the same
``reactor-runtime``. Both follow the package version in ``pyproject.toml``, so
the change that bumps the version bumps every example with it, and no example
falls behind the release it ships beside.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).parents[3]
_WORKSPACES = sorted(manifest.parent for manifest in (_ROOT / "examples").glob("*/reactor.yaml"))
_PIN = re.compile(r"^reactor-runtime==(?P<version>\S+)\s*$", re.MULTILINE)
_BUMP = "bump it with the package version; see skills/maintaining-examples"


def _package_version() -> str:
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text())["project"]
    return str(project["version"])


def test_the_examples_are_the_three_workspaces() -> None:
    # A new feature joins one of these; see skills/maintaining-examples.
    assert [workspace.name for workspace in _WORKSPACES] == ["echo", "starter", "waypoint"]


@pytest.mark.parametrize("workspace", _WORKSPACES, ids=lambda path: path.name)
def test_requirements_pin_the_package_version(workspace: Path) -> None:
    pins = _PIN.findall((workspace / "requirements.txt").read_text())
    assert pins == [_package_version()], f"{workspace.name}/requirements.txt: {_BUMP}"


@pytest.mark.parametrize("workspace", _WORKSPACES, ids=lambda path: path.name)
def test_the_build_installs_the_package_version(workspace: Path) -> None:
    manifest = yaml.safe_load((workspace / "reactor.yaml").read_text())
    version = str(manifest["build"]["runtime_version"])
    assert version == _package_version(), f"{workspace.name}/reactor.yaml: {_BUMP}"
