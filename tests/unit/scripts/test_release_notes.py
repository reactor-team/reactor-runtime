"""The release-notes renderer: what each kind tells the reader, and the wire version."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "release-notes.py"

_VERSION = "3.0.1"
_WIRE = "1.20260722.6"
_COMMIT = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def notes() -> ModuleType:
    """Load the script as a module so its rendering is testable."""
    spec = importlib.util.spec_from_file_location("release_notes", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pinned(notes: ModuleType, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Render against fixed versions so the assertions do not track the real bump."""
    monkeypatch.setattr(notes, "versions", lambda: (_VERSION, _WIRE))
    return notes


class TestVersions:
    def test_reads_both_versions_from_pyproject(self, notes: ModuleType) -> None:
        version, wire = notes.versions()
        # The package is semver and the wire protocol is CalVer on its own line.
        assert len(version.split(".")) >= 3
        assert wire.startswith("1.")


class TestBody:
    def test_release_installs_the_exact_version_from_pypi(self, pinned: ModuleType) -> None:
        body = pinned.body("release", _COMMIT)
        assert f"pip install reactor-runtime=={_VERSION}" in body

    def test_prerelease_sends_the_reader_to_the_assets(self, pinned: ModuleType) -> None:
        body = pinned.body("prerelease", _COMMIT)
        assert "not on PyPI" in body
        assert "pip install" not in body

    @pytest.mark.parametrize("kind", ["release", "prerelease"])
    def test_every_kind_names_the_wire_release_and_the_commit(
        self, pinned: ModuleType, kind: str
    ) -> None:
        body = pinned.body(kind, _COMMIT)
        assert f"[`{_WIRE}`]" in body
        assert f"/releases/tag/wire/v{_WIRE})" in body
        assert _COMMIT in body
