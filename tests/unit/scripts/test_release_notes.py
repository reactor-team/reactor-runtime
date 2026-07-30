"""The release-notes renderer: what each kind states, and where its change list starts."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "release-notes.py"

_VERSION = "3.0.1"
_WIRE = "1.20260722.6"
_COMMIT = "0123456789abcdef0123456789abcdef01234567"

type _Git = Callable[..., str | None]


@pytest.fixture
def notes() -> ModuleType:
    """Load the script as a module so its rendering is testable."""
    spec = importlib.util.spec_from_file_location("release_notes", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_git(calls: list[tuple[str, ...]], describe: str | None, log: str) -> _Git:
    """Build a ``_git`` stand-in that records its calls and answers from canned output."""

    def run(*args: str) -> str | None:
        calls.append(args)
        if args[0] == "describe":
            return describe
        return log

    return run


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


class TestStartTag:
    def test_a_release_measures_from_the_previous_release(
        self, notes: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[str, ...]] = []
        monkeypatch.setattr(notes, "_git", _fake_git(calls, "v3.0.0", ""))
        assert notes.start_tag("release", _COMMIT) == "v3.0.0"
        assert "--match=v*" in calls[0]
        assert "--match=pre/v*" not in calls[0]

    def test_a_prerelease_measures_from_whatever_shipped_last(
        self, notes: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[str, ...]] = []
        monkeypatch.setattr(notes, "_git", _fake_git(calls, "pre/v3.0.0.dev7", ""))
        assert notes.start_tag("prerelease", _COMMIT) == "pre/v3.0.0.dev7"
        assert "--match=pre/v*" in calls[0]
        assert "--match=v*" in calls[0]

    def test_describes_from_the_parent_so_the_new_tag_cannot_shift_the_span(
        self, notes: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[str, ...]] = []
        monkeypatch.setattr(notes, "_git", _fake_git(calls, "v3.0.0", ""))
        notes.start_tag("release", _COMMIT)
        assert calls[0][-1] == f"{_COMMIT}^"


class TestChanges:
    def test_lists_one_entry_per_commit(self, notes: ModuleType, monkeypatch) -> None:
        log = "Encode recordings in process with PyAV (#90)\nServe the metrics (#81)"
        monkeypatch.setattr(notes, "_git", _fake_git([], "v3.0.0", log))
        assert notes.changes("v3.0.0", _COMMIT) == [
            "Encode recordings in process with PyAV (#90)",
            "Serve the metrics (#81)",
        ]

    def test_the_first_build_of_all_has_nothing_to_measure_from(
        self, notes: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(notes, "_git", _fake_git([], None, ""))
        assert notes.changes(None, _COMMIT) == []


class TestBody:
    @pytest.fixture(autouse=True)
    def _git(self, pinned: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
        log = "Encode recordings in process with PyAV (#90)"
        monkeypatch.setattr(pinned, "_git", _fake_git([], "v3.0.0", log))

    def test_release_installs_the_exact_version_from_pypi(self, pinned: ModuleType) -> None:
        assert f"pip install reactor-runtime=={_VERSION}" in pinned.body("release", _COMMIT)

    def test_prerelease_says_what_the_build_is(self, pinned: ModuleType) -> None:
        body = pinned.body("prerelease", _COMMIT)
        assert "Development build from `main`" in body
        assert "pip install" not in body

    @pytest.mark.parametrize("kind", ["release", "prerelease"])
    def test_every_kind_names_the_wire_release_and_the_commit(
        self, pinned: ModuleType, kind: str
    ) -> None:
        body = pinned.body(kind, _COMMIT)
        assert f"[`{_WIRE}`]" in body
        assert f"/releases/tag/wire/v{_WIRE})" in body
        assert _COMMIT in body

    @pytest.mark.parametrize("kind", ["release", "prerelease"])
    def test_every_kind_lists_the_changes_and_names_their_starting_point(
        self, pinned: ModuleType, kind: str
    ) -> None:
        body = pinned.body(kind, _COMMIT)
        assert "## Changes since `v3.0.0`" in body
        assert "- Encode recordings in process with PyAV (#90)" in body

    def test_the_version_override_wins_over_pyproject(self, pinned: ModuleType) -> None:
        body = pinned.body("prerelease", _COMMIT, "3.0.0.dev8")
        assert "**3.0.0.dev8**" in body
        assert _VERSION not in body

    def test_a_build_with_no_starting_point_carries_no_change_list(
        self, pinned: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pinned, "_git", _fake_git([], None, ""))
        assert "## Changes" not in pinned.body("release", _COMMIT)
