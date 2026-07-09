"""The contract gate's classifier: diff class, mandated bump, enforcement."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "check-http-spec.py"

type _Replies = dict[str, tuple[int, str]]
type _Runner = Callable[[str, Path, str], subprocess.CompletedProcess[str]]


@pytest.fixture
def gate() -> ModuleType:
    """Load the script as a module so its logic is testable without docker."""
    spec = importlib.util.spec_from_file_location("check_http_spec", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_oasdiff(replies: _Replies) -> _Runner:
    """Build a ``_run_oasdiff`` stand-in answering each subcommand from *replies*."""

    def run(subcommand: str, specs_dir: Path, fmt: str) -> subprocess.CompletedProcess[str]:
        code, stdout = replies[subcommand]
        return subprocess.CompletedProcess(args=[subcommand], returncode=code, stdout=stdout)

    return run


def _spec_pair(tmp_path: Path) -> tuple[Path, Path]:
    previous = tmp_path / "previous.json"
    current = tmp_path / "current.json"
    previous.write_text("{}")
    current.write_text("{}")
    return previous, current


_BREAKING_REPLIES: _Replies = {
    "breaking": (1, json.dumps([{"id": "api-removed"}])),
    "changelog": (1, "### removed /clips\n"),
}


class TestClassifyBump:
    def test_recognises_each_component(self, gate: ModuleType) -> None:
        assert gate.classify_bump("1.2.3", "2.0.0") == "major"
        assert gate.classify_bump("1.2.3", "1.3.0") == "minor"
        assert gate.classify_bump("1.2.3", "1.2.4") == "patch"
        assert gate.classify_bump("1.2.3", "1.2.3") == "none"

    def test_rejects_non_semver(self, gate: ModuleType) -> None:
        with pytest.raises(gate.ToolError):
            gate.classify_bump("1.2", "1.3")
        with pytest.raises(gate.ToolError):
            gate.classify_bump("a.b.c", "1.2.3")


class TestRequiredBump:
    def test_breaking_mandates_minor_never_major(self, gate: ModuleType) -> None:
        assert gate.required_bump("breaking") == "minor"

    def test_changed_mandates_patch(self, gate: ModuleType) -> None:
        assert gate.required_bump("changed") == "patch"

    def test_no_change_mandates_nothing(self, gate: ModuleType) -> None:
        assert gate.required_bump("none") == "none"


class TestClassifyDiff:
    def test_breaking_findings_win(
        self, gate: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(gate, "_run_oasdiff", _fake_oasdiff(_BREAKING_REPLIES))
        diff_class, _ = gate.classify_diff(*_spec_pair(tmp_path))
        assert diff_class == "breaking"

    def test_info_only_diff_is_no_change(
        self, gate: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        replies: _Replies = {
            "breaking": (0, "[]"),
            "diff": (0, json.dumps({"info": {"version": {"from": "1", "to": "2"}}})),
        }
        monkeypatch.setattr(gate, "_run_oasdiff", _fake_oasdiff(replies))
        diff_class, _ = gate.classify_diff(*_spec_pair(tmp_path))
        assert diff_class == "none"

    def test_non_breaking_path_diff_is_changed(
        self, gate: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        replies: _Replies = {
            "breaking": (0, "[]"),
            "diff": (0, json.dumps({"paths": {"added": ["/new"]}})),
        }
        monkeypatch.setattr(gate, "_run_oasdiff", _fake_oasdiff(replies))
        diff_class, _ = gate.classify_diff(*_spec_pair(tmp_path))
        assert diff_class == "changed"

    def test_tool_error_fails_closed(
        self, gate: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(gate, "_run_oasdiff", _fake_oasdiff({"breaking": (2, "")}))
        with pytest.raises(gate.ToolError):
            gate.classify_diff(*_spec_pair(tmp_path))

    def test_invalid_json_fails_closed(
        self, gate: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(gate, "_run_oasdiff", _fake_oasdiff({"breaking": (1, "not json")}))
        with pytest.raises(gate.ToolError):
            gate.classify_diff(*_spec_pair(tmp_path))


class TestModes:
    def _main(
        self,
        gate: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        replies: _Replies,
        argv_tail: list[str],
    ) -> int:
        monkeypatch.setattr(gate, "_run_oasdiff", _fake_oasdiff(replies))
        previous, current = _spec_pair(tmp_path)
        argv = ["check-http-spec.py", *argv_tail]
        argv += ["--previous", str(previous), "--current", str(current)]
        monkeypatch.setattr(sys, "argv", argv)
        result = gate.main()
        assert isinstance(result, int)
        return result

    def test_breaking_diff_rejects_patch_bump(
        self, gate: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        tail = ["enforce", "--previous-version", "0.1.0", "--current-version", "0.1.1"]
        assert self._main(gate, monkeypatch, tmp_path, _BREAKING_REPLIES, tail) == 1

    def test_breaking_diff_accepts_minor_bump(
        self, gate: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        tail = ["enforce", "--previous-version", "0.1.1", "--current-version", "0.2.0"]
        assert self._main(gate, monkeypatch, tmp_path, _BREAKING_REPLIES, tail) == 0

    def test_no_diff_accepts_no_bump(
        self, gate: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        replies: _Replies = {"breaking": (0, "[]"), "diff": (0, "{}")}
        tail = ["enforce", "--previous-version", "0.1.0", "--current-version", "0.1.0"]
        assert self._main(gate, monkeypatch, tmp_path, replies, tail) == 0

    def test_report_mode_never_gates(
        self, gate: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        assert self._main(gate, monkeypatch, tmp_path, _BREAKING_REPLIES, ["report"]) == 0

    def test_report_mode_degrades_on_tool_error(
        self, gate: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        replies: _Replies = {"breaking": (2, "")}
        assert self._main(gate, monkeypatch, tmp_path, replies, ["report"]) == 0

    def test_enforce_mode_fails_closed_on_tool_error(
        self, gate: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        replies: _Replies = {"breaking": (2, "")}
        tail = ["enforce", "--previous-version", "0.1.0", "--current-version", "0.2.0"]
        with pytest.raises(gate.ToolError):
            self._main(gate, monkeypatch, tmp_path, replies, tail)

    def test_changelog_failure_decides_nothing(
        self, gate: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        replies: _Replies = {
            "breaking": (1, json.dumps([{"id": "api-removed"}])),
            "changelog": (2, ""),
        }
        tail = ["enforce", "--previous-version", "0.1.0", "--current-version", "0.2.0"]
        assert self._main(gate, monkeypatch, tmp_path, replies, tail) == 0
