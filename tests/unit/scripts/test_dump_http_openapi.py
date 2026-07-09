"""The spec dumper: writes the committed contract, and gates on drift."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "dump-http-openapi.py"

_SPEC = '{"openapi": "3.1.0"}\n'


@pytest.fixture
def dumper() -> ModuleType:
    """Load the script as a module so its logic is testable without a render."""
    spec = importlib.util.spec_from_file_location("dump_http_openapi", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(
    dumper: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    committed: str | None,
    rendered: str,
    check: bool,
) -> int:
    spec_path = tmp_path / "api" / "openapi.json"
    if committed is not None:
        spec_path.parent.mkdir(parents=True)
        spec_path.write_text(committed)
    monkeypatch.setattr(dumper, "SPEC_PATH", spec_path)
    monkeypatch.setattr(dumper, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(dumper, "render_spec_json", lambda: rendered)
    argv = ["dump-http-openapi.py"]
    if check:
        argv.append("--check")
    monkeypatch.setattr(sys, "argv", argv)
    return dumper.main()


class TestCheck:
    def test_fresh_spec_passes(
        self, dumper: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        assert _run(dumper, monkeypatch, tmp_path, committed=_SPEC, rendered=_SPEC, check=True) == 0

    def test_drifted_spec_fails(
        self, dumper: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        assert (
            _run(dumper, monkeypatch, tmp_path, committed="{}\n", rendered=_SPEC, check=True) == 1
        )

    def test_missing_spec_fails(
        self, dumper: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        assert _run(dumper, monkeypatch, tmp_path, committed=None, rendered=_SPEC, check=True) == 1

    def test_check_never_writes(
        self, dumper: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _run(dumper, monkeypatch, tmp_path, committed="{}\n", rendered=_SPEC, check=True)
        assert (tmp_path / "api" / "openapi.json").read_text() == "{}\n"


class TestWrite:
    def test_writes_when_stale(
        self, dumper: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        assert (
            _run(dumper, monkeypatch, tmp_path, committed="{}\n", rendered=_SPEC, check=False) == 0
        )
        assert (tmp_path / "api" / "openapi.json").read_text() == _SPEC

    def test_creates_when_missing(
        self, dumper: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        assert _run(dumper, monkeypatch, tmp_path, committed=None, rendered=_SPEC, check=False) == 0
        assert (tmp_path / "api" / "openapi.json").read_text() == _SPEC
