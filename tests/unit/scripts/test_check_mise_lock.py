#!/usr/bin/env python3
#
# Tests for the lint:mise-lock task (mise-tasks/lint/mise-lock.py).
#
# Runs the script against fixture mise.toml/mise.lock pairs via MISE_LOCK_CHECK_ROOT.
# Run: pytest tests/unit/scripts/test_check_mise_lock.py

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent.parent.parent / "mise-tasks" / "lint" / "mise-lock.py"
FIXTURES = HERE / "fixtures" / "check-mise-lock"


def _clean_env() -> dict[str, str]:
    # Git exports repo-pinning variables (GIT_DIR and friends) to hooks, so a
    # suite run from a hook would point every git call below at the developer's
    # checkout instead of the scratch repo the test built.
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def _run(case: str) -> subprocess.CompletedProcess[str]:
    # Copy the fixture out of the repo so the script's git short-circuit
    # doesn't see untracked fixture files and incorrectly skip the check.
    with tempfile.TemporaryDirectory(prefix="mise-lock-test-") as tmp:
        for name in ("mise.toml", "mise.lock"):
            src = FIXTURES / case / name
            if src.exists():
                shutil.copy(src, Path(tmp) / name)
        env = _clean_env()
        env["MISE_LOCK_CHECK_ROOT"] = tmp
        return subprocess.run(
            [str(SCRIPT)],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )


def test_valid_lockfile_passes():
    result = _run("valid")
    assert result.returncode == 0, result.stderr
    assert "mise.lock OK" in result.stdout
    assert "2 tool(s)" in result.stdout
    assert "2 platform(s)" in result.stdout


def test_missing_tool_fails():
    result = _run("missing-tool")
    assert result.returncode == 1
    assert "tool-c" in result.stderr
    assert "no matching entry" in result.stderr
    assert "Run: mise lock" in result.stderr


def test_missing_platform_fails():
    result = _run("missing-platform")
    assert result.returncode == 1
    assert "tool-a" in result.stderr
    assert "missing platforms" in result.stderr
    assert "macos-arm64" in result.stderr


def test_version_mismatch_fails():
    result = _run("version-mismatch")
    assert result.returncode == 1
    assert "tool-a" in result.stderr
    assert "no matching entry" in result.stderr
    assert "1.0.0" in result.stderr


def test_version_prefix_match_accepted():
    # "valid" fixture declares tool-b = "2.0" and lock has 2.0.1 - the prefix
    # match should accept this without complaint.
    result = _run("valid")
    assert result.returncode == 0, result.stderr


def test_configured_platform_missing_from_lock_fails():
    # mise.toml's lockfile_platforms declares linux-arm64, but no tool has
    # an entry for it - the check must surface this even though the lockfile
    # union (the old, weaker heuristic) wouldn't include linux-arm64.
    result = _run("missing-configured-platform")
    assert result.returncode == 1, result.stdout
    assert "tool-a" in result.stderr
    assert "missing platforms" in result.stderr
    assert "linux-arm64" in result.stderr


def test_go_backend_tool_exempt_from_platform_check():
    # go: backend tools are compiled from source and have no per-platform
    # binary artifacts. mise lock never emits platform sub-tables for them,
    # so the check must not flag them as missing platforms.
    result = _run("go-backend")
    assert result.returncode == 0, result.stderr
    assert "mise.lock OK" in result.stdout


def test_npm_backend_tool_exempt_from_platform_check():
    # npm: backend tools are platform-independent JavaScript and have no
    # per-platform binary artifacts. mise lock never emits platform
    # sub-tables for them, so the check must not flag them as missing
    # platforms.
    result = _run("npm-backend")
    assert result.returncode == 0, result.stderr
    assert "mise.lock OK" in result.stdout


def test_pipx_backend_tool_exempt_from_platform_check():
    # pipx: backend tools are platform-independent Python packages with no
    # per-platform binary artifacts, and they are pinned as an inline table
    # ({ version = "...", uvx = true }). The check must read the table's version
    # and skip the platform coverage check.
    result = _run("pipx-backend")
    assert result.returncode == 0, result.stderr
    assert "mise.lock OK" in result.stdout


def test_malformed_mise_toml_dies():
    result = _run("malformed-toml")
    assert result.returncode != 0
    assert "failed to parse" in result.stderr
    assert "mise.toml" in result.stderr
    # Must NOT silently succeed.
    assert "mise.lock OK" not in result.stdout


def test_malformed_mise_lock_dies():
    result = _run("malformed-lock")
    assert result.returncode != 0
    assert "failed to parse" in result.stderr
    assert "mise.lock" in result.stderr
    assert "mise.lock OK" not in result.stdout


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(
        ["git", *args], cwd=cwd, env=_clean_env(), check=True, capture_output=True, text=True
    )


def _init_repo_with_fixture(tmp: Path, fixture: str) -> None:
    """Initialize a git repo at tmp, commit the named fixture's mise.{toml,lock}."""
    _git("init", "-q", "-b", "main", cwd=tmp)
    _git("config", "commit.gpgsign", "false", cwd=tmp)
    _git("config", "user.email", "test@reactor.test", cwd=tmp)
    _git("config", "user.name", "Test", cwd=tmp)
    for name in ("mise.toml", "mise.lock"):
        shutil.copy(FIXTURES / fixture / name, tmp / name)
    _git("add", "mise.toml", "mise.lock", cwd=tmp)
    _git("commit", "-q", "-m", "initial", cwd=tmp)


def _run_script(
    tmp: Path, source: str | None = None, base_ref: str = "does-not-exist"
) -> subprocess.CompletedProcess[str]:
    env = _clean_env()
    env["MISE_LOCK_CHECK_ROOT"] = str(tmp)
    env["BASE_REF"] = base_ref  # point at a missing ref by default to disable the short-circuit
    if source is not None:
        env["MISE_LOCK_CHECK_FROM"] = source
    return subprocess.run([str(SCRIPT)], capture_output=True, text=True, env=env, check=False)


# Exercise MISE_LOCK_CHECK_FROM=index|head and merge-base short-circuit.


def test_index_mode_uses_staged_content():
    # Index has the valid fixture; worktree is overwritten with a broken
    # mise.toml (missing-tool). FROM=index must pass because it reads
    # the staged version, not the worktree.
    with tempfile.TemporaryDirectory(prefix="mise-lock-git-") as tmp_str:
        tmp = Path(tmp_str)
        _init_repo_with_fixture(tmp, "valid")
        shutil.copy(FIXTURES / "missing-tool" / "mise.toml", tmp / "mise.toml")
        result = _run_script(tmp, source="index")
        assert result.returncode == 0, result.stderr
        assert "mise.lock OK" in result.stdout


def test_index_mode_detects_staged_breakage():
    # Index has broken (missing-tool) mise.toml; worktree restored to valid.
    # FROM=index must fail because it reads the staged version.
    with tempfile.TemporaryDirectory(prefix="mise-lock-git-") as tmp_str:
        tmp = Path(tmp_str)
        _init_repo_with_fixture(tmp, "valid")
        shutil.copy(FIXTURES / "missing-tool" / "mise.toml", tmp / "mise.toml")
        _git("add", "mise.toml", cwd=tmp)
        # Now restore worktree to the valid version so only the index is broken.
        shutil.copy(FIXTURES / "valid" / "mise.toml", tmp / "mise.toml")
        result = _run_script(tmp, source="index")
        assert result.returncode == 1, result.stdout
        assert "tool-c" in result.stderr


def test_head_mode_ignores_staged_and_worktree():
    # HEAD has valid. Stage AND write a broken version in the worktree.
    # FROM=head must pass because it reads only the committed version.
    with tempfile.TemporaryDirectory(prefix="mise-lock-git-") as tmp_str:
        tmp = Path(tmp_str)
        _init_repo_with_fixture(tmp, "valid")
        shutil.copy(FIXTURES / "missing-tool" / "mise.toml", tmp / "mise.toml")
        _git("add", "mise.toml", cwd=tmp)
        result = _run_script(tmp, source="head")
        assert result.returncode == 0, result.stderr


def test_head_mode_detects_committed_breakage():
    # HEAD has broken (missing-tool). FROM=head must fail.
    with tempfile.TemporaryDirectory(prefix="mise-lock-git-") as tmp_str:
        tmp = Path(tmp_str)
        _init_repo_with_fixture(tmp, "missing-tool")
        result = _run_script(tmp, source="head")
        assert result.returncode == 1, result.stdout
        assert "tool-c" in result.stderr


def test_short_circuit_when_source_matches_base():
    # Source content matches merge-base content, so the script should skip
    # the full check and print the skipped message.
    with tempfile.TemporaryDirectory(prefix="mise-lock-git-") as tmp_str:
        tmp = Path(tmp_str)
        _init_repo_with_fixture(tmp, "valid")
        # BASE_REF=HEAD makes merge-base=HEAD; source=worktree equals HEAD.
        result = _run_script(tmp, base_ref="HEAD")
        assert result.returncode == 0, result.stderr
        assert "check skipped" in result.stdout


def test_invalid_source_value_rejected():
    # Unknown MISE_LOCK_CHECK_FROM values fail with a clear message.
    with tempfile.TemporaryDirectory(prefix="mise-lock-git-") as tmp_str:
        tmp = Path(tmp_str)
        shutil.copy(FIXTURES / "valid" / "mise.toml", tmp / "mise.toml")
        shutil.copy(FIXTURES / "valid" / "mise.lock", tmp / "mise.lock")
        result = _run_script(tmp, source="garbage")
        assert result.returncode != 0
        assert "MISE_LOCK_CHECK_FROM" in result.stderr
