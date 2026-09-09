# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""The source-checkout command selects one exact wheel, even after version bumps."""

import os
import subprocess
from pathlib import Path


def test_build_task_rejects_unknown_arguments_without_building(tmp_path: Path) -> None:
    repo = Path(__file__).parents[3]
    result = subprocess.run(
        ["bash", str(repo / "mise-tasks" / "example" / "multi-gpu"), "--dry-run"],
        env={**os.environ, "REPO_ROOT": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "no arguments" in result.stderr
    assert not (tmp_path / "examples").exists()


def test_build_task_stages_the_selected_wheel_and_calls_the_cli(tmp_path: Path) -> None:
    repo = Path(__file__).parents[3]
    tools = tmp_path / "tools"
    tools.mkdir()
    uv = tools / "uv"
    uv.write_text("#!/bin/sh\nprintf '3.3.0\\n'\n")
    uv.chmod(0o755)
    reactor = tools / "reactor"
    reactor.write_text('#!/bin/sh\nprintf "CLI %s in %s\\n" "$*" "$PWD"\n')
    reactor.chmod(0o755)
    wheels = tmp_path / "dist"
    wheels.mkdir()
    selected = "reactor_runtime-3.3.0-py3-none-any.whl"
    (wheels / selected).write_bytes(b"selected wheel")
    stage = tmp_path / "examples" / "multi_gpu" / "runtime-wheel"
    stage.mkdir(parents=True)
    older = stage / "reactor_runtime-3.2.0-py3-none-any.whl"
    older.write_bytes(b"old wheel")

    result = subprocess.run(
        ["bash", str(repo / "mise-tasks" / "example" / "multi-gpu")],
        env={**os.environ, "REPO_ROOT": str(tmp_path), "PATH": f"{tools}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=True,
    )

    assert (stage / selected).read_bytes() == b"selected wheel"
    assert older.read_bytes() == b"old wheel"
    assert (stage / "requirements.txt").read_text() == f"/tmp/runtime-wheel/{selected}\n"
    assert f"CLI build in {stage.parent}" in result.stdout
    assert "reactor run --gpus all" in result.stdout
    dockerfile = (repo / "examples" / "multi_gpu" / "Dockerfile").read_text()
    assert "pip install --no-cache-dir -r /tmp/runtime-wheel/requirements.txt" in dockerfile
    assert dockerfile.index("torch==") < dockerfile.index("COPY runtime-wheel/")
