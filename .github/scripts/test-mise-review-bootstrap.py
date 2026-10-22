#!/usr/bin/env python3
# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Check that reviews bootstrap mise independently of the target PR files."""

import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/codex-review.yml"
VERSION = "2025.4.9"
DIGEST = "a" * 64
CONFIG = f'min_version = "{VERSION}"\n[_]\nmise_sha256_linux_x64 = "{DIGEST}"\n'


def _step(name):
    match = re.search(
        rf"^      - name: {re.escape(name)}\n(.*?)(?=^      - |\Z)",
        WORKFLOW.read_text(),
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"Missing workflow step: {name}"
    return match.group(1)


class MiseReviewBootstrapTests(unittest.TestCase):
    """Keep the workflow pin separate from an old or modified PR checkout."""

    def test_trusted_checkout_contract(self):
        """Read only the workflow revision pin without changing the PR checkout."""
        head = _step("Checkout PR head")
        trusted = _step("Checkout workflow mise pin")
        assert "ref: ${{ needs.acknowledge.outputs.target_sha }}" in head
        assert "path:" not in head
        assert "persist-credentials: false" in head
        checkout_action = re.search(r"uses: (actions/checkout@\S+)", head)
        assert checkout_action is not None
        assert f"uses: {checkout_action.group(1)}" in trusted
        for field in (
            "ref: ${{ github.workflow_sha }}",
            "path: .mise-bootstrap",
            "sparse-checkout-cone-mode: false",
            "persist-credentials: false",
        ):
            assert field in trusted
        sparse = re.search(r"sparse-checkout: \|\n((?:            .+\n)+)", trusted)
        assert sparse is not None
        assert sparse.group(1).split() == ["mise.toml", ".github/scripts/mise-pin.sh"]
        workflow = WORKFLOW.read_text()
        assert (
            workflow.index("name: Checkout PR head\n")
            < workflow.index("name: Checkout workflow mise pin\n")
            < workflow.index("name: Read mise pin\n")
            < workflow.index("uses: jdx/mise-action@")
        )
        read = _step("Read mise pin")
        assert "shell: bash" in read
        assert 'run: bash .mise-bootstrap/.github/scripts/mise-pin.sh >> "$GITHUB_ENV"' in read

    def test_action_does_not_load_pr_configuration(self):
        """Install the trusted mise binary without reading or installing PR tools."""
        setup = _step("Set up mise")
        for field in (
            "working_directory: ${{ runner.temp }}",
            "install: false",
            "env: false",
            "version: ${{ env.MISE_VERSION }}",
            "sha256: ${{ env.MISE_SHA256 }}",
        ):
            assert field in setup
        ci_steps = re.split(
            r"^      - ", (ROOT / ".github/workflows/ci.yml").read_text(), flags=re.MULTILINE
        )
        checks = [
            step for step in ci_steps if ".github/scripts/test-mise-review-bootstrap.py" in step
        ]
        assert len(checks) == 1
        assert re.search(r"^          MISE_REVIEW_TEST_BINARY: mise$", checks[0], re.MULTILINE)

    @unittest.skipUnless(
        os.environ.get("MISE_REVIEW_TEST_BINARY"),
        "Set MISE_REVIEW_TEST_BINARY to test with an explicit mise binary",
    )
    def test_action_commands_ignore_future_pr_config(self):
        """Keep higher floors and unknown TOML outside the action subprocesses."""
        binary = shutil.which(os.environ["MISE_REVIEW_TEST_BINARY"])
        assert binary is not None, "MISE_REVIEW_TEST_BINARY must name an executable mise binary"
        setup = _step("Set up mise")
        inputs = dict(re.findall(r"^          ([a-z_]+): (.+)$", setup, re.MULTILINE))
        for config in (
            'min_version = "2099.1.1"\n[env]\nPR_ONLY_ENV = "present"\n',
            'min_version = "2099.1.1"\nfuture_syntax = [\n',
        ):
            with self.subTest(config=config), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                target = root / "pr"
                runner_temp = root / "runner-temp"
                target.mkdir()
                runner_temp.mkdir()
                pin = target / "mise.toml"
                pin.write_text(config)
                env = {
                    "PATH": os.environ["PATH"],
                    "HOME": str(root / "home"),
                    "MISE_DATA_DIR": str(root / "data"),
                    "MISE_CACHE_DIR": str(root / "cache"),
                    "MISE_CONFIG_DIR": str(root / "config"),
                    "MISE_TRUSTED_CONFIG_PATHS": str(target),
                    "MISE_AUTO_INSTALL": "false",
                }
                baseline = subprocess.run(
                    [binary, "ls"], cwd=target, env=env, capture_output=True, check=False
                )
                assert baseline.returncode != 0
                if "PR_ONLY_ENV" in config:
                    assert b"mise version 2099.1.1 is required" in baseline.stderr
                cwd = (
                    runner_temp
                    if inputs.get("working_directory") == "${{ runner.temp }}"
                    else target
                )
                commands = [["--version"], ["ls"]]
                if inputs.get("install", "true") != "false":
                    commands.insert(1, ["install"])
                if inputs.get("env", "true") != "false":
                    commands.append(["env", "--json"])
                for command in [*commands, ["env", "--json"]]:
                    result = subprocess.run(
                        [binary, *command], cwd=cwd, env=env, capture_output=True, check=False
                    )
                    assert result.returncode == 0, result.stderr.decode()
                    if command == ["env", "--json"]:
                        assert "PR_ONLY_ENV" not in json.loads(result.stdout)
                assert pin.read_text() == config

    def test_old_target_uses_trusted_pin_and_rejects_invalid_config(self):
        """Run the workflow reader when the target has no bootstrap files."""
        read = _step("Read mise pin")
        command = re.search(r"^        run: (.+)$", read, re.MULTILINE)
        assert command is not None
        for config in (CONFIG, CONFIG.replace(DIGEST, "short"), CONFIG.replace(DIGEST, "")):
            with self.subTest(config=config), tempfile.TemporaryDirectory() as directory:
                target = Path(directory)
                trusted = target / ".mise-bootstrap"
                helper = trusted / ".github/scripts/mise-pin.sh"
                helper.parent.mkdir(parents=True)
                shutil.copyfile(ROOT / ".github/scripts/mise-pin.sh", helper)
                (trusted / "mise.toml").write_text(config)
                environment = target / "github-env"
                assert not (target / "mise.toml").exists()
                assert not (target / ".github").exists()
                result = subprocess.run(
                    ["bash", "-e", "-c", command.group(1)],
                    cwd=target,
                    env={
                        **os.environ,
                        "RUNNER_OS": "Linux",
                        "RUNNER_ARCH": "X64",
                        "GITHUB_ENV": str(environment),
                    },
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if config == CONFIG:
                    assert result.returncode == 0, result.stderr
                    assert (
                        environment.read_text() == f"MISE_VERSION={VERSION}\nMISE_SHA256={DIGEST}\n"
                    )
                else:
                    assert result.returncode != 0
                    assert environment.read_text() == ""


if __name__ == "__main__":
    unittest.main()
