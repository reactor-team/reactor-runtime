#!/usr/bin/env python3
# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Check that reviews bootstrap mise independently of the target PR files."""

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
