#!/usr/bin/env python3
# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Test the CI mise pin contract without an installed mise binary."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

HELPER = Path(__file__).with_name("mise-pin.sh")
VERSION = "2025.4.9"
HASHES = {
    "linux_x64": "1" * 64,
    "linux_arm64": "2" * 64,
    "macos_x64": "3" * 64,
    "macos_arm64": "4" * 64,
    "windows_x64": "5" * 64,
}
CONFIG = f'min_version = "{VERSION}"\n[_]\n' + "".join(
    f'mise_sha256_{platform} = "{digest}"\n' for platform, digest in HASHES.items()
)


class MisePinTests(unittest.TestCase):
    """Check the bootstrap environment contract for CI runners."""

    def _run_helper(self, config, runner_os="Linux", runner_arch="X64"):
        with tempfile.TemporaryDirectory(prefix="mise pin ") as directory:
            root = Path(directory)
            helper = root / ".github/scripts/mise-pin.sh"
            helper.parent.mkdir(parents=True)
            shutil.copyfile(HELPER, helper)
            (root / "mise.toml").write_bytes(config.encode())
            return subprocess.run(
                ["bash", str(helper)],
                cwd=root.parent,
                env={**os.environ, "RUNNER_OS": runner_os, "RUNNER_ARCH": runner_arch},
                capture_output=True,
                text=True,
                check=False,
            )

    def test_runner_platforms(self):
        """Select the checksum for each supported runner."""
        for runner_os, runner_arch, key in (
            ("Linux", "X64", "linux_x64"),
            ("Linux", "ARM64", "linux_arm64"),
            ("macOS", "X64", "macos_x64"),
            ("macOS", "ARM64", "macos_arm64"),
            ("Windows", "X64", "windows_x64"),
        ):
            with self.subTest(runner_os=runner_os, runner_arch=runner_arch):
                result = self._run_helper(CONFIG, runner_os, runner_arch)
                assert result.returncode == 0, result.stderr
                assert result.stdout == f"MISE_VERSION={VERSION}\nMISE_SHA256={HASHES[key]}\n"

    def test_comments_and_windows_line_endings(self):
        """Accept checkout line endings and ignore unrelated table keys."""
        config = CONFIG.replace(f'"{VERSION}"', f'"{VERSION}" # CI floor')
        config += '\n[tools]\nmin_version = "ignore this table"\n'
        result = self._run_helper(config.replace("\n", "\r\n"))
        assert result.returncode == 0, result.stderr
        assert result.stdout == f"MISE_VERSION={VERSION}\nMISE_SHA256={HASHES['linux_x64']}\n"

    def test_invalid_config_emits_no_environment(self):
        """Reject incomplete or invalid pins without partial output."""
        version = f'min_version = "{VERSION}"\n'
        digest = f'mise_sha256_linux_x64 = "{HASHES["linux_x64"]}"\n'
        configs = {
            "missing version": CONFIG.replace(version, ""),
            "empty version": CONFIG.replace(VERSION, ""),
            "duplicate version": version + CONFIG,
            "prefixed version": CONFIG.replace(VERSION, "v" + VERSION),
            "partial version": CONFIG.replace(VERSION, "2025.4"),
            "version injection": CONFIG.replace(VERSION, "2025.4.9; echo bad"),
            "unquoted version": CONFIG.replace(f'"{VERSION}"', VERSION),
            "missing checksum": CONFIG.replace(digest, ""),
            "empty checksum": CONFIG.replace(HASHES["linux_x64"], ""),
            "duplicate checksum": CONFIG + digest,
            "short checksum": CONFIG.replace(HASHES["linux_x64"], "a" * 63),
            "nonhex checksum": CONFIG.replace(HASHES["linux_x64"], "z" * 64),
            "misplaced checksum": CONFIG.replace("[_]\n", "[tools]\n"),
            "misplaced version": "[tools]\n" + CONFIG,
        }
        for label, config in configs.items():
            with self.subTest(label=label):
                result = self._run_helper(config)
                assert result.returncode != 0
                assert result.stdout == ""

    def test_unsupported_or_missing_runner(self):
        """Reject runners without a pinned binary."""
        for runner_os, runner_arch in (
            ("Windows", "ARM64"),
            ("Linux", "ARM"),
            ("", "X64"),
            ("Linux", ""),
        ):
            with self.subTest(runner_os=runner_os, runner_arch=runner_arch):
                result = self._run_helper(CONFIG, runner_os, runner_arch)
                assert result.returncode != 0
                assert result.stdout == ""


if __name__ == "__main__":
    unittest.main()
