"""Boot tests: managed secrets reach the environment before the model module imports.

Each test runs a real runtime entry point in a child interpreter: the session
runtime's :func:`reactor_runtime.serve.main` and the model server's ``main()``.
The model module records ``API_TOKEN`` at import time, and the process then
stops by itself. Requires the native ``reactor_webrtc`` wheel, because
``serve.main()`` mounts the libwebrtc engine.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("reactor_webrtc")

from reactor_runtime.secrets import EXPECTED_KEYS_ENV

_API_TOKEN = "synth-boot-café-0001\n".encode()

_DRIVER = """\
import importlib
import sys
from pathlib import Path

from reactor_runtime import secrets

secrets.SECRETS_ROOT = Path(sys.argv[1])
importlib.import_module(sys.argv[2]).main()
"""

_MODEL = """\
import os
from pathlib import Path

from reactor_runtime import ReactorApp

Path(os.environ["BOOT_RECORD"]).write_bytes(os.environb[b"API_TOKEN"])


class Model(ReactorApp):
    def load(self, config_path):
        raise RuntimeError("the record is written; refuse to load")
"""

_SERVER_MANIFEST = """\
model:
  resources:
    workers:
      shared_memory:
        input: 64Ki
        output: 64Ki
runtime:
  import: model:Model
  model_server:
    import: model:Model
"""


def _boot(
    tmp_path: Path, entry: str, manifest: str, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    mount = tmp_path / "secrets"
    mount.mkdir()
    (mount / "API_TOKEN").write_bytes(_API_TOKEN)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "reactor.yaml").write_text(manifest)
    (workspace / "model.py").write_text(_MODEL)
    (workspace / "driver.py").write_text(_DRIVER)
    env = os.environ | env
    env |= {
        EXPECTED_KEYS_ENV: '["API_TOKEN"]',
        "API_TOKEN": "synth-image-default-0000",
        "BOOT_RECORD": str(tmp_path / "record"),
        "HOST": "127.0.0.1",
    }
    return subprocess.run(
        [sys.executable, str(workspace / "driver.py"), str(mount), entry],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )


def test_the_exact_bytes_reach_the_model_before_import(tmp_path: Path) -> None:
    manifest = "runtime:\n  import: model:Model\n"

    result = _boot(tmp_path, "reactor_runtime.serve", manifest, {"PORT": "0"})

    assert result.returncode == 0, result.stderr
    # The mounted value replaced the image default, byte for byte.
    assert (tmp_path / "record").read_bytes() == _API_TOKEN
    assert _API_TOKEN.decode() not in result.stdout + result.stderr


def test_the_exact_bytes_reach_the_model_server_before_import(tmp_path: Path) -> None:
    (tmp_path / "s0").mkdir()
    env = {
        "REACTOR_MAX_SESSIONS": "1",
        "REACTOR_SHM_SLOT_DIR": str(tmp_path / "s{slot}"),
        "REACTOR_MODEL_SERVER_SOCKET": str(tmp_path / "socket"),
        "REACTOR_MODEL_SERVER_HEALTH_PORT": "0",
    }

    result = _boot(tmp_path, "reactor_runtime.model_server.__main__", _SERVER_MANIFEST, env)

    # The model is not a model server, so the process stops after the import.
    assert (tmp_path / "record").read_bytes() == _API_TOKEN, result.stderr
    assert _API_TOKEN.decode() not in result.stdout + result.stderr
