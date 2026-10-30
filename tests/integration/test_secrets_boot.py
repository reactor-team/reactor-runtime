"""Boot test: managed secrets reach the environment before the model module imports.

The test runs the real :func:`reactor_runtime.serve.main` in a child interpreter.
The model module records ``API_TOKEN`` at import time and then refuses to load,
so the runtime stops by itself. Requires the native ``reactor_webrtc`` wheel,
because ``main()`` mounts the libwebrtc engine.
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
import sys
from pathlib import Path

from reactor_runtime import secrets, serve

secrets.SECRETS_ROOT = Path(sys.argv[1])
serve.main()
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


def test_the_exact_bytes_reach_the_model_before_import(tmp_path: Path) -> None:
    mount = tmp_path / "secrets"
    mount.mkdir()
    (mount / "API_TOKEN").write_bytes(_API_TOKEN)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "reactor.yaml").write_text("runtime:\n  import: model:Model\n")
    (workspace / "model.py").write_text(_MODEL)
    (workspace / "driver.py").write_text(_DRIVER)
    record = tmp_path / "record"
    env = os.environ | {
        EXPECTED_KEYS_ENV: '["API_TOKEN"]',
        "API_TOKEN": "synth-image-default-0000",
        "BOOT_RECORD": str(record),
        "HOST": "127.0.0.1",
        "PORT": "0",
    }

    result = subprocess.run(
        [sys.executable, str(workspace / "driver.py"), str(mount)],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert result.returncode == 0, result.stderr
    # The mounted value replaced the image default, byte for byte.
    assert record.read_bytes() == _API_TOKEN
    assert _API_TOKEN.decode() not in result.stdout + result.stderr
