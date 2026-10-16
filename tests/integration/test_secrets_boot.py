"""Boot test: managed secrets reach the environment before the model module imports.

The test runs the real :func:`reactor_runtime.serve.main` in a child
interpreter, from a temporary workspace with a ``reactor.yaml`` and a model
module. The child points the mount root at a temporary directory with the
layout that the deployment platform mounts, then calls ``main()``. At import
time, the model module records what the environment holds. Its ``load()`` then
raises, so the runtime stops by itself.

Requires the native ``reactor_webrtc`` wheel, because ``main()`` mounts the
libwebrtc engine; the module skips cleanly when it is absent.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("reactor_webrtc")

from reactor_runtime.secrets import EXPECTED_KEYS_ENV

_API_TOKEN = "synth-boot-café-0001\n".encode()
_SECOND = b"synth-boot-second-0002\n"
_IMAGE_DEFAULT = "synth-image-default-0000"
_BOOT_TIMEOUT_S = 90

_DRIVER = """\
import sys
from pathlib import Path

from reactor_runtime import secrets, serve

secrets.SECRETS_ROOT = Path(sys.argv[1])
serve.main()
"""

_MODEL = """\
import base64
import json
import os
from pathlib import Path

from reactor_runtime import ReactorApp

KEYS = ("API_TOKEN", "SECOND")
RECORD = Path(os.environ["BOOT_RECORD"])

RECORD.write_text(
    json.dumps(
        {
            key: base64.b64encode(value).decode()
            for key in KEYS
            if (value := os.environb.get(key.encode())) is not None
        }
    )
)


class Model(ReactorApp):
    def load(self, config_path):
        raise RuntimeError("the record is written; refuse to load")
"""


def _mount(root: Path, files: dict[str, bytes]) -> Path:
    """Build the platform layout: ``..data -> ..<timestamp>``, one symlink per key."""
    stamp = root / "..2026_10_13_04_00_00.123456789"
    stamp.mkdir(parents=True)
    for name, data in files.items():
        (stamp / name).write_bytes(data)
    (root / "..data").symlink_to(stamp.name)
    for name in files:
        (root / name).symlink_to(f"..data/{name}")
    return root


def _boot(tmp_path: Path, mount: Path, expected: list[str]) -> subprocess.CompletedProcess[str]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "reactor.yaml").write_text("runtime:\n  import: model:Model\n")
    (workspace / "model.py").write_text(_MODEL)
    (workspace / "driver.py").write_text(_DRIVER)
    env = dict(os.environ)
    env.pop("SECOND", None)
    env.update(
        {
            EXPECTED_KEYS_ENV: json.dumps(expected),
            "API_TOKEN": _IMAGE_DEFAULT,
            "BOOT_RECORD": str(tmp_path / "record.json"),
            "HOST": "127.0.0.1",
            "PORT": "0",
        }
    )
    return subprocess.run(
        [sys.executable, str(workspace / "driver.py"), str(mount)],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        timeout=_BOOT_TIMEOUT_S,
    )


def _record(tmp_path: Path, result: subprocess.CompletedProcess[str]) -> dict[str, bytes]:
    path = tmp_path / "record.json"
    assert path.is_file(), f"the model module never imported\n{result.stderr}"
    return {key: base64.b64decode(value) for key, value in json.loads(path.read_text()).items()}


def test_the_exact_bytes_reach_the_model_before_import(tmp_path: Path) -> None:
    mount = _mount(tmp_path / "secrets", {"API_TOKEN": _API_TOKEN, "SECOND": _SECOND})

    result = _boot(tmp_path, mount, ["API_TOKEN", "SECOND"])

    assert result.returncode == 0, result.stderr
    # The mounted value replaced the image default, byte for byte.
    assert _record(tmp_path, result) == {"API_TOKEN": _API_TOKEN, "SECOND": _SECOND}
    assert _API_TOKEN.decode() not in result.stdout + result.stderr
