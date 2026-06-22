from pathlib import Path

import pytest

from examples.passthrough import Passthrough
from reactor_runtime.core import RuntimeConfig
from reactor_runtime.http import HttpServer
from reactor_runtime.model import ModelContract
from reactor_runtime.runner import Runner
from reactor_runtime.serve import _assemble, _load_config, _version, main

_MANIFEST = """\
model:
  name: demo
  version: 0.1.0
runtime:
  import: pipeline:Demo
  config: config.yml
"""


def test_assemble_hooks_on_runner_then_http() -> None:
    service = _assemble(RuntimeConfig(model_ref="fake:Model"))

    components = service._components
    assert set(components) == {"runner", "http"}
    assert isinstance(components["runner"], Runner)
    assert isinstance(components["http"], HttpServer)
    assert components["http"].depends_on == ("runner",)


def test_assemble_wires_the_runner_shutdown_to_the_service() -> None:
    service = _assemble(RuntimeConfig(model_ref="fake:Model"))

    runner = service._components["runner"]
    assert isinstance(runner, Runner)
    assert runner.request_shutdown == service.request_shutdown


def test_version_is_a_non_empty_string() -> None:
    assert isinstance(_version(), str)
    assert _version()


def test_load_config_reads_the_model_reference_from_runtime_import(tmp_path: Path) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text(_MANIFEST)

    cfg = _load_config(manifest)

    assert cfg.model_ref == "pipeline:Demo"


def test_load_config_refuses_a_manifest_without_runtime_import(tmp_path: Path) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text("model:\n  name: demo\n")

    with pytest.raises(SystemExit):
        _load_config(manifest)


def test_main_refuses_when_no_manifest_in_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit):
        main()


def test_example_model_has_a_valid_contract() -> None:
    contract = ModelContract.of(Passthrough)

    assert "video" in contract.tracks
    assert contract.tracks["video"].direction.value == "out"
    assert "set_brightness" in contract.commands


def test_example_model_loads() -> None:
    model = Passthrough()
    model.load({"brightness": 200})

    assert model._brightness == 200
