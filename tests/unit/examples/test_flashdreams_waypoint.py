"""The flashdreams-waypoint workspace: two YAML files and a requirements file, no Python.

The model's behaviour is tested in ``tests/unit/flashdreams``; this checks the
workspace itself, without FlashDreams installed: the manifest resolves to the
family class the runtime ships, its schema renders, the config names an
action2v slug, and every FlashDreams package pins the same commit.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from reactor_runtime.flashdreams import FlashDreamsOutput, RolloutRestarted
from reactor_runtime.flashdreams.action2v import Action2V
from reactor_runtime.interface.model.contract import ModelContract
from reactor_runtime.manifest import import_model_class, load_config

_EXAMPLE_DIR = Path(__file__).parents[3] / "examples" / "flashdreams-waypoint"
_COMMIT = re.compile(r"github\.com/NVIDIA/flashdreams@([0-9a-f]+)#subdirectory=")


@pytest.fixture(autouse=True)
def _seed_registries(
    isolate_interface_registries: None,
    register_model: Callable[[type], None],
    register: Callable[..., None],
) -> None:
    register_model(Action2V)
    register(FlashDreamsOutput, RolloutRestarted)


def test_the_workspace_has_no_python() -> None:
    assert sorted(p.name for p in _EXAMPLE_DIR.iterdir()) == [
        "README.md",
        "config.yml",
        "reactor.yaml",
        "requirements.txt",
    ]


def test_the_manifest_resolves_to_the_family_class_the_runtime_ships() -> None:
    cfg = load_config(_EXAMPLE_DIR / "reactor.yaml")
    assert cfg.model_ref == "reactor_runtime.flashdreams.action2v:Action2V"
    assert import_model_class(cfg.model_ref) is Action2V
    assert cfg.config_path == _EXAMPLE_DIR / "config.yml"


def test_the_schema_renders_the_family_surface() -> None:
    schema = ModelContract.of(Action2V).render_schema()
    assert set(schema.commands) == {
        "set_image",
        "set_keys",
        "move",
        "set_seed",
        "set_paused",
        "reset",
    }
    assert list(schema.tracks) == ["main_video"]
    assert "rollout_restarted" in schema.messages


def test_the_config_names_an_action2v_slug_and_the_family_keys() -> None:
    config = yaml.safe_load((_EXAMPLE_DIR / "config.yml").read_text())
    assert config["application"].startswith("action2v-")
    assert config["example_image"] is True
    assert isinstance(config["warmup_steps"], int)
    assert set(config) == {"application", "example_image", "warmup_steps"}


def test_every_flashdreams_package_pins_the_same_commit() -> None:
    requirements = (_EXAMPLE_DIR / "requirements.txt").read_text()
    manifest = yaml.safe_load((_EXAMPLE_DIR / "reactor.yaml").read_text())
    run_steps = " ".join(manifest["build"]["run"])
    commits = set(_COMMIT.findall(requirements)) | set(_COMMIT.findall(run_steps))
    assert len(commits) == 1, commits
    # Core with its dependencies from requirements; the family and model
    # packages without theirs from build.run, which is where --no-deps lives.
    assert "subdirectory=flashdreams" in requirements
    assert "--no-deps" in run_steps
    assert "subdirectory=apps/action2v" in run_steps
    assert "subdirectory=integrations_v2/waypoint" in run_steps
    assert "reactor-runtime" not in requirements.replace("reactor-runtime itself", "")


def test_the_build_block_carries_flashdreams_requirements() -> None:
    build = yaml.safe_load((_EXAMPLE_DIR / "reactor.yaml").read_text())["build"]
    assert build["cuda_version"].startswith("13.")
    assert "git" in build["system_packages"]
    assert build["build_env"]["UV_INDEX_STRATEGY"] == "unsafe-best-match"
