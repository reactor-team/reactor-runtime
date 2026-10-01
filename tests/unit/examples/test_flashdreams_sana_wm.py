"""The flashdreams-sana-wm workspace: two YAML files and a requirements file, no Python.

The family's behaviour is tested in ``tests/unit/flashdreams``; this checks the
workspace itself, without FlashDreams installed: the manifest resolves to the
family class the runtime ships, the config names the SANA-WM slug and the
adapter's example data, every FlashDreams package pins the same commit, and
the requirements carry what the SANA-WM package imports, since the package
itself is installed without its declared dependencies.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from reactor_runtime.flashdreams import FlashDreamsOutput, RolloutRestarted
from reactor_runtime.flashdreams.cam2v import Cam2V
from reactor_runtime.manifest import import_model_class, load_config

_EXAMPLE_DIR = Path(__file__).parents[3] / "examples" / "flashdreams-sana-wm"
_LINGBOT_DIR = Path(__file__).parents[3] / "examples" / "flashdreams-lingbot"
_COMMIT = re.compile(r"github\.com/NVIDIA/flashdreams@([0-9a-f]+)#subdirectory=")


@pytest.fixture(autouse=True)
def _seed_registries(
    isolate_interface_registries: None,
    register_model: Callable[[type], None],
    register: Callable[..., None],
) -> None:
    register_model(Cam2V)
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
    assert cfg.model_ref == "reactor_runtime.flashdreams.cam2v:Cam2V"
    assert import_model_class(cfg.model_ref) is Cam2V
    assert cfg.config_path == _EXAMPLE_DIR / "config.yml"


def test_the_config_names_the_sana_wm_slug_and_the_adapters_example_data() -> None:
    config = yaml.safe_load((_EXAMPLE_DIR / "config.yml").read_text())
    assert config["application"] == "cam2v-sana-wm-streaming"
    assert config["example_data"] is True
    # SANA-WM ships five examples.
    assert config["example_idx"] in range(5)
    assert set(config) == {"application", "example_data", "example_idx", "warmup_steps"}


def test_every_flashdreams_package_pins_the_commit_the_lingbot_example_pins() -> None:
    requirements = (_EXAMPLE_DIR / "requirements.txt").read_text()
    manifest = yaml.safe_load((_EXAMPLE_DIR / "reactor.yaml").read_text())
    run_steps = " ".join(manifest["build"]["run"])
    commits = set(_COMMIT.findall(requirements)) | set(_COMMIT.findall(run_steps))
    assert len(commits) == 1, commits
    # The two FlashDreams examples serve the same FlashDreams commit, so one
    # bump moves both.
    lingbot = (_LINGBOT_DIR / "requirements.txt").read_text()
    assert commits == set(_COMMIT.findall(lingbot))
    assert "flashdreams[runners] @" in requirements
    assert "--no-deps" in run_steps
    assert "subdirectory=apps/cam2v" in run_steps
    assert "subdirectory=integrations_v2/sana_wm" in run_steps


def test_the_requirements_carry_what_the_sana_wm_package_imports() -> None:
    """flashdreams-sana-wm installs without its dependencies; these are them."""
    names = {
        re.split(r"[\[ >=<@]", line, maxsplit=1)[0]
        for line in (_EXAMPLE_DIR / "requirements.txt").read_text().splitlines()
        if line and not line.startswith("#")
    }
    assert {"accelerate", "diffusers", "sentencepiece", "torchvision", "pillow"} <= names


def test_the_build_block_carries_flashdreams_requirements() -> None:
    build = yaml.safe_load((_EXAMPLE_DIR / "reactor.yaml").read_text())["build"]
    assert build["cuda_version"].startswith("13.")
    assert "git" in build["system_packages"]
    assert build["build_env"]["UV_INDEX_STRATEGY"] == "unsafe-best-match"
