"""Every ``reactor_runtime`` name a skill cites still exists.

Skills under ``skills/`` teach agents the runtime by naming its public symbols
(``reactor_runtime.get_weights_path``, ``reactor_runtime.ReactorApp.process_output``).
A rename or a removal that leaves a skill naming the old symbol fails here, so
the PR that makes the statement wrong also updates the skill. A name a skill
cites only to say it is gone is listed in ``_GONE`` and must stay unimportable.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[2]
_CITATION = re.compile(r"\breactor_runtime(?:\.[A-Za-z_][A-Za-z0-9_]*)+")
_GONE = {
    # porting-models-to-standalone-runtime: the profiler module was removed.
    "reactor_runtime.profiling",
}
_FIX = "a skill cites this name; update the skill in the same PR (see AGENTS.md, Skills)"


def _cited() -> list[str]:
    names = {
        name.rstrip(".")
        for skill in sorted((_ROOT / "skills").rglob("*.md"))
        for name in _CITATION.findall(skill.read_text())
    }
    return sorted(names - _GONE)


def _resolve(dotted: str) -> object:
    parts = dotted.split(".")
    for split in range(len(parts), 0, -1):
        try:
            target: object = importlib.import_module(".".join(parts[:split]))
        except ModuleNotFoundError:
            continue
        for attribute in parts[split:]:
            target = getattr(target, attribute)
        return target
    raise ModuleNotFoundError(dotted)


def test_skills_cite_some_runtime_names() -> None:
    # Guards the scan itself: an empty list would pass every case below.
    assert len(_cited()) >= 10


@pytest.mark.parametrize("name", _cited())
def test_cited_name_resolves(name: str) -> None:
    try:
        _resolve(name)
    except (ModuleNotFoundError, AttributeError) as exc:
        pytest.fail(f"{name} does not resolve ({exc}); {_FIX}")


@pytest.mark.parametrize("name", sorted(_GONE))
def test_name_cited_as_gone_stays_gone(name: str) -> None:
    with pytest.raises((ModuleNotFoundError, AttributeError)):
        _resolve(name)


@pytest.mark.parametrize("name", _cited())
def test_cited_name_is_public(name: str) -> None:
    private = [part for part in name.split(".") if part.startswith("_")]
    assert not private, (
        f"{name} names a private symbol; cite the public one whose docstring states the fact"
    )
