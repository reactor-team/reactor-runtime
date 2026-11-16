"""The contract module, and the import boundary the model half keeps.

The model half runs behind a ``DistributedRunner`` and in a notebook with only
FlashDreams installed, so it must never import the runtime's authoring surface.
The check reads the modules' import statements rather than the loaded module
graph, so it holds whatever else the test process has imported.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from reactor_runtime.flashdreams import (
    FlashDreamsResult,
    PromptSwapUnsupported,
    RolloutExhausted,
    RolloutNotStarted,
)

_PACKAGE = Path(__file__).parents[3] / "src" / "reactor_runtime" / "flashdreams"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def test_the_model_half_imports_nothing_from_the_interface() -> None:
    for module in ("contract.py", "model.py"):
        names = _imports(_PACKAGE / module)
        offending = {name for name in names if name.startswith("reactor_runtime.interface")}
        assert not offending, f"{module} imports {sorted(offending)}"
        assert "reactor_runtime" not in names, f"{module} imports the top-level package"


def test_the_model_half_imports_flashdreams_and_torch_inside_functions_only() -> None:
    tree = ast.parse((_PACKAGE / "model.py").read_text())
    top_level = {
        alias.name.partition(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.partition(".")[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert top_level.isdisjoint({"flashdreams", "torch", "loguru", "huggingface_hub"})


def test_the_result_is_a_frozen_dataclass_of_plain_data() -> None:
    frames = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    result = FlashDreamsResult(frames=frames, index=3, rollout_id=2)
    assert result.frames is frames
    assert (result.index, result.rollout_id) == (3, 2)
    try:
        result.index = 4  # type: ignore[ty:invalid-assignment]
    except AttributeError:
        pass
    else:
        raise AssertionError("FlashDreamsResult must be frozen")


def test_the_three_errors_are_distinct_plain_exceptions() -> None:
    errors = (RolloutNotStarted, RolloutExhausted, PromptSwapUnsupported)
    for error in errors:
        assert issubclass(error, Exception)
        assert not issubclass(error, tuple(e for e in errors if e is not error))
    assert RolloutExhausted(7).args == (7,)
