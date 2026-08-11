"""Package-level smoke tests."""

import subprocess
import sys

import reactor_runtime


def test_version_is_exposed() -> None:
    assert reactor_runtime.__version__


def test_multi_gpu_surface_is_importable_from_the_top_level() -> None:
    # The supported surface is what the top-level package re-exports, so the
    # multi-GPU names have to be reachable without knowing the submodule.
    for name in (
        "WorkerGroup",
        "DistributedWorker",
        "SharedFrameBuffer",
        "WorkerError",
        "WorkerCrashed",
    ):
        assert hasattr(reactor_runtime, name), name
        assert name in reactor_runtime.__all__, name


def test_importing_the_package_does_not_pull_in_torch() -> None:
    # The controller side of a multi-GPU model runs torch-free: torch is
    # imported lazily, inside the functions that need it, so a model image
    # without torch can still import the runtime. Nothing in the source
    # structurally prevents a module-level `import torch` from creeping in
    # later, which is why this is a test and not a convention.
    probe = "import sys; import reactor_runtime; sys.exit(1 if 'torch' in sys.modules else 0)"
    result = subprocess.run([sys.executable, "-c", probe], check=False)
    assert result.returncode == 0, "importing reactor_runtime imported torch"
