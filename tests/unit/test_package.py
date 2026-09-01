"""Package-level smoke tests."""

import subprocess
import sys

import reactor_runtime


def test_version_is_exposed() -> None:
    assert reactor_runtime.__version__


def test_multi_gpu_surface_stays_off_the_package_root() -> None:
    # The multi-GPU primitives are experimental: importable from
    # reactor_runtime.distributed, deliberately not re-exported from the
    # package root, so nothing hardens into the stable top-level surface
    # before the runtime-managed path exists. Re-exporting one of these
    # names at the root is a supported-API promise and must be a
    # deliberate decision, not a drive-by import.
    import reactor_runtime.distributed as distributed

    for name in (
        "WorkerGroup",
        "DistributedWorker",
        "SharedFrameBuffer",
        "WorkerError",
        "WorkerCrashed",
    ):
        assert hasattr(distributed, name), name
        assert name in distributed.__all__, name
        assert not hasattr(reactor_runtime, name), name
        assert name not in reactor_runtime.__all__, name


def test_importing_the_package_does_not_pull_in_torch() -> None:
    # The controller side of a multi-GPU model runs torch-free: torch is
    # imported lazily, inside the functions that need it, so a model image
    # without torch can still import the runtime. Nothing in the source
    # structurally prevents a module-level `import torch` from creeping in
    # later, which is why this is a test and not a convention. The probe
    # imports the distributed package explicitly now that the root no
    # longer re-exports it.
    probe = (
        "import sys; import reactor_runtime; import reactor_runtime.distributed; "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    result = subprocess.run([sys.executable, "-c", probe], check=False)
    assert result.returncode == 0, "importing reactor_runtime imported torch"
