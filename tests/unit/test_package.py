"""Package-level smoke tests."""

import subprocess
import sys

import reactor_runtime


def test_version_is_exposed() -> None:
    assert reactor_runtime.__version__


def test_multi_gpu_surface_stays_off_the_package_root() -> None:
    # The worker primitives remain experimental;
    # a package-root export would promise a stable authoring contract.
    import reactor_runtime.distributed as distributed

    for name in (
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
    # Importability must not depend on the model image supplying torch.
    probe = (
        "import sys; import reactor_runtime; import reactor_runtime.distributed; "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    result = subprocess.run([sys.executable, "-c", probe], check=False)
    assert result.returncode == 0, "importing reactor_runtime imported torch"
