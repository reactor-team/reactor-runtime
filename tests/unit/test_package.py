"""Package-level smoke tests."""

import reactor_runtime


def test_version_is_exposed() -> None:
    assert reactor_runtime.__version__
