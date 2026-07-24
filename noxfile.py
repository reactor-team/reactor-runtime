"""Nox sessions for the reactor-runtime test suite.

mise pins nox (see mise.toml) and the ``test`` / ``test-matrix`` tasks drive it.
uv is the venv backend: nox asks uv to build each per-version environment and to
download the interpreter when it is missing. Parametrizing the session over
PYTHON_VERSIONS produces one dedicated session per interpreter (``tests-3.12``,
``tests-3.13``, ...), so a single version runs on its own and the whole matrix
runs by selecting them all.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import nox

# The interpreters the suite supports. Each becomes its own session
# (tests-<version>): the local matrix runs them all, and each CI matrix leg runs
# the single session for its version. CI declares the same version list in its
# workflow matrix, so a version added here is added there too.
PYTHON_VERSIONS = ["3.12", "3.13"]

nox.options.default_venv_backend = "uv"
# Reuse the per-version environments across runs: the locked `uv sync` in each
# session refreshes dependencies anyway, and this keeps a second local run from
# failing when uv declines to recreate an environment that already exists. CI
# runners start clean, so they always build fresh.
nox.options.reuse_existing_virtualenvs = True


def _gstreamer_env() -> dict[str, str]:
    """Return environment additions that let PyGObject find GStreamer.

    macOS strips ``DYLD_*`` from some subprocess chains, so PyGObject cannot load
    the Homebrew GStreamer libraries the transport tests import. Re-add
    Homebrew's library and typelib paths. Empty on Linux, where the loader finds
    them through the standard search paths.
    """
    if sys.platform != "darwin" or not shutil.which("brew"):
        return {}
    prefix = subprocess.run(
        ["brew", "--prefix"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return {
        "DYLD_LIBRARY_PATH": f"{prefix}/lib:{os.environ.get('DYLD_LIBRARY_PATH', '')}",
        "GI_TYPELIB_PATH": (
            f"{prefix}/lib/girepository-1.0:{os.environ.get('GI_TYPELIB_PATH', '')}"
        ),
    }


def _install_locked(session: nox.Session) -> None:
    """Generate the wire bindings and install the locked environment.

    The wire bindings must exist before the editable install maps them in;
    generate them from the in-repo proto (no released wheel or token needed),
    the packaged build vendors the pinned release instead.
    """
    session.run("mise", "run", "//proto:gen", external=True)
    session.run_install(
        "uv",
        "sync",
        "--locked",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Install the locked environment and run the unit suite under pytest.

    Integration tests live in their own ``integration`` session; this one is the
    fast, hermetic suite (unit and contract), so a push gate need not stand up a
    live peer connection.
    """
    _install_locked(session)
    session.run(
        "pytest", "-q", "--ignore=tests/integration", *session.posargs, env=_gstreamer_env()
    )


@nox.session(python=PYTHON_VERSIONS)
def integration(session: nox.Session) -> None:
    """Install the locked environment and run the integration suite under pytest.

    The integration tests negotiate real peer connections (e.g. the libwebrtc
    loopback), so they need the media backends the locked environment installs —
    notably the ``reactor_webrtc`` wheel — present rather than skipped.
    """
    _install_locked(session)
    session.run("pytest", "-q", "tests/integration", *session.posargs, env=_gstreamer_env())
