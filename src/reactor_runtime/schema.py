"""Rendering a model's schema without serving it — the ``schema`` entry.

A code generator and a release pipeline both need the OpenAPI document a model
publishes, and neither can afford to stand a server up and ask for it over a
data channel. This module renders that document from the model class alone::

    python -m reactor_runtime.schema --version 1.4.0 --out schema.json

The model is imported, never loaded: the contract is assembled when the class is
created, so nothing here reads weights. The manifest is read by the same code
that boots the model, so the schema and the running process can never disagree
on which model a directory holds — and that reader lives in a module of its
own, so rendering a schema never imports the server or the transport stack.

The command keeps standard output clean so a caller can redirect it straight
into a file: whatever the model prints as it imports is rerouted to standard
error. The reroute catches Python-level writes; a native extension writing to
file descriptor 1 directly bypasses it, and ``--out`` sidesteps even that.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from reactor_runtime.interface.model import ModelContract
from reactor_runtime.manifest import MANIFEST, import_model_class, load_config

_DEFAULT_VERSION = "v0.0.0"

_VERSION_PATTERN = re.compile(r"v?\d+\.\d+\.\d+(?:-g[0-9a-f]+)?")

_VERSION_HINT = (
    "format 'X.Y.Z' or 'X.Y.Z-g<hex>', with an optional 'v' prefix "
    "(for example '1.4.0', 'v1.4.0', '1.4.0-gac767ec')"
)


def render(path: Path, version: str = _DEFAULT_VERSION) -> dict[str, Any]:
    """Render the OpenAPI document of the model the manifest in *path* names.

    Call this once per process: a model registers its messages and its tracks as
    it is declared, and those registrations outlive the call.

    Importing the model runs its module top to bottom, so whatever it prints or
    raises reaches the caller unchanged; the streams are left alone.

    Args:
        path: Directory that holds the ``reactor.yaml`` manifest.
        version: Release tag to stamp into ``info.version``, with or without a
            leading ``v``. The emitted tag always carries the ``v``.

    Returns:
        The model's schema as an OpenAPI 3.1 document.

    Raises:
        ValueError: If *version* is not a release tag.
        SystemExit: If the directory holds no manifest, or the manifest names no
            model — the same way the runtime's own manifest reader answers.
        Exception: Whatever importing the model's own module raises, with its
            traceback, which is what names the line that failed.
    """
    tag = _tag(version)
    manifest = path / MANIFEST
    if not manifest.is_file():
        raise SystemExit(f"no {MANIFEST} found in {path}")
    config = load_config(manifest)
    previous_path = list(sys.path)
    sys.path.insert(0, str(manifest.parent))
    try:
        model_cls = import_model_class(config.model_ref)
    finally:
        sys.path[:] = previous_path
    contract = ModelContract.of(model_cls)
    return contract.render_schema(tag, config.model_name).to_openapi()


def _tag(value: str) -> str:
    """Validate a release tag and return it with its leading ``v``.

    Raises:
        ValueError: If *value* is not a release tag.
    """
    if _VERSION_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{value!r} is not a release tag: {_VERSION_HINT}")
    return value if value.startswith("v") else f"v{value}"


def _release_tag(value: str | None) -> str:
    """Read the ``--version`` argument as a release tag.

    Raises:
        SystemExit: If the argument is not a release tag.
    """
    if value is None:
        return _DEFAULT_VERSION
    try:
        return _tag(value)
    except ValueError as error:
        raise SystemExit(f"--version {error}") from error


def main() -> None:
    """Render the schema of the model in a directory, and emit it.

    Raises:
        SystemExit: If an argument is malformed, the directory holds no
            manifest, or the manifest names no model.
        Exception: Whatever importing the model's own module raises.
    """
    parser = argparse.ArgumentParser(
        prog="python -m reactor_runtime.schema",
        description="Print or write the OpenAPI schema of a model.",
    )
    parser.add_argument(
        "--path",
        "-p",
        type=Path,
        default=None,
        help=f"directory that holds the model's {MANIFEST} (default: the working directory)",
    )
    parser.add_argument(
        "--version",
        "-v",
        default=None,
        help=(
            f"release tag to stamp into info.version — {_VERSION_HINT}. "
            f"The emitted tag always carries the 'v'. Default: {_DEFAULT_VERSION}"
        ),
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=None,
        help="write the JSON to this file instead of standard output, creating parent directories",
    )
    args = parser.parse_args()

    version = _release_tag(args.version)
    path = args.path.expanduser().resolve() if args.path else Path.cwd()
    # A model prints as it imports — a device banner, a deprecation notice, the
    # chatter of a library it pulls in. On standard output that lands ahead of
    # the document and leaves the JSON unparseable, so the import runs with
    # stdout rerouted to stderr. The reroute is the command's concern, not the
    # library's: a caller of render() keeps its own streams.
    with contextlib.redirect_stdout(sys.stderr):
        schema = render(path, version)
    document = json.dumps(schema, indent=2)
    if args.out is None:
        print(document)
        return
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"{document}\n")
    print(f"schema written to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
