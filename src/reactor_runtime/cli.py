"""The ``reactor-runtime`` command.

``reactor-runtime serve`` boots the runtime from the ``reactor.yaml`` in the
working directory, as ``python -m reactor_runtime.serve`` does. ``reactor-runtime
serve-worker`` serves the model's worker half, as ``python -m
reactor_runtime.model_worker`` does; see :mod:`reactor_runtime.model_worker.__main__`
for its options. Both serve until signalled.
"""

from __future__ import annotations

import sys

_USAGE = "usage: reactor-runtime {serve,serve-worker} [options]"


def main(argv: list[str] | None = None) -> None:
    """Dispatch to the subcommand *argv* names.

    Raises:
        SystemExit: With the subcommand's exit status, or 2 for an unknown one.
    """
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] in ("-h", "--help"):
        print(_USAGE)
        raise SystemExit(0 if args else 2)
    command, rest = args[0], args[1:]
    if command == "serve":
        from reactor_runtime.serve import main as serve

        serve()
        return
    if command == "serve-worker":
        from reactor_runtime.model_worker.__main__ import main as serve_worker

        raise SystemExit(serve_worker(rest))
    print(f"reactor-runtime: unknown command {command!r}\n{_USAGE}", file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
