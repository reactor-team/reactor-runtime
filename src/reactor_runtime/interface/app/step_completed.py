"""What one finished step produced, :class:`StepCompleted`."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from reactor_runtime.interface.tracks import Output

RESERVED_FILE_NAMES = frozenset({"result.json", "output.mp4"})
"""File names the runtime writes into a step result itself."""


@dataclass(frozen=True)
class StepCompleted:
    """The report that one step of the model is done.

    The default step loop reports each step that ran, through
    :meth:`ReactorCore.complete_step`, and ``process_output()`` may return one to
    write that report itself; a model with its own ``run()`` calls that method
    itself each time it finishes a unit of work. The runtime counts these
    reports against a session's step limit, and can save each one as the step's
    result.

    Attributes:
        output: The media the step produced, or ``None`` when it produced none.
        files: Extra files to keep with the step, keyed by file name. A name is
            one path segment, and ``result.json`` and ``output.mp4`` are taken
            by the runtime.
        error: Why the step failed, or ``None`` when it worked.
        elapsed: Wall-clock seconds the step took, when measured.

    Raises:
        ValueError: A file name is empty, holds a path separator, is ``.`` or
            ``..``, or is reserved.
        TypeError: A file's contents are not ``bytes``.
    """

    output: Output | None = None
    files: Mapping[str, bytes] = field(default_factory=dict)
    error: str | None = None
    elapsed: float | None = None

    def __post_init__(self) -> None:
        """Check the extra files and keep a copy the caller cannot change."""
        for name, data in self.files.items():
            _check_file_name(name)
            if not isinstance(data, bytes):
                raise TypeError(f"step file {name!r} must be bytes, got {type(data).__name__}")
        object.__setattr__(self, "files", dict(self.files))


def _check_file_name(name: str) -> None:
    if not isinstance(name, str) or not name or name in {".", ".."}:
        raise ValueError(f"step file name {name!r} is not a file name")
    if "/" in name or "\\" in name:
        raise ValueError(f"step file name {name!r} must not contain a path separator")
    if name in RESERVED_FILE_NAMES:
        raise ValueError(f"step file name {name!r} is reserved for the runtime")
