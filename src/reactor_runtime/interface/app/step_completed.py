"""What one finished step produced, :class:`StepCompleted`."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from reactor_runtime.core.local_file import LocalFile
from reactor_runtime.interface.tracks import Output
from reactor_runtime.interface.tracks.output import encode_metadata

RESERVED_FILE_NAMES = frozenset({"result.json", "output.mp4"})
"""File names that hold a step result's own files.

The runtime writes ``result.json``, the step's manifest. ``output.mp4`` is the
step's video: the runtime encodes it from the step's frames, or a step with no
frames supplies it as a :class:`LocalFile`. :func:`_check_file_name` holds the
rule.
"""


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
        files: Files to keep with the step, as bytes or :class:`LocalFile`
            values, keyed by file name. A name is one path segment, and it
            sets the content type the file is served with. ``result.json`` is
            reserved. ``output.mp4`` takes a ``LocalFile`` only when ``output``
            is ``None``; with frames, the runtime encodes ``output.mp4`` itself.
            The runtime copies a ``LocalFile`` when the step is reported:
            before ``complete_step()`` returns, or after ``process_output()``
            returns on the default step loop. See :class:`LocalFile` for when
            the original may be removed, the copy's cost, and its errors.
        error: Why the step failed, or ``None`` when it worked.
        elapsed: Wall-clock seconds the step took, when measured.
        data: What the step returned besides its media, for the client that
            requested it: an action chunk, a score, a dict. Bytes are sent as
            they are, and a dict is encoded as compact UTF-8 JSON when the
            report is built, the same rule as a frame's metadata, so the
            attribute reads back as bytes. It travels in the step's result,
            in the same message as its frames, so a model whose output is data
            alone declares no output track. It counts toward the result's size:
            a result larger than the connection's largest message is answered
            ``RESULT_TOO_LARGE``, with no media and no data. A failed step
            sends none. The runtime does not save it with the step's result
            files; keep it in ``files`` too to have it there.

    Raises:
        ValueError: A file name is empty, holds a path separator, is ``.`` or
            ``..``, or is reserved for what the file holds, or ``data`` is a
            dict JSON cannot encode.
        TypeError: A file's contents are not ``bytes`` or ``LocalFile``, or
            ``data`` is neither ``bytes`` nor a dict.
    """

    output: Output | None = None
    files: Mapping[str, bytes | LocalFile] = field(default_factory=dict)
    error: str | None = None
    elapsed: float | None = None
    data: bytes | dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Check the extra files, keep a copy the caller cannot change, and encode the data."""
        for name, data in self.files.items():
            if not isinstance(data, bytes | LocalFile):
                raise TypeError(
                    f"step file {name!r} must be bytes or LocalFile, got {type(data).__name__}"
                )
            _check_file_name(name, data, has_output=self.output is not None)
        object.__setattr__(self, "files", dict(self.files))
        if self.data is None:
            return
        if not isinstance(self.data, bytes | dict):
            raise TypeError(f"step data must be bytes or a dict, got {type(self.data).__name__}")
        try:
            object.__setattr__(self, "data", encode_metadata(self.data))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"step data is not JSON-serialisable: {exc}") from exc


def _check_file_name(name: str, data: bytes | LocalFile, *, has_output: bool) -> None:
    """Refuse a step file name that is not one path segment, or that a step result holds itself.

    ``output.mp4`` is the one reserved name a model can fill: with a
    :class:`LocalFile`, in a step that has no frames for the runtime to encode.
    """
    if not isinstance(name, str) or not name or name in {".", ".."}:
        raise ValueError(f"step file name {name!r} is not a file name")
    if "/" in name or "\\" in name:
        raise ValueError(f"step file name {name!r} must not contain a path separator")
    if name == "output.mp4" and isinstance(data, LocalFile) and not has_output:
        return
    if name in RESERVED_FILE_NAMES:
        raise ValueError(f"step file name {name!r} is reserved for the runtime")
