"""Keep completed files on disk while a step result is saved."""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LocalFile:
    """A finished file on disk to keep with a step, byte for byte, without encoding it.

    Use one as a value in ``StepCompleted.files``, beside ``bytes`` values::

        StepCompleted(files={"output.mp4": LocalFile(path), "notes.json": b"{}"})

    The constructor checks only that ``path`` is a :class:`pathlib.Path`; it
    does not open the file. The runtime takes the file over when the step is
    reported: it opens the path without following a symbolic link, requires a
    regular file, copies it in 64 KiB chunks into a directory of its own, and
    checks that its size and modification time did not change during the copy.
    It never moves or deletes the original, so removing it is the model's job.
    When the model may do so depends on how the step is reported:

    * With ``complete_step()``, the copy is made before the call returns.
      Remove the file once it returns.
    * Returned from ``process_output()`` on the default step loop, the copy is
      made after the method returns, when the loop reports the step. The file
      must still exist then, so a ``TemporaryDirectory`` opened inside
      ``process_output()`` is already gone. Keep the file in a directory the
      model owns, and remove it on a later step or in ``@session_ended``.

    The copy runs on a worker thread, so the model's event loop keeps running,
    and the call that reports the step waits until it ends. The runtime puts no
    limit on the file's size; bound what the model hands over. A step reported
    with ``error`` set never opens its files.

    Handing a file over does not promise it is stored: with step results off,
    or when a step is not saved, the runtime deletes its copy at once.

    Attributes:
        path: The file to keep. A symbolic link is refused.

    Raises:
        TypeError: ``path`` is not a :class:`pathlib.Path`, when constructed.
        OSError: The file cannot be opened or read, when the step is reported.
        ValueError: The file is not a regular file, or it changed during the
            copy, when the step is reported. ``complete_step()`` raises either
            error to its caller, which can report a failed step instead. On the
            default step loop the step is reported as failed, with the reason
            in its ``error``, and the session goes on.
    """

    path: Path

    def __post_init__(self) -> None:
        """Require a local path, leaving file access to the step handoff."""
        if not isinstance(self.path, Path):
            raise TypeError("LocalFile.path must be a pathlib.Path")


class OwnedLocalFile:
    """Hold a private snapshot until the step store saves or discards it."""

    def __init__(self, source: LocalFile) -> None:
        """Copy a regular source into a private directory using a 64 KiB buffer."""
        self._directory = tempfile.TemporaryDirectory(prefix="reactor-step-file-")
        self.path = Path(self._directory.name) / "file"
        try:
            fd = os.open(source.path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb") as reader:
                before = os.fstat(reader.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise ValueError("LocalFile must refer to a regular file")
                with self.path.open("wb") as writer:
                    shutil.copyfileobj(reader, writer, length=64 * 1024)
                after = os.fstat(reader.fileno())
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError("LocalFile changed during handoff")
                self.size = after.st_size
        except BaseException:
            self.close()
            raise

    def move_to(self, destination: Path) -> None:
        """Move the owned snapshot into a result folder without encoding it."""
        shutil.move(self.path, destination)
        self.close()

    def close(self) -> None:
        """Remove the owned snapshot, also when called more than once."""
        self._directory.cleanup()


def adopt_files(files: Mapping[str, bytes | LocalFile]) -> dict[str, bytes | OwnedLocalFile]:
    """Snapshot local files before the report leaves the model thread."""
    adopted: dict[str, bytes | OwnedLocalFile] = {}
    try:
        for name, value in files.items():
            adopted[name] = OwnedLocalFile(value) if isinstance(value, LocalFile) else value
    except BaseException:
        discard_files(adopted)
        raise
    return adopted


def discard_files(files: Mapping[str, bytes | OwnedLocalFile]) -> None:
    """Release all owned snapshots in a saved or rejected report."""
    for value in files.values():
        if isinstance(value, OwnedLocalFile):
            value.close()
