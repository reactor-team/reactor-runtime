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
    """Supply one finished local file as a step result, without encoding it.

    Use a value in ``StepCompleted.files``, for example
    ``files={"output.mp4": LocalFile(path)}``. The file must be regular and
    closed by its writer. Keep it unchanged until ``complete_step()`` returns;
    the runtime copies it into owned staging before returning, so the caller
    can then remove the original. The copy uses bounded memory. Failed steps
    discard their files. Saving can still fail or be refused when disabled or
    full; handing over a file does not promise durable storage.

    Attributes:
        path: The path to a regular file. Symbolic links are refused.
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
