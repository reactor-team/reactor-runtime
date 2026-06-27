"""Local recording — the recorder and its clip-serving surface.

A standalone runtime records the model's output to local disk and serves the
clips straight back over HTTP. This package holds the :class:`Recorder` the
runner owns, the fMP4 chunk encoder it drives, and the marker bookkeeping behind
the clip math. There is no object store here: a director pulls the bytes from
``/clips`` and ships them onward.
"""

from reactor_runtime.recording.recorder import (
    ClipManifest,
    ClipReadyCallback,
    ClipResult,
    ClipSessionGoneError,
    Gone,
    NoMediaYetError,
    Pending,
    Recorder,
    RecorderDisabledError,
    RecorderError,
)

__all__ = [
    "ClipManifest",
    "ClipReadyCallback",
    "ClipResult",
    "ClipSessionGoneError",
    "Gone",
    "NoMediaYetError",
    "Pending",
    "Recorder",
    "RecorderDisabledError",
    "RecorderError",
]
