"""Models that build a whole shot of video in one call and play it when a client asks.

Subclass :class:`ReactorDirectorApp` and write ``load()`` and ``generate()``::

    from reactor_runtime.apps.director import ReactorDirectorApp, Shot

The class reads ``generate()``'s parameters into the ``enqueue`` command, and
owns the queues, the continuations, playback, and the step each shot reports.
"""

from reactor_runtime.apps.director.app import ReactorDirectorApp
from reactor_runtime.apps.director.shots import Shot

__all__ = ["ReactorDirectorApp", "Shot"]
