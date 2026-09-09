"""The echo example: loop the client's webcam and mic back with a video effect.

Run it from this directory with ``reactor build && reactor run`` — the CLI
builds the image from the ``build:`` block in ``reactor.yaml`` and serves the
model it names.
"""

from examples.echo.echo import Echo

__all__ = ["Echo"]
