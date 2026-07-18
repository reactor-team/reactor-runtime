"""The echo example: loop the client's webcam and mic back with a video effect.

Run it from this directory with ``python -m reactor_runtime.serve`` — the
``reactor.yaml`` here points the runtime at the model.
"""

from examples.echo.echo import Echo

__all__ = ["Echo"]
