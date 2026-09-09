"""The brightness example: a generated, client-controllable gradient with audio.

Run it from this directory with ``reactor build && reactor run`` — the CLI
builds the image from the ``build:`` block in ``reactor.yaml`` and serves the
model it names.
"""

from examples.brightness.brightness import Brightness

__all__ = ["Brightness"]
