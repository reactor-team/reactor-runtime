"""The brightness example: a generated, client-controllable gradient with audio.

Run it from this directory with ``reactor build && reactor run`` — the CLI
builds the image from the ``Dockerfile`` here and serves the model the
``reactor.yaml`` names.
"""

from examples.brightness.brightness import Brightness

__all__ = ["Brightness"]
