"""Decoding the image a client uploads to start a world from.

Every family that starts a rollout from an image takes the upload through
:func:`as_rgb_png`: the declared size is checked before any pixel is decoded,
the EXIF orientation is applied, and the image is kept as an RGB PNG at its
own size, so the adapter's own loader, which sizes it for the model, sees a
mode it accepts whatever the client sent.

Pillow is a dependency of the workspace, not of the runtime: the FlashDreams
model packages depend on it, and a workspace that serves a family lists it.
"""

from __future__ import annotations

import io

MAX_SEED_PIXELS = 4096 * 4096
"""The most pixels a seed upload may declare.

The adapter sizes the seed to the model's frame itself, so nothing is gained
above this, and the bound is what keeps a small file that declares a huge
image from allocating hundreds of megabytes when it is decoded.
"""


class ImageTooLargeError(ValueError):
    """The upload declares more pixels than :data:`MAX_SEED_PIXELS`."""


def as_rgb_png(data: bytes) -> bytes:
    """Decode an upload, apply its EXIF orientation, and re-encode it as an RGB PNG.

    Args:
        data: The uploaded bytes, PNG or JPEG.

    Returns:
        The same image as an RGB PNG, at its own size.

    Raises:
        ImageTooLargeError: The header declares more than :data:`MAX_SEED_PIXELS`
            pixels. Raised before any pixel is decoded.
        ImportError: Pillow is not installed.
        Exception: Whatever Pillow raises for a file it cannot decode.
    """
    from PIL import Image, ImageOps  # a dependency of the workspace, not of the runtime

    with Image.open(io.BytesIO(data)) as decoded:
        width, height = decoded.size
        if width * height > MAX_SEED_PIXELS:
            raise ImageTooLargeError(
                f"The image is {width}x{height}; a seed image may hold at most "
                f"{MAX_SEED_PIXELS} pixels."
            )
        image = ImageOps.exif_transpose(decoded).convert("RGB")
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()
